r"""
Sparse2Inverse for ultrafast ultrasound -- stage 2: self-supervised training
============================================================================

Self-supervised reconstruction from a *fast* plane-wave acquisition, trained
without ever seeing a clean image, and without the blur that Noise2Inverse
inherits from its targets. Run ``precompute_n2i_ultrasound.py`` first: it
beamforms each of the 87 transmits of every in-vivo frame into its own
single-angle image, demodulates it to complex baseband (IQ) and caches them.
This script reads the same cache as ``train_n2i_deepinv.py``.

Noise2Inverse :footcite:t:`hendriksen2020noise2inverse` splits the transmits in
two, feeds the network the sub-reconstruction of one half and asks it to predict
the sub-reconstruction of the other half. Its target is an *image*, so it carries
the whole point spread function of a few-transmit acquisition: the sidelobes and
the grating clutter of a sparse transmit sequence are not noise, they are a
deterministic function of the anatomy, identical in both halves, and the network
is therefore trained to keep them. Sparse2Inverse
:footcite:t:`gruber2024sparse2inverse` keeps the splitting but moves the loss to
the *measurement* domain: the network output is projected back to the channel
data of the held-out transmits and compared to what those transmits actually
recorded,

.. math::
    \mathcal{L}(\psi) = \sum_{t} \sum_{\Theta}
    \left\| A_{\Theta}\, \Phi_\psi\!\left( \tfrac{1}{|g \setminus \Theta|}
    \textstyle\sum_{a \in g \setminus \Theta} A_a^{\top} y_a \right)
    - y_{\Theta} \right\|^2 ,

so nothing fixes the null-space component of the reconstruction and the network
is free to sharpen, as long as it stays consistent with the data it has not
seen.

Sparse-view CT acquires all its angles at once; here the "sparse data" is one
fast acquisition of ``87 / K`` transmits, and every cached frame holds ``K``
disjoint ones. A training step draws a fast acquisition ``g``, holds out
``HELD_OUT`` of its transmits, feeds the network the sub-reconstruction of the
rest and pays the residual of the held-out transmits. At inference the estimate
is averaged over the leave-one-out inputs, as in Eq. (2) of the paper.

The forward operator is needed in the loop, but only on the held-out transmits,
and the cache is enough to evaluate the loss. Expanding the square,

.. math::
    \|A_\Theta u - y_\Theta\|^2 = \|A_\Theta u\|^2
    - 2 \langle u, A_\Theta^{\top} y_\Theta \rangle + \|y_\Theta\|^2 ,

the middle term only involves the cached single-angle images, whose sum over
:math:`\Theta` *is* :math:`A_\Theta^{\top} y_\Theta`, and the last one does not
depend on the network. The loss below is that expansion without
:math:`\|y_\Theta\|^2`, which the cache does not hold: it has the exact gradient
of the Sparse2Inverse loss but is shifted by a (negative) constant per frame, so
its value is only meaningful relative to another network on the same frames --
the identity is printed next to it for that reason. One application of :math:`A`
per step is all the physics that is left.

How much signal that loss carries is worth knowing before trusting it. On one
abdominal frame of this dataset, the best scalar multiple of a two-transmit
sub-reconstruction explains 0.6% of the energy of a held-out transmit's channel
data; the 87-transmit CPWC image explains 2.5%, and the held-out transmit's
*own* single-angle image 5.5% (10% when a depth-dependent gain is fitted too). A
linear, single-scattering, constant-speed-of-sound model of in-vivo tissue leaves
most of the record unexplained -- out-of-plane scattering, aberration,
attenuation and electronic noise are not in :math:`A` -- where filtered
backprojection accounts for nearly all of a CT sinogram. The loss therefore sits
on a large floor and moves in its fourth digit, which is why it is accumulated in
double precision and why only differences between reconstructions of the same
frames are reported.

Training, validation and test sets are whole folders of the cache, so the network
is validated and tested on volunteers it has never seen.
"""

import glob
import json
import math
import os
from pathlib import Path

import matplotlib

# Figures are only written to files: without this, matplotlib picks an
# interactive backend whenever DISPLAY is set, e.g. under ssh -X, and draws
# every figure through the X tunnel.
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

import deepinv as dinv
from n2i_models import AttentionResUNet, AttentionUNet, ResUNet

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script

# Network, as in ``train_n2i_deepinv.py``: "unet", "resunet", "attunet",
# "attresunet" or "drunet". Each architecture keeps its own checkpoints. The
# S2I_ARCH environment variable overrides it, to queue several networks:
#   for a in unet drunet; do S2I_ARCH=$a python ...; done
ARCH = os.environ.get("S2I_ARCH", "unet")
# Runs of a different data split go to their own folders, so RESUME = "auto"
# never continues a run trained on another split.
RUN_SUFFIX = "_test002"
SAVE_PATH = f"./checkpoints/s2i_ultrasound_{ARCH}{RUN_SUFFIX}"

# Restarting an interrupted run, as in ``train_n2i_deepinv.py``:
#   ""      train from scratch
#   "auto"  continue the most recent run of this architecture, from its latest
#           checkpoint
#   a path  continue from that checkpoint file
# Weights, optimiser, scheduler, epoch counter, loss histories and the
# reflectivity scale of the loss are restored, so EPOCHS stays the *total*
# number of epochs and the loss keeps the same definition.
RESUME = "auto"

# Folders of the cache (relative to CACHE_ROOT), searched recursively: a whole
# volunteer or a single region. The split of round 3 of ``train_n2i_deepinv.py``,
# so that both methods are tested on the same unseen volunteer. The training set
# is every cached in-vivo frame in neither the validation nor the test set; set
# TRAIN_DIRS to a list of folders to restrict it instead. Phantom frames are
# never used.
VAL_DIRS = ["volunteer_005/carotid", "volunteer_006/carotid"]
TEST_DIRS = ["volunteer_002"]  # unseen volunteer; empty skips the test
TRAIN_DIRS = None  # None = all the remaining in-vivo data

# The fast acquisitions: ``K`` disjoint sub-sequences of ``87 / K`` transmits,
# interleaved so that each one spans the full angular aperture. One of them is
# drawn per training step and plays the role of the sparse-view data; the other
# ``K - 1`` are simply other fast acquisitions of the same anatomy.
K = 29
GROUPS = [list(range(j, 87, K)) for j in range(K)]
HELD_OUT = 1  # transmits of the drawn group kept out of the input, for the loss

# Scalar converting the cached adjoint images into the reflectivity units the
# forward operator expects (see below). ``None`` estimates it from the first
# ``SCALE_FRAMES`` training frames; set it to the printed value to skip that.
SCALE = None
SCALE_FRAMES = 16

SCALES = 4  # unet only; the ported nets have 5 scales, DRUNet 4
CHANNELS = 32
RESIDUAL = False  # unet only
BLOCKS = 2  # drunet only
UPSAMPLE = "bilinear"  # ported nets only

EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
NUM_WORKERS = 4
SEED = 0

DYNAMIC_RANGE_DB = 50.0  # for the B-mode preview

CENTER_FREQ = 5.3e6
FRAC_BW = 0.75 / math.sqrt(2)
N_ELEMENTS = 192
PITCH = 2.3e-4
ELEMENT_WIDTH = 6.1206151719371e-05
SAMPLING_FREQ = 20833333.333333332
SOUND_SPEED = 1540.0
T0 = 4.272e-06 + (4.1e-07 - 1.92e-07)
F_NUMBER = 1.5
Z_MIN, X_HALF = 2e-3, 18e-3

torch.manual_seed(SEED)
device = dinv.utils.get_device()

# %%
# The cache
# ---------
find = lambda dirs, name: sorted(
    f
    for d in dirs
    for f in glob.glob(os.path.join(CACHE_ROOT, d, "**", name), recursive=True)
)
for d in (TRAIN_DIRS or []) + VAL_DIRS + TEST_DIRS:
    if not find([d], "invivo_*.npy"):
        raise FileNotFoundError(f"No cached frames under {os.path.join(CACHE_ROOT, d)}")
val_files, test_files = find(VAL_DIRS, "invivo_*.npy"), find(TEST_DIRS, "invivo_*.npy")
if overlap := set(val_files) & set(test_files):
    raise ValueError(f"{len(overlap)} frames are in both VAL_DIRS and TEST_DIRS")
held_out = set(val_files) | set(test_files)
train_files = [f for f in find(TRAIN_DIRS or ["volunteer_*"], "invivo_*.npy") if f not in held_out]
if not train_files:
    raise FileNotFoundError(f"No cached frames left for training under {CACHE_ROOT}")
folders = lambda files: sorted({os.path.relpath(os.path.dirname(f), CACHE_ROOT) for f in files})
train_folders = folders(train_files)
manifests = [
    json.load(open(os.path.join(CACHE_ROOT, d, "manifest.json")))
    for d in folders(train_files + val_files + test_files)
]
manifest = manifests[0]
if any(m["img_size"] != manifest["img_size"] for m in manifests):
    raise ValueError("All folders must be beamformed on the same pixel grid")
if any(m["n_samples"] != manifest["n_samples"] for m in manifests):
    raise ValueError("All folders must have the same number of time samples")
flat = sorted(a for g in GROUPS for a in g)
if flat != sorted(set(flat)) or not set(flat) <= set(range(manifest["n_angles"])):
    raise ValueError("GROUPS must be disjoint subsets of the transmit indices")
if not 1 <= HELD_OUT < min(len(g) for g in GROUPS):
    raise ValueError("HELD_OUT must leave at least one transmit in the input")

# The RF pixel grid the cached images were beamformed on, before demodulation:
# ``DECIMATION`` times finer axially, same lateral sampling.
DECIMATION = manifest["decimation"]
n_z_iq, n_x = manifest["img_size"]
n_z = n_z_iq * DECIMATION
pixel_size = (manifest["pixel_size"][0] / DECIMATION, manifest["pixel_size"][1])
carrier_bin = round(manifest["demodulation_wavenumber"] * n_z * pixel_size[0])
angles = torch.deg2rad(torch.tensor(manifest["angles_deg"]))

if abs(pixel_size[1] - PITCH / 1.5) > 1e-12 or abs(pixel_size[0] - SOUND_SPEED / CENTER_FREQ / 8) > 1e-12:
    raise ValueError("Pixel grid of the cache does not match the constants above")


# %%
# Sparse2Inverse batches
# ----------------------
def pad_to_multiple(x, multiple=32):
    """Reflect-pad the last two dims up to a multiple of ``multiple``, for the U-Net."""
    z, w = x.shape[-2:]
    pad_z, pad_w = (multiple - z % multiple) % multiple, (multiple - w % multiple) % multiple
    return F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_z), mode="reflect").squeeze(0)


class Sparse2InverseDataset(Dataset):
    r"""One (input, held-out adjoint) pair per cached frame, batched.

    ``__getitem__`` takes the *list* of frame indices of a whole batch, because
    the held-out transmits must be the same for all its frames: the forward
    operator is applied once per batch, with those steering angles. At train
    time the fast acquisition and the transmits held out of it are drawn at
    random, otherwise the first group and its first transmits are used, so
    validation and test losses are deterministic. The cache is memory-mapped, so
    only the planes actually used are read from disk.

    ``'y'`` is the network input, the mean of the single-angle images of the
    transmits kept in the acquisition; ``'x'`` is not a target but the data term
    of the loss, :math:`A_{\Theta}^{\top} y_{\Theta}`, i.e. the *sum* of the
    single-angle images of the held-out transmits. Both are ``(2, Z, X)`` real
    tensors (I and Q channels). ``'angles'`` are the held-out steering angles.

    No lateral flip here: unlike in Noise2Inverse, the loss goes through the
    forward operator, and mirroring the image is only exact if the pixel grid is
    laterally symmetric about the array (this one is off by 1.8 pixels) and the
    steering angles are mirrored with it.
    """

    def __init__(self, files, groups, angles, held_out, train):
        self.files, self.groups, self.angles = list(files), groups, angles
        self.held_out, self.train = held_out, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, indices):
        indices = [indices] if isinstance(indices, int) else indices
        group = self.groups[torch.randint(len(self.groups), ())] if self.train else self.groups[0]
        order = torch.randperm(len(group)) if self.train else torch.arange(len(group))
        held = [group[i] for i in order[: self.held_out]]
        kept = [group[i] for i in order[self.held_out :]]

        inputs, data_terms = [], []
        for idx in indices:
            planes = np.load(self.files[idx], mmap_mode="r")  # (n_angles, Z, X) complex64
            inputs.append(torch.from_numpy(planes[kept].mean(0)))
            data_terms.append(torch.from_numpy(planes[held].sum(0)))

        iq = lambda t: pad_to_multiple(torch.view_as_real(t).permute(2, 0, 1)).contiguous()
        return {
            "y": torch.stack([iq(t) for t in inputs]),
            "x": torch.stack([iq(t) for t in data_terms]),
            "angles": self.angles[held],
        }


loader = lambda files, train: DataLoader(
    Sparse2InverseDataset(files, GROUPS, angles, HELD_OUT, train),
    batch_size=None,
    sampler=BatchSampler(
        RandomSampler(files) if train else SequentialSampler(files),
        BATCH_SIZE if train else 1,
        drop_last=False,
    ),
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    persistent_workers=NUM_WORKERS > 0,
)
train_dataloader, val_dataloader = loader(train_files, True), loader(val_files, False)

print(
    f"S2I over transmits: {len(GROUPS)} fast acquisitions of {len(GROUPS[0])} transmits, "
    f"{HELD_OUT} held out for the loss\n"
    f"  frames {n_z_iq}x{n_x} IQ, re-projected on {n_z}x{n_x} RF pixels\n"
    f"  train: {len(train_files)} frames from {train_folders}\n"
    f"  val:   {len(val_files)} frames from {VAL_DIRS}\n"
    f"  test:  {len(test_files)} frames from {TEST_DIRS}"
)

# %%
# Physics: re-projecting an IQ image
# ----------------------------------
# The cached images are demodulated and decimated, the forward operator works on
# the RF grid, so the loss first undoes the demodulation of the precompute
# script. ``remodulate`` is its exact right inverse: interpolate the baseband
# back to the fine grid by zero-padding its spectrum, shift it back to the
# carrier and take the real part. What it discards is out of the band the cache
# kept, so the re-projected image is the band-pass part of the network output,
# which is all the pulse can see anyway.
element_x = torch.arange(N_ELEMENTS) * PITCH + ELEMENT_WIDTH / 2
element_x = element_x - element_x.mean()
element_positions = torch.stack([element_x, torch.zeros_like(element_x)], dim=-1)

sigma_t = math.sqrt(2 * math.log(2)) / (math.pi * FRAC_BW * CENTER_FREQ)
n_half = math.ceil(3.5 * sigma_t * SAMPLING_FREQ)
t = (np.arange(2 * n_half + 1) - n_half) / SAMPLING_FREQ
pulse = np.exp(-(t**2) / (2 * sigma_t**2)) * np.cos(2 * np.pi * CENTER_FREQ * t)

physics = dinv.physics.UltrasoundPlaneWave(
    img_size=(n_z, n_x),
    angles=angles[:HELD_OUT],  # replaced by the held-out ones at every step
    element_positions=element_positions.to(device),
    n_samples=manifest["n_samples"],
    sampling_frequency=SAMPLING_FREQ,
    sound_speed=SOUND_SPEED,
    pixel_size=pixel_size,
    pixel_origin=(Z_MIN, -X_HALF),
    t0=T0,
    pulse=torch.as_tensor(pulse, dtype=torch.float32),
    f_number=F_NUMBER,
    receive_apod_window="hann",
    normalize=False,
    device=device,
)


def remodulate(x):
    """``(B, 2, Z', X)`` padded IQ channels to a ``(B, 1, Z, X)`` real RF image."""
    iq = torch.complex(x[:, 0, :n_z_iq, :n_x], x[:, 1, :n_z_iq, :n_x])
    spectrum = torch.fft.fftshift(torch.fft.fft(iq, dim=-2), dim=-2) * (DECIMATION / 2)
    full = iq.new_zeros((iq.shape[0], n_z, n_x))
    full[:, (n_z - n_z_iq) // 2 : (n_z + n_z_iq) // 2] = spectrum
    full = torch.roll(torch.fft.ifftshift(full, dim=-2), carrier_bin, dims=-2)
    return 2 * torch.fft.ifft(full, dim=-2).real.unsqueeze(1)


# The cached images are adjoint images :math:`A^{\top} y`, whose amplitude is
# that of the data times the gain of the normal operator: a few hundred times
# the reflectivity that :math:`A` expects here. Rather than ask the network to
# learn that factor, it keeps working in the units of its input and the loss
# converts with one scalar, the multiple of the input sub-reconstruction that
# minimises the loss itself. The identity then means "the best rescaling of the
# input", which is the baseline the network has to beat. A resumed run takes the
# scale of the run it continues, so that its loss keeps the same definition.
def find_checkpoint(spec):
    """Resolve ``RESUME`` to a checkpoint file, or ``None`` to train from scratch."""
    if not spec:
        return None
    if spec != "auto":
        if not os.path.exists(spec):
            raise FileNotFoundError(f"RESUME checkpoint {spec} not found")
        return spec
    # The most recently written checkpoint of the latest run is the furthest along.
    for run in sorted(glob.glob(os.path.join(SAVE_PATH, "*")), reverse=True):
        if ckpts := glob.glob(os.path.join(run, "ckp_*.pth.tar")):
            return max(ckpts, key=os.path.getmtime)
    return None


resume_from = find_checkpoint(RESUME)
if SCALE is None and resume_from:
    SCALE = torch.load(resume_from, map_location="cpu", weights_only=False).get("s2i_scale")
    if SCALE is not None:
        print(f"reflectivity scale {SCALE:.4g} (1 / {1 / SCALE:.0f}), from {resume_from}")
if SCALE is None:
    probe = Sparse2InverseDataset(train_files, GROUPS, angles, HELD_OUT, train=True)
    quadratic, linear, ratios = 0.0, 0.0, []
    for i in torch.randperm(len(train_files))[:SCALE_FRAMES].tolist():
        batch = probe[i]
        physics.update_parameters(angles=batch["angles"].to(device))
        u, d = remodulate(batch["y"].to(device)), remodulate(batch["x"].to(device))
        with torch.no_grad():
            q = physics.A(u).pow(2).sum(dtype=torch.float64).item()
        l = (u * d).sum(dtype=torch.float64).item()
        quadratic, linear = quadratic + q, linear + l
        ratios.append(l / q)
    # The minimiser of the total loss over the draws, not the mean of the
    # per-draw minimisers: the gain of the operator depends on the steering
    # angle, and a single image has to answer for every transmit anyway.
    SCALE = linear / quadratic
    print(
        f"reflectivity scale {SCALE:.4g} (1 / {1 / SCALE:.0f}), per-draw spread "
        f"{np.std(ratios) / abs(np.mean(ratios)):.0%} over {len(ratios)} draws"
    )


class Sparse2InverseLoss(dinv.loss.Loss):
    r"""Data consistency of the network output on the held-out transmits.

    Returns :math:`\|A_\Theta u\|^2 - 2 \langle u, A_\Theta^{\top} y_\Theta\rangle`
    per frame with :math:`u = \mathrm{SCALE} \cdot R(x_\mathrm{net})`, normalized
    by the number of measurement samples: the Sparse2Inverse loss (4) minus the
    constant :math:`\|y_\Theta\|^2`, which the cache does not hold. The two terms
    are large and nearly cancel, so they are summed in double precision. ``x``
    carries :math:`A_\Theta^{\top} y_\Theta` and the physics carries the steering
    angles of :math:`\Theta`.
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x_net, x, y, physics, **kwargs):
        u = self.scale * remodulate(x_net)
        n_meas = len(physics.angles) * N_ELEMENTS * physics.n_samples
        quadratic = physics.A(u).pow(2).sum((1, 2, 3, 4), dtype=torch.float64)
        linear = (u * remodulate(x)).sum((1, 2, 3), dtype=torch.float64)
        return ((quadratic - 2 * linear) / n_meas).float()


# %%
# Model and training
# ------------------
# The network sees the sub-reconstruction directly, so the reconstruction is a
# plain forward pass; the physics only appears in the loss.
if ARCH == "unet":
    denoiser = dinv.models.UNet(
        in_channels=2,  # I and Q
        out_channels=2,
        scales=SCALES,
        channels_per_scale=[CHANNELS * 2**k for k in range(SCALES)],
        residual=RESIDUAL,
        batch_norm=False,
    )
elif ARCH in ("resunet", "attunet", "attresunet"):
    net_cls = {"resunet": ResUNet, "attunet": AttentionUNet, "attresunet": AttentionResUNet}
    denoiser = net_cls[ARCH](
        in_channels=2, out_channels=2, base_channels=CHANNELS, upsample_mode=UPSAMPLE
    )
elif ARCH == "drunet":
    # The ultrasound physics has no noise level to feed DRUNet's extra channel,
    # which therefore stays at its ``None`` default (a zero map).
    denoiser = dinv.models.DRUNet(
        in_channels=2,
        out_channels=2,
        nc=[CHANNELS * 2**k for k in range(4)],
        nb=BLOCKS,
        pretrained=None,
    )
else:
    raise ValueError(f"Unknown ARCH {ARCH!r}")
print(f"{ARCH}: {sum(p.numel() for p in denoiser.parameters()) / 1e6:.2f} M parameters")
model = dinv.models.ArtifactRemoval(denoiser, mode="direct", device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


class Sparse2InverseTrainer(dinv.Trainer):
    """Trainer whose batches also carry the steering angles of the loss.

    It also resumes like the ``ResumableTrainer`` of ``train_n2i_deepinv.py``:
    :meth:`deepinv.Trainer.load_model` leaves the loss histories empty, and
    without them a resumed run would treat its first validation as the best ever
    and early-stop at once. Every checkpoint also stores the reflectivity scale
    of the loss.
    """

    HISTORIES = {
        "loss_history": "loss",
        "eval_loss_history": "eval_loss",
        "train_metrics_history": "train_metrics",
        "eval_metrics_history": "eval_metrics",
    }

    def save_model(self, filename, epoch, state=None):
        super().save_model(filename, epoch, (state or {}) | {"s2i_scale": SCALE})

    def load_model(self, ckpt_pretrained=None, strict=True):
        checkpoint = super().load_model(ckpt_pretrained, strict)
        if checkpoint is not None:
            for attr, key in self.HISTORIES.items():
                if saved := checkpoint.get(key):
                    setattr(self, attr, {k: list(v) for k, v in saved.items()})
            # Put the cosine schedule back on the horizon of this run, in closed
            # form, since CosineAnnealingLR steps recursively from the current LR.
            sched = self.scheduler
            if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                sched.T_max = self.epochs
                for group, base in zip(self.optimizer.param_groups, sched.base_lrs):
                    group["lr"] = sched.eta_min + (base - sched.eta_min) * (
                        1 + math.cos(math.pi * sched.last_epoch / sched.T_max)
                    ) / 2
        return checkpoint

    def get_samples_offline(self, iterators, g):
        batch = next(iterators[g])
        physics = self.physics[g]
        physics.update_parameters(angles=batch["angles"].to(self.device))
        return (
            batch["x"].to(self.device, non_blocking=self.non_blocking_transfers),
            batch["y"].to(self.device, non_blocking=self.non_blocking_transfers),
            physics,
        )


trainer = Sparse2InverseTrainer(
    ckpt_pretrained=resume_from,
    model=model,
    physics=physics,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    epochs=EPOCHS,
    losses=Sparse2InverseLoss(SCALE),
    metrics=None,
    compute_train_metrics=False,
    compute_eval_losses=True,
    early_stop=EARLY_STOP_PATIENCE,
    early_stop_on_losses=True,
    online_measurements=False,
    grad_clip=GRAD_CLIP,
    check_grad=True,  # required by the trainer whenever grad_clip is set
    device=device,
    save_path=SAVE_PATH,
    ckp_interval=10,
    plot_images=False,
    verbose=True,
    freq_update_progress_bar=10,  # refreshing the loss display forces a GPU sync
)

if resume_from:
    print(f"Resuming from {resume_from}")
print(f"Training on {device} up to epoch {EPOCHS}; checkpoints -> {SAVE_PATH}")
trainer.train()

# %%
# Inference and test
# ------------------
# Eq. (2) of the paper: the estimate of a fast acquisition is the mean of the
# network applied to each of its leave-one-out sub-reconstructions, which is also
# what the network was trained on -- feeding it all the transmits at once would
# be an input it has never seen.
# The best checkpoint lives in this run's folder, unless the run was resumed and
# never improved on the validation loss it inherited.
best_ckpt = Path(trainer.save_path) / "ckp_best.pth.tar"
if not best_ckpt.exists() and resume_from:
    best_ckpt = Path(resume_from).parent / "ckp_best.pth.tar"
if best_ckpt.exists():
    trainer.load_model(best_ckpt)
else:
    print("No best checkpoint saved (no validation step ran); using the final weights")
model.eval()


def reconstruct(planes, group):
    """Leave-one-out average of the network over the transmits of ``group``."""
    out = 0
    for held in range(0, len(group), HELD_OUT):
        kept = [a for i, a in enumerate(group) if not held <= i < held + HELD_OUT]
        sub = torch.view_as_real(planes[kept].mean(0)).permute(2, 0, 1)
        with torch.no_grad():
            out = out + model(pad_to_multiple(sub).unsqueeze(0).to(device), physics)
    return out / math.ceil(len(group) / HELD_OUT)


# Data consistency of the best checkpoint on the held-out transmits of the test
# folders, against the identity (the sub-reconstruction fed straight through).
# Both are shifted by the same unknown constant, only their difference means
# anything: it is the residual energy the network recovered, per sample.
if test_files:
    test_dataloader = loader(test_files, False)
    criterion = Sparse2InverseLoss(SCALE)
    loss_s2i, loss_identity = [], []
    for batch in test_dataloader:
        physics.update_parameters(angles=batch["angles"].to(device))
        x, y = batch["x"].to(device), batch["y"].to(device)
        with torch.no_grad():
            loss_s2i.append(criterion(model(y, physics), x, y, physics).mean().item())
            loss_identity.append(criterion(y, x, y, physics).mean().item())
    print(
        f"Test on {len(test_files)} frames from {TEST_DIRS}: S2I loss "
        f"{np.mean(loss_s2i):.4g} (identity {np.mean(loss_identity):.4g}, "
        f"gain {np.mean(loss_identity) - np.mean(loss_s2i):.4g})"
    )

# %%
# B-mode preview on one held-out frame
# ------------------------------------
# Fast acquisition, its Sparse2Inverse reconstruction, and the full-aperture
# CPWC reference (mean of all 87 single-angle images).
preview_file = (test_files or val_files)[0]
planes = torch.from_numpy(np.load(preview_file))  # complex64
cpwc = planes.mean(0, keepdim=True)
fast, n_tx = planes[GROUPS[0]].mean(0, keepdim=True), len(GROUPS[0])

rec = reconstruct(planes, GROUPS[0])[0, :, :n_z_iq, :n_x].cpu()
rec = torch.complex(rec[0], rec[1]).unsqueeze(0)

# The envelope of an IQ image is its modulus, which ``bmode`` uses directly.
bmode = lambda img: dinv.utils.bmode(img.unsqueeze(0), amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=False)
out = Path(trainer.save_path) / "preview_bmode.png"
dinv.utils.plot(
    {
        f"DAS, {n_tx} transmits": bmode(fast),
        f"Sparse2Inverse, {n_tx} transmits": bmode(rec),
        f"CPWC, {manifest['n_angles']} transmits": bmode(cpwc),
    },
    rescale_mode="clip",
    vmin=-DYNAMIC_RANGE_DB,
    vmax=0.0,
    cmap="gray",
    extent=[
        -manifest["x_half"] * 1e3,
        manifest["x_half"] * 1e3,
        manifest["z_max"] * 1e3,
        manifest["z_min"] * 1e3,
    ],
    aspect="equal",
    figsize=(12, 6),
    suptitle=f"{os.path.relpath(preview_file, CACHE_ROOT)}, {DYNAMIC_RANGE_DB:.0f} dB",
    save_fn=str(out),
    show=False,
)
print(f"Preview written to {out}")
