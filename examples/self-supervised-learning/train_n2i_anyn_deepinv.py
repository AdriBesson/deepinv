r"""
Noise2Inverse for ultrafast ultrasound -- any number of plane waves
===================================================================

Variant of ``train_n2i_deepinv.py`` (kept as the baseline) for a single network
that denoises DAS images compounded from any number of steering angles, from 1
to all 87. The baseline only ever sees the mean of 3 transmits: applied to 1
angle it under-denoises, and on 11 or 87 angles it still applies the correction
learned for 3 (``nangles_n2i_deepinv.py``).

Here every training pair draws its own fast acquisition:

* the number of transmits ``n`` is log-uniform between ``N_MIN`` and ``N_MAX``,
  since the noise falls as :math:`1/\sqrt{n}` and most of the difficulty is at
  small ``n``;
* the transmits are evenly spaced in steering angle over a random span
  (``SPAN_DEG``) at a random position, like an ``n``-angle compounding sequence;
* the input is their mean (2 channels, I and Q) and the target the mean of all
  the other transmits (X:1), so the noise of the target stays independent of
  that of the input and Noise2Inverse remains valid for every ``n``. Near
  ``N_MAX`` the target averages few transmits: it is noisier, but unbiased.
  ``n = 87`` cannot be trained (no transmit is left for the target); inference
  on CPWC is a small extrapolation beyond ``N_MAX``.

The network is told ``n`` through DRUNet's noise-level channel,
:math:`\sigma(n) = \sigma_\mathrm{ref} \sqrt{n_\mathrm{ref} / n}` (``SIGMA_REF``
at ``N_REF`` transmits, the constant of the baseline at 3), passed per sample as
``params["sigma"]`` and routed by the trainer to the physics. The input is
divided by its RMS (scale equivariance) before the network sees it, so
:math:`\sigma` is relative to the image level. Other architectures have no
noise-level input and train blind.

Everything else follows the baseline: splits by volunteer, scale equivariance,
TGC augmentation, lateral flips and optional vertical flips (``AUG_VFLIP``,
default off here, as in the round-4 baseline this is compared with). Training
and validation read the transmit cache of ``train_n2i_multitx_deepinv.py``
(same ``GROUPS``, so the same folder; it is built if missing): any subset of
transmits plus the mean of all of them, from which the target follows.

Validation and test use fixed subsets (``EVAL_SETS``: 1, 3, 11 and 43 angles),
and the loss is reported per subset at the end of training.

Checkpoints go to ``./checkpoints/n2i_ultrasound_<ARCH>_anyn[_vflip]<RUN_SUFFIX>`` and
record ``checkpoint["input"] = "mean_anyn"`` with the noise-level rule.
"""

import glob
import hashlib
import itertools
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import deepinv as dinv
import n2i_invitro
from n2i_models import AttentionResUNet, AttentionUNet, MSDNet, PaddedArtifactRemoval, ResUNet

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
TX_CACHE_ROOT = "/data/n2i_tx_cache"  # shared with train_n2i_multitx_deepinv.py

# DRUNet is the only network with a noise-level input. N2I_ARCH overrides it.
ARCH = os.environ.get("N2I_ARCH", "drunet")
# Same split as the round-4 baseline, so the two are directly comparable.
RUN_SUFFIX = "_test002_006_008"

# "" from scratch, "auto" the latest run of this architecture, or a checkpoint path.
RESUME = "auto"

# Splits: whole volunteers, as in the round-4 baseline (see train_n2i_deepinv.py).
VAL_DIRS = ["volunteer_005"]
TEST_DIRS = ["volunteer_002", "volunteer_006", "volunteer_008"]  # empty skips the test
TRAIN_DIRS = ["volunteer_001", "volunteer_003", "volunteer_004", "volunteer_007"]  # None = all the rest
INVITRO_TRAIN = True
INVITRO_HELD_OUT = ["invitro_CIRS054GS/054GS"]
INVITRO_TEST = True

# Layout of the transmit cache (the groups of the multi-transmit script); only
# used to locate the transmits in it.
K = 29
GROUPS = [list(range(j, 87, K)) for j in range(K)]

# Fast acquisitions drawn for training.
N_MIN, N_MAX = 1, 80  # number of transmits, log-uniform
SPAN_DEG = (10.0, 32.6)  # total angular span of the transmits (n >= 2), uniform
# Noise-level channel: sigma(n) = SIGMA_REF * sqrt(N_REF / n).
SIGMA_REF, N_REF = 0.1, 3
# Fixed acquisitions of validation and test: steering angles (deg), each mapped to
# the nearest transmit. The keys are labels.
EVAL_SETS = {
    1: [0.0],
    3: [-10.0, 0.0, 10.0],
    11: list(np.linspace(-10.0, 10.0, 11)),
    43: list(np.linspace(-16.0, 16.0, 43)),
}
TEST_MAX_FRAMES = 300  # evenly spaced over the test frames (each read in full)

SCALES = 4  # unet only
CHANNELS = 32
RESIDUAL = False  # unet only
BLOCKS = 2  # drunet only
MSD_DEPTH, MSD_WIDTH, MSD_DILATION = 100, 1, 10  # msd only
UPSAMPLE = "bilinear"  # ported nets only
SCALE_EQUIVARIANT = True

AUG_TGC_DB = 10.0
AUG_TGC_KNOTS = 5
AUG_GAIN_DB = 0.0 if SCALE_EQUIVARIANT else 10.0
# Vertical flip of half the training pairs; N2I_VFLIP=1 turns it on.
AUG_VFLIP = os.environ.get("N2I_VFLIP", "0") != "0"
SAVE_PATH = f"./checkpoints/n2i_ultrasound_{ARCH}_anyn{'_vflip' if AUG_VFLIP else ''}{RUN_SUFFIX}"

# Optimisation
EPOCHS = 130
BATCH_SIZE = 8
LR = 1e-3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 30
NUM_WORKERS = 4
SEED = 0

DYNAMIC_RANGE_DB = 50.0  # for the B-mode figures

torch.manual_seed(SEED)
device = dinv.utils.get_device()


def noise_level(n):
    """Noise-level channel of the network for a mean of ``n`` transmits."""
    return SIGMA_REF * math.sqrt(N_REF / n)


# %%
# Inputs
# ------
def random_gain_db(n_z):
    """Random gain in dB per depth: zero-mean TGC curve plus global gain, ``(n_z,)``."""
    knots = (2 * torch.rand(AUG_TGC_KNOTS) - 1) * AUG_TGC_DB
    z_knots = torch.linspace(0, n_z - 1, AUG_TGC_KNOTS)
    curve = torch.from_numpy(np.interp(np.arange(n_z), z_knots.numpy(), knots.numpy())).float()
    return curve - curve.mean() + (2 * torch.rand(1) - 1) * AUG_GAIN_DB


def nearest_transmits(angles, angles_deg):
    """Distinct transmits nearest to the steering ``angles`` (deg), by increasing angle."""
    angles_deg = np.asarray(angles_deg)
    return sorted({int(np.abs(angles_deg - a).argmin()) for a in angles}, key=angles_deg.__getitem__)


class AnyNDataset(Dataset):
    r"""One (input, target) pair per frame, from a fast acquisition of any size.

    Reads the transmit cache, ``(n_total + 1, 2, Z, X)`` float32 per frame: every
    transmit (in the order ``positions`` maps them to) then the mean over all of
    them. Training draws the acquisition (:func:`draw`); otherwise every frame
    gives one item per acquisition of ``eval_sets``. The input is the mean of the
    acquisition, the target the mean of all the other transmits.

    Returns ``'y'`` and ``'x'`` of shape ``(2, Z, X)``, and ``'params'`` with the
    noise level ``'sigma'`` of the acquisition.
    """

    def __init__(self, files, positions, angles_deg, train, eval_sets=None):
        self.files, self.positions, self.train = list(files), positions, train
        self.n_total = len(positions)
        self.by_angle = list(np.argsort(angles_deg))  # rank -> transmit
        self.step_deg = (max(angles_deg) - min(angles_deg)) / (self.n_total - 1)
        self.eval_sets = [nearest_transmits(a, angles_deg) for a in (eval_sets or {}).values()]

    def __len__(self):
        return len(self.files) * (1 if self.train else len(self.eval_sets))

    def draw(self):
        """Random transmits: log-uniform count, evenly spaced over a random span."""
        n = int(math.exp(np.random.uniform(math.log(N_MIN), math.log(N_MAX + 1))))
        n = min(max(n, N_MIN), N_MAX)
        if n == 1:
            ranks = [np.random.randint(self.n_total)]
        else:
            lo, hi = (round(s / self.step_deg) + 1 for s in SPAN_DEG)
            span = np.random.randint(max(n, lo), max(n, min(hi, self.n_total)) + 1)
            start = np.random.randint(self.n_total - span + 1)
            ranks = start + np.round(np.linspace(0, span - 1, n)).astype(int)
        return [self.by_angle[r] for r in ranks]

    def __getitem__(self, idx):
        if self.train:
            data = np.load(self.files[idx], mmap_mode="r")
            tx = self.draw()
        else:
            data = np.load(self.files[idx // len(self.eval_sets)], mmap_mode="r")
            tx = self.eval_sets[idx % len(self.eval_sets)]
        rows = sorted(self.positions[t] for t in tx)  # in file order, for the disk
        total = torch.from_numpy(np.array(data[rows])).sum(0)  # (2, Z, X)
        n = len(tx)
        inp = total / n
        tgt = (self.n_total * torch.from_numpy(np.array(data[-1])) - total) / (self.n_total - n)

        if self.train and torch.rand(1) > 0.5:
            inp, tgt = torch.flip(inp, (-1,)), torch.flip(tgt, (-1,))
        if self.train and AUG_VFLIP and torch.rand(1) > 0.5:
            inp, tgt = torch.flip(inp, (-2,)), torch.flip(tgt, (-2,))
        if self.train and (AUG_TGC_DB or AUG_GAIN_DB):
            gain = 10 ** (random_gain_db(inp.shape[-2]) / 20)[:, None]  # amplitude, (Z, 1)
            inp, tgt = inp * gain, tgt * gain

        return {
            "x": tgt.contiguous(),
            "y": inp.contiguous(),
            "params": {"sigma": torch.tensor(noise_level(n), dtype=torch.float32)},
        }


find = lambda dirs, name: sorted(
    {f for d in dirs for f in glob.glob(os.path.join(CACHE_ROOT, d, "**", name), recursive=True)}
)
for d in VAL_DIRS + TEST_DIRS:
    if d.startswith("invitro"):
        raise ValueError(f"{d}: VAL_DIRS and TEST_DIRS are in-vivo only")
for d in (TRAIN_DIRS or []) + VAL_DIRS + TEST_DIRS:
    if not find([d], "invi*_*.npy"):
        raise FileNotFoundError(f"No cached frames under {os.path.join(CACHE_ROOT, d)}")

val_files, test_files = find(VAL_DIRS, "invivo_*.npy"), find(TEST_DIRS, "invivo_*.npy")
if overlap := set(val_files) & set(test_files):
    raise ValueError(f"{len(overlap)} frames are in both VAL_DIRS and TEST_DIRS")
held_out = set(val_files) | set(test_files) | {os.path.join(CACHE_ROOT, n2i_invitro.FRAME)}
held_out |= set(find(INVITRO_HELD_OUT, "invitro_*.npy"))
train_dirs = (TRAIN_DIRS or ["volunteer_*"]) + (["invitro_*"] if INVITRO_TRAIN else [])
train_files = [f for f in find(train_dirs, "invi*_*.npy") if f not in held_out]
if not train_files:
    raise FileNotFoundError(f"No cached frames left for training under {CACHE_ROOT}")
folders = lambda files: sorted({os.path.relpath(os.path.dirname(f), CACHE_ROOT) for f in files})
train_folders = folders(train_files)
volunteers = lambda files: {Path(f).relative_to(CACHE_ROOT).parts[0] for f in files if "invivo_" in f}
for (a, fa), (b, fb) in itertools.combinations(
    [("train", train_files), ("val", val_files), ("test", test_files)], 2
):
    if shared := volunteers(fa) & volunteers(fb):
        raise ValueError(f"Volunteers {sorted(shared)} are in both the {a} and {b} sets")
for f in train_files + val_files + test_files:
    try:
        np.load(f, mmap_mode="r")
    except ValueError as e:
        raise ValueError(f"{f} is truncated; is the precompute script still running?") from e

manifests = [
    json.load(open(os.path.join(CACHE_ROOT, d, "manifest.json")))
    for d in folders(train_files + val_files + test_files)
]
manifest = manifests[0]
if any(m["img_size"] != manifest["img_size"] for m in manifests):
    raise ValueError("All folders must be beamformed on the same pixel grid")
angles_deg = manifest["angles_deg"]
if any(not np.allclose(m["angles_deg"], angles_deg) for m in manifests):
    raise ValueError("All folders must share the same transmit angles")
N_TOTAL = manifest["n_angles"]
if sorted(a for g in GROUPS for a in g) != list(range(N_TOTAL)):
    raise ValueError("GROUPS must partition all the transmits (the cache holds them all)")
if not 1 <= N_MIN <= N_MAX < N_TOTAL:
    raise ValueError(f"Need 1 <= N_MIN <= N_MAX < {N_TOTAL} (the target needs a transmit)")
if any(len(nearest_transmits(a, angles_deg)) >= N_TOTAL for a in EVAL_SETS.values()):
    raise ValueError("Every EVAL_SETS acquisition must leave transmits for the target")

# %%
# Transmit cache
# --------------
# The cache of train_n2i_multitx_deepinv.py: per frame the transmits of every
# group, group after group and each group sorted by angle, then the mean over
# all of them. Same folder (hash of GROUPS) and same build, so a cache built by
# either script serves both.
sort_by_angle = lambda tx: sorted(tx, key=lambda a: angles_deg[a])
ordered = [a for g in GROUPS for a in sort_by_angle(g)]
positions = {t: p for p, t in enumerate(ordered)}  # transmit -> row of the cache
tx_root = os.path.join(
    TX_CACHE_ROOT, f"K{len(GROUPS)}_" + hashlib.sha1(json.dumps(GROUPS).encode()).hexdigest()[:10]
)
tx_path = lambda f: os.path.join(tx_root, os.path.relpath(f, CACHE_ROOT))


def build_tx_cache(files):
    """Write the transmit file of every frame whose one is missing or older than it."""
    stale = [
        f
        for f in files
        if not os.path.exists(tx_path(f)) or os.path.getmtime(tx_path(f)) < os.path.getmtime(f)
    ]
    if not stale:
        return
    os.makedirs(tx_root, exist_ok=True)
    with open(os.path.join(tx_root, "layout.json"), "w") as fh:
        json.dump({"groups": GROUPS, "ordered_transmits": ordered}, fh)
    for f in tqdm(stale, desc=f"Transmit cache -> {tx_root}"):
        planes = np.load(f)  # the whole file in one sequential read
        stack = np.concatenate([planes[ordered], planes[ordered].mean(0, keepdims=True)])
        out = np.stack([stack.real, stack.imag], axis=1)  # float32
        os.makedirs(os.path.dirname(tx_path(f)), exist_ok=True)
        with open(tx_path(f) + ".tmp", "wb") as fh:  # never leave a truncated file
            np.save(fh, out)
        os.replace(tx_path(f) + ".tmp", tx_path(f))


build_tx_cache(train_files + val_files)

train_dataset = AnyNDataset(map(tx_path, train_files), positions, angles_deg, train=True)
val_dataset = AnyNDataset(map(tx_path, val_files), positions, angles_deg, False, EVAL_SETS)

loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    persistent_workers=NUM_WORKERS > 0,
    # Different random acquisitions in every worker.
    worker_init_fn=lambda w: np.random.seed((torch.initial_seed() + w) % 2**32),
)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)

eval_sizes = {k: len(nearest_transmits(a, angles_deg)) for k, a in EVAL_SETS.items()}
print(
    f"N2I over transmits, any number of transmits: n log-uniform in [{N_MIN}, {N_MAX}], "
    f"span {SPAN_DEG[0]:g}-{SPAN_DEG[1]:g} deg, target: mean of the others\n"
    f"  noise level sigma(n) = {SIGMA_REF:g} sqrt({N_REF}/n)"
    + ("" if ARCH == "drunet" else f" (unused: {ARCH} has no noise-level input)") + "\n"
    f"  frames {tuple(train_dataset[0]['y'].shape[-2:])}\n"
    f"  train: {len(train_files)} frames from {train_folders}\n"
    f"  val:   {len(val_files)} frames from {VAL_DIRS}, acquisitions of {list(eval_sizes.values())} transmits\n"
    f"  test:  {len(test_files)} frames from {TEST_DIRS} (at most {TEST_MAX_FRAMES})\n"
    f"  scale-equivariant: {SCALE_EQUIVARIANT}; augmentation: TGC +-{AUG_TGC_DB} dB "
    f"({AUG_TGC_KNOTS} knots), global gain +-{AUG_GAIN_DB} dB, lateral flip"
    f"{', vertical flip' if AUG_VFLIP else ''}"
)

# %%
# Model and training
# ------------------
# The trainer sets the noise level of every batch from its params.
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(SIGMA_REF)).to(device)

if ARCH == "unet":
    denoiser = dinv.models.UNet(
        in_channels=2,
        out_channels=2,
        scales=SCALES,
        channels_per_scale=[CHANNELS * 2**k for k in range(SCALES)],
        residual=RESIDUAL,
        batch_norm=False,
    )
elif ARCH in ("resunet", "attunet", "attresunet"):
    net_cls = {"resunet": ResUNet, "attunet": AttentionUNet, "attresunet": AttentionResUNet}
    denoiser = net_cls[ARCH](in_channels=2, out_channels=2, base_channels=CHANNELS, upsample_mode=UPSAMPLE)
elif ARCH == "drunet":
    denoiser = dinv.models.DRUNet(
        in_channels=2,
        out_channels=2,
        nc=[CHANNELS * 2**k for k in range(4)],
        nb=BLOCKS,
        pretrained=None,
    )
elif ARCH == "msd":
    denoiser = MSDNet(
        in_channels=2, out_channels=2, depth=MSD_DEPTH, width=MSD_WIDTH, max_dilation=MSD_DILATION
    )
else:
    raise ValueError(f"Unknown ARCH {ARCH!r}")
print(f"{ARCH}: {sum(p.numel() for p in denoiser.parameters()) / 1e6:.3f} M parameters")
model = PaddedArtifactRemoval(
    denoiser, mode="direct", device=device, scale_equivariant=SCALE_EQUIVARIANT
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# %%
# Resuming an interrupted run
# ---------------------------
def find_checkpoint(spec):
    """Resolve ``RESUME`` to a checkpoint file, or ``None`` to train from scratch."""
    if not spec:
        return None
    if spec != "auto":
        if not os.path.exists(spec):
            raise FileNotFoundError(f"RESUME checkpoint {spec} not found")
        return spec
    for run in sorted(glob.glob(os.path.join(SAVE_PATH, "*")), reverse=True):
        if ckpts := glob.glob(os.path.join(run, "ckp_*.pth.tar")):
            return max(ckpts, key=os.path.getmtime)
    return None


class ResumableTrainer(dinv.Trainer):
    """Trainer that also restores the loss histories held in a checkpoint.

    See ``train_n2i_deepinv.py``: without the histories a resumed run early-stops
    at once, and the cosine schedule must be put back on its horizon.
    """

    HISTORIES = {
        "loss_history": "loss",
        "eval_loss_history": "eval_loss",
        "train_metrics_history": "train_metrics",
        "eval_metrics_history": "eval_metrics",
    }

    def load_model(self, ckpt_pretrained=None, strict=True):
        checkpoint = super().load_model(ckpt_pretrained, strict)
        if checkpoint is not None:
            for attr, key in self.HISTORIES.items():
                if saved := checkpoint.get(key):
                    setattr(self, attr, {k: list(v) for k, v in saved.items()})
            sched = self.scheduler
            if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                sched.T_max = self.epochs
                for group, base in zip(self.optimizer.param_groups, sched.base_lrs):
                    group["lr"] = sched.eta_min + (base - sched.eta_min) * (
                        1 + math.cos(math.pi * sched.last_epoch / sched.T_max)
                    ) / 2
        return checkpoint

    def save_model(self, filename, epoch, state=None):
        # The evaluation scripts rebuild the model and its input from these.
        extra = {
            "scale_equivariant": SCALE_EQUIVARIANT,
            "augmentation": {
                "tgc_db": AUG_TGC_DB,
                "tgc_knots": AUG_TGC_KNOTS,
                "gain_db": AUG_GAIN_DB,
                "vflip": AUG_VFLIP,
            },
            "input": "mean_anyn",  # mean of any number of transmits
            "noise_level": {"sigma_ref": SIGMA_REF, "n_ref": N_REF},  # sigma_ref sqrt(n_ref / n)
            "n_range": [N_MIN, N_MAX],
            "span_deg": list(SPAN_DEG),
        }
        return super().save_model(filename, epoch, (state or {}) | extra)


resume_from = find_checkpoint(RESUME)

trainer = ResumableTrainer(
    ckpt_pretrained=resume_from,
    model=model,
    physics=physics,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    epochs=EPOCHS,
    losses=dinv.loss.SupLoss(metric=dinv.loss.metric.MSE(reduction=None)),
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
    freq_update_progress_bar=10,
)

if resume_from:
    print(f"Resuming from {resume_from}")
print(f"Training on {device} up to epoch {EPOCHS}; checkpoints -> {SAVE_PATH}")
trainer.train()

# %%
# Test on unseen volunteers, per acquisition size
# -----------------------------------------------
best_ckpt = Path(trainer.save_path) / "ckp_best.pth.tar"
if not best_ckpt.exists() and resume_from:
    best_ckpt = Path(resume_from).parent / "ckp_best.pth.tar"
if best_ckpt.exists():
    trainer.load_model(best_ckpt)
else:
    print("No best checkpoint saved (no validation step ran); using the final weights")
model.eval()


def denoise(img, n):
    """Network output for a ``(Z, X)`` complex mean of ``n`` transmits."""
    physics.update(sigma=torch.tensor([noise_level(n)], device=device))
    net_in = torch.view_as_real(img).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        out = model(net_in, physics)[0].cpu()
    return torch.complex(out[0], out[1])


eval_tx = {k: nearest_transmits(a, angles_deg) for k, a in EVAL_SETS.items()}
mse = lambda a, b: (a - b).abs().pow(2).mean().item() / 2  # per real channel, as the loss
# N2I loss (mean of the other transmits as target) of the network and of its input,
# on the validation frames (transmit cache) and on the test frames (read in full).
for split, files in [("val", val_files), ("test", test_files)]:
    if not files:
        continue
    if len(files) > TEST_MAX_FRAMES:
        files = [files[i] for i in np.linspace(0, len(files) - 1, TEST_MAX_FRAMES).round().astype(int)]
    loss, loss_identity = {k: [] for k in eval_tx}, {k: [] for k in eval_tx}
    for f in tqdm(files, desc=f"Per-size {split} loss"):
        planes = torch.from_numpy(np.load(f))  # (87, Z, X) complex64
        total = planes.sum(0)
        for k, tx in eval_tx.items():
            inp = planes[tx].mean(0)
            tgt = (total - planes[tx].sum(0)) / (N_TOTAL - len(tx))
            loss[k].append(mse(denoise(inp, len(tx)), tgt))
            loss_identity[k].append(mse(inp, tgt))
    print(f"N2I loss on {len(files)} {split} frames, network (identity):")
    for k, tx in eval_tx.items():
        print(f"  {len(tx):2d} transmits: {np.mean(loss[k]):.4f} ({np.mean(loss_identity[k]):.4f})")

# %%
# In-vitro test on the CIRS phantom, and B-mode preview
# -----------------------------------------------------
bmode = lambda img: dinv.utils.bmode(img[None, None], amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=False)


def comparison(planes):
    """DAS and network for every EVAL_SETS acquisition and for all transmits."""
    images = {}
    for tx in list(eval_tx.values()) + [list(range(N_TOTAL))]:
        das = planes[tx].mean(0)
        images[f"DAS, {len(tx)} tx"], images[f"N2I any-n, {len(tx)} tx"] = das, denoise(das, len(tx))
    return images


invitro_path = os.path.join(CACHE_ROOT, n2i_invitro.FRAME)
if INVITRO_TEST and not os.path.exists(invitro_path):
    print(f"In-vitro test skipped: {invitro_path} not cached (run the precompute script)")
elif INVITRO_TEST:
    planes, invitro_manifest = n2i_invitro.load_frame(CACHE_ROOT)
    images = comparison(planes)
    print(f"In-vitro test on {n2i_invitro.FRAME}:")
    n2i_invitro.print_table({k: n2i_invitro.metrics(v, invitro_manifest) for k, v in images.items()})
    out = Path(trainer.save_path) / "invitro_bmode.png"
    n2i_invitro.plot(images, invitro_manifest, out, DYNAMIC_RANGE_DB)
    print(f"In-vitro B-mode with the regions written to {out}")

preview_file = (test_files or val_files)[0]
images = {k: bmode(v) for k, v in comparison(torch.from_numpy(np.load(preview_file))).items()}
out = Path(trainer.save_path) / "preview_bmode.png"
dinv.utils.plot(
    images,
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
    figsize=(2.5 * len(images), 6),
    suptitle=f"{os.path.relpath(preview_file, CACHE_ROOT)}, {DYNAMIC_RANGE_DB:.0f} dB",
    save_fn=str(out),
    show=False,
)
print(f"Preview written to {out}")
