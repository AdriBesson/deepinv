r"""
Noise2Inverse for ultrafast ultrasound -- multi-transmit input
==============================================================

Variant of ``train_n2i_deepinv.py`` (kept as the baseline) in which the network
sees the single-angle images of a fast acquisition separately instead of their
mean: the three transmits of a group are stacked as ``3 x 2 = 6`` real channels
(I and Q of each, sorted by steering angle), and the network still outputs one IQ
image (2 channels). The target is unchanged: the mean of the other transmits.

Why: with the mean of the three transmits as its only input, a network cannot
tell coherent speckle from incoherent noise, which look alike locally, so it
falls back on context. The baseline networks learned an orientation prior from
the in-vivo frames ("deep, or below a structure, means noise, so shrink"): on the
held-out CIRS phantom they darken the speckle below the cysts and at depth, the
patch follows the content when the input is cropped and vanishes when it is
flipped upside down, and flipping costs them about 2 dB of in-vivo error. Whether
the transmits agree is exactly the information they lack: the coherence of a
3-transmit image with the other transmits is 0.97-0.99 at all depths on the
phantom but falls to 0.2-0.3 deep in vivo. With the transmits as separate
channels the network can measure their local agreement itself. Noise2Inverse
stays valid, since the input still uses only the transmits of one group and the
noise of the other groups (the target) is independent of them.

Everything else follows the baseline: splits by volunteer, the K = 29
interleaved groups, the X:1 target, scale equivariance (RMS over all six
channels), TGC augmentation, lateral flips and (``AUG_VFLIP``) vertical flips. A
lateral flip mirrors the steering angles, so it also reverses the order of the
transmit channels; with interleaved groups the mirrored angles fall within one
angle step (0.38 deg) of another group.

Data: a per-transmit cache (``TX_CACHE_ROOT``) holds per frame the transmits of
every group, group after group and each group sorted by angle, followed by the
mean over all of them, as ``(n + 1, 2, Z, X)`` float32 (77 MB per frame). A
training step then reads one contiguous block of three planes and the mean. It is
built for the training and validation frames only (about 300 GB); the test frames
are read from the per-transmit cache of the precompute script.

Checkpoints go to ``./checkpoints/n2i_ultrasound_<ARCH>_multitx[_vflip]<RUN_SUFFIX>`` and
record ``checkpoint["input"] = "transmits"`` and the channel layout.
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
TX_CACHE_ROOT = "/data/n2i_tx_cache"  # transmits ordered by group, built by this script

# Network, as in the baseline. The N2I_ARCH environment variable overrides it.
ARCH = os.environ.get("N2I_ARCH", "unet")
# Same split as the round-4 baseline, so the two are directly comparable.
RUN_SUFFIX = "_test002_006_008"
# SAVE_PATH is set with the augmentation below: AUG_VFLIP runs get their own folder.

# "" from scratch, "auto" the latest run of this architecture, or a checkpoint path.
RESUME = "auto"

# Splits: whole volunteers, as in the round-4 baseline (see train_n2i_deepinv.py).
VAL_DIRS = ["volunteer_005"]
TEST_DIRS = ["volunteer_002", "volunteer_006", "volunteer_008"]  # empty skips the test
TRAIN_DIRS = ["volunteer_001", "volunteer_003", "volunteer_004", "volunteer_007"]  # None = all the rest
INVITRO_TRAIN = True
INVITRO_HELD_OUT = ["invitro_CIRS054GS/054GS"]
INVITRO_TEST = True

# Partition of the 87 transmits; every group must have the same size, which sets
# the number of input channels (2 per transmit).
K = 29
GROUPS = [list(range(j, 87, K)) for j in range(K)]
TARGET = "others"  # "others" (mean of all the other transmits) or "group"
VAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

SCALES = 4  # unet only
CHANNELS = 32
RESIDUAL = False  # unet only; needs as many input as output channels, so unused here
BLOCKS = 2  # drunet only
MSD_DEPTH, MSD_WIDTH, MSD_DILATION = 100, 1, 10  # msd only
UPSAMPLE = "bilinear"  # ported nets only
SCALE_EQUIVARIANT = True

AUG_TGC_DB = 10.0
AUG_TGC_KNOTS = 5
AUG_GAIN_DB = 0.0 if SCALE_EQUIVARIANT else 10.0
# Vertical flip of half the training pairs, as in the baseline (see
# train_n2i_deepinv.py); it leaves the steering angles, hence the channel order,
# unchanged. N2I_VFLIP=0 resumes or reproduces a run without it.
AUG_VFLIP = os.environ.get("N2I_VFLIP", "1") != "0"
SAVE_PATH = f"./checkpoints/n2i_ultrasound_{ARCH}_multitx{'_vflip' if AUG_VFLIP else ''}{RUN_SUFFIX}"

# Optimisation
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 30
NUM_WORKERS = 4
SEED = 0

DYNAMIC_RANGE_DB = 50.0  # for the B-mode preview

torch.manual_seed(SEED)
device = dinv.utils.get_device()

N_TX = len(GROUPS[0])
if any(len(g) != N_TX for g in GROUPS):
    raise ValueError("Every group must have the same number of transmits (input channels)")
IN_CHANNELS = 2 * N_TX


# %%
# Inputs
# ------
def random_gain_db(n_z):
    """Random gain in dB per depth: zero-mean TGC curve plus global gain, ``(n_z,)``."""
    knots = (2 * torch.rand(AUG_TGC_KNOTS) - 1) * AUG_TGC_DB
    z_knots = torch.linspace(0, n_z - 1, AUG_TGC_KNOTS)
    curve = torch.from_numpy(np.interp(np.arange(n_z), z_knots.numpy(), knots.numpy())).float()
    return curve - curve.mean() + (2 * torch.rand(1) - 1) * AUG_GAIN_DB


def sort_by_angle(tx, angles_deg):
    """Transmit indices ``tx`` in increasing steering angle (the channel order)."""
    return sorted(tx, key=lambda a: angles_deg[a])


def stack_transmits(planes):
    """``(n, Z, X)`` complex single-angle images -> ``(2 n, Z, X)`` real channels.

    The channels are I and Q of the first transmit, then of the second, ... the
    layout of the transmit cache reshaped to ``(2 n, Z, X)``.
    """
    return torch.view_as_real(planes).permute(0, 3, 1, 2).reshape(-1, *planes.shape[-2:])


class MultiTransmitDataset(Dataset):
    r"""One (input, target) pair per frame, the input being the transmits of a group.

    ``source="tx"`` reads the transmit cache built below, ``(n + 1, 2, Z, X)``
    float32 per frame: the transmits of group 0 sorted by angle, then of group 1,
    ..., then the mean over all of them. ``source="raw"`` reads the
    ``(87, Z, X)`` complex64 files of the precompute script directly (slower,
    used for the test set). Training draws the input group at random, the target
    being set by ``TARGET``; otherwise every frame gives one item per
    ``(input, target)`` pair of ``pairs``, the target being the mean of the
    target group.

    Training pairs are flipped laterally at random, which reverses the order of
    the transmits (the angles change sign), vertically too if ``AUG_VFLIP``, and
    multiplied by a random depth-gain curve. Returns ``'y'`` of shape ``(2 n_tx, Z, X)`` and ``'x'`` of ``(2, Z, X)``.
    """

    def __init__(self, files, groups, angles_deg, train, pairs=((0, 1),), source="tx"):
        self.files, self.train, self.pairs, self.source = list(files), train, list(pairs), source
        self.groups = [sort_by_angle(g, angles_deg) for g in groups]
        self.offsets = np.cumsum([0] + [len(g) for g in groups])
        self.n_total = self.offsets[-1]

    def __len__(self):
        return len(self.files) * (1 if self.train else len(self.pairs))

    def group(self, data, j):
        """``(n_j, 2, Z, X)`` transmits of group ``j``, sorted by angle."""
        if self.source == "tx":
            return torch.from_numpy(np.array(data[self.offsets[j] : self.offsets[j + 1]]))
        planes = torch.from_numpy(np.array(data[self.groups[j]]))  # complex64
        return torch.view_as_real(planes).permute(0, 3, 1, 2)

    def __getitem__(self, idx):
        if self.train:
            data = np.load(self.files[idx], mmap_mode="r")
            j, k = torch.randperm(len(self.groups))[:2].tolist()
        else:
            data = np.load(self.files[idx // len(self.pairs)], mmap_mode="r")
            j, k = self.pairs[idx % len(self.pairs)]
        inp = self.group(data, j)  # (n_j, 2, Z, X)
        if self.train and TARGET == "others":
            # Mean of the transmits outside group j, from the mean over all of them.
            n, n_j = self.n_total, inp.shape[0]
            tgt = (n * torch.from_numpy(np.array(data[-1])) - inp.sum(0)) / (n - n_j)
        else:
            tgt = self.group(data, k).mean(0)

        if self.train and torch.rand(1) > 0.5:
            # A lateral flip mirrors the steering angles: reverse the transmits too.
            inp, tgt = torch.flip(inp, (0, -1)), torch.flip(tgt, (-1,))
        if self.train and AUG_VFLIP and torch.rand(1) > 0.5:
            inp, tgt = torch.flip(inp, (-2,)), torch.flip(tgt, (-2,))
        if self.train and (AUG_TGC_DB or AUG_GAIN_DB):
            gain = 10 ** (random_gain_db(inp.shape[-2]) / 20)[:, None]  # amplitude, (Z, 1)
            inp, tgt = inp * gain, tgt * gain

        return {"x": tgt.contiguous(), "y": inp.reshape(-1, *inp.shape[-2:]).contiguous()}


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
# The channel order follows the steering angles, so they must agree everywhere.
angles_deg = manifest["angles_deg"]
if any(not np.allclose(m["angles_deg"], angles_deg) for m in manifests):
    raise ValueError("All folders must share the same transmit angles")
flat = sorted(a for g in GROUPS for a in g)
if flat != sorted(set(flat)) or not set(flat) <= set(range(manifest["n_angles"])):
    raise ValueError("GROUPS must be disjoint subsets of the transmit indices")
if TARGET not in ("others", "group"):
    raise ValueError(f"Unknown TARGET {TARGET!r}")

# %%
# Transmit cache
# --------------
# Reading a group's three planes from the precompute cache means three seeks
# across a 76 MB file, and the target needs the mean of all of them. Each
# training and validation frame is therefore rewritten once with the transmits of
# every group next to each other (sorted by angle) and the mean over all of them
# appended. The folder name hashes GROUPS, and a frame is rebuilt when its source
# is newer.
ordered = [a for g in GROUPS for a in sort_by_angle(g, angles_deg)]
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

train_dataset = MultiTransmitDataset(map(tx_path, train_files), GROUPS, angles_deg, train=True)
val_dataset = MultiTransmitDataset(map(tx_path, val_files), GROUPS, angles_deg, False, VAL_PAIRS)
test_dataset = MultiTransmitDataset(test_files, GROUPS, angles_deg, train=False, source="raw")

loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    persistent_workers=NUM_WORKERS > 0,
)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)

print(
    f"N2I over transmits, multi-transmit input: {len(GROUPS)} groups of {N_TX} transmits "
    f"-> {IN_CHANNELS} input channels, target: {TARGET}\n"
    f"  frames {tuple(train_dataset[0]['y'].shape[-2:])}\n"
    f"  train: {len(train_files)} frames from {train_folders}\n"
    f"  val:   {len(val_files)} frames from {VAL_DIRS}, group pairs {VAL_PAIRS}\n"
    f"  test:  {len(test_files)} frames from {TEST_DIRS}\n"
    f"  scale-equivariant: {SCALE_EQUIVARIANT}; augmentation: TGC +-{AUG_TGC_DB} dB "
    f"({AUG_TGC_KNOTS} knots), global gain +-{AUG_GAIN_DB} dB, lateral flip"
    f"{', vertical flip' if AUG_VFLIP else ''}"
)

# %%
# Model and training
# ------------------
physics = dinv.physics.Denoising().to(device)

if ARCH == "unet":
    denoiser = dinv.models.UNet(
        in_channels=IN_CHANNELS,
        out_channels=2,
        scales=SCALES,
        channels_per_scale=[CHANNELS * 2**k for k in range(SCALES)],
        residual=RESIDUAL,
        batch_norm=False,
    )
elif ARCH in ("resunet", "attunet", "attresunet"):
    net_cls = {"resunet": ResUNet, "attunet": AttentionUNet, "attresunet": AttentionResUNet}
    denoiser = net_cls[ARCH](
        in_channels=IN_CHANNELS, out_channels=2, base_channels=CHANNELS, upsample_mode=UPSAMPLE
    )
elif ARCH == "drunet":
    denoiser = dinv.models.DRUNet(
        in_channels=IN_CHANNELS,
        out_channels=2,
        nc=[CHANNELS * 2**k for k in range(4)],
        nb=BLOCKS,
        pretrained=None,
    )
elif ARCH == "msd":
    denoiser = MSDNet(
        in_channels=IN_CHANNELS,
        out_channels=2,
        depth=MSD_DEPTH,
        width=MSD_WIDTH,
        max_dilation=MSD_DILATION,
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
            "input": "transmits",  # the baseline's checkpoints have no key: "group mean"
            "n_tx": N_TX,
            "channel_order": "I, Q of each transmit, by increasing steering angle",
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
# Test on unseen volunteers
# -------------------------
# Noise2Inverse loss of the best checkpoint, group 0 -> group 1 as in the
# baseline; the identity reference is the mean of the input transmits.
best_ckpt = Path(trainer.save_path) / "ckp_best.pth.tar"
if not best_ckpt.exists() and resume_from:
    best_ckpt = Path(resume_from).parent / "ckp_best.pth.tar"
if best_ckpt.exists():
    trainer.load_model(best_ckpt)
else:
    print("No best checkpoint saved (no validation step ran); using the final weights")
model.eval()
if test_files:
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    loss_n2i, loss_identity = [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test"):
            x, y = batch["x"].to(device), batch["y"].to(device)
            loss_n2i.append(F.mse_loss(model(y, physics), x).item())
            y_mean = y.reshape(y.shape[0], N_TX, 2, *y.shape[-2:]).mean(1)
            loss_identity.append(F.mse_loss(y_mean, x).item())
    print(
        f"Test on {len(test_files)} frames from {TEST_DIRS}: "
        f"N2I loss {np.mean(loss_n2i):.4f} (identity {np.mean(loss_identity):.4f})"
    )


# %%
# In-vitro test on the CIRS phantom
# ---------------------------------
def denoise(planes, tx):
    """Denoise the fast acquisition of transmits ``tx`` of ``(87, Z, X)`` complex planes."""
    net_in = stack_transmits(planes[sort_by_angle(tx, angles_deg)])[None].to(device)
    with torch.no_grad():
        out = model(net_in, physics)[0].cpu()
    return torch.complex(out[0], out[1])


invitro_path = os.path.join(CACHE_ROOT, n2i_invitro.FRAME)
if INVITRO_TEST and not os.path.exists(invitro_path):
    print(f"In-vitro test skipped: {invitro_path} not cached (run the precompute script)")
elif INVITRO_TEST:
    planes, invitro_manifest = n2i_invitro.load_frame(CACHE_ROOT)
    images = {
        f"DAS, {N_TX} transmits": planes[GROUPS[0]].mean(0),
        f"N2I multi-transmit, {N_TX} transmits": denoise(planes, GROUPS[0]),
        f"CPWC, {invitro_manifest['n_angles']} transmits": planes.mean(0),
    }
    print(f"In-vitro test on {n2i_invitro.FRAME}:")
    n2i_invitro.print_table({k: n2i_invitro.metrics(v, invitro_manifest) for k, v in images.items()})
    out = Path(trainer.save_path) / "invitro_bmode.png"
    n2i_invitro.plot(images, invitro_manifest, out, DYNAMIC_RANGE_DB)
    print(f"In-vitro B-mode with the regions written to {out}")

# %%
# B-mode preview on one held-out frame
# ------------------------------------
preview_file = (test_files or val_files)[0]
planes = torch.from_numpy(np.load(preview_file))  # complex64
cpwc = planes.mean(0, keepdim=True)
fast = planes[GROUPS[0]].mean(0, keepdim=True)
denoised = denoise(planes, GROUPS[0]).unsqueeze(0)

bmode = lambda img: dinv.utils.bmode(img.unsqueeze(0), amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=False)
out = Path(trainer.save_path) / "preview_bmode.png"
dinv.utils.plot(
    {
        f"DAS, {N_TX} transmits": bmode(fast),
        f"N2I multi-transmit, {N_TX} transmits": bmode(denoised),
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
