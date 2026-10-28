r"""
Noise2Inverse for ultrafast ultrasound -- stage 2: self-supervised training
===========================================================================

Self-supervised denoising of plane-wave ultrasound reconstructions, trained
without ever seeing a clean image. Run ``precompute_n2i_ultrasound.py`` first: it
beamforms each of the 87 plane-wave transmits of every in-vivo frame into its own
single-angle image, demodulates it to complex baseband (IQ) and caches them. The
network sees the IQ images as two real channels (in-phase and quadrature).

This script partitions the 87 transmits into ``K`` disjoint sub-sequences; the
sub-reconstruction of a sub-sequence is the mean of its cached single-angle
images (the adjoint is linear in the transmits). Noise2Inverse
:footcite:t:`hendriksen2020noise2inverse` then trains a network to map one
sub-reconstruction to the mean of the other ones of the same frame. Because the
noise of disjoint transmit subsets is independent while the imaged anatomy is
not, the network can only fit the anatomy, and at inference it denoises a
single fast acquisition of ``87 / K`` plane waves. The sub-reconstructions are
condensed once into a second, smaller cache (``GROUP_CACHE_ROOT``), so that a
training step reads two images instead of six scattered single-angle ones. Splitting the loss this way is
equivalent, up to a term that does not depend on the network, to the
measurement-domain splitting of :class:`deepinv.loss.SplittingLoss`, but no
forward operator is needed in the loop, so a step costs what the U-Net costs.

Training, validation and test sets are whole folders of the cache (one per
volunteer and region), so the network is validated and tested on volunteers it
has never seen. There is no ground truth, so validation monitors the
Noise2Inverse loss itself, which also drives best-checkpoint saving and early
stopping; the test set reports the same loss with the best checkpoint.

The best checkpoint is also tested on the in-vitro CIRS phantom frame
``invitro_02162`` (contrast, CNR and gCNR of an anechoic and a hypoechoic cyst,
speckle SNR and speckle size, see ``n2i_invitro.py``). That frame never enters
training or validation; the other phantom frames are part of the training set
when ``INVITRO_TRAIN`` is set, except those of the ``INVITRO_HELD_OUT``
folders, and never of the validation set.
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
GROUP_CACHE_ROOT = "/data/n2i_group_cache"  # sub-reconstructions, built by this script
TX_CACHE_ROOT = "/data/n2i_tx_cache"  # of train_n2i_multitx_deepinv.py, used for K = 87

# Network. Each architecture keeps its own checkpoints.
#   "unet"       deepinv U-Net, concatenating skips, ``SCALES`` deep
#   "resunet"    residual U-Net of the ai_for_image_quality project (n2i_models.py)
#   "attunet"    attention-gated U-Net of the same project
#   "attresunet" both residual blocks and attention gates
#   "drunet"     residual U-Net of :footcite:t:`zhang2021plug`, ``BLOCKS`` blocks
#                per scale, additive skips (:class:`deepinv.models.DRUNet`)
#   "msd"        mixed-scale dense network of :footcite:t:`pelt2018mixed`, the
#                network of the Noise2Inverse paper: no down-sampling, MSD_DEPTH
#                layers of MSD_WIDTH channels, each fed by all the earlier ones
# The N2I_ARCH environment variable overrides it, to queue several networks:
#   for a in unet resunet attresunet drunet; do N2I_ARCH=$a python ...; done
ARCH = os.environ.get("N2I_ARCH", "unet")
# Runs of a different data split go to their own folders, so RESUME = "auto"
# never continues a run trained on another split.
RUN_SUFFIX = "_test002_006_008"  # "_test002" for round 3, "" for the rounds 1-2 split
# SAVE_PATH is set with the augmentation below: AUG_VFLIP runs get their own folder.

# Restarting an interrupted run. Every run writes to its own timestamped folder
# under SAVE_PATH, so resuming never overwrites the run it continues.
#   ""      train from scratch
#   "auto"  continue the most recent run of this architecture, from its latest
#           periodic checkpoint (or its best one if no periodic one was reached)
#   a path  continue from that checkpoint file
# Weights, optimiser, scheduler, epoch counter and loss histories are restored,
# so EPOCHS below stays the *total* number of epochs, cosine annealing picks up
# where it stopped, and early stopping keeps counting from the real best epoch.
RESUME = "auto"

# Folders of the cache (relative to CACHE_ROOT) assigned to each split, searched
# recursively: a whole volunteer or a single region. Frames of one acquisition
# are highly correlated, so splits are made by volunteer. The training set is
# every cached in-vivo frame (``volunteer_*`` folders) that is in neither the
# validation nor the test set, plus the phantom frames if INVITRO_TRAIN; set
# TRAIN_DIRS to a list of folders to restrict it instead. VAL_DIRS and TEST_DIRS
# are in-vivo only.
# Splits are whole volunteers (checked below), listed explicitly so that a newly
# cached volunteer never joins training silently. Frames per region:
#   train  001 (abdomen 1008), 003 (abdomen 35, breast 679), 004 (breast 513,
#          carotid 594), 007 (abdomen 449, breast 63)            -> 3341 frames
#   val    005 (abdomen 353, carotid 100)                          ->  453 frames
#   test   002 (abdomen 298, carotid 161), 006 (carotid 289),
#          008 (abdomen 602, back 542, carotid 381)               -> 2273 frames
# Breast is only in training and back only in the test set (unseen anatomy).
# Round 3 (RUN_SUFFIX "_test002") validated on 005 + 006 carotid and tested on 002.
VAL_DIRS = ["volunteer_005"]
TEST_DIRS = ["volunteer_002", "volunteer_006", "volunteer_008"]  # empty skips the test
# N2I_TEST=0 skips it too, so that a run with another K caches no test frames
# (eval_n2i_deepinv.py tests from the per-transmit cache anyway).
if os.environ.get("N2I_TEST", "1") == "0":
    TEST_DIRS = []
TRAIN_DIRS = ["volunteer_001", "volunteer_003", "volunteer_004", "volunteer_007"]  # None = all the rest
# CIRS phantom (``invitro_*`` folders): every cached frame but the test frame
# invitro_02162 of n2i_invitro.py and INVITRO_HELD_OUT joins the training set.
INVITRO_TRAIN = True
# Phantom folders kept out of training anyway. The other frames of the 054GS
# section likely image the same phantom position as invitro_02162, so the test
# frame is only independent of training if the whole section is held out.
INVITRO_HELD_OUT = ["invitro_CIRS054GS/054GS"]
# In-vitro test on invitro_02162 after training; needs that frame in the cache.
INVITRO_TEST = True

# Partition of the 87 transmits into K sub-sequences, each giving one
# sub-reconstruction of 87 / K plane waves. The default is interleaved,
# ``group j = {j, j+K, j+2K, ...}``: every group spans the full angular aperture,
# so the sub-reconstructions differ in noise rather than in point spread
# function. Replace ``GROUPS`` by any list of disjoint lists of transmit indices
# (acquisition order: alternating +/-, widest angle first, 0 deg last).
# The N2I_K environment variable overrides K (runs with K != 29 save under a
# "_K<K>" folder); the groups then have 87 // K or 87 // K + 1 transmits. With
# K = 87 (single transmits), the transmit cache of train_n2i_multitx_deepinv.py
# serves as the group cache: same layout, so no 300 GB copy of it is made.
K = int(os.environ.get("N2I_K", 29))
GROUPS = [list(range(j, 87, K)) for j in range(K)]
if K == 87:
    groups_29 = [list(range(j, 87, 29)) for j in range(29)]
    tx_root = os.path.join(
        TX_CACHE_ROOT, "K29_" + hashlib.sha1(json.dumps(groups_29).encode()).hexdigest()[:10]
    )
    with open(os.path.join(tx_root, "layout.json")) as fh:
        GROUPS = [[t] for t in json.load(fh)["ordered_transmits"]]
# Training target for an input sub-reconstruction:
#   "others"  the mean of all the other groups (the X:1 strategy of
#             Noise2Inverse): same expectation as a single group, but its noise
#             variance is K - 1 times lower, so the gradients are far less noisy
#   "group"   one other group drawn at random (rounds 1-2)
#   "cpwc"    supervised baseline, not N2I: the compound of all the transmits
#             (87-transmit CPWC), input group included, as in supervised
#             few-to-many-transmit networks; its runs save under a "_sup" folder
# Validation and test map one group to another one (VAL_PAIRS), as in the
# earlier rounds and in eval_n2i_deepinv.py; with "cpwc" they map each input
# group of VAL_PAIRS to CPWC instead, so the supervised loss selects the
# checkpoint. The N2I_TARGET environment variable overrides it.
TARGET = os.environ.get("N2I_TARGET", "others")
# Pairs of groups (input, target) the validation loss is averaged over. A single
# pair leaves the loss that selects the best checkpoint and drives early stopping
# noisier than the differences between epochs. The test loss uses (0, 1) alone,
# the definition of eval_n2i_deepinv.py.
VAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

SCALES = 4  # unet only; the ported nets have 5 scales, DRUNet 4
CHANNELS = 32  # width at the finest scale, doubled at each scale
RESIDUAL = False  # unet only: global skip, the network predicts the residual
BLOCKS = 2  # drunet only: residual blocks per scale (4 in the original)
# msd only: layers, channels per layer and largest dilation (dilations cycle
# 1..MSD_DILATION); 100, 1, 10 as in the Noise2Inverse paper
MSD_DEPTH, MSD_WIDTH, MSD_DILATION = 100, 1, 10
UPSAMPLE = "bilinear"  # ported nets only: "bilinear" or "conv_transpose"
# Divide each input by its RMS and multiply the output back, so that the network
# is exactly scale-equivariant, f(a y) = a f(y), and cannot key on absolute
# brightness. Rounds 1-3 were trained without it: on the held-out phantom every
# network then hallucinated dark patches at depth, 2-4x larger when the input was
# scaled by 0.5 or 2. Stored in the checkpoints (see PaddedArtifactRemoval).
SCALE_EQUIVARIANT = True

# Brightness augmentation of the training pairs, the same real gain applied to the
# input and the target, so their noises stay independent and the N2I loss valid:
#   AUG_TGC_DB   random depth-gain curve, like the TGC sliders of a scanner: gains
#                drawn uniformly in +-AUG_TGC_DB at AUG_TGC_KNOTS equally spaced
#                depths, linear in dB between them, then shifted to zero mean, so
#                that "deep" no longer implies "weak" and "weak" no longer
#                implies "noise"
#   AUG_GAIN_DB  random global gain, uniform in +-AUG_GAIN_DB; a no-op on the input
#                of a scale-equivariant network (it would only reweight the loss
#                of each frame), so only used without SCALE_EQUIVARIANT
# 0 disables either. Validation and test frames are never augmented.
AUG_TGC_DB = 10.0
AUG_TGC_KNOTS = 5
AUG_GAIN_DB = 0.0 if SCALE_EQUIVARIANT else 10.0
# Vertical (axial) flip of half the training pairs, input and target alike. Not
# physically valid (attenuation, depth-dependent PSF and focusing are not
# symmetric in depth), but the N2I loss stays valid, and it takes away the
# orientation cue the round-4 networks used: on the held-out phantom they darken
# the speckle below the cysts and at depth, and those dark regions vanish when the
# input is flipped upside down. The N2I_VFLIP environment variable overrides it
# (N2I_VFLIP=0 to resume or reproduce a round-4 run).
AUG_VFLIP = os.environ.get("N2I_VFLIP", "1") != "0"
# Seed of the weight initialisation and of the training pairs. Runs with another
# seed than 0 (N2I_SEED) save under a "_seed<n>" folder, so that RESUME = "auto"
# never continues the run of another seed.
SEED = int(os.environ.get("N2I_SEED", 0))
# Linear learning-rate warm-up over the first WARMUP_EPOCHS epochs, on top of the
# cosine annealing (0 = none, as in rounds 1-4); runs with a warm-up save under a
# "_warmup<n>" folder. The N2I_WARMUP environment variable overrides it.
WARMUP_EPOCHS = int(os.environ.get("N2I_WARMUP", 0))
SAVE_PATH = (
    f"./checkpoints/n2i_ultrasound_{ARCH}{'_vflip' if AUG_VFLIP else ''}"
    f"{'_sup' if TARGET == 'cpwc' else ''}{RUN_SUFFIX}{f'_K{K}' if K != 29 else ''}"
    f"{f'_warmup{WARMUP_EPOCHS}' if WARMUP_EPOCHS else ''}{f'_seed{SEED}' if SEED else ''}"
)

# Optimisation
EPOCHS = 100
BATCH_SIZE = 8  # whole 480x256 IQ frames
LR = 1e-3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 30  # epochs without validation improvement
NUM_WORKERS = 4

DYNAMIC_RANGE_DB = 50.0  # for the B-mode preview

torch.manual_seed(SEED)
device = dinv.utils.get_device()


# %%
# Noise2Inverse pairs
# -------------------
def random_gain_db(n_z):
    """Random gain in dB per depth: zero-mean TGC curve plus global gain, ``(n_z,)``."""
    knots = (2 * torch.rand(AUG_TGC_KNOTS) - 1) * AUG_TGC_DB
    z_knots = torch.linspace(0, n_z - 1, AUG_TGC_KNOTS)
    curve = torch.from_numpy(np.interp(np.arange(n_z), z_knots.numpy(), knots.numpy())).float()
    return curve - curve.mean() + (2 * torch.rand(1) - 1) * AUG_GAIN_DB


class Noise2InverseDataset(Dataset):
    r"""One (input, target) pair of sub-reconstructions per cached frame.

    Reads the group cache built below: per frame, the ``K`` sub-reconstructions
    followed by the mean over every transmit of ``GROUPS``, as ``(K + 1, 2, Z, X)``
    float32 (I and Q channels). At train time the input group is drawn at random
    and the target is set by ``TARGET``; otherwise every frame gives one item per
    ``(input, target)`` group pair of ``pairs``, so validation and test losses are
    deterministic. The cache is memory-mapped, so only the two images actually
    used are read from disk.

    Training pairs are flipped laterally at random, vertically too if
    ``AUG_VFLIP``, and multiplied by a random depth-gain curve and global gain
    (``AUG_TGC_DB``, ``AUG_GAIN_DB``).

    Returns the deepinv convention: ``'x'`` is the target of the loss, ``'y'``
    the network input, both as ``(2, Z, X)`` real tensors (I and Q channels).
    """

    def __init__(self, files, groups, train, pairs=((0, 1),)):
        self.files, self.train, self.pairs = list(files), train, list(pairs)
        self.sizes = [len(g) for g in groups]
        self.n_total = sum(self.sizes)

    def __len__(self):
        return len(self.files) * (1 if self.train else len(self.pairs))

    def __getitem__(self, idx):
        if self.train:
            means = np.load(self.files[idx], mmap_mode="r")  # (K + 1, 2, Z, X) float32
            j, k = torch.randperm(len(self.sizes))[:2].tolist()
        else:
            means = np.load(self.files[idx // len(self.pairs)], mmap_mode="r")
            j, k = self.pairs[idx % len(self.pairs)]
        inp = torch.from_numpy(np.array(means[j]))
        if TARGET == "cpwc":
            tgt = torch.from_numpy(np.array(means[-1]))
        elif self.train and TARGET == "others":
            # Mean of the transmits outside group j, from the mean over all of them.
            n, n_j = self.n_total, self.sizes[j]
            tgt = (n * torch.from_numpy(np.array(means[-1])) - n_j * inp) / (n - n_j)
        else:
            tgt = torch.from_numpy(np.array(means[k]))

        if self.train and torch.rand(1) > 0.5:  # lateral flip only: physics-valid
            inp, tgt = torch.flip(inp, (-1,)), torch.flip(tgt, (-1,))
        if self.train and AUG_VFLIP and torch.rand(1) > 0.5:
            inp, tgt = torch.flip(inp, (-2,)), torch.flip(tgt, (-2,))
        if self.train and (AUG_TGC_DB or AUG_GAIN_DB):
            gain = random_gain_db(inp.shape[-2])
            gain = 10 ** (gain / 20)[:, None]  # amplitude, (Z, 1)
            inp, tgt = inp * gain, tgt * gain

        return {"x": tgt.contiguous(), "y": inp.contiguous()}


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
# The in-vitro test frame is held out of training whatever the folders.
held_out = set(val_files) | set(test_files) | {os.path.join(CACHE_ROOT, n2i_invitro.FRAME)}
held_out |= set(find(INVITRO_HELD_OUT, "invitro_*.npy"))
train_dirs = (TRAIN_DIRS or ["volunteer_*"]) + (["invitro_*"] if INVITRO_TRAIN else [])
train_files = [f for f in find(train_dirs, "invi*_*.npy") if f not in held_out]
if not train_files:
    raise FileNotFoundError(f"No cached frames left for training under {CACHE_ROOT}")
# Folders (volunteer/region) each split draws from, for the log.
folders = lambda files: sorted({os.path.relpath(os.path.dirname(f), CACHE_ROOT) for f in files})
train_folders = folders(train_files)
volunteers = lambda files: {Path(f).relative_to(CACHE_ROOT).parts[0] for f in files if "invivo_" in f}
for (a, fa), (b, fb) in itertools.combinations(
    [("train", train_files), ("val", val_files), ("test", test_files)], 2
):
    if shared := volunteers(fa) & volunteers(fb):
        raise ValueError(f"Volunteers {sorted(shared)} are in both the {a} and {b} sets")
# A frame still being written by the precompute script fails to memory-map.
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
flat = sorted(a for g in GROUPS for a in g)
if flat != sorted(set(flat)) or not set(flat) <= set(range(manifest["n_angles"])):
    raise ValueError("GROUPS must be disjoint subsets of the transmit indices")

if TARGET not in ("others", "group", "cpwc"):
    raise ValueError(f"Unknown TARGET {TARGET!r}")

# %%
# Group cache
# -----------
# A training sample needs two sub-reconstructions of a frame. Averaged on the fly
# from the per-transmit cache, that is six planes scattered across a 76 MB file,
# which on a hard drive takes longer than the network step (0.4 s per frame).
# So each frame is condensed once into its K sub-reconstructions followed by the
# mean over every transmit of GROUPS, from which the mean of the other groups
# follows (26 MB per frame for K = 29, in the float32 of the per-transmit cache).
# The folder name hashes GROUPS, so another partition gets its own cache, and a
# frame is rebuilt when its per-transmit source is newer.
group_root = os.path.join(
    GROUP_CACHE_ROOT,
    f"K{len(GROUPS)}_" + hashlib.sha1(json.dumps(GROUPS).encode()).hexdigest()[:10],
)
if K == 87:
    group_root = tx_root
group_path = lambda f: os.path.join(group_root, os.path.relpath(f, CACHE_ROOT))


def build_group_cache(files):
    """Write the group file of every frame whose one is missing or older than it."""
    stale = [
        f
        for f in files
        if not os.path.exists(group_path(f))
        or os.path.getmtime(group_path(f)) < os.path.getmtime(f)
    ]
    if not stale:
        return
    os.makedirs(group_root, exist_ok=True)
    with open(os.path.join(group_root, "groups.json"), "w") as fh:
        json.dump(GROUPS, fh)
    every = sorted(a for g in GROUPS for a in g)
    for f in tqdm(stale, desc=f"Group cache -> {group_root}"):
        planes = np.load(f)  # the whole file in one sequential read
        means = np.stack([planes[g].mean(0) for g in GROUPS] + [planes[every].mean(0)])
        out = np.stack([means.real, means.imag], axis=1)  # float32
        os.makedirs(os.path.dirname(group_path(f)), exist_ok=True)
        with open(group_path(f) + ".tmp", "wb") as fh:  # never leave a truncated file
            np.save(fh, out)
        os.replace(group_path(f) + ".tmp", group_path(f))


build_group_cache(train_files + val_files + test_files)

train_dataset = Noise2InverseDataset(map(group_path, train_files), GROUPS, train=True)
val_dataset = Noise2InverseDataset(map(group_path, val_files), GROUPS, False, VAL_PAIRS)
test_dataset = Noise2InverseDataset(map(group_path, test_files), GROUPS, train=False)

loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    persistent_workers=NUM_WORKERS > 0,
)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)

print(
    f"N2I over transmits: {len(GROUPS)} sub-sequences of sizes {[len(g) for g in GROUPS]}, "
    f"target: {TARGET}\n"
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
# The cache already holds :math:`A^{\top} y`, so the network sees images directly
# and the trainer's physics is the identity.
physics = dinv.physics.Denoising().to(device)

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
    # DRUNet appends a noise-level map to its input; the trainer feeds it the
    # (constant) sigma of the Denoising physics, which is just an extra constant
    # channel here since every sub-reconstruction averages the same number of
    # transmits.
    denoiser = dinv.models.DRUNet(
        in_channels=2,
        out_channels=2,
        nc=[CHANNELS * 2**k for k in range(4)],
        nb=BLOCKS,
        pretrained=None,
    )
elif ARCH == "msd":
    denoiser = MSDNet(
        in_channels=2,
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
if WARMUP_EPOCHS:
    # Closed form in the epoch, so a resumed run picks up the schedule exactly.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: min(1.0, (e + 1) / WARMUP_EPOCHS) * (1 + math.cos(math.pi * e / EPOCHS)) / 2,
    )
else:
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
    # Runs sort by their timestamped folder, and within a run the checkpoints are
    # written in epoch order, so the most recently written one is the furthest
    # along: ckp_best.pth.tar when the last epoch improved on the validation
    # loss, the last periodic ckp_<epoch>.pth.tar otherwise.
    for run in sorted(glob.glob(os.path.join(SAVE_PATH, "*")), reverse=True):
        if ckpts := glob.glob(os.path.join(run, "ckp_*.pth.tar")):
            return max(ckpts, key=os.path.getmtime)
    return None


class ResumableTrainer(dinv.Trainer):
    """Trainer that also restores the loss histories held in a checkpoint.

    :meth:`deepinv.Trainer.load_model` restores the weights, the optimiser, the
    scheduler and the epoch counter, but leaves the histories empty. They are
    what :meth:`deepinv.Trainer.save_best_model` compares against and what
    :meth:`deepinv.Trainer.stop_criterion` measures patience from, so without
    them a resumed run would treat its first validation as the best ever and
    then early-stop at once, the current epoch already being far past epoch 0.
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
            # The scheduler state also carries the horizon it was built with, so
            # a resume that raises EPOCHS would keep annealing to the old one.
            # CosineAnnealingLR steps recursively from the current learning
            # rate, so put it back on the schedule in closed form as well.
            sched = self.scheduler
            if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                sched.T_max = self.epochs
                for group, base in zip(self.optimizer.param_groups, sched.base_lrs):
                    group["lr"] = sched.eta_min + (base - sched.eta_min) * (
                        1 + math.cos(math.pi * sched.last_epoch / sched.T_max)
                    ) / 2
        return checkpoint

    def save_model(self, filename, epoch, state=None):
        # The evaluation scripts rebuild the model from these.
        extra = {
            "scale_equivariant": SCALE_EQUIVARIANT,
            "target": TARGET,
            "groups": GROUPS,
            "warmup_epochs": WARMUP_EPOCHS,
            "augmentation": {
                "tgc_db": AUG_TGC_DB,
                "tgc_knots": AUG_TGC_KNOTS,
                "gain_db": AUG_GAIN_DB,
                "vflip": AUG_VFLIP,
            },
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
    freq_update_progress_bar=10,  # refreshing the loss display forces a GPU sync
)

if resume_from:
    print(f"Resuming from {resume_from}")
print(f"Training on {device} up to epoch {EPOCHS}; checkpoints -> {SAVE_PATH}")
trainer.train()

# %%
# Test on unseen volunteers
# -------------------------
# Noise2Inverse loss of the best checkpoint on the test folders. As a reference,
# the loss of the identity (feeding the input sub-reconstruction straight
# through) is the noise floor a denoiser has to beat.
# The best checkpoint lives in this run's folder, unless the run was resumed and
# never improved on the validation loss it inherited, in which case it is still
# the one of the run we continued.
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
        for batch in test_dataloader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            loss_n2i.append(F.mse_loss(model(y, physics), x).item())
            loss_identity.append(F.mse_loss(y, x).item())
    print(
        f"Test on {len(test_files)} frames from {TEST_DIRS}: "
        f"{'loss against CPWC' if TARGET == 'cpwc' else 'N2I loss'} {np.mean(loss_n2i):.4f} "
        f"(identity {np.mean(loss_identity):.4f})"
    )

# %%
# In-vitro test on the CIRS phantom
# ---------------------------------
# Contrast, CNR and gCNR of the two cysts of invitro_02162, speckle SNR and
# speckle size, for the fast acquisition (first group, as the network sees it
# during training), its denoising, and the full-aperture CPWC reference.
def denoise(fast):
    """Denoise a ``(Z, X)`` complex image with the trained model."""
    net_in = torch.view_as_real(fast).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(net_in, physics)[0].cpu()
    return torch.complex(out[0], out[1])


invitro_path = os.path.join(CACHE_ROOT, n2i_invitro.FRAME)
if INVITRO_TEST and not os.path.exists(invitro_path):
    print(f"In-vitro test skipped: {invitro_path} not cached (run the precompute script)")
elif INVITRO_TEST:
    planes, invitro_manifest = n2i_invitro.load_frame(CACHE_ROOT)
    fast, n_tx = planes[GROUPS[0]].mean(0), len(GROUPS[0])
    images = {
        f"DAS, {n_tx} transmits": fast,
        f"Noise2Inverse, {n_tx} transmits": denoise(fast),
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
# Fast acquisition (the sub-reconstruction of the first group, as the network
# sees it during training), its Noise2Inverse denoising, and the full-aperture
# CPWC reference (mean of all single-angle images).
preview_file = (test_files or val_files)[0]
planes = torch.from_numpy(np.load(preview_file))  # complex64
cpwc = planes.mean(0, keepdim=True)
fast, n_tx = planes[GROUPS[0]].mean(0, keepdim=True), len(GROUPS[0])

denoised = denoise(fast[0]).unsqueeze(0)

# The envelope of an IQ image is its modulus, which ``bmode`` uses directly.
bmode = lambda img: dinv.utils.bmode(img.unsqueeze(0), amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=False)
out = Path(trainer.save_path) / "preview_bmode.png"
dinv.utils.plot(
    {
        f"DAS, {n_tx} transmits": bmode(fast),
        f"Noise2Inverse, {n_tx} transmits": bmode(denoised),
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
