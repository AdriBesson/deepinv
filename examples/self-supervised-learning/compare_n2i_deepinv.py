r"""
Noise2Inverse for ultrafast ultrasound -- network comparison
============================================================

Compare the networks trained by ``train_n2i_deepinv.py`` (U-Net, ResUNet,
AttResUNet, DRUNet, ...) on the same fast plane-wave acquisitions. As in
``infer_n2i_deepinv.py``, a fast acquisition is the sub-reconstruction of the few
steering angles ``ANGLES_DEG``.

Two kinds of networks can be compared, alone or side by side (``INPUTS``): those
of the baseline ``train_n2i_deepinv.py`` (``"mean"``), which see the mean of the
transmits of the fast acquisition, and those of ``train_n2i_multitx_deepinv.py``
(``"transmits"``), which see them as separate channels (I and Q of each, sorted
by steering angle), each with or without vertical-flip augmentation
(``"_vflip"``). Each checkpoint says which input it expects
(``checkpoint["input"]``, absent for the baseline) and how it was augmented.
The networks of ``train_s2i_deepinv.py`` (``"s2i"``, recognised by the
``s2i_scale`` of their checkpoint) can be compared with them: as in that script,
their estimate is the mean of the network over the leave-one-out
sub-reconstructions of the fast acquisition. So can the networks of
``train_n2i_anyn_deepinv.py`` (``"mean_anyn"``), trained on the mean of any
number of transmits: they see the mean of the fast acquisition, whatever its
number of angles, and DRUNet is told that number through its noise-level
channel, with the rule recorded in the checkpoint.

For every frame of the requested folders, the script saves a B-mode figure with
the fast DAS input, its denoising by every network and the full-aperture CPWC
reference. It then runs the in-vitro test of ``n2i_invitro.py`` on the CIRS
phantom frame ``invitro_02162``: contrast, CNR and gCNR of its anechoic and
hypoechoic cysts, speckle SNR and speckle size, printed and written to a CSV for
every method, with the same B-mode figure and the measurement regions drawn.
Figures are only saved to files, never displayed.
"""

import csv
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

import deepinv as dinv
import n2i_invitro
from n2i_models import AttentionResUNet, AttentionUNet, MSDNet, PaddedArtifactRemoval, ResUNet

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
# Folders to evaluate: use ones the networks were not trained on (the validation
# folder of the training script, or a test volunteer).
INPUT_DIRS = ["volunteer_002/carotid", "volunteer_006/carotid", "volunteer_008"]  # test volunteers

# Kinds of networks to compare, any of
#   "mean"       baseline networks, fed the mean of the transmits (2 channels)
#   "transmits"  multi-transmit networks, fed each transmit (2 channels each)
# and the same trained with vertical-flip augmentation (AUG_VFLIP of the training
# scripts), "mean_vflip" and "transmits_vflip"; and
#   "s2i"        Sparse2Inverse networks of train_s2i_deepinv.py
#   "mean_anyn"  networks of train_n2i_anyn_deepinv.py, fed the mean of any
#                number of transmits ("mean_anyn_vflip" with vertical flips)
#   "mean_sup"   supervised baselines of train_n2i_deepinv.py with TARGET="cpwc"
#                ("mean_vflip_sup" with vertical flips); "transmits_sup" those of
#                train_n2i_multitx_deepinv.py
# The N2I_INPUTS environment variable overrides it, e.g. N2I_INPUTS=transmits or
# N2I_INPUTS=mean,s2i.
INPUTS = ["transmits_vflip"]
INPUTS = os.environ.get("N2I_INPUTS", ",".join(INPUTS)).split(",")
# Checkpoint folder suffix of each kind, as in the SAVE_PATH of the training scripts.
SAVE_SUFFIX = {
    "mean": "",
    "transmits": "_multitx",
    "mean_vflip": "_vflip",
    "transmits_vflip": "_multitx_vflip",
    "s2i": "",
    "mean_anyn": "_anyn",
    "mean_anyn_vflip": "_anyn_vflip",
    "mean_sup": "_sup",
    "mean_vflip_sup": "_vflip_sup",
    "transmits_sup": "_multitx_sup",
}
# Checkpoint folder prefix, s2i_ultrasound for train_s2i_deepinv.py.
SAVE_PREFIX = {"s2i": "s2i_ultrasound"}
if not INPUTS or not set(INPUTS) <= set(SAVE_SUFFIX):
    raise ValueError(f"INPUTS must be among {list(SAVE_SUFFIX)}, got {INPUTS}")

# Networks to compare, with their checkpoint; None = ``ckp_best`` of the latest
# training run under ``./checkpoints/<prefix>_<arch><suffix><SPLIT_SUFFIX>``
# for every kind of INPUTS, prefix being SAVE_PREFIX of that kind (default
# n2i_ultrasound) and suffix its SAVE_SUFFIX (for the
# U-Net without any suffix, the early runs saved under
# ``./checkpoints/n2i_ultrasound`` are also searched). An explicit path is used
# for the kind its checkpoint records, if that kind is in INPUTS.
# Networks without any checkpoint are skipped.
CHECKPOINTS = {
    "unet": None,
    "resunet": None,
    "attresunet": None,
    "drunet": None,
    "msd": None,
}
# The N2I_ARCHS environment variable keeps only some of them, e.g. N2I_ARCHS=drunet.
if archs := os.environ.get("N2I_ARCHS"):
    CHECKPOINTS = {a: CHECKPOINTS.get(a) for a in archs.split(",")}
# RUN_SUFFIX of the training scripts, e.g. "_test002" (round 3); the
# N2I_SPLIT_SUFFIX environment variable overrides it.
SPLIT_SUFFIX = os.environ.get("N2I_SPLIT_SUFFIX", "_test002_006_008")
# RUN_SUFFIX of train_s2i_deepinv.py, whose split is that of N2I round 3
# (volunteer_006/carotid is one of its validation folders, not a test one).
S2I_SPLIT_SUFFIX = "_test002"
S2I_HELD_OUT = 1  # HELD_OUT of train_s2i_deepinv.py
# Each data split and choice of inputs writes to its own folder, so a run never
# overwrites the figures and tables of another one, e.g. "_multitx_vs_mean".
TAG = lambda kind: "s2i" if kind == "s2i" else SAVE_SUFFIX[kind].lstrip("_") or "mean"
OUT_TAG = (
    SAVE_SUFFIX[INPUTS[0]] if len(set(INPUTS)) == 1 and INPUTS[0] != "s2i"
    else "_" + "_vs_".join(TAG(i) for i in INPUTS)
)
OUT_DIR = f"./results/n2i_ultrasound_comparison{OUT_TAG}{SPLIT_SUFFIX}"
NAMES = {
    "unet": "UNet",
    "resunet": "ResUNet",
    "attunet": "AttUNet",
    "attresunet": "AttResUNet",
    "drunet": "DRUNet",
    "msd": "MS-D",
}

# Steering angles of the fast acquisition, in degrees; each is mapped to the
# nearest transmit of the sequence (0.38 deg steps between -16.3 and +16.3).
# A multi-transmit network needs as many angles as it was trained with (n_tx in
# its checkpoint); its training groups are the interleaved {j, j+29, j+58}, i.e.
# about 16.3 deg apart (group 28: -11.0, 0.0, +5.7 deg). An any-n network takes
# any number of angles (1 to 80 in training).
ANGLES_DEG = [-10.0, 0.0, 10.0]

# Networks: must match the training script.
SCALES = 4  # unet only
CHANNELS = 32
RESIDUAL = False  # unet only
BLOCKS = 2  # drunet only
UPSAMPLE = "bilinear"  # ported nets only
MSD_DEPTH, MSD_WIDTH, MSD_DILATION = 100, 1, 10  # msd only

N_FRAMES = 5  # cap on frames per folder (None = all)
INVITRO_TEST = True  # phantom frame invitro_02162, if cached
DYNAMIC_RANGE_DB = 50.0

device = dinv.utils.get_device()


# %%
# Networks
# --------
def build_denoiser(arch, in_channels=2):
    """``in_channels``: 2 (I and Q) for the baseline, ``2 n_tx`` for multi-transmit."""
    if arch in ("resunet", "attunet", "attresunet"):
        net_cls = {"resunet": ResUNet, "attunet": AttentionUNet, "attresunet": AttentionResUNet}
        return net_cls[arch](
            in_channels=in_channels, out_channels=2, base_channels=CHANNELS, upsample_mode=UPSAMPLE
        )
    if arch == "unet":
        return dinv.models.UNet(
            in_channels=in_channels,
            out_channels=2,
            scales=SCALES,
            channels_per_scale=[CHANNELS * 2**k for k in range(SCALES)],
            residual=RESIDUAL and in_channels == 2,  # needs as many in as out channels
            batch_norm=False,
        )
    if arch == "drunet":
        return dinv.models.DRUNet(
            in_channels=in_channels,
            out_channels=2,
            nc=[CHANNELS * 2**k for k in range(4)],
            nb=BLOCKS,
            pretrained=None,
        )
    if arch == "msd":
        return MSDNet(
            in_channels=in_channels,
            out_channels=2,
            depth=MSD_DEPTH,
            width=MSD_WIDTH,
            max_dilation=MSD_DILATION,
        )
    raise ValueError(f"Unknown arch {arch!r}")


def find_checkpoint(arch, suffix, prefix="n2i_ultrasound"):
    save_paths = [f"./checkpoints/{prefix}_{arch}{suffix}"]
    if arch == "unet" and not suffix and prefix == "n2i_ultrasound":
        save_paths.append("./checkpoints/n2i_ultrasound")
    for save_path in save_paths:
        found = sorted(glob.glob(os.path.join(save_path, "*", "ckp_best.pth.tar")))
        if found:
            return found[-1]
    return None


# name -> (model, n_tx, kind, noise), n_tx = None for a network fed the mean of
# the transmits (or, for "s2i", of the leave-one-out subsets of them), noise the
# noise-level rule of a "mean_anyn" network (None for the others).
models = {}
for inputs in INPUTS:
    for arch, ckpt_path in CHECKPOINTS.items():
        prefix = SAVE_PREFIX.get(inputs, "n2i_ultrasound")
        suffix = SAVE_SUFFIX[inputs] + (S2I_SPLIT_SUFFIX if inputs == "s2i" else SPLIT_SUFFIX)
        ckpt_path = ckpt_path or find_checkpoint(arch, suffix, prefix)
        if ckpt_path is None:
            print(f"{NAMES[arch]} ({prefix}{suffix}): no checkpoint found, skipped")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Baseline checkpoints have no "input" key, older ones no "vflip"; S2I
        # checkpoints are the only ones with "s2i_scale".
        vflip = checkpoint.get("augmentation", {}).get("vflip", False)
        supervised = checkpoint.get("target") == "cpwc"
        kind = checkpoint.get("input", "mean") + ("_vflip" if vflip else "") + ("_sup" if supervised else "")
        if "s2i_scale" in checkpoint:
            kind = "s2i"
        if kind != inputs:
            if CHECKPOINTS[arch] is None:
                raise ValueError(f"{ckpt_path}: a '{kind}' checkpoint, not '{inputs}'")
            continue  # explicit path, loaded in the loop of its own kind
        n_tx = checkpoint["n_tx"] if checkpoint.get("input") == "transmits" else None
        if n_tx is not None and n_tx != len(ANGLES_DEG):
            print(f"{ckpt_path}: trained on {n_tx} transmits, not {len(ANGLES_DEG)}, skipped")
            continue
        model = PaddedArtifactRemoval(
            build_denoiser(arch, 2 * n_tx if n_tx else 2),
            mode="direct",
            device=device,
            scale_equivariant=checkpoint.get("scale_equivariant", False),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        name = (
            NAMES[arch] + (" multi-tx" if n_tx else "") + (" vflip" if vflip else "")
            + (" S2I" if kind == "s2i" else "") + (" any-n" if kind.startswith("mean_anyn") else "")
            + (" supervised" if supervised else "")
        )
        noise = checkpoint.get("noise_level") if kind.startswith("mean_anyn") else None
        models[name] = model, n_tx, kind, noise
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {ckpt_path} (epoch {checkpoint['epoch']}, {n_params / 1e6:.2f} M params)")
if not models:
    raise RuntimeError("No checkpoint found for any network")
physics = dinv.physics.Denoising().to(device)
# The S2I physics has no noise level: DRUNet's noise-level channel was zero.
physics_s2i = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.0)).to(device)
# The noise level of an any-n network follows the size of its input.
physics_anyn = dinv.physics.Denoising().to(device)


def denoise(model, n_tx, kind, noise, planes, tx):
    r"""Denoise the fast acquisition of transmits ``tx`` of ``(n_angles, Z, X)`` complex planes.

    A baseline network (``n_tx`` None) sees their mean as I/Q channels, a
    multi-transmit one the I/Q channels of each transmit, in the order of ``tx``
    (which must be by increasing steering angle). A Sparse2Inverse network
    (``kind`` "s2i") is averaged over the means of the transmits left after
    holding out S2I_HELD_OUT of them in turn, as in its training script. An
    any-n network (``noise`` its rule ``{"sigma_ref", "n_ref"}``) sees their mean
    with the noise level :math:`\sigma_\mathrm{ref} \sqrt{n_\mathrm{ref} / n}` of a
    mean of ``n = len(tx)`` transmits.
    """
    if kind == "s2i":
        subsets = [
            [t for i, t in enumerate(tx) if not held <= i < held + S2I_HELD_OUT]
            for held in range(0, len(tx), S2I_HELD_OUT)
        ]
    else:
        subsets = [tx]
    out = 0
    for sub in subsets:
        if n_tx is None:
            net_in = torch.view_as_real(planes[sub].mean(0)).permute(2, 0, 1)
        else:
            net_in = torch.view_as_real(planes[sub]).permute(0, 3, 1, 2).flatten(0, 1)
        # The model pads and crops back; recombine I and Q.
        net_in = net_in[None].to(device)
        if noise is not None:
            sigma = noise["sigma_ref"] * math.sqrt(noise["n_ref"] / len(sub))
            physics_anyn.update(sigma=torch.tensor([sigma], device=device))
        phys = physics_s2i if kind == "s2i" else physics_anyn if noise is not None else physics
        with torch.no_grad():
            out = out + model(net_in, phys)[0].cpu()
    out = out / len(subsets)
    return torch.complex(out[0], out[1])


# %%
# Plot every frame
# ----------------
bmode = lambda img: dinv.utils.bmode(img[None, None], amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=False)

for folder in INPUT_DIRS:
    pattern = os.path.join(CACHE_ROOT, folder, "**", "manifest.json")
    for manifest_path in sorted(glob.glob(pattern, recursive=True)):
        cache_dir = os.path.dirname(manifest_path)
        rel = os.path.relpath(cache_dir, CACHE_ROOT)
        out_dir = os.path.join(OUT_DIR, rel)
        os.makedirs(out_dir, exist_ok=True)
        with open(manifest_path) as f:
            manifest = json.load(f)
        extent = [
            -manifest["x_half"] * 1e3,
            manifest["x_half"] * 1e3,
            manifest["z_max"] * 1e3,
            manifest["z_min"] * 1e3,
        ]
        angles_deg = np.asarray(manifest["angles_deg"])
        # By increasing angle: the channel order of the multi-transmit networks.
        ANGLES = sorted(
            (int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG), key=angles_deg.__getitem__
        )
        files = sorted(glob.glob(os.path.join(cache_dir, "invi*_*.npy")))[:N_FRAMES]
        print(
            f"{rel}: {len(files)} frames, fast acquisition from transmits {ANGLES} "
            f"({', '.join(f'{a:+.2f}' for a in angles_deg[ANGLES])} deg) -> {out_dir}"
        )

        for path in files:
            planes = torch.from_numpy(np.load(path))  # (n_angles, Z, X) complex64
            fast = planes[ANGLES].mean(0)  # (Z, X)
            cpwc = planes.mean(0)
            images = {f"DAS, {len(ANGLES)} transmits": bmode(fast)}
            images |= {
                name: bmode(denoise(model, n_tx, kind, noise, planes, ANGLES))
                for name, (model, n_tx, kind, noise) in models.items()
            }
            images[f"CPWC, {manifest['n_angles']} transmits"] = bmode(cpwc)
            dinv.utils.plot(
                images,
                rescale_mode="clip",
                vmin=-DYNAMIC_RANGE_DB,
                vmax=0.0,
                cmap="gray",
                extent=extent,
                aspect="equal",
                figsize=(3 * len(images), 6),
                suptitle=f"{rel}/{Path(path).stem}, {DYNAMIC_RANGE_DB:.0f} dB",
                save_fn=os.path.join(out_dir, Path(path).stem + "_comparison.png"),
                show=False,
                close=True,
            )


# %%
# In-vitro test on the CIRS phantom
# ---------------------------------
invitro_path = os.path.join(CACHE_ROOT, n2i_invitro.FRAME)
if INVITRO_TEST and not os.path.exists(invitro_path):
    print(f"In-vitro test skipped: {invitro_path} not cached (run the precompute script)")
elif INVITRO_TEST:
    planes, manifest = n2i_invitro.load_frame(CACHE_ROOT)
    angles_deg = np.asarray(manifest["angles_deg"])
    ANGLES = sorted(
        (int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG), key=angles_deg.__getitem__
    )
    fast = planes[ANGLES].mean(0)
    images = {f"DAS, {len(ANGLES)} transmits": fast}
    images |= {
        name: denoise(model, n_tx, kind, noise, planes, ANGLES)
        for name, (model, n_tx, kind, noise) in models.items()
    }
    images[f"CPWC, {manifest['n_angles']} transmits"] = planes.mean(0)

    results = {name: n2i_invitro.metrics(img, manifest) for name, img in images.items()}
    print(f"In-vitro test on {n2i_invitro.FRAME}, fast acquisition from transmits {ANGLES}:")
    n2i_invitro.print_table(results)

    out_dir = os.path.join(OUT_DIR, os.path.dirname(n2i_invitro.FRAME))
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(n2i_invitro.FRAME).stem
    with open(os.path.join(out_dir, stem + "_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *next(iter(results.values()))])
        writer.writeheader()
        writer.writerows({"method": name, **m} for name, m in results.items())
    n2i_invitro.plot(
        images, manifest, os.path.join(out_dir, stem + "_comparison.png"), DYNAMIC_RANGE_DB
    )
    print(f"  -> {out_dir}")
