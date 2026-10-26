r"""
Noise2Inverse for ultrafast ultrasound -- in-vivo metrics
=========================================================

Score the networks trained by ``train_n2i_deepinv.py`` on in-vivo frames that
have no ground truth. Those of ``train_n2i_multitx_deepinv.py``, which see the
transmits of the input as separate channels (I and Q of each, sorted by steering
angle), and the vertical-flip variants of both can be scored too (``INPUTS``);
each checkpoint says which input it expects. Each output is compared with the **held-out
reconstruction**, the mean of the transmits that are *not* in the input. Its
noise is independent of the input, so the expected squared error against it is
the error against the noiseless image plus a constant: the ranking of the
methods is unbiased, while the absolute PSNR underestimates the true one.

Two inputs are scored on every frame:

- *interleaved*: group 0 of the training partition, ``{0, K, 2K}``, i.e. the
  training distribution;
- *fast*: the transmits nearest to ``ANGLES_DEG``, a realistic fast sequence
  that is not one of the training groups.

with

- PSNR on the complex IQ image, :math:`10 \log_{10}(\max|x_{\bar g}|^2 / \mathrm{mean}|\hat x - x_{\bar g}|^2)`;
- SSIM between the B-modes, each normalised to its maximum and clipped to a
  ``DYNAMIC_RANGE_DB`` window mapped to [0, 1];
- the N2I loss (group 0 -> group 1, MSE on the I/Q channels) on the frame
  itself. The validation loss of the training script is the same MSE on the
  reflect-padded frame, so it is lower (0.2848 instead of 0.3031 for the DAS
  input on volunteer 005, carotid) and cannot be compared with this one;
- the median GPU time per frame (batch 1, fp32, after warm-up).

Per-frame values go to a CSV per folder; the script prints, per folder, the
mean and standard deviation of every metric and the paired differences of each
network from ``REFERENCE`` (mean, fraction of frames on which the network is
better, Wilcoxon signed-rank p-value).
"""

import csv
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

import deepinv as dinv
from n2i_models import AttentionResUNet, AttentionUNet, MSDNet, PaddedArtifactRemoval, ResUNet

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
# In-vivo folders to score, each reported separately: use folders the networks
# were not trained on (the validation folder, or the TEST_DIRS of training).
# Volunteer 008 was never trained on; 006 carotid was in the round-3 validation set
# (checkpoint selection, early stopping), so its round-3 scores are slightly optimistic.
EVAL_DIRS = ["volunteer_002", "volunteer_006", "volunteer_008"]  # test volunteers

# Kinds of networks to score, any of
#   "mean"       baseline networks, fed the mean of the transmits (2 channels)
#   "transmits"  multi-transmit networks, fed each transmit (2 channels each)
# and the same trained with vertical-flip augmentation, "mean_vflip" and
# "transmits_vflip"; and the supervised baselines of train_n2i_deepinv.py with
# TARGET="cpwc", "mean_sup" and "mean_vflip_sup". The N2I_INPUTS environment
# variable overrides it, e.g. N2I_INPUTS=mean,transmits,transmits_vflip.
INPUTS = ["mean"]
INPUTS = os.environ.get("N2I_INPUTS", ",".join(INPUTS)).split(",")
# Checkpoint folder suffix of each kind, as in the SAVE_PATH of the training scripts.
SAVE_SUFFIX = {
    "mean": "",
    "transmits": "_multitx",
    "mean_vflip": "_vflip",
    "transmits_vflip": "_multitx_vflip",
    "mean_sup": "_sup",
    "mean_vflip_sup": "_vflip_sup",
}
if not INPUTS or not set(INPUTS) <= set(SAVE_SUFFIX):
    raise ValueError(f"INPUTS must be among {list(SAVE_SUFFIX)}, got {INPUTS}")

# Networks to score, with their checkpoint; None = ``ckp_best`` of the latest
# training run under ``./checkpoints/n2i_ultrasound_<arch><suffix><RUN_SUFFIX>``
# for every kind of INPUTS, suffix being its SAVE_SUFFIX. An explicit path is
# used for the kind its checkpoint records. Networks without any checkpoint are
# skipped.
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
# As in the SAVE_PATH of the training script, e.g. "_test002" (round 3) or
# "_test002_006_008" (round 4); the N2I_RUN_SUFFIX environment variable overrides it.
RUN_SUFFIX = os.environ.get("N2I_RUN_SUFFIX", "_test002")
# Each data split and choice of inputs writes to its own folder, so a run never
# overwrites the tables of another one, e.g. "_mean_vs_multitx_vs_multitx_vflip".
TAG = lambda kind: SAVE_SUFFIX[kind].lstrip("_") or "mean"
OUT_TAG = SAVE_SUFFIX[INPUTS[0]] if len(set(INPUTS)) == 1 else "_" + "_vs_".join(map(TAG, INPUTS))
OUT_DIR = f"./results/n2i_ultrasound_metrics{OUT_TAG}{RUN_SUFFIX}"
NAMES = {
    "unet": "UNet",
    "resunet": "ResUNet",
    "attunet": "AttUNet",
    "attresunet": "AttResUNet",
    "drunet": "DRUNet",
    "msd": "MS-D",
}
# Paired differences are taken from this network (the N2I_REFERENCE environment
# variable overrides it); if it is not scored, from the first network scored.
REFERENCE = os.environ.get("N2I_REFERENCE", "UNet")

# Partition of the training script; group 0 is the interleaved input and
# group 0 -> group 1 the N2I loss.
K = 29
GROUPS = [list(range(j, 87, K)) for j in range(K)]
# Steering angles of the fast acquisition, in degrees; each is mapped to the
# nearest transmit of the sequence (0.38 deg steps between -16.3 and +16.3).
ANGLES_DEG = [-10.0, 0.0, 10.0]

# Networks: must match the training script.
SCALES = 4  # unet only
CHANNELS = 32
RESIDUAL = False  # unet only
BLOCKS = 2  # drunet only
UPSAMPLE = "bilinear"  # ported nets only
MSD_DEPTH, MSD_WIDTH, MSD_DILATION = 100, 1, 10  # msd only

N_FRAMES = None  # cap on frames per folder (None = all)
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


def find_checkpoint(arch, suffix):
    save_path = f"./checkpoints/n2i_ultrasound_{arch}{suffix}{RUN_SUFFIX}"
    found = sorted(glob.glob(os.path.join(save_path, "*", "ckp_best.pth.tar")))
    return found[-1] if found else None


# name -> (model, n_tx), n_tx = None for a network fed the mean of the transmits.
models = {}
for inputs in INPUTS:
    for arch, ckpt_path in CHECKPOINTS.items():
        ckpt_path = ckpt_path or find_checkpoint(arch, SAVE_SUFFIX[inputs])
        if ckpt_path is None:
            print(f"{NAMES[arch]} ({inputs}): no checkpoint found, skipped")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Baseline checkpoints have no "input" key, older ones no "vflip" or "target".
        vflip = checkpoint.get("augmentation", {}).get("vflip", False)
        supervised = checkpoint.get("target") == "cpwc"
        kind = checkpoint.get("input", "mean") + ("_vflip" if vflip else "") + ("_sup" if supervised else "")
        if kind not in SAVE_SUFFIX:
            raise ValueError(f"{ckpt_path}: a '{kind}' network, not one of {list(SAVE_SUFFIX)}")
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
            + (" supervised" if supervised else "")
        )
        models[name] = model, n_tx
        print(f"{name}: {ckpt_path} (epoch {checkpoint['epoch']})")
if not models:
    raise RuntimeError("No checkpoint found for any network")
if REFERENCE not in models:
    REFERENCE = next(iter(models))
physics = dinv.physics.Denoising().to(device)


def to_channels(planes, n_tx):
    """``(n, Z, X)`` complex transmits (by increasing angle) -> I/Q batch on the device.

    A baseline network (``n_tx`` None) sees their mean, ``(1, 2, Z, X)``; a
    multi-transmit one each of them, ``(1, 2 n, Z, X)``. The model pads it.
    """
    if n_tx is None:
        return torch.view_as_real(planes.mean(0)).permute(2, 0, 1)[None].to(device)
    return torch.view_as_real(planes).permute(0, 3, 1, 2).flatten(0, 1)[None].to(device)


def run(model, net_in):
    """Network output on a padded I/Q batch, and its GPU time in seconds."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        out = model(net_in, physics)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t_start


def to_complex(out, shape):
    """Crop a padded I/Q batch back to ``shape`` and recombine it."""
    out = out[0, :, : shape[0], : shape[1]].cpu()
    return torch.complex(out[0], out[1])


# %%
# Metrics
# -------
ssim = dinv.metric.SSIM(max_pixel=1.0)
bmode01 = lambda img: dinv.utils.bmode(
    img[None, None], amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=True
)


def psnr(img, ref):
    return (10 * torch.log10(ref.abs().max() ** 2 / (img - ref).abs().pow(2).mean())).item()


def score_folder(cache_dir, manifest):
    """Per-frame metrics of DAS and every network on the frames of one folder."""
    angles_deg = np.asarray(manifest["angles_deg"])
    # By increasing angle: the channel order of the multi-transmit networks.
    by_angle = lambda tx: sorted(tx, key=angles_deg.__getitem__)
    inputs = {
        "fast": by_angle(int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG),
        "interleaved": by_angle(GROUPS[0]),
    }
    for name, idx in inputs.items():
        print(f"  {name} input: transmits {idx} ({', '.join(f'{a:+.2f}' for a in angles_deg[idx])} deg)")
    files = sorted(glob.glob(os.path.join(cache_dir, "invivo_*.npy")))[:N_FRAMES]

    zeros = torch.zeros(len(ANGLES_DEG), *manifest["img_size"], dtype=torch.complex64)
    for model, n_tx in models.values():  # warm-up, so the first frame is timed fairly
        run(model, to_channels(zeros, n_tx))

    rows, times = [], {name: [] for name in models}
    for path in files:
        planes = np.load(path, mmap_mode="r")  # (n_angles, Z, X) complex64
        row = {"frame": os.path.basename(path)}
        for setting, idx in inputs.items():
            tx = torch.from_numpy(planes[idx])
            inp = tx.mean(0)
            held_out = [a for a in range(planes.shape[0]) if a not in idx]
            ref = torch.from_numpy(planes[held_out].mean(0))
            outs = {"DAS": inp}
            for name, (model, n_tx) in models.items():
                out, dt = run(model, to_channels(tx, n_tx))
                times[name].append(dt)
                outs[name] = to_complex(out, inp.shape)
            for name, img in outs.items():
                row[f"{name} PSNR {setting}"] = psnr(img, ref)
                row[f"{name} SSIM {setting}"] = ssim(bmode01(img), bmode01(ref)).item()
            if setting == "interleaved":
                target = torch.view_as_real(torch.from_numpy(planes[GROUPS[1]].mean(0)))
                for name, img in outs.items():
                    row[f"{name} N2I loss"] = F.mse_loss(torch.view_as_real(img), target).item()
        rows.append(row)
    return rows, {name: 1e3 * np.median(t) for name, t in times.items()}


def report(rows, times):
    """Print the mean-std table and the paired differences from REFERENCE."""
    names = ["DAS", *models]
    columns = [
        ("PSNR fast", "{:.2f} ± {:.2f}"),
        ("SSIM fast", "{:.3f} ± {:.3f}"),
        ("PSNR interleaved", "{:.2f} ± {:.2f}"),
        ("SSIM interleaved", "{:.3f} ± {:.3f}"),
    ]
    values = lambda name, col: np.array([r[f"{name} {col}"] for r in rows])
    print(f"| Method | {' | '.join(c for c, _ in columns)} | N2I loss | Time (ms) |")
    print(f"|---{'|---' * len(columns)}|---:|---:|")
    for name in names:
        cells = [fmt.format(values(name, c).mean(), values(name, c).std()) for c, fmt in columns]
        time_ms = f"{times[name]:.1f}" if name in times else "—"
        print(f"| {name} | {' | '.join(cells)} | {values(name, 'N2I loss').mean():.4f} | {time_ms} |")

    if REFERENCE not in models:
        return
    print(f"\nPaired differences from {REFERENCE}: mean (frames where better, Wilcoxon p)")
    print(f"| Network | {' | '.join('Δ' + c for c, _ in columns)} | ΔN2I loss |")
    print(f"|---{'|---' * (len(columns) + 1)}|")
    for name in models:
        if name == REFERENCE:
            continue
        cells = []
        for col, sign in [(c, 1) for c, _ in columns] + [("N2I loss", -1)]:
            diff = values(name, col) - values(REFERENCE, col)
            better = np.mean(sign * diff > 0) * 100
            p = wilcoxon(diff).pvalue if np.any(diff) else 1.0
            unit = " dB" if col.startswith("PSNR") else ""
            cells.append(f"{diff.mean():+.{4 if col == 'N2I loss' else 3}f}{unit} ({better:.0f}%, p={p:.1g})")
        print(f"| {name} | {' | '.join(cells)} |")


# %%
# Score every folder
# ------------------
for folder in EVAL_DIRS:
    pattern = os.path.join(CACHE_ROOT, folder, "**", "manifest.json")
    for manifest_path in sorted(glob.glob(pattern, recursive=True)):
        cache_dir = os.path.dirname(manifest_path)
        rel = os.path.relpath(cache_dir, CACHE_ROOT)
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"\n{rel}:")
        rows, times = score_folder(cache_dir, manifest)
        print(f"  {len(rows)} frames\n")
        report(rows, times)

        out_dir = os.path.join(OUT_DIR, rel)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-frame metrics written to {out_dir}/metrics.csv")
