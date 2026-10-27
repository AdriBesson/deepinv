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

With ``ROI_JSON`` (environment variable ``N2I_ROI_JSON``), the ROIs drawn with
``roi_tool_n2i.py`` are scored too: lumen against tissue and wall against lumen
(contrast, CNR, gCNR) and the speckle SNR of the tissue, on the fast input.

With ``LOSS_GROUPS`` (environment variable ``N2I_LOSS_GROUPS``, e.g. ``all``),
the loss of the interleaved input is also computed against each of these
groups alone and against the mean of all the groups but 0, to see whether the
ranking of the networks depends on the angles of the target.

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
from n2i_invitro import gcnr
from n2i_models import AttentionResUNet, AttentionUNet, MSDNet, PaddedArtifactRemoval, ResUNet
from roi_tool_n2i import roi_masks

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
# In-vivo folders to score, each reported separately: use folders the networks
# were not trained on (the validation folder, or the TEST_DIRS of training).
# Volunteer 008 was never trained on; 006 carotid was in the round-3 validation set
# (checkpoint selection, early stopping), so its round-3 scores are slightly optimistic.
EVAL_DIRS = ["volunteer_002", "volunteer_006", "volunteer_008"]  # test volunteers
# The N2I_EVAL_DIRS environment variable overrides it; empty (N2I_EVAL_DIRS=)
# skips the scoring of whole folders, e.g. to compute the ROI metrics alone.
if (eval_dirs := os.environ.get("N2I_EVAL_DIRS")) is not None:
    EVAL_DIRS = [d for d in eval_dirs.split(",") if d]
# ROIs drawn with roi_tool_n2i.py (lumen, wall, tissue); when set, the ROI
# metrics of the fast input are computed on their frames (N2I_ROI_JSON).
ROI_JSON = os.environ.get("N2I_ROI_JSON")

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
# Groups the N2I loss of the interleaved input is also computed against, each
# alone, next to the mean of all the groups but 0; the N2I_LOSS_GROUPS
# environment variable sets them, e.g. "2,28" or "all" (groups 1 to K - 1).
# Group 1 mirrors the angles of group 0; group 2 is the nearest in angle. Only
# group 1 enters the tables above; the others get their own table and CSV.
LOSS_GROUPS = os.environ.get("N2I_LOSS_GROUPS", "1")
LOSS_GROUPS = list(range(1, K)) if LOSS_GROUPS == "all" else [int(g) for g in LOSS_GROUPS.split(",")]
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
if (n_frames := os.environ.get("N2I_N_FRAMES")) is not None:  # e.g. N2I_N_FRAMES=4 for a quick test
    N_FRAMES = None if n_frames == "all" else int(n_frames)
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

    rows, group_rows, times = [], [], {name: [] for name in models}
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
                if LOSS_GROUPS != [1]:
                    group_row = {"frame": row["frame"]}
                    targets = {f"g{j}": planes[GROUPS[j]].mean(0) for j in LOSS_GROUPS} | {"rest": ref.numpy()}
                    for tag, target in targets.items():
                        target = torch.view_as_real(torch.from_numpy(target))
                        for name, img in outs.items():
                            group_row[f"{name} loss {tag}"] = F.mse_loss(torch.view_as_real(img), target).item()
                    group_rows.append(group_row)
        rows.append(row)
    return rows, group_rows, {name: 1e3 * np.median(t) for name, t in times.items()}


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


def report_groups(group_rows, manifest):
    """Loss of the interleaved input against each group of LOSS_GROUPS, and against the rest."""
    angles_deg = np.asarray(manifest["angles_deg"])
    g0 = np.sort(angles_deg[GROUPS[0]])
    values = lambda name, tag: np.array([r[f"{name} loss {tag}"] for r in group_rows])
    names = ["DAS", *models]
    print(f"\nLoss of the interleaved input (group 0, {', '.join(f'{a:+.2f}' for a in g0)} deg) against single "
          f"groups and the rest; distance = mean |angle - angle of group 0| (sorted), in deg")
    others = [n for n in models if n != REFERENCE]
    print(f"| Target | Angles (deg) | Distance | {' | '.join(names)} | "
          + " | ".join(f"Δ {n} (better)" for n in others) + " |")
    print(f"|---|---|---:|{'---:|' * len(names)}{'---|' * len(others)}")
    for tag in [f"g{j}" for j in LOSS_GROUPS] + ["rest"]:
        if tag == "rest":
            angles, distance = "mean of groups 1–28", "—"
        else:
            a = np.sort(angles_deg[GROUPS[int(tag[1:])]])
            angles, distance = ", ".join(f"{x:+.2f}" for x in a), f"{np.abs(a - g0).mean():.2f}"
        cells = [f"{values(n, tag).mean():.4f}" for n in names]
        if REFERENCE in models:
            for n in others:
                diff = values(n, tag) - values(REFERENCE, tag)
                cells.append(f"{diff.mean():+.4f} ({np.mean(diff < 0) * 100:.0f}%)")
        print(f"| {tag} | {angles} | {distance} | {' | '.join(cells)} |")


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
        rows, group_rows, times = score_folder(cache_dir, manifest)
        print(f"  {len(rows)} frames\n")
        report(rows, times)
        if group_rows:
            report_groups(group_rows, manifest)

        out_dir = os.path.join(OUT_DIR, rel)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-frame metrics written to {out_dir}/metrics.csv")
        if group_rows:
            with open(os.path.join(out_dir, "loss_groups.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(group_rows[0]))
                writer.writeheader()
                writer.writerows(group_rows)
            print(f"Per-frame losses against single groups written to {out_dir}/loss_groups.csv")


# %%
# ROI metrics
# -----------
# On the envelope |x| of the fast input, its denoising and CPWC, in the
# hand-drawn ROIs: lumen against tissue and wall against lumen (contrast, CNR,
# gCNR as in n2i_invitro.py), and the speckle SNR mean / std in the tissue ROI
# (1.91 for fully developed speckle, higher where the speckle is smoothed).
def roi_metrics(env, masks):
    out = {}
    for a_label, b_label in [("lumen", "tissue"), ("wall", "lumen")]:
        if a_label in masks and b_label in masks:
            a, b = env[masks[a_label]], env[masks[b_label]]
            key = f"{a_label}/{b_label}"
            out[f"{key} contrast_db"] = 20 * np.log10(a.mean() / b.mean())
            out[f"{key} cnr"] = abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var())
            out[f"{key} gcnr"] = gcnr(a, b)
    if "tissue" in masks:
        t = env[masks["tissue"]]
        out["tissue speckle_snr"] = t.mean() / t.std()
    return out


if ROI_JSON:
    masks_by_frame = roi_masks(ROI_JSON)
    roi_rows = []
    for key, masks in masks_by_frame.items():
        folder, stem = key.rsplit("/", 1)
        with open(os.path.join(CACHE_ROOT, folder, "manifest.json")) as f:
            manifest = json.load(f)
        angles_deg = np.asarray(manifest["angles_deg"])
        fast = sorted((int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG), key=angles_deg.__getitem__)
        planes = np.load(os.path.join(CACHE_ROOT, folder, stem + ".npy"), mmap_mode="r")
        tx = torch.from_numpy(planes[fast])
        images = {"DAS": tx.mean(0)}
        for name, (model, n_tx) in models.items():
            images[name] = to_complex(run(model, to_channels(tx, n_tx))[0], tx.shape[1:])
        images["CPWC"] = torch.from_numpy(np.asarray(planes).mean(0))
        for name, img in images.items():
            roi_rows.append({"frame": key, "method": name} | roi_metrics(img.abs().double().numpy(), masks))

    metric_keys = [k for k in roi_rows[0] if k not in ("frame", "method")]
    methods = list(dict.fromkeys(r["method"] for r in roi_rows))
    print(f"\nROI metrics, fast input, {len(masks_by_frame)} frames of {ROI_JSON}: mean ± std over the frames")
    print(f"| Method | {' | '.join(metric_keys)} |")
    print(f"|---{'|---:' * len(metric_keys)}|")
    for m in methods:
        cells = []
        for k in metric_keys:
            v = np.array([r[k] for r in roi_rows if r["method"] == m and k in r])
            cells.append(f"{v.mean():.2f} ± {v.std():.2f}" if len(v) else "—")
        print(f"| {m} | {' | '.join(cells)} |")
    out = os.path.join(OUT_DIR, "roi_metrics.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "method", *metric_keys])
        writer.writeheader()
        writer.writerows(roi_rows)
    print(f"Per-frame ROI metrics written to {out}")
