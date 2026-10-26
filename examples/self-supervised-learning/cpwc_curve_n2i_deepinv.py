r"""
Noise2Inverse for ultrafast ultrasound -- equivalent number of compounded transmits
==================================================================================

How many compounded plane waves is a network fed three transmits worth? DAS
images compounded from N = 3, 5, 9, 15 and 29 transmits, evenly spaced over the
full aperture like the 87-transmit CPWC, give a quality-against-N curve; each
network, on the fast input {-10, 0, +10} deg and on the interleaved group 0 of
the training script, is placed on that curve. The N at which DAS reaches the
network's PSNR (SSIM) is its equivalent number of transmits, and N / 3 the gain
in frame rate at equal image quality.

Every image of a frame is scored against the same reference: the mean of the
transmits in none of the CPWC sets and none of the network inputs. Its noise is
independent of all of them, so the ranking is unbiased; with fewer reference
transmits than the 84 of ``eval_n2i_deepinv.py``, the absolute PSNR is lower
than in Table I of ``summary.md``. The metrics are those of the eval script:
IQ PSNR against the peak of the reference, SSIM of the 50 dB B-modes.

On the held-out phantom frame ``invitro_02162`` the in-vitro metrics of
``n2i_invitro.py`` (cyst contrast, gCNR, speckle) are printed for every DAS
set and network, for the same equivalence in cyst contrast.

Outputs in ``OUT_DIR``: per-frame metrics per folder (``metrics.csv``), the
equivalent numbers of transmits (``equivalent_n.csv``), the phantom metrics and
the curves (``psnr_vs_n.png``, ``ssim_vs_n.png``).
"""

import csv
import glob
import json
import os

import matplotlib

# Figures are only written to files, never drawn through an ssh -X tunnel.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import deepinv as dinv
import n2i_invitro
from n2i_models import PaddedArtifactRemoval

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
SPLIT_SUFFIX = "_test002_006_008"  # RUN_SUFFIX of the training scripts
OUT_DIR = f"./results/n2i_ultrasound_cpwc_curve{SPLIT_SUFFIX}"
# Networks to place on the curve: label -> (arch, checkpoint folder suffix), the
# ckp_best of the latest run under ./checkpoints/n2i_ultrasound_<arch><suffix>
# <SPLIT_SUFFIX>. Networks without a checkpoint are skipped, so the supervised
# baselines (TARGET="cpwc" of train_n2i_deepinv.py) join once trained.
NETWORKS = {
    "U-Net": ("unet", ""),
    "DRUNet": ("drunet", ""),
    "DRUNet multi-tx": ("drunet", "_multitx"),
    "U-Net supervised": ("unet", "_sup"),
    "DRUNet supervised": ("drunet", "_sup"),
}
# Numbers of compounded transmits of the DAS curve, evenly spaced in angle over
# the whole sequence and mapped to the nearest transmits.
N_SET = [3, 5, 9, 15, 29]
# Network inputs, as in eval_n2i_deepinv.py: the fast acquisition, and group 0 of
# the training partition (transmits j, j + 29, j + 58 of the sequence).
FAST_DEG = [-10.0, 0.0, 10.0]
K = 29
INTERLEAVED = list(range(0, 87, K))
EVAL_DIRS = ["volunteer_002", "volunteer_006", "volunteer_008"]  # test volunteers
# Frames per folder, evenly spaced over it; None = all. The N2I_N_FRAMES
# environment variable overrides it ("all" or a number).
N_FRAMES = None
if (n_frames := os.environ.get("N2I_N_FRAMES")) is not None:
    N_FRAMES = None if n_frames == "all" else int(n_frames)
DYNAMIC_RANGE_DB = 50.0

# Networks: must match the training scripts.
SCALES = 4  # unet only
CHANNELS = 32
BLOCKS = 2  # drunet only

device = dinv.utils.get_device()

# %%
# Networks
# --------
def build_denoiser(arch, in_channels):
    if arch == "unet":
        return dinv.models.UNet(
            in_channels=in_channels,
            out_channels=2,
            scales=SCALES,
            channels_per_scale=[CHANNELS * 2**k for k in range(SCALES)],
            residual=False,
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
    raise ValueError(f"Unknown arch {arch!r}")


models = {}  # label -> (model, n_tx), n_tx None for a network fed the mean
for label, (arch, suffix) in NETWORKS.items():
    found = sorted(glob.glob(f"./checkpoints/n2i_ultrasound_{arch}{suffix}{SPLIT_SUFFIX}/*/ckp_best.pth.tar"))
    if not found:
        print(f"{label}: no checkpoint found, skipped")
        continue
    checkpoint = torch.load(found[-1], map_location=device, weights_only=False)
    n_tx = checkpoint["n_tx"] if checkpoint.get("input") == "transmits" else None
    if checkpoint.get("input", "mean") not in ("mean", "transmits"):
        raise ValueError(f"{found[-1]}: a '{checkpoint['input']}' network")
    if n_tx is not None and n_tx != len(FAST_DEG):
        raise ValueError(f"{found[-1]}: trained on {n_tx} transmits, not {len(FAST_DEG)}")
    model = PaddedArtifactRemoval(
        build_denoiser(arch, 2 * n_tx if n_tx else 2),
        mode="direct",
        device=device,
        scale_equivariant=checkpoint.get("scale_equivariant", False),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    models[label] = model, n_tx
    print(f"{label}: {found[-1]} (epoch {checkpoint['epoch']})")
if not models:
    raise RuntimeError("No checkpoint found for any network")
physics = dinv.physics.Denoising().to(device)


def denoise(model, n_tx, planes):
    """Network output for ``(n, Z, X)`` complex transmits sorted by increasing angle.

    A mean-input network sees their mean, a multi-transmit one each of them as
    I/Q channel pairs, as in ``eval_n2i_deepinv.py``.
    """
    if n_tx is None:
        net_in = torch.view_as_real(planes.mean(0)).permute(2, 0, 1)[None]
    else:
        net_in = torch.view_as_real(planes).permute(0, 3, 1, 2).flatten(0, 1)[None]
    with torch.no_grad():
        out = model(net_in.to(device), physics)[0].cpu()
    return torch.complex(out[0], out[1])


# %%
# Transmit sets and metrics
# -------------------------
def transmit_sets(angles_deg):
    """Transmits of every DAS set and network input, by increasing angle, and the reference."""
    by_angle = np.argsort(angles_deg)
    nearest = lambda targets: sorted(
        {int(np.abs(angles_deg - t).argmin()) for t in targets}, key=angles_deg.__getitem__
    )
    span = angles_deg[by_angle[[0, -1]]]
    sets = {n: nearest(np.linspace(*span, n)) for n in N_SET}
    for n, tx in sets.items():
        if len(tx) != n:
            raise ValueError(f"{n} angles map to only {len(tx)} distinct transmits")
    inputs = {"fast": nearest(FAST_DEG), "interleaved": sorted(INTERLEAVED, key=angles_deg.__getitem__)}
    used = {t for tx in [*sets.values(), *inputs.values()] for t in tx}
    reference = [t for t in range(len(angles_deg)) if t not in used]
    return sets, inputs, reference


ssim = dinv.metric.SSIM(max_pixel=1.0)
bmode01 = lambda img: dinv.utils.bmode(img[None, None], amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=True)


def psnr(img, ref):
    return (10 * torch.log10(ref.abs().max() ** 2 / (img - ref).abs().pow(2).mean())).item()


def equivalent_n(value, curve):
    """N at which the DAS curve reaches ``value``, linear in log N between the points.

    Outside the curve, ``"<3"`` or ``">29"`` (no extrapolation).
    """
    ns, vs = np.array(N_SET, float), np.array([curve[n] for n in N_SET])
    if value < vs[0]:
        return f"<{N_SET[0]}"
    if value > vs.max():
        return f">{N_SET[-1]}"
    i = int(np.argmax(vs >= value))  # first point at or above, the curve rising with N
    if i == 0:
        return f"{ns[0]:.1f}"
    t = (value - vs[i - 1]) / (vs[i] - vs[i - 1])
    return f"{np.exp(np.log(ns[i - 1]) + t * np.log(ns[i] / ns[i - 1])):.1f}"


# %%
# In-vivo curves
# --------------
os.makedirs(OUT_DIR, exist_ok=True)
folders = sorted(
    os.path.relpath(os.path.dirname(m), CACHE_ROOT)
    for d in EVAL_DIRS
    for m in glob.glob(os.path.join(CACHE_ROOT, d, "**", "manifest.json"), recursive=True)
)
methods = [f"DAS {n}" for n in N_SET] + [f"{m} {s}" for s in ("fast", "interleaved") for m in ("DAS", *models)]
summary = {}  # folder -> method -> metric -> mean
for folder in folders:
    with open(os.path.join(CACHE_ROOT, folder, "manifest.json")) as f:
        manifest = json.load(f)
    angles_deg = np.asarray(manifest["angles_deg"])
    sets, inputs, reference = transmit_sets(angles_deg)
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, folder, "invivo_*.npy")))
    if N_FRAMES is not None:
        files = [files[i] for i in np.linspace(0, len(files) - 1, N_FRAMES).round().astype(int)]
    print(
        f"\n{folder}: {len(files)} frames; reference = mean of {len(reference)} transmits "
        f"outside the sets {N_SET} and the fast and interleaved inputs"
    )
    rows = []
    for path in files:
        planes = torch.from_numpy(np.load(path))  # (n_angles, Z, X) complex64
        ref = planes[reference].mean(0)
        images = {f"DAS {n}": planes[tx].mean(0) for n, tx in sets.items()}
        for setting, tx in inputs.items():
            images[f"DAS {setting}"] = planes[tx].mean(0)
            for label, (model, n_tx) in models.items():
                images[f"{label} {setting}"] = denoise(model, n_tx, planes[tx])
        ref01 = bmode01(ref)
        row = {"frame": os.path.basename(path)}
        for name, img in images.items():
            row[f"{name} PSNR"] = psnr(img, ref)
            row[f"{name} SSIM"] = ssim(bmode01(img), ref01).item()
        rows.append(row)
    out_dir = os.path.join(OUT_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary[folder] = {
        m: {k: np.mean([r[f"{m} {k}"] for r in rows]) for k in ("PSNR", "SSIM")} for m in methods
    }
    print(f"  {'method':32s} {'PSNR (dB)':>10s} {'SSIM':>7s}")
    for m in methods:
        print(f"  {m:32s} {summary[folder][m]['PSNR']:10.2f} {summary[folder][m]['SSIM']:7.3f}")

# Equivalent number of compounded transmits of every 3-transmit image.
eq_rows = []
print("\nEquivalent number of compounded transmits (DAS curve reaching the same PSNR / SSIM)")
for folder, s in summary.items():
    for m in methods[len(N_SET):]:
        eq = {k: equivalent_n(s[m][k], {n: s[f"DAS {n}"][k] for n in N_SET}) for k in ("PSNR", "SSIM")}
        eq_rows.append({"folder": folder, "method": m, "N PSNR": eq["PSNR"], "N SSIM": eq["SSIM"]})
        print(f"  {folder:24s} {m:32s} PSNR: N = {eq['PSNR']:>5s}   SSIM: N = {eq['SSIM']:>5s}")
with open(os.path.join(OUT_DIR, "equivalent_n.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(eq_rows[0]))
    writer.writeheader()
    writer.writerows(eq_rows)

# %%
# Figures
# -------
# One panel per folder: the DAS curve against N (neutral ink), and every network
# on the fast input as a horizontal line at its value, labelled directly at the
# right end; the fast DAS input as a hollow marker at N = 3.
INK, MUTED, GRID = "#1f1e1c", "#6b6a64", "#e4e2dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical, fixed order
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
for metric, unit in (("PSNR", " (dB)"), ("SSIM", "")):
    n_cols = 3
    n_rows = int(np.ceil(len(summary) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax, (folder, s) in zip(axs.flat, summary.items()):
        ax.plot(N_SET, [s[f"DAS {n}"][metric] for n in N_SET], color=INK, lw=2, marker="o", ms=5,
                label="DAS, N transmits")
        ax.plot([3], [s["DAS fast"][metric]], ls="none", marker="o", ms=7, mfc="white", mec=MUTED,
                mew=1.5, label="DAS, fast input")
        for (label, _), color, style in zip(models.items(), SERIES, STYLES):
            v = s[f"{label} fast"][metric]
            ax.axhline(v, color=color, lw=2, ls=style, label=label)
        ax.set_xscale("log")
        ax.set_xticks(N_SET, [str(n) for n in N_SET])
        ax.minorticks_off()
        ax.set_title(folder.replace("volunteer_", "vol. ").replace("/", " "), fontsize=10, color=INK)
        ax.set_xlabel("Compounded transmits N", fontsize=9, color=MUTED)
        ax.set_ylabel(metric + unit, fontsize=9, color=MUTED)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    for ax in axs.flat[len(summary):]:
        ax.set_visible(False)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(f"{metric} against a common held-out reference: DAS of N transmits vs networks "
                 f"on the 3-transmit fast input", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(os.path.join(OUT_DIR, f"{metric.lower()}_vs_n.png"), dpi=150)
    plt.close(fig)

# %%
# Phantom
# -------
planes, manifest = n2i_invitro.load_frame(CACHE_ROOT)
sets, inputs, _ = transmit_sets(np.asarray(manifest["angles_deg"]))
images = {f"DAS, {n} transmits": planes[tx].mean(0) for n, tx in sets.items()}
images["DAS, fast input"] = planes[inputs["fast"]].mean(0)
for label, (model, n_tx) in models.items():
    images[f"{label}, fast input"] = denoise(model, n_tx, planes[inputs["fast"]])
images["CPWC, 87 transmits"] = planes.mean(0)
results = {name: n2i_invitro.metrics(img, manifest) for name, img in images.items()}
print(f"\nIn-vitro metrics on {n2i_invitro.FRAME}")
n2i_invitro.print_table(results)
with open(os.path.join(OUT_DIR, "invitro_02162_metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", *next(iter(results.values()))])
    writer.writeheader()
    writer.writerows({"method": name, **m} for name, m in results.items())
print(f"\nFigures and tables in {OUT_DIR}")
