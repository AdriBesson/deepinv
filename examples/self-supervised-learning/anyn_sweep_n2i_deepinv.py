r"""
Noise2Inverse for ultrafast ultrasound -- any-n network: transmits and noise level
=================================================================================

The any-n DRUNets of ``train_n2i_anyn_deepinv.py`` darken 18-43% of the held-out
CIRS phantom from one to three transmits (Section V-G of ``summary.md``). Two
sweeps test where this comes from:

1. **Number of transmits** ``N_SWEEP``: the DAS of ``n`` transmits evenly spaced
   over ``SPAN_DEG`` (nearest transmits), the network being told ``n`` through
   its noise-level channel, :math:`\sigma(n) = \sigma_\mathrm{ref}
   \sqrt{n_\mathrm{ref} / n}`, in finer steps than Table XII.
2. **Noise level** at fixed ``n`` (``N_SIGMA``): the same input with
   :math:`\sigma(n)` multiplied by each of ``SIGMA_FACTORS``. If the network
   shrinks what is weak *relative to the noise level it is told*, a lower
   factor should shrink the dark area on the phantom, at some cost in vivo.

Measurements, for DAS and each network:

- phantom homogeneity on the eight frames of Section V-D, as in
  ``artifacts_n2i_deepinv.py``: local intensity against CPWC (Gaussian of
  ``SMOOTHING`` pixels, median-normalised), mean (max) area below ``DARK_DB``,
  darkest point and mean over the deepest fifth;
- in vivo, the NMSE against the mean of the transmits outside every set of the
  sweep (the same reference for every ``n``), on ``N_FRAMES`` frames evenly
  spaced over each test folder.

Tables are printed (pipe them to a log) and written to CSV files in ``OUT_DIR``,
with a figure of both sweeps; figures are only saved, never displayed.
"""

import csv
import glob
import json
import math
import os

import matplotlib

# Figures are only written to files, never drawn through an ssh -X tunnel.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import deepinv as dinv
from n2i_models import PaddedArtifactRemoval

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
# Checkpoints; None = ckp_best of the latest run under
# ./checkpoints/n2i_ultrasound_drunet<suffix><SPLIT_SUFFIX>.
CHECKPOINTS = {
    "DRUNet any-n": None,  # suffix "_anyn"
    "DRUNet any-n vflip": None,  # suffix "_anyn_vflip", trained with vertical flips
}
SAVE_SUFFIX = {"DRUNet any-n": "_anyn", "DRUNet any-n vflip": "_anyn_vflip"}
SPLIT_SUFFIX = "_test002_006_008"  # RUN_SUFFIX of the training scripts
OUT_DIR = f"./results/n2i_ultrasound_anyn_sweep{SPLIT_SUFFIX}"

# Sweep 1: numbers of transmits, evenly spaced over SPAN_DEG (deg), centred on 0.
N_SWEEP = list(range(1, 12))
SPAN_DEG = 20.0  # +/-10 deg, the span of the 3- and 11-transmit sets of Table XII
# Sweep 2: factors applied to sigma(n), at each n of N_SIGMA.
N_SIGMA = [1, 3]
SIGMA_FACTORS = [0.25, 0.5, 0.71, 1.0, 1.41, 2.0]

# In-vivo test folders and frames per folder (evenly spaced) for the NMSE; the
# N2I_N_FRAMES environment variable overrides it ("all" or a number).
INVIVO_DIRS = [
    "volunteer_002/carotid",
    "volunteer_002/abdomen",
    "volunteer_006/carotid",
    "volunteer_008/abdomen",
    "volunteer_008/back",
    "volunteer_008/carotid",
]
N_FRAMES = 6
if (n_frames := os.environ.get("N2I_N_FRAMES")) is not None:
    N_FRAMES = None if n_frames == "all" else int(n_frames)
# Held-out phantom frames of the homogeneity test (Section V-D).
PHANTOM_DIR = "invitro_CIRS054GS/054GS"
PHANTOM_FRAMES = [
    "invitro_02162", "invitro_00112", "invitro_00193", "invitro_00274",
    "invitro_00355", "invitro_00436", "invitro_00517", "invitro_00598",
]
SMOOTHING = 8  # pixels, Gaussian sigma of the local intensity
DARK_DB = -3.0

# Networks: must match the training scripts.
CHANNELS = 32
BLOCKS = 2

device = dinv.utils.get_device()

# %%
# Networks
# --------
# name -> (model, noise-level rule of the checkpoint)
models = {}
for name, ckpt_path in CHECKPOINTS.items():
    if ckpt_path is None:
        found = sorted(glob.glob(
            f"./checkpoints/n2i_ultrasound_drunet{SAVE_SUFFIX[name]}{SPLIT_SUFFIX}/*/ckp_best.pth.tar"
        ))
        if not found:
            print(f"{name}: no checkpoint with suffix {SAVE_SUFFIX[name] + SPLIT_SUFFIX!r}, skipped")
            continue
        ckpt_path = found[-1]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if checkpoint.get("input") != "mean_anyn":
        raise ValueError(f"{ckpt_path}: not an any-n network")
    model = PaddedArtifactRemoval(
        dinv.models.DRUNet(
            in_channels=2,
            out_channels=2,
            nc=[CHANNELS * 2**k for k in range(4)],
            nb=BLOCKS,
            pretrained=None,
        ),
        mode="direct",
        device=device,
        scale_equivariant=checkpoint.get("scale_equivariant", False),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    models[name] = model, checkpoint["noise_level"]
    print(f"{name}: {ckpt_path} (epoch {checkpoint['epoch']})")
if not models:
    raise RuntimeError("No any-n checkpoint found")
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1)).to(device)


def denoise(name, img, n, factor=1.0):
    """Output of network ``name`` for a ``(Z, X)`` complex mean of ``n`` transmits,
    told the noise level ``factor * sigma(n)``."""
    model, noise = models[name]
    sigma = factor * noise["sigma_ref"] * math.sqrt(noise["n_ref"] / n)
    physics.update(sigma=torch.tensor([sigma], device=device))
    net_in = torch.view_as_real(img).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        out = model(net_in, physics)[0].cpu()
    return torch.complex(out[0], out[1])


def transmit_set(angles_deg, n):
    """The ``n`` transmits nearest to ``n`` angles evenly spaced over SPAN_DEG."""
    targets = [0.0] if n == 1 else np.linspace(-SPAN_DEG / 2, SPAN_DEG / 2, n)
    tx = {int(np.abs(angles_deg - t).argmin()) for t in targets}
    if len(tx) != n:
        raise ValueError(f"{n} angles over {SPAN_DEG} deg map to only {len(tx)} transmits")
    return sorted(tx, key=angles_deg.__getitem__)


def load(rel, stem):
    with open(os.path.join(CACHE_ROOT, rel, "manifest.json")) as f:
        manifest = json.load(f)
    planes = torch.from_numpy(np.load(os.path.join(CACHE_ROOT, rel, stem + ".npy")))
    return planes, np.asarray(manifest["angles_deg"])


def nmse_db(x, ref):
    return 10 * np.log10(((x - ref).abs() ** 2).sum() / (ref.abs() ** 2).sum()).item()


def local_db(img, ref):
    """Local intensity of ``img`` against ``ref`` (dB), median-normalized."""
    smooth = lambda z: gaussian_filter(np.abs(z) ** 2, SMOOTHING)
    r = 10 * np.log10(smooth(img) / smooth(ref))
    return r - np.median(r)


# Every (n, factor) point of both sweeps, and the methods scored at each.
points = [(n, 1.0) for n in N_SWEEP] + [(n, f) for n in N_SIGMA for f in SIGMA_FACTORS if f != 1.0]
methods = ["DAS", *models]
os.makedirs(OUT_DIR, exist_ok=True)


def score(planes, angles_deg, scorer):
    """``{(n, factor, method): scorer(image)}`` over every point of the sweeps."""
    out = {}
    for n in sorted({n for n, _ in points}):
        das = planes[transmit_set(angles_deg, n)].mean(0)
        for f in sorted({f for m, f in points if m == n}):
            if f == 1.0:
                out[(n, f, "DAS")] = scorer(das)
            for name in models:
                out[(n, f, name)] = scorer(denoise(name, das, n, f))
    return out


# %%
# Phantom homogeneity
# -------------------
phantom = {}  # (n, factor, method) -> [(darkest, area, deepest fifth) per frame]
for stem in PHANTOM_FRAMES:
    planes, angles_deg = load(PHANTOM_DIR, stem)
    cpwc = planes.mean(0).numpy()

    def homogeneity(img):
        r = local_db(img.numpy(), cpwc)
        return r.min(), 100 * (r < DARK_DB).mean(), np.array_split(r, 5, 0)[-1].mean()

    for key, value in score(planes, angles_deg, homogeneity).items():
        phantom.setdefault(key, []).append(value)
    print(f"phantom {stem} done", flush=True)

# %%
# In-vivo NMSE
# ------------
invivo = {}  # (n, factor, method) -> NMSE of every frame, all folders
by_folder = {}  # rel -> {(n, factor, method): mean NMSE}
for rel in INVIVO_DIRS:
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, rel, "invivo_*.npy")))
    picks = files if N_FRAMES is None else [
        files[i] for i in np.linspace(0, len(files) - 1, N_FRAMES).round().astype(int)
    ]
    folder = {}
    for path in picks:
        planes, angles_deg = load(rel, os.path.basename(path)[:-4])
        used = {t for n in N_SWEEP for t in transmit_set(angles_deg, n)}
        ref = planes[[i for i in range(len(angles_deg)) if i not in used]].mean(0)
        for key, value in score(planes, angles_deg, lambda img: nmse_db(img, ref)).items():
            folder.setdefault(key, []).append(value)
            invivo.setdefault(key, []).append(value)
    by_folder[rel] = {key: np.mean(v) for key, v in folder.items()}
    print(f"{rel}: {len(picks)} frames done", flush=True)

# %%
# Tables
# ------
rows = []
for (n, f, m), v in phantom.items():
    v = np.array(v)
    rows.append({
        "n": n, "sigma_factor": f, "method": m,
        "dark_area_mean_pct": v[:, 1].mean(), "dark_area_max_pct": v[:, 1].max(),
        "darkest_db": v[:, 0].min(), "deepest_fifth_db": v[:, 2].mean(),
        "invivo_nmse_db": np.mean(invivo[(n, f, m)]),
    } | {f"nmse_db {rel}": s[(n, f, m)] for rel, s in by_folder.items()})
rows.sort(key=lambda r: (r["sigma_factor"] != 1.0, r["n"], r["sigma_factor"], methods.index(r["method"])))
with open(os.path.join(OUT_DIR, "sweep.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(
    f"\nPhantom ({len(PHANTOM_FRAMES)} frames): area below {DARK_DB:.0f} dB mean (max), darkest point, deepest "
    f"fifth; in vivo: NMSE (dB) against the transmits outside every set, mean over "
    f"{'all' if N_FRAMES is None else N_FRAMES} frames of each of {len(INVIVO_DIRS)} folders"
)
print(f"  {'n':>3s} {'sigma x':>8s}  {'method':20s} {'dark area':>15s} {'darkest':>8s} {'deep 5th':>9s} {'NMSE':>7s}")
for r in rows:
    print(
        f"  {r['n']:3d} {r['sigma_factor']:8.2f}  {r['method']:20s} "
        f"{r['dark_area_mean_pct']:6.1f}% ({r['dark_area_max_pct']:5.1f}%) {r['darkest_db']:8.1f} "
        f"{r['deepest_fifth_db']:+9.2f} {r['invivo_nmse_db']:7.2f}"
    )

fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
for m in methods:
    sel = [r for r in rows if r["method"] == m and r["sigma_factor"] == 1.0]
    ax[0, 0].plot([r["n"] for r in sel], [r["dark_area_mean_pct"] for r in sel], "o-", label=m)
    ax[1, 0].plot([r["n"] for r in sel], [r["invivo_nmse_db"] for r in sel], "o-", label=m)
    for n, style in zip(N_SIGMA, ["o-", "s--"]):
        if m == "DAS":
            continue
        sel = sorted((r for r in rows if r["method"] == m and r["n"] == n), key=lambda r: r["sigma_factor"])
        ax[0, 1].plot([r["sigma_factor"] for r in sel], [r["dark_area_mean_pct"] for r in sel], style, label=f"{m}, n={n}")
        ax[1, 1].plot([r["sigma_factor"] for r in sel], [r["invivo_nmse_db"] for r in sel], style, label=f"{m}, n={n}")
ax[0, 0].set(ylabel=f"phantom area < {DARK_DB:.0f} dB (%)", title="sweep over n (sigma(n))")
ax[0, 1].set(title="sweep over the noise level at fixed n")
ax[1, 0].set(xlabel="number of transmits n", ylabel="in-vivo NMSE (dB)")
ax[1, 1].set(xlabel="noise level / sigma(n)", xscale="log")
for a in ax.ravel():
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sweep.png"), dpi=110)
plt.close(fig)
print(f"\nTables and figure in {OUT_DIR}")
