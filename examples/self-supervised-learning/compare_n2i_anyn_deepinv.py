r"""
Noise2Inverse for ultrafast ultrasound -- any-n network against the baseline
============================================================================

Compare, on DAS images compounded from 1, 3, 11 and all 87 steering angles
(``ANGLE_SETS``, nested), two DRUNets:

* the baseline of ``train_n2i_deepinv.py``, trained on the mean of 3 transmits
  with a fixed noise level (``NOISE_LEVEL``), whatever the number of angles;
* the any-n network of ``train_n2i_anyn_deepinv.py``, trained on the mean of any
  number of transmits and told that number through its noise-level channel,
  :math:`\sigma(n) = \sigma_\mathrm{ref} \sqrt{n_\mathrm{ref} / n}` (the rule
  recorded in its checkpoint).

The measurements are those of ``nangles_n2i_deepinv.py``, for DAS and both
networks:

* NMSE against the mean of the transmits outside the largest finite set (76
  transmits), the same reference for every set. With all 87 transmits no
  independent reference is left; the NMSE of each network output against its
  input (CPWC) measures how much the network still changes it.
* Homogeneity of the CIRS phantom: local intensity of each image against CPWC,
  darkest value, area below -3 dB and mean over the deepest fifth.
* The in-vitro metrics of ``n2i_invitro.py`` on frame ``invitro_02162``, also
  written to a CSV.

Figures (B-modes of DAS and of both networks for every set, one frame per folder,
and the phantom with its measurement regions) are only saved to ``OUT_DIR``.
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
import n2i_invitro
from n2i_models import PaddedArtifactRemoval

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
# Checkpoints; None = ckp_best of the latest run under
# ./checkpoints/n2i_ultrasound_drunet<suffix><SPLIT_SUFFIX>.
CHECKPOINTS = {
    "DRUNet": None,  # baseline, suffix ""
    "DRUNet any-n": None,  # suffix "_anyn"
    "DRUNet any-n vflip": None,  # suffix "_anyn_vflip", trained with vertical flips
}
SAVE_SUFFIX = {"DRUNet": "", "DRUNet any-n": "_anyn", "DRUNet any-n vflip": "_anyn_vflip"}
SPLIT_SUFFIX = "_test002_006_008"  # RUN_SUFFIX of the training scripts
OUT_DIR = f"./results/n2i_ultrasound_anyn_vs_mean{SPLIT_SUFFIX}"

# Steering angles (deg) of each set, mapped to the nearest transmits; None = all.
# The sets must be nested; the NMSE reference is every transmit outside the
# largest finite one.
ANGLE_SETS = {
    1: [0.0],
    3: [-10.0, 0.0, 10.0],
    11: list(np.linspace(-10.0, 10.0, 11)),
    87: None,
}
# Test folders (never seen in training), and frames per folder for the NMSE.
INPUT_DIRS = [
    "volunteer_002/carotid",
    "volunteer_002/abdomen",
    "volunteer_006/carotid",
    "volunteer_008/abdomen",
    "volunteer_008/back",
    "volunteer_008/carotid",
    "invitro_CIRS054GS/054GS",
]
# Frames per folder, evenly spaced over it; None = all. The N2I_N_FRAMES environment
# variable overrides it ("all" or a number); a run on every frame writes its tables
# to OUT_DIR + "_all".
N_FRAMES = 6
if (n_frames := os.environ.get("N2I_N_FRAMES")) is not None:
    N_FRAMES = None if n_frames == "all" else int(n_frames)
if N_FRAMES is None:
    OUT_DIR += "_all"
# Held-out phantom frames of the homogeneity test.
PHANTOM_DIR = "invitro_CIRS054GS/054GS"
PHANTOM_FRAMES = [
    "invitro_02162", "invitro_00112", "invitro_00193", "invitro_00274",
    "invitro_00355", "invitro_00436", "invitro_00517", "invitro_00598",
]
SMOOTHING = 8  # pixels, Gaussian sigma of the local intensity
DARK_DB = -3.0
DYNAMIC_RANGE_DB = 50.0

# Networks: must match the training scripts.
CHANNELS = 32
BLOCKS = 2
NOISE_LEVEL = 0.1  # sigma of the baseline's Denoising physics, fed to DRUNet

device = dinv.utils.get_device()

# %%
# Networks
# --------
# name -> (model, noise), noise the noise-level rule of an any-n network, None
# for the baseline.
models = {}
for name, ckpt_path in CHECKPOINTS.items():
    if ckpt_path is None:
        found = sorted(glob.glob(
            f"./checkpoints/n2i_ultrasound_drunet{SAVE_SUFFIX[name]}{SPLIT_SUFFIX}/*/ckp_best.pth.tar"
        ))
        if not found:
            raise FileNotFoundError(f"No {name} checkpoint with suffix {SAVE_SUFFIX[name] + SPLIT_SUFFIX!r}")
        ckpt_path = found[-1]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    kind = checkpoint.get("input", "mean")
    if kind not in ("mean", "mean_anyn"):
        raise ValueError(f"{ckpt_path}: a '{kind}' network, not a mean-input one")
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
    models[name] = model, checkpoint["noise_level"] if kind == "mean_anyn" else None
    print(f"{name}: {ckpt_path} (epoch {checkpoint['epoch']})")
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(NOISE_LEVEL)).to(device)


def denoise(name, img, n):
    """Output of network ``name`` for a ``(Z, X)`` complex mean of ``n`` transmits."""
    model, noise = models[name]
    sigma = NOISE_LEVEL if noise is None else noise["sigma_ref"] * math.sqrt(noise["n_ref"] / n)
    physics.update(sigma=torch.tensor([sigma], device=device))
    net_in = torch.view_as_real(img).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        out = model(net_in, physics)[0].cpu()
    return torch.complex(out[0], out[1])


def outputs(planes, tx):
    """DAS of transmits ``tx`` and the output of every network on it."""
    das = planes[tx].mean(0)
    return {"DAS": das} | {name: denoise(name, das, len(tx)) for name in models}


def transmit_sets(angles_deg):
    """Transmits of every set, by increasing angle, and those of the NMSE reference."""
    sets = {
        n: list(range(len(angles_deg))) if a is None
        else sorted({int(np.abs(angles_deg - t).argmin()) for t in a}, key=angles_deg.__getitem__)
        for n, a in ANGLE_SETS.items()
    }
    largest = max((n for n, a in ANGLE_SETS.items() if a is not None), key=lambda n: len(sets[n]))
    reference = [i for i in range(len(angles_deg)) if i not in sets[largest]]
    return sets, reference


def load(rel, stem):
    with open(os.path.join(CACHE_ROOT, rel, "manifest.json")) as f:
        manifest = json.load(f)
    planes = torch.from_numpy(np.load(os.path.join(CACHE_ROOT, rel, stem + ".npy")))
    return planes, manifest  # (n_angles, Z, X) complex64


def nmse_db(x, ref):
    return 10 * np.log10(((x - ref).abs() ** 2).sum() / (ref.abs() ** 2).sum()).item()


def local_db(img, ref):
    """Local intensity of ``img`` against ``ref`` (dB), median-normalized."""
    smooth = lambda z: gaussian_filter(np.abs(z) ** 2, SMOOTHING)
    r = 10 * np.log10(smooth(img) / smooth(ref))
    return r - np.median(r)


bmode = lambda img: dinv.utils.bmode(img[None, None], amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=False)[0, 0]
extent = lambda m: [-m["x_half"] * 1e3, m["x_half"] * 1e3, m["z_max"] * 1e3, m["z_min"] * 1e3]
methods = ["DAS", *models]
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# NMSE against the transmits outside every finite set
# ---------------------------------------------------
finite = [n for n, a in ANGLE_SETS.items() if a is not None]
full = [n for n, a in ANGLE_SETS.items() if a is None]
print(
    "\nNMSE (dB) against the mean of the transmits outside every finite set"
    + (f"; {full[0]} angles: NMSE of each network output against its input" if full else "")
)
nmse_rows = []
for rel in INPUT_DIRS:
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, rel, "invi*_*.npy")))
    if not files:
        print(f"  {rel}: no frames, skipped")
        continue
    picks = files if N_FRAMES is None else [
        files[i] for i in np.linspace(0, len(files) - 1, N_FRAMES).round().astype(int)
    ]
    scores = {}  # (n, method) -> NMSE of every frame
    for k, path in enumerate(picks):
        stem = os.path.basename(path)[:-4]
        planes, manifest = load(rel, stem)
        sets, reference = transmit_sets(np.asarray(manifest["angles_deg"]))
        ref = planes[reference].mean(0)
        images = {n: outputs(planes, tx) for n, tx in sets.items()}
        for n, imgs in images.items():
            for m, img in imgs.items():
                if n in finite:
                    scores.setdefault((n, m), []).append(nmse_db(img, ref))
                elif m != "DAS":
                    scores.setdefault((n, m), []).append(nmse_db(img, imgs["DAS"]))
        if k == len(picks) // 2:  # B-modes of the middle frame
            fig, ax = plt.subplots(
                len(methods), len(images), figsize=(3.3 * len(images), 5.5 * len(methods)), squeeze=False
            )
            for j, (n, imgs) in enumerate(images.items()):
                for i, (m, img) in enumerate(imgs.items()):
                    ax[i, j].imshow(bmode(img), cmap="gray", vmin=-DYNAMIC_RANGE_DB, vmax=0,
                                    extent=extent(manifest))
                    ax[i, j].set_title(f"{m}, {n} angles", fontsize=10)
            fig.suptitle(f"{rel}/{stem}, {DYNAMIC_RANGE_DB:.0f} dB")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"{rel.replace('/', '_')}_{stem}.png"), dpi=110)
            plt.close(fig)
    nmse_rows.append((rel, {key: np.mean(v) for key, v in scores.items()}))

cols = [(n, m) for n in finite for m in methods] + [(n, m) for n in full for m in models]
label = lambda n, m: f"{m.replace('DRUNet', 'Net')} {n}"
width = max(len(label(*c)) for c in cols) + 1
print(f"  {'folder':26s}" + "".join(f"{label(*c):>{width}s}" for c in cols))
for rel, s in nmse_rows:
    print(f"  {rel:26s}" + "".join(f"{s[c]:{width}.2f}" for c in cols))
with open(os.path.join(OUT_DIR, "nmse_db.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["folder", *(label(*c) for c in cols)])
    writer.writerows([rel, *(f"{s[c]:.3f}" for c in cols)] for rel, s in nmse_rows)

# %%
# Phantom homogeneity
# -------------------
print(
    f"\nPhantom homogeneity against CPWC ({len(PHANTOM_FRAMES)} frames): darkest, "
    f"area below {DARK_DB:.0f} dB mean (max), deepest fifth"
)
stats = {}
for stem in PHANTOM_FRAMES:
    planes, manifest = load(PHANTOM_DIR, stem)
    sets, _ = transmit_sets(np.asarray(manifest["angles_deg"]))
    cpwc = planes.mean(0).numpy()
    for n, tx in sets.items():
        for m, img in outputs(planes, tx).items():
            r = local_db(img.numpy(), cpwc)
            stats.setdefault(f"{m} {n}", []).append(
                (r.min(), 100 * (r < DARK_DB).mean(), np.array_split(r, 5, 0)[-1].mean())
            )
for name, s in stats.items():
    s = np.array(s)
    print(f"  {name:17s} {s[:, 0].min():6.1f} dB  {s[:, 1].mean():5.1f}% ({s[:, 1].max():5.1f}%)  "
          f"deepest fifth {s[:, 2].mean():+.2f} dB")

# %%
# In-vitro metrics on invitro_02162
# ---------------------------------
planes, manifest = n2i_invitro.load_frame(CACHE_ROOT)
sets, _ = transmit_sets(np.asarray(manifest["angles_deg"]))
images = {f"{m} {n}": img for n, tx in sets.items() for m, img in outputs(planes, tx).items()}
results = {name: n2i_invitro.metrics(img, manifest) for name, img in images.items()}
print(f"\nIn-vitro metrics on {n2i_invitro.FRAME}")
n2i_invitro.print_table(results)
with open(os.path.join(OUT_DIR, "invitro_02162_metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["method", *next(iter(results.values()))])
    writer.writeheader()
    writer.writerows({"method": name, **m} for name, m in results.items())
n2i_invitro.plot(images, manifest, os.path.join(OUT_DIR, "invitro_02162_comparison.png"), DYNAMIC_RANGE_DB)
print(f"\nFigures and tables in {OUT_DIR}")
