r"""
Noise2Inverse for ultrafast ultrasound -- number of steering angles
===================================================================

Apply a network trained by ``train_n2i_deepinv.py`` on fast acquisitions of 3
transmits (mean input) to DAS images compounded from other numbers of steering
angles: 1, 3, 11 and all 87. The angle sets are nested (``ANGLE_SETS``), so the
network sees the same scene at decreasing noise levels.

Three measurements, printed:

* NMSE of DAS and of the network against the mean of the transmits outside the
  largest finite set (76 transmits), the same reference for every set. With all
  87 transmits no independent reference is left; the NMSE of the network output
  against its own input (CPWC) measures how much the network still changes it.
* Homogeneity of the CIRS phantom (as in ``artifacts_n2i_deepinv.py``): local
  intensity of each image against CPWC, darkest value, area below -3 dB and
  mean over the deepest fifth.
* The in-vitro metrics of ``n2i_invitro.py`` on frame ``invitro_02162``.

Figures (B-modes of DAS and network for every set, one frame per folder, and
the phantom with its measurement regions) are only saved to ``OUT_DIR``.
"""

import glob
import json
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
# Mean-input DRUNet trained on 3 transmits; None = ckp_best of the latest run
# under ./checkpoints/n2i_ultrasound_drunet<SPLIT_SUFFIX>.
CHECKPOINT = None
SPLIT_SUFFIX = "_test002_006_008"  # RUN_SUFFIX of the training script
OUT_DIR = f"./results/n2i_ultrasound_nangles{SPLIT_SUFFIX}"

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
N_FRAMES = 6  # evenly spaced over each folder
# Held-out phantom frames of the homogeneity test (Table IV of summary.md).
PHANTOM_DIR = "invitro_CIRS054GS/054GS"
PHANTOM_FRAMES = [
    "invitro_02162", "invitro_00112", "invitro_00193", "invitro_00274",
    "invitro_00355", "invitro_00436", "invitro_00517", "invitro_00598",
]
SMOOTHING = 8  # pixels, Gaussian sigma of the local intensity
DARK_DB = -3.0
DYNAMIC_RANGE_DB = 50.0

# Network: must match the training script.
CHANNELS = 32
BLOCKS = 2
NOISE_LEVEL = 0.1  # sigma of the Denoising physics, fed to DRUNet

device = dinv.utils.get_device()

# %%
# Network
# -------
if CHECKPOINT is None:
    found = sorted(glob.glob(f"./checkpoints/n2i_ultrasound_drunet{SPLIT_SUFFIX}/*/ckp_best.pth.tar"))
    if not found:
        raise FileNotFoundError(f"No DRUNet checkpoint with suffix {SPLIT_SUFFIX!r}")
    CHECKPOINT = found[-1]
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
if checkpoint.get("input", "mean") != "mean":
    raise ValueError(f"{CHECKPOINT}: a '{checkpoint['input']}' network, not a mean-input one")
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
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(NOISE_LEVEL)).to(device)
print(f"DRUNet: {CHECKPOINT} (epoch {checkpoint['epoch']})")


def denoise(img):
    """Network output for a ``(Z, X)`` complex image."""
    net_in = torch.view_as_real(img).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        out = model(net_in, physics)[0].cpu()
    return torch.complex(out[0], out[1])


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
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# NMSE against the transmits outside every finite set
# ---------------------------------------------------
finite = [n for n, a in ANGLE_SETS.items() if a is not None]
full = [n for n, a in ANGLE_SETS.items() if a is None]
print(
    "\nNMSE (dB) against the mean of the transmits outside every finite set"
    + (f"; {full[0]}: NMSE of the network output against its input" if full else "")
)
header = " ".join(f"{'DAS ' + str(n):>8s} {'Net ' + str(n):>8s}" for n in finite)
print(f"  {'folder':26s} {header}" + "".join(f" {'change ' + str(n):>10s}" for n in full))
for rel in INPUT_DIRS:
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, rel, "invi*_*.npy")))
    if not files:
        print(f"  {rel}: no frames, skipped")
        continue
    picks = [files[i] for i in np.linspace(0, len(files) - 1, N_FRAMES).round().astype(int)]
    scores = {}
    for k, path in enumerate(picks):
        stem = os.path.basename(path)[:-4]
        planes, manifest = load(rel, stem)
        sets, reference = transmit_sets(np.asarray(manifest["angles_deg"]))
        ref = planes[reference].mean(0)
        images = {}
        for n, tx in sets.items():
            das = planes[tx].mean(0)
            images[n] = das, denoise(das)
            if n in finite:
                scores.setdefault(f"das{n}", []).append(nmse_db(images[n][0], ref))
                scores.setdefault(f"net{n}", []).append(nmse_db(images[n][1], ref))
            else:
                scores.setdefault(f"change{n}", []).append(nmse_db(images[n][1], das))
        if k == len(picks) // 2:  # B-modes of the middle frame
            fig, ax = plt.subplots(2, len(images), figsize=(3.3 * len(images), 11), squeeze=False)
            for j, (n, (das, out)) in enumerate(images.items()):
                for i, (img, title) in enumerate([(das, f"DAS, {n} angles"), (out, f"DRUNet on DAS {n}")]):
                    ax[i, j].imshow(bmode(img), cmap="gray", vmin=-DYNAMIC_RANGE_DB, vmax=0,
                                    extent=extent(manifest))
                    ax[i, j].set_title(title, fontsize=10)
            fig.suptitle(f"{rel}/{stem}, {DYNAMIC_RANGE_DB:.0f} dB")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"{rel.replace('/', '_')}_{stem}.png"), dpi=110)
            plt.close(fig)
    row = " ".join(f"{np.mean(scores[f'das{n}']):8.2f} {np.mean(scores[f'net{n}']):8.2f}" for n in finite)
    print(f"  {rel:26s} {row}" + "".join(f" {np.mean(scores[f'change{n}']):10.2f}" for n in full))

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
        das = planes[tx].mean(0)
        for name, img in [(f"DAS {n}", das), (f"DRUNet {n}", denoise(das))]:
            r = local_db(img.numpy(), cpwc)
            stats.setdefault(name, []).append(
                (r.min(), 100 * (r < DARK_DB).mean(), np.array_split(r, 5, 0)[-1].mean())
            )
for name, s in stats.items():
    s = np.array(s)
    print(f"  {name:11s} {s[:, 0].min():6.1f} dB  {s[:, 1].mean():5.1f}% ({s[:, 1].max():5.1f}%)  "
          f"deepest fifth {s[:, 2].mean():+.2f} dB")

# %%
# In-vitro metrics on invitro_02162
# ---------------------------------
planes, manifest = n2i_invitro.load_frame(CACHE_ROOT)
sets, _ = transmit_sets(np.asarray(manifest["angles_deg"]))
images = {}
for n, tx in sets.items():
    das = planes[tx].mean(0)
    images[f"DAS {n}"], images[f"DRUNet {n}"] = das, denoise(das)
print(f"\nIn-vitro metrics on {n2i_invitro.FRAME}")
n2i_invitro.print_table({name: n2i_invitro.metrics(img, manifest) for name, img in images.items()})
n2i_invitro.plot(images, manifest, os.path.join(OUT_DIR, "invitro_02162_comparison.png"), DYNAMIC_RANGE_DB)
print(f"\nFigures in {OUT_DIR}")
