r"""
Noise2Inverse for ultrafast ultrasound -- dark artifacts on the phantom
=======================================================================

Analysis behind Section V-D and the Discussion of ``summary.md``: the networks
trained by ``train_n2i_deepinv.py`` darken parts of the homogeneous CIRS 054GS
phantom (held out of training) that neither the three-transmit input nor CPWC
shows. Three measurements, on the fast acquisition {-10, 0, +10} deg of
``compare_n2i_deepinv.py``:

1. **Homogeneity** (Table IV). Local brightness of each output against CPWC:
   the ratio of the two intensities :math:`|x|^2`, each smoothed with a Gaussian
   of ``SIGMA_PX`` pixels, in dB, minus its median over the frame. Per frame,
   its minimum and the fraction of the image below ``DARK_DB``; per method, the
   mean over the frames by depth fifth.
2. **Rescaling** the input by ``GAINS``. A denoiser should satisfy
   :math:`f(a y) = a f(y)`; the script prints the relative error
   :math:`\|f(a y)/a - f(y)\| / \|f(y)\|` and the homogeneity of :math:`f(a y)/a`.
   For a DRUNet without the scale-equivariant wrapper, it repeats the test with
   the noise level of its extra input channel scaled by the same gain, which
   makes it exactly equivariant.
3. **Coherence** of the input with the rest of the sequence: the local magnitude
   of the normalised complex correlation between the three-transmit input and
   the mean of the other transmits, by depth fifth, on phantom and in-vivo
   frames, next to the local gain of each network (output against input
   amplitude, dB). An MMSE estimator shrinks its input roughly by this
   correlation.

Tables are printed (pipe them to a log), and written to CSV files with the
figures in ``OUT_DIR``. Figures are only saved to files, never displayed.
"""

import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import deepinv as dinv
from n2i_models import AttentionResUNet, AttentionUNet, MSDNet, PaddedArtifactRemoval, ResUNet

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
PHANTOM = "invitro_CIRS054GS/054GS"  # held out of training since round 3

# Networks to analyse, with their checkpoint; None = ``ckp_best`` of the latest
# training run under ``./checkpoints/n2i_ultrasound_<arch><RUN_SUFFIX>``.
# Networks without any checkpoint are skipped.
CHECKPOINTS = {
    "unet": None,
    "resunet": None,
    "attresunet": None,
    "drunet": None,
    "msd": None,
}
RUN_SUFFIX = "_test002"  # as in the SAVE_PATH of the training script
OUT_DIR = f"./results/n2i_ultrasound_artifacts{RUN_SUFFIX}"

# Steering angles of the fast acquisition, in degrees, as in compare_n2i_deepinv.py.
ANGLES_DEG = [-10.0, 0.0, 10.0]

# Frames of the held-out phantom section. The first one is the in-vitro test
# frame of n2i_invitro.py, whose maps are plotted.
HOMOGENEITY_FRAMES = [
    "invitro_02162", "invitro_00112", "invitro_00193", "invitro_00274",
    "invitro_00355", "invitro_00436", "invitro_00517", "invitro_00598",
]
GAIN_FRAMES = ["invitro_02162", "invitro_00112", "invitro_00301", "invitro_00490"]
GAINS = [0.25, 0.5, 1.0, 2.0, 4.0]
# Folders for the coherence measurement, with N_COHERENCE frames spread over each.
COHERENCE_FOLDERS = [
    "invitro_CIRS054GS/054GS",
    "invitro_CIRS054GS/FullSpeckle",
    "volunteer_001/abdomen",
    "volunteer_004/carotid",
    "volunteer_003/breast",
    "volunteer_002/abdomen",
    "volunteer_002/carotid",
]
N_COHERENCE = 4

SIGMA_PX = 8  # Gaussian smoothing of the intensities, pixels (about 1.2 mm)
DARK_DB = -3.0  # threshold of the dark area
DEPTH_BANDS = 5  # depth fifths

# Networks: must match the training script.
SCALES = 4  # unet only
CHANNELS = 32
RESIDUAL = False  # unet only
BLOCKS = 2  # drunet only
UPSAMPLE = "bilinear"  # ported nets only
MSD_DEPTH, MSD_WIDTH, MSD_DILATION = 100, 1, 10  # msd only
DRUNET_SIGMA = 0.1  # the noise level of the Denoising physics used in training

device = dinv.utils.get_device()


# %%
# Networks
# --------
def build_denoiser(arch):
    if arch in ("resunet", "attunet", "attresunet"):
        net_cls = {"resunet": ResUNet, "attunet": AttentionUNet, "attresunet": AttentionResUNet}
        return net_cls[arch](
            in_channels=2, out_channels=2, base_channels=CHANNELS, upsample_mode=UPSAMPLE
        )
    if arch == "unet":
        return dinv.models.UNet(
            in_channels=2,
            out_channels=2,
            scales=SCALES,
            channels_per_scale=[CHANNELS * 2**k for k in range(SCALES)],
            residual=RESIDUAL,
            batch_norm=False,
        )
    if arch == "drunet":
        return dinv.models.DRUNet(
            in_channels=2,
            out_channels=2,
            nc=[CHANNELS * 2**k for k in range(4)],
            nb=BLOCKS,
            pretrained=None,
        )
    if arch == "msd":
        return MSDNet(
            in_channels=2,
            out_channels=2,
            depth=MSD_DEPTH,
            width=MSD_WIDTH,
            max_dilation=MSD_DILATION,
        )
    raise ValueError(f"Unknown arch {arch!r}")


def find_checkpoint(arch):
    save_path = f"./checkpoints/n2i_ultrasound_{arch}{RUN_SUFFIX}"
    found = sorted(glob.glob(os.path.join(save_path, "*", "ckp_best.pth.tar")))
    return found[-1] if found else None


models = {}
for arch, ckpt_path in CHECKPOINTS.items():
    ckpt_path = ckpt_path or find_checkpoint(arch)
    if ckpt_path is None:
        print(f"{arch}: no checkpoint found, skipped")
        continue
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PaddedArtifactRemoval(
        build_denoiser(arch),
        mode="direct",
        device=device,
        scale_equivariant=checkpoint.get("scale_equivariant", False),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    models[arch] = model
    print(f"{arch}: {ckpt_path} (epoch {checkpoint['epoch']}, scale-equivariant: {model.scale_equivariant})")
if not models:
    raise RuntimeError("No checkpoint found for any network")
os.makedirs(OUT_DIR, exist_ok=True)


def denoise(model, fast, sigma=DRUNET_SIGMA):
    """Denoise a ``(Z, X)`` complex image; ``sigma`` only matters for DRUNet."""
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma)).to(device)
    net_in = torch.view_as_real(fast).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        out = model(net_in, physics)[0].cpu()
    return torch.complex(out[0], out[1])


# %%
# Measurements
# ------------
def load(rel, stem):
    """Fast input, CPWC and mean of the other transmits of a cached frame."""
    folder = os.path.join(CACHE_ROOT, rel)
    with open(os.path.join(folder, "manifest.json")) as f:
        manifest = json.load(f)
    angles_deg = np.asarray(manifest["angles_deg"])
    fast_idx = [int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG]
    rest = [a for a in range(len(angles_deg)) if a not in fast_idx]
    planes = torch.from_numpy(np.load(os.path.join(folder, stem + ".npy")))
    return planes[fast_idx].mean(0), planes.mean(0), planes[rest].mean(0), manifest


def smooth(img):
    return gaussian_filter(img, SIGMA_PX)


def brightness_db(img, ref):
    """Local intensity of ``img`` against ``ref`` (dB), minus its median."""
    r = 10 * np.log10(smooth(np.abs(img) ** 2) / smooth(np.abs(ref) ** 2))
    return r - np.median(r)


def local_gain_db(out, inp):
    """Local amplitude of ``out`` against ``inp`` (dB)."""
    return 10 * np.log10(np.sqrt(smooth(np.abs(out) ** 2) / smooth(np.abs(inp) ** 2)))


def coherence(a, b):
    """Local magnitude of the normalised complex correlation of two IQ images."""
    cross = a * b.conj()
    num = np.abs(smooth(cross.real) + 1j * smooth(cross.imag))
    return num / np.sqrt(smooth(np.abs(a) ** 2) * smooth(np.abs(b) ** 2))


def bands(img):
    """Mean of a ``(Z, X)`` map over each depth band."""
    return [band.mean() for band in np.array_split(img, DEPTH_BANDS, axis=0)]


def dark(r):
    return r.min(), 100 * (r < DARK_DB).mean()


def write_csv(name, rows):
    with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


band_header = " ".join(f"{f'z{i + 1}':>6s}" for i in range(DEPTH_BANDS))

# %%
# 1. Homogeneity
# --------------
methods = ["DAS"] + list(models)
maps = {m: [] for m in methods}
rows = []
for stem in HOMOGENEITY_FRAMES:
    fast, cpwc, _, manifest = load(PHANTOM, stem)
    outputs = {"DAS": fast} | {arch: denoise(model, fast) for arch, model in models.items()}
    for m, img in outputs.items():
        r = brightness_db(img.numpy(), cpwc.numpy())
        maps[m].append(r)
        darkest, area = dark(r)
        rows.append({"frame": stem, "method": m, "darkest_db": darkest, "dark_area_pct": area})
write_csv("homogeneity.csv", rows)

print(
    f"\n1. Homogeneity on {len(HOMOGENEITY_FRAMES)} frames of {PHANTOM}: local brightness against CPWC, "
    f"darkest point (dB) and area below {DARK_DB:.0f} dB, range over the frames (mean)"
)
for m in methods:
    d = [r["darkest_db"] for r in rows if r["method"] == m]
    a = [r["dark_area_pct"] for r in rows if r["method"] == m]
    print(f"  {m:11s} {min(d):6.1f} to {max(d):6.1f} dB   {min(a):5.1f}-{max(a):5.1f}% ({np.mean(a):4.1f}%)")
print(f"Mean over the frames by depth band, shallow to deep (dB):\n  {'':11s} {band_header}")
for m in methods:
    print(f"  {m:11s} " + " ".join(f"{v:6.2f}" for v in bands(np.mean(maps[m], 0))))

extent = [
    -manifest["x_half"] * 1e3, manifest["x_half"] * 1e3,
    manifest["z_max"] * 1e3, manifest["z_min"] * 1e3,
]
fig, ax = plt.subplots(2, len(methods), figsize=(3 * len(methods), 11), squeeze=False)
for j, m in enumerate(methods):
    for i, (img, title) in enumerate(
        [(maps[m][0], HOMOGENEITY_FRAMES[0]), (np.mean(maps[m], 0), f"mean of {len(HOMOGENEITY_FRAMES)} frames")]
    ):
        im = ax[i, j].imshow(img, cmap="RdBu_r", vmin=-8, vmax=8, extent=extent)
        ax[i, j].set_title(f"{m}, {title}", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.6, label="local intensity vs CPWC (dB)")
fig.savefig(os.path.join(OUT_DIR, "homogeneity_maps.png"), dpi=110)
plt.close(fig)

# %%
# 2. Rescaling the input
# ----------------------
# Every variant: the networks as they are, and DRUNet with its noise level
# scaled like the input, when its wrapper does not already make it equivariant.
variants = {arch: (model, False) for arch, model in models.items()}
if "drunet" in models and not models["drunet"].scale_equivariant:
    variants["drunet, sigma x gain"] = (models["drunet"], True)

rows, gain_maps = [], {}
for stem in GAIN_FRAMES:
    fast, cpwc, _, _ = load(PHANTOM, stem)
    for name, (model, scale_sigma) in variants.items():
        ref = denoise(model, fast)
        for g in GAINS:
            sigma = DRUNET_SIGMA * g if scale_sigma else DRUNET_SIGMA
            out = denoise(model, fast * g, sigma) / g
            r = brightness_db(out.numpy(), cpwc.numpy())
            darkest, area = dark(r)
            error = float((out - ref).abs().norm() / ref.abs().norm())
            rows.append({"frame": stem, "network": name, "gain": g, "equivariance_error": error,
                         "darkest_db": darkest, "dark_area_pct": area})
            if stem == GAIN_FRAMES[0]:
                gain_maps[(name, g)] = r
write_csv("rescaling.csv", rows)

print(
    f"\n2. Rescaling on {len(GAIN_FRAMES)} frames: per gain, equivariance error, darkest point (dB) "
    f"and area below {DARK_DB:.0f} dB, means over the frames"
)
print(f"  {'':21s}" + "".join(f"{'x' + str(g):>22s}" for g in GAINS))
for name in variants:
    cells = []
    for g in GAINS:
        sel = [r for r in rows if r["network"] == name and r["gain"] == g]
        cells.append(
            f"{np.mean([r['equivariance_error'] for r in sel]):7.3f} "
            f"{np.mean([r['darkest_db'] for r in sel]):6.1f} "
            f"{np.mean([r['dark_area_pct'] for r in sel]):5.1f}%"
        )
    print(f"  {name:21s}" + "  ".join(cells))

fig, ax = plt.subplots(len(variants), len(GAINS), figsize=(2.4 * len(GAINS), 3.6 * len(variants)), squeeze=False)
for i, name in enumerate(variants):
    for j, g in enumerate(GAINS):
        ax[i, j].imshow(gain_maps[(name, g)], cmap="RdBu_r", vmin=-8, vmax=8)
        ax[i, j].set_title(f"{name} x{g}", fontsize=8)
        ax[i, j].axis("off")
fig.suptitle(f"{GAIN_FRAMES[0]}: local intensity vs CPWC (dB), input x gain, output / gain")
fig.savefig(os.path.join(OUT_DIR, "rescaling_maps.png"), dpi=90)
plt.close(fig)

# %%
# 3. Coherence with the rest of the sequence
# ------------------------------------------
print(
    f"\n3. Coherence of the input with the other transmits, and local gain of each network "
    f"(output against input, dB), by depth band\n  {'':31s} {band_header}"
)
rows = []
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for rel in COHERENCE_FOLDERS:
    stems = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(CACHE_ROOT, rel, "invi*_*.npy")))
    if rel == PHANTOM:
        stems = GAIN_FRAMES
    elif rel.startswith("invitro"):
        stems = stems[::30][:N_COHERENCE]
    else:
        stems = stems[:: max(1, len(stems) // N_COHERENCE)][:N_COHERENCE]
    cohs, gains = [], {arch: [] for arch in models}
    for stem in stems:
        fast, _, rest, manifest = load(rel, stem)
        cohs.append(coherence(fast.numpy(), rest.numpy()))
        for arch, model in models.items():
            gains[arch].append(local_gain_db(denoise(model, fast).numpy(), fast.numpy()))
    coh = np.mean(cohs, 0)
    print(f"  {rel} ({len(stems)} frames)\n  {'  coherence':31s} " + " ".join(f"{v:6.2f}" for v in bands(coh)))
    rows.append({"folder": rel, "quantity": "coherence"} | {f"z{i + 1}": v for i, v in enumerate(bands(coh))})
    for arch in models:
        g = np.mean(gains[arch], 0)
        print(f"  {'  ' + arch:31s} " + " ".join(f"{v:6.2f}" for v in bands(g)))
        rows.append({"folder": rel, "quantity": f"gain_db {arch}"} | {f"z{i + 1}": v for i, v in enumerate(bands(g))})
    z_mm = np.linspace(manifest["z_min"], manifest["z_max"], coh.shape[0]) * 1e3
    axs[0].plot(z_mm, coh.mean(1), label=rel)
    axs[1].plot(z_mm, np.mean(gains[next(iter(models))], 0).mean(1), label=rel)
write_csv("coherence.csv", rows)
axs[0].set(xlabel="depth (mm)", ylabel="coherence", title="3-tx input vs other transmits")
axs[1].set(xlabel="depth (mm)", ylabel="dB", title=f"{next(iter(models))}: output / input amplitude")
for a in axs:
    a.legend(fontsize=7)
fig.savefig(os.path.join(OUT_DIR, "coherence_depth.png"), dpi=100)
plt.close(fig)

# Do the dark regions on the phantom follow a loss of coherence?
fast, _, rest, _ = load(PHANTOM, HOMOGENEITY_FRAMES[0])
coh = coherence(fast.numpy(), rest.numpy())
print(f"\n{HOMOGENEITY_FRAMES[0]}: pixel-wise correlation of the coherence with the local gain of each network")
for arch, model in models.items():
    g = local_gain_db(denoise(model, fast).numpy(), fast.numpy())
    print(f"  {arch:11s} r = {np.corrcoef(coh.ravel(), g.ravel())[0, 1]:+.2f}")
print(f"\nTables and figures written to {OUT_DIR}")
