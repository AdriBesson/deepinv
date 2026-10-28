r"""
Noise2Inverse for ultrafast ultrasound -- dark artifacts on the phantom
=======================================================================

Analysis behind Sections V-D to V-I and the Discussion of ``summary.md``: the
networks trained by ``train_n2i_deepinv.py`` (and its multi-transmit, any-n and
supervised variants) darken parts of the homogeneous CIRS 054GS phantom (held
out of training) that neither the three-transmit input nor CPWC shows. Five
measurements, on the fast acquisition {-10, 0, +10} deg of
``compare_n2i_deepinv.py``:

1. **Homogeneity** (Tables IV, VI, IX, XVIII). Local brightness of each output
   against CPWC: the ratio of the two intensities :math:`|x|^2`, each smoothed
   with a Gaussian of ``SIGMA_PX`` pixels, in dB, minus its median over the
   frame. Per frame, its minimum and the fraction of the image below
   ``DARK_DB``; per method, the mean over the frames by depth fifth.
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
4. **Vertical flip** (Tables VII, IX, X). The input is flipped upside down and
   the output flipped back: the homogeneity of measurement 1 with the flipped
   input, and the NMSE against the mean of the transmits not in the input, with
   the plain and the flipped input, on ``N_NMSE`` frames of each of
   ``NMSE_FOLDERS``.
5. **Cropping** (Section V-E). The top ``CROP_ROWS`` rows of the input of
   ``CROP_FRAME`` are removed; the homogeneity is measured on the remaining
   rows, against CPWC cropped alike. A dark region that stays below the cysts
   rather than at fixed rows is not caused by the image border.
6. **Consistency across groups** (Discussion, *Temporal consistency*). The
   ``K`` interleaved groups of one acquisition are ``K`` images of the same
   scene with independent noise. Every group is denoised and, in the signal
   region (CPWC above ``SIGNAL_DB``), the script measures across the outputs
   the coefficient of variation of the envelope, the incoherence
   :math:`1 - |\bar y|^2 / \overline{|y|^2}`, and the lag-one correlation
   between groups adjacent in time, :math:`R_1 = \overline{y_{g+1} y_g^*}`
   normalised, as a Doppler estimator would: its magnitude and the spread of
   its phase (the phase-noise floor of a static scene), for DAS, each network
   and the compounds of ``COMPOUND_SPLIT`` disjoint interleaved subsets of the
   transmits. On the phantom frames of measurement 1, the dark area
   of each group and of their union show whether the dark regions move from one
   group to the next.

The networks are chosen as in ``eval_n2i_deepinv.py``: ``INPUTS`` (environment
variable ``N2I_INPUTS``) selects the kinds of networks, ``N2I_ARCHS`` the
architectures and ``N2I_RUN_SUFFIX`` the data split; ``N2I_SECTIONS`` runs only
some of the measurements, e.g. ``N2I_SECTIONS=1,4``. Tables are printed (pipe
them to a log), and written to CSV files with the figures in ``OUT_DIR``.
Figures are only saved to files, never displayed.
"""

import csv
import glob
import json
import math
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

# Kinds of networks to analyse, as in eval_n2i_deepinv.py:
#   "mean"       baseline networks, fed the mean of the transmits (2 channels)
#   "transmits"  multi-transmit networks (train_n2i_multitx_deepinv.py), fed
#                each transmit, sorted by steering angle (2 channels each)
#   "mean_anyn"  networks of train_n2i_anyn_deepinv.py, fed the mean of the
#                transmits and told their number through the DRUNet noise level
# each also trained with vertical flips ("mean_vflip", "transmits_vflip",
# "mean_anyn_vflip"); and the supervised baselines of train_n2i_deepinv.py with
# TARGET="cpwc", "mean_sup" and "mean_vflip_sup" ("transmits_sup" for those of
# train_n2i_multitx_deepinv.py). The N2I_INPUTS environment
# variable overrides it, e.g. N2I_INPUTS=mean,transmits,transmits_vflip.
INPUTS = ["mean"]
INPUTS = os.environ.get("N2I_INPUTS", ",".join(INPUTS)).split(",")
# Checkpoint folder suffix of each kind, as in the SAVE_PATH of the training scripts.
SAVE_SUFFIX = {
    "mean": "",
    "transmits": "_multitx",
    "mean_anyn": "_anyn",
    "mean_vflip": "_vflip",
    "transmits_vflip": "_multitx_vflip",
    "mean_anyn_vflip": "_anyn_vflip",
    "mean_sup": "_sup",
    "mean_vflip_sup": "_vflip_sup",
    "transmits_sup": "_multitx_sup",
}
if not INPUTS or not set(INPUTS) <= set(SAVE_SUFFIX):
    raise ValueError(f"INPUTS must be among {list(SAVE_SUFFIX)}, got {INPUTS}")

# Networks to analyse, with their checkpoint; None = ``ckp_best`` of the latest
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
# Each data split and choice of inputs writes to its own folder, as in eval_n2i_deepinv.py.
TAG = lambda kind: SAVE_SUFFIX[kind].lstrip("_") or "mean"
OUT_TAG = SAVE_SUFFIX[INPUTS[0]] if len(set(INPUTS)) == 1 else "_" + "_vs_".join(map(TAG, INPUTS))
OUT_DIR = f"./results/n2i_ultrasound_artifacts{OUT_TAG}{RUN_SUFFIX}"
NAMES = {
    "unet": "UNet",
    "resunet": "ResUNet",
    "attunet": "AttUNet",
    "attresunet": "AttResUNet",
    "drunet": "DRUNet",
    "msd": "MS-D",
}
# Measurements to run (see the docstring); the N2I_SECTIONS environment variable
# overrides it, e.g. N2I_SECTIONS=4,5.
SECTIONS = {int(s) for s in os.environ.get("N2I_SECTIONS", "1,2,3,4,5,6").split(",")}

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
# Folders of the NMSE flip test, with N_NMSE frames taken at regular steps
# through each (Table X): the test folders of round 4 and the phantom section.
NMSE_FOLDERS = [
    "volunteer_002/carotid",
    "volunteer_002/abdomen",
    "volunteer_006/carotid",
    "volunteer_008/abdomen",
    "volunteer_008/back",
    "volunteer_008/carotid",
    "invitro_CIRS054GS/054GS",
]
N_NMSE = 6
# Cropping test: rows removed from the top of the input of CROP_FRAME.
CROP_FRAME = HOMOGENEITY_FRAMES[0]
CROP_ROWS = [0, 64, 128, 192]

# Consistency across groups: the partition of the training script, the folders
# (N_CONSISTENCY frames spread over each) and the number of disjoint compounds
# (each the mean of 87 / COMPOUND_SPLIT interleaved transmits) to compare with.
K = 29
GROUPS = [list(range(j, 87, K)) for j in range(K)]
CONSISTENCY_FOLDERS = NMSE_FOLDERS
N_CONSISTENCY = 4
COMPOUND_SPLIT = 3
SIGNAL_DB = -40.0  # signal region: CPWC envelope above this, relative to its maximum

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


# name -> (model, n_tx, noise), n_tx = None for a network fed the mean of the
# transmits, noise the noise-level rule of an any-n network (None otherwise).
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
        anyn = checkpoint.get("input") == "mean_anyn"
        name = (
            NAMES[arch] + (" multi-tx" if n_tx else "") + (" any-n" if anyn else "")
            + (" vflip" if vflip else "") + (" supervised" if supervised else "")
        )
        models[name] = model, n_tx, checkpoint["noise_level"] if anyn else None
        print(
            f"{name}: {ckpt_path} (epoch {checkpoint['epoch']}, scale-equivariant: {model.scale_equivariant})"
        )
if not models:
    raise RuntimeError("No checkpoint found for any network")
os.makedirs(OUT_DIR, exist_ok=True)
W = max(11, *(len(name) for name in models))  # width of the name column


def denoise(name, tx, gain=1.0, scale_sigma=False, flip=False):
    """Denoise the ``(n, Z, X)`` complex transmits ``tx`` (by increasing angle).

    The input is multiplied by ``gain`` and, with ``flip``, flipped upside
    down; the output is flipped back but not divided by the gain. The DRUNet
    noise level is ``DRUNET_SIGMA``, or that of ``n`` transmits for an any-n
    network, times ``gain`` with ``scale_sigma``; other networks ignore it.
    """
    model, n_tx, noise = models[name]
    sigma = DRUNET_SIGMA if noise is None else noise["sigma_ref"] * math.sqrt(noise["n_ref"] / len(tx))
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma * (gain if scale_sigma else 1.0)))
    tx = tx.flip(-2) if flip else tx
    if n_tx is None:
        net_in = torch.view_as_real(tx.mean(0) * gain).permute(2, 0, 1)
    else:
        net_in = torch.view_as_real(tx * gain).permute(0, 3, 1, 2).flatten(0, 1)
    with torch.no_grad():
        out = model(net_in[None].to(device), physics.to(device))[0].cpu()
    out = torch.complex(out[0], out[1])
    return out.flip(-2) if flip else out


# %%
# Measurements
# ------------
def load(rel, stem):
    """Fast transmits ``(n, Z, X)`` by increasing angle, CPWC and mean of the other transmits."""
    folder = os.path.join(CACHE_ROOT, rel)
    with open(os.path.join(folder, "manifest.json")) as f:
        manifest = json.load(f)
    angles_deg = np.asarray(manifest["angles_deg"])
    # By increasing angle: the channel order of the multi-transmit networks.
    fast_idx = sorted((int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG), key=angles_deg.__getitem__)
    rest = [a for a in range(len(angles_deg)) if a not in fast_idx]
    planes = torch.from_numpy(np.load(os.path.join(folder, stem + ".npy")))
    return planes[fast_idx], planes.mean(0), planes[rest].mean(0), manifest


def folder_stems(rel, n):
    """``n`` frames taken at regular steps through a folder."""
    stems = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(CACHE_ROOT, rel, "invi*_*.npy")))
    return stems[:: max(1, len(stems) // n)][:n]


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


def nmse_db(img, ref):
    return 10 * math.log10(float((img - ref).abs().pow(2).sum() / ref.abs().pow(2).sum()))


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


def homogeneity(flip=False):
    """Brightness maps and per-frame rows of measurement 1, plain or flipped input."""
    methods = (["DAS"] if not flip else []) + list(models)
    maps, rows = {m: [] for m in methods}, []
    for stem in HOMOGENEITY_FRAMES:
        tx, cpwc, _, manifest = load(PHANTOM, stem)
        outputs = ({"DAS": tx.mean(0)} if not flip else {}) | {name: denoise(name, tx, flip=flip) for name in models}
        for m, img in outputs.items():
            r = brightness_db(img.numpy(), cpwc.numpy())
            maps[m].append(r)
            darkest, area = dark(r)
            rows.append({"frame": stem, "method": m, "flipped_input": flip, "darkest_db": darkest,
                         "dark_area_pct": area, "deepest_fifth_db": bands(r)[-1]})
    return methods, maps, rows, manifest


def print_homogeneity(methods, rows):
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        d, a = [r["darkest_db"] for r in sel], [r["dark_area_pct"] for r in sel]
        print(
            f"  {m:{W}s} {min(d):6.1f} to {max(d):6.1f} dB   {min(a):5.1f}-{max(a):5.1f}% ({np.mean(a):4.1f}%)"
            f"   deepest fifth {np.mean([r['deepest_fifth_db'] for r in sel]):+5.2f} dB"
        )


band_header = " ".join(f"{f'z{i + 1}':>6s}" for i in range(DEPTH_BANDS))

# %%
# 1. Homogeneity
# --------------
if 1 in SECTIONS:
    methods, maps, rows, manifest = homogeneity()
    write_csv("homogeneity.csv", rows)
    print(
        f"\n1. Homogeneity on {len(HOMOGENEITY_FRAMES)} frames of {PHANTOM}: local brightness against CPWC, "
        f"darkest point (dB) and area below {DARK_DB:.0f} dB, range over the frames (mean)"
    )
    print_homogeneity(methods, rows)
    print(f"Mean over the frames by depth band, shallow to deep (dB):\n  {'':{W}s} {band_header}")
    for m in methods:
        print(f"  {m:{W}s} " + " ".join(f"{v:6.2f}" for v in bands(np.mean(maps[m], 0))))

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
if 2 in SECTIONS:
    # Every variant: the networks as they are, and each DRUNet with its noise
    # level scaled like the input, when its wrapper does not already make it
    # equivariant.
    variants = {name: (name, False) for name in models}
    for name, (model, _, _) in models.items():
        if name.startswith(NAMES["drunet"]) and not model.scale_equivariant:
            variants[f"{name}, sigma x gain"] = (name, True)

    rows, gain_maps = [], {}
    for stem in GAIN_FRAMES:
        tx, cpwc, _, _ = load(PHANTOM, stem)
        for variant, (name, scale_sigma) in variants.items():
            ref = denoise(name, tx)
            for g in GAINS:
                out = denoise(name, tx, gain=g, scale_sigma=scale_sigma) / g
                r = brightness_db(out.numpy(), cpwc.numpy())
                darkest, area = dark(r)
                error = float((out - ref).abs().norm() / ref.abs().norm())
                rows.append({"frame": stem, "network": variant, "gain": g, "equivariance_error": error,
                             "darkest_db": darkest, "dark_area_pct": area})
                if stem == GAIN_FRAMES[0]:
                    gain_maps[(variant, g)] = r
    write_csv("rescaling.csv", rows)

    VW = max(21, *(len(v) for v in variants))
    print(
        f"\n2. Rescaling on {len(GAIN_FRAMES)} frames: per gain, equivariance error, darkest point (dB) "
        f"and area below {DARK_DB:.0f} dB, means over the frames"
    )
    print(f"  {'':{VW}s}" + "".join(f"{'x' + str(g):>22s}" for g in GAINS))
    for variant in variants:
        cells = []
        for g in GAINS:
            sel = [r for r in rows if r["network"] == variant and r["gain"] == g]
            cells.append(
                f"{np.mean([r['equivariance_error'] for r in sel]):7.3f} "
                f"{np.mean([r['darkest_db'] for r in sel]):6.1f} "
                f"{np.mean([r['dark_area_pct'] for r in sel]):5.1f}%"
            )
        print(f"  {variant:{VW}s}" + "  ".join(cells))

    fig, ax = plt.subplots(len(variants), len(GAINS), figsize=(2.4 * len(GAINS), 3.6 * len(variants)), squeeze=False)
    for i, variant in enumerate(variants):
        for j, g in enumerate(GAINS):
            ax[i, j].imshow(gain_maps[(variant, g)], cmap="RdBu_r", vmin=-8, vmax=8)
            ax[i, j].set_title(f"{variant} x{g}", fontsize=8)
            ax[i, j].axis("off")
    fig.suptitle(f"{GAIN_FRAMES[0]}: local intensity vs CPWC (dB), input x gain, output / gain")
    fig.savefig(os.path.join(OUT_DIR, "rescaling_maps.png"), dpi=90)
    plt.close(fig)

# %%
# 3. Coherence with the rest of the sequence
# ------------------------------------------
if 3 in SECTIONS:
    print(
        f"\n3. Coherence of the input with the other transmits, and local gain of each network "
        f"(output against input, dB), by depth band\n  {'':{W + 4}s} {band_header}"
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
        cohs, gains = [], {name: [] for name in models}
        for stem in stems:
            tx, _, rest, manifest = load(rel, stem)
            fast = tx.mean(0)
            cohs.append(coherence(fast.numpy(), rest.numpy()))
            for name in models:
                gains[name].append(local_gain_db(denoise(name, tx).numpy(), fast.numpy()))
        coh = np.mean(cohs, 0)
        print(f"  {rel} ({len(stems)} frames)\n  {'  coherence':{W + 4}s} " + " ".join(f"{v:6.2f}" for v in bands(coh)))
        rows.append({"folder": rel, "quantity": "coherence"} | {f"z{i + 1}": v for i, v in enumerate(bands(coh))})
        for name in models:
            g = np.mean(gains[name], 0)
            print(f"  {'  ' + name:{W + 4}s} " + " ".join(f"{v:6.2f}" for v in bands(g)))
            rows.append({"folder": rel, "quantity": f"gain_db {name}"} | {f"z{i + 1}": v for i, v in enumerate(bands(g))})
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
    tx, _, rest, _ = load(PHANTOM, HOMOGENEITY_FRAMES[0])
    fast = tx.mean(0)
    coh = coherence(fast.numpy(), rest.numpy())
    print(f"\n{HOMOGENEITY_FRAMES[0]}: pixel-wise correlation of the coherence with the local gain of each network")
    for name in models:
        g = local_gain_db(denoise(name, tx).numpy(), fast.numpy())
        print(f"  {name:{W}s} r = {np.corrcoef(coh.ravel(), g.ravel())[0, 1]:+.2f}")

# %%
# 4. Vertical flip
# ----------------
if 4 in SECTIONS:
    methods, maps, rows, manifest = homogeneity(flip=True)
    write_csv("flip_homogeneity.csv", rows)
    print(
        f"\n4a. Homogeneity with the input flipped upside down (output flipped back), same frames and "
        f"columns as 1"
    )
    print_homogeneity(methods, rows)
    print("Per frame with the flipped input, dark area (%) / darkest point (dB):")
    for name in models:
        cells = []
        for stem in HOMOGENEITY_FRAMES:
            r = next(r for r in rows if r["frame"] == stem and r["method"] == name)
            cells.append(f"{stem[-5:]}: {r['dark_area_pct']:.1f}%/{r['darkest_db']:.1f}")
        print(f"  {name:{W}s} " + "  ".join(cells))

    print(
        f"\n4b. NMSE (dB) against the mean of the transmits not in the input, {N_NMSE} frames per folder at "
        f"regular steps; plain / flipped input"
    )
    print(f"  {'folder':26s} {'DAS':>7s} " + " ".join(f"{name:>{max(17, len(name))}s}" for name in models))
    rows = []
    for rel in NMSE_FOLDERS:
        das, res = [], {name: ([], []) for name in models}
        for stem in folder_stems(rel, N_NMSE):
            tx, _, rest, _ = load(rel, stem)
            das.append(nmse_db(tx.mean(0), rest))
            for name in models:
                res[name][0].append(nmse_db(denoise(name, tx), rest))
                res[name][1].append(nmse_db(denoise(name, tx, flip=True), rest))
            rows.append({"folder": rel, "frame": stem, "method": "DAS", "nmse_db": das[-1], "nmse_flipped_db": ""})
            rows += [{"folder": rel, "frame": stem, "method": name, "nmse_db": res[name][0][-1],
                      "nmse_flipped_db": res[name][1][-1]} for name in models]
        print(f"  {rel:26s} {np.mean(das):7.2f} " + " ".join(
            f"{f'{np.mean(res[n][0]):.2f} / {np.mean(res[n][1]):.2f}':>{max(17, len(n))}s}" for n in models
        ), flush=True)
    write_csv("flip_nmse.csv", rows)

# %%
# 5. Cropping the top of the input
# --------------------------------
if 5 in SECTIONS:
    tx, cpwc, _, _ = load(PHANTOM, CROP_FRAME)
    rows = []
    for c in CROP_ROWS:
        for m in ["DAS"] + list(models):
            out = tx[:, c:].mean(0) if m == "DAS" else denoise(m, tx[:, c:])
            darkest, area = dark(brightness_db(out.numpy(), cpwc[c:].numpy()))
            rows.append({"rows_removed": c, "method": m, "darkest_db": darkest, "dark_area_pct": area})
    write_csv("crop.csv", rows)
    print(
        f"\n5. {CROP_FRAME} with the top rows of the input removed: dark area (%) / darkest point (dB) "
        f"on the remaining rows, against CPWC cropped alike"
    )
    print(f"  {'':{W}s}" + "".join(f"{f'-{c} rows':>16s}" for c in CROP_ROWS))
    for m in ["DAS"] + list(models):
        print(f"  {m:{W}s}" + "".join(
            f"{f'{r['dark_area_pct']:.1f}% / {r['darkest_db']:.1f}':>16s}"
            for r in rows if r["method"] == m
        ))

# %%
# 6. Consistency across the groups of one acquisition
# ---------------------------------------------------
def consistency(images, signal):
    """Stability of a stack ``(G, Z, X)`` of complex images of the same scene, in ``signal``."""
    y = images[:, signal]  # (G, pixels)
    power = (np.abs(y) ** 2).mean(0)
    r1 = (y[1:] * y[:-1].conj()).mean(0) / np.sqrt((np.abs(y[1:]) ** 2).mean(0) * (np.abs(y[:-1]) ** 2).mean(0))
    phase = np.angle(r1, deg=True)
    return {
        "envelope_cv": float((np.abs(y).std(0) / np.abs(y).mean(0)).mean()),
        "incoherence": float((1 - np.abs(y.mean(0)) ** 2 / power).mean()),
        "r1_magnitude": float(np.abs(r1).mean()),
        "r1_phase_std_deg": float(np.sqrt((phase**2).mean())),  # about 0: RMS = spread
    }


if 6 in SECTIONS:
    methods = ["DAS", *models, f"DAS {87 // COMPOUND_SPLIT} tx"]
    rows, phantom_rows = [], []
    for rel in CONSISTENCY_FOLDERS:
        stems = HOMOGENEITY_FRAMES if rel == PHANTOM else folder_stems(rel, N_CONSISTENCY)
        with open(os.path.join(CACHE_ROOT, rel, "manifest.json")) as f:
            angles_deg = np.asarray(json.load(f)["angles_deg"])
        for stem in stems:
            planes = torch.from_numpy(np.load(os.path.join(CACHE_ROOT, rel, stem + ".npy")))
            cpwc = planes.mean(0)
            env = cpwc.abs().numpy()
            signal = 20 * np.log10(env / env.max()) > SIGNAL_DB
            stacks = {m: [] for m in methods}
            for g in GROUPS:
                tx = planes[sorted(g, key=angles_deg.__getitem__)]
                stacks["DAS"].append(tx.mean(0))
                for name in models:
                    stacks[name].append(denoise(name, tx))
            stacks[methods[-1]] = [planes[r::COMPOUND_SPLIT].mean(0) for r in range(COMPOUND_SPLIT)]
            for m, stack in stacks.items():
                stack = torch.stack(stack).numpy()
                rows.append({"folder": rel, "frame": stem, "method": m} | consistency(stack, signal))
                if rel == PHANTOM and m in models:
                    dark_maps = [brightness_db(img, cpwc.numpy()) < DARK_DB for img in stack]
                    areas = [100 * d.mean() for d in dark_maps]
                    phantom_rows.append({"frame": stem, "method": m, "mean_area_pct": np.mean(areas),
                                         "min_area_pct": min(areas), "max_area_pct": max(areas),
                                         "union_area_pct": 100 * np.any(dark_maps, 0).mean(),
                                         "intersection_area_pct": 100 * np.all(dark_maps, 0).mean()})
        print(f"6. {rel}: {len(stems)} frames done", flush=True)
    write_csv("consistency.csv", rows)
    keys = ["envelope_cv", "incoherence", "r1_magnitude", "r1_phase_std_deg"]
    MW = max(W, len(methods[-1]))
    print(
        f"\n6. Consistency of the outputs of the {K} groups of one acquisition, in the signal region "
        f"(CPWC above {SIGNAL_DB:.0f} dB), mean over the frames; the last row compounds "
        f"{COMPOUND_SPLIT} disjoint sets of {87 // COMPOUND_SPLIT} transmits"
    )
    for rel in CONSISTENCY_FOLDERS:
        print(f"  {rel}\n  {'':{MW}s} " + " ".join(f"{k:>17s}" for k in keys))
        for m in methods:
            sel = [r for r in rows if r["folder"] == rel and r["method"] == m]
            print(f"  {m:{MW}s} " + " ".join(f"{np.mean([r[k] for r in sel]):17.3f}" for k in keys))
    if phantom_rows:
        write_csv("consistency_phantom_dark.csv", phantom_rows)
        print(
            f"\nPhantom, dark area (< {DARK_DB:.0f} dB against CPWC) of each of the {K} groups: mean, range over "
            f"the groups, union and intersection of the {K} dark regions; mean over the frames (%)"
        )
        for name in models:
            sel = [r for r in phantom_rows if r["method"] == name]
            mean = lambda k: np.mean([r[k] for r in sel])
            print(
                f"  {name:{W}s} mean {mean('mean_area_pct'):5.1f}  range {mean('min_area_pct'):5.1f}-"
                f"{mean('max_area_pct'):5.1f}  union {mean('union_area_pct'):5.1f}  "
                f"intersection {mean('intersection_area_pct'):5.1f}"
            )

print(f"\nTables and figures written to {OUT_DIR}")
