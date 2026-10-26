r"""
Ultrasound despeckling with RAM 2
=================================

Post-processing of the full-aperture CPWC image (mean of all 87 transmits) of
the EPFL ultrafast dataset with the Reconstruct Anything Model 2 of Blur Labs
(https://www.blurlabs.ai/ram2/, 359 M parameters, GPL-3.0), used zero-shot as a
denoiser (``physics=None``): speckle is the noise.

RAM 2 is not installed (its ``pip`` dependencies would replace the editable
deepinv by deepinv main): its repository is cloned next to deepinv and imported
from there (``RAM2_REPO``). The weights are gated: request access at
https://huggingface.co/blurlabs/ram2, then ``hf auth login`` (or set HF_TOKEN).

RAM 2 works on the log-compressed B-mode, as in ``demo_ultrasound_despeckling.py``
(``DYNAMIC_RANGE_DB`` below the maximum mapped to [0, 1]), where the speckle is
roughly additive Gaussian noise. Its reference ``sigma`` is the standard deviation
of the log of single-look speckle, 5.57 dB, divided by the dynamic range; the
script sweeps multiples of it (``LOG_GAINS``), with BM3D at the reference sigma
as a baseline. (Handing RAM 2 the intensity with its Gamma noise model was tried
and dropped: it fills anechoic cysts and blacks out dark regions in vivo.)

For every frame of ``INPUT_DIRS``
the script saves a B-mode figure (CPWC and every method), then measures the
CIRS phantom frame ``invitro_02162`` as in ``n2i_invitro.py``: despeckling should
raise the speckle SNR and CNR of the cysts while keeping their contrast. Figures
are only saved to files, never displayed.
"""

import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

# Figures are only written to files: without this, matplotlib picks an
# interactive backend whenever DISPLAY is set, e.g. under ssh -X.
matplotlib.use("Agg")

import numpy as np
import torch

import deepinv as dinv
import n2i_invitro

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
INPUT_DIRS = ["volunteer_002/carotid", "volunteer_006/carotid", "volunteer_008"]
N_FRAMES = 3  # cap on frames per folder (None = all)
OUT_DIR = "./results/ram2_despeckling"
RAM2_REPO = os.environ.get("RAM2_REPO", os.path.expanduser("~/dev/blurlabs"))
# A checkpoint path, "download" (Hugging Face) or "none" (random weights, smoke test).
RAM2_WEIGHTS = os.environ.get("RAM2_WEIGHTS", "download")

# One column per value: multiplier of the log-domain sigma (larger = stronger
# despeckling), e.g. RAM2_LOG_GAINS=0.5,1,2.
LOG_GAINS = [
    float(v) for v in os.environ.get("RAM2_LOG_GAINS", "0.25,0.5,0.75,1,1.5,2").split(",")
]
# BM3D baseline in the log domain, at the reference sigma (gain 1); needs ~3 GB of GPU.
BM3D = os.environ.get("BM3D", "1") == "1"
# eres_v2 of ginger with the B-mode preset of its linear probes (Vermon LA2536):
# its filter radii are in pixels, so it also runs on the image resampled to that
# preset's beamforming grid (ERES_GRID_MM, axial x lateral) and back.
ERES = os.environ.get("ERES", "1") == "1"
ERES_GRID_MM = (0.4205, 0.1905)
# "bmode": eres on the B-mode given to RAM 2 (DYNAMIC_RANGE_DB mapped to [0, 1]);
# "envelope": eres on the envelope normalized by its peak, then log compression;
# "log": eres inside ginger's chain, on the S-curve log-compressed image.
ERES_DOMAIN = os.environ.get("ERES_DOMAIN", "bmode")
DYNAMIC_RANGE_DB = 50.0
# Standard deviation of 10 log10 of single-look (exponential) speckle intensity.
LOG_SPECKLE_STD_DB = 10 / math.log(10) * math.pi / math.sqrt(6)  # 5.57 dB

device = dinv.utils.get_device()


# %%
# Models
# ------
sys.path.insert(0, RAM2_REPO)
from blurlabs.models import RAM2  # noqa: E402

ram2 = RAM2(pretrained=None if RAM2_WEIGHTS == "none" else RAM2_WEIGHTS).to(device).eval()
bm3d = dinv.models.BM3D(use_legacy=False, device=device) if BM3D else None


class Eres:
    """eres_v2 of ginger's default B-mode pipeline for a linear probe, on ``(Z, X)``
    amplitudes.

    In the ``"bmode"`` domain, eres filters the B-mode that RAM 2 sees (``to_log``)
    and its output is mapped back by ``from_log``. In the ``"envelope"`` domain, it filters the envelope divided by its peak and
    the result, times the peak, is log compressed like the other methods. In the
    ``"log"`` domain it runs inside the ``compress -> eres_v2 -> compress_2``
    chain of the pipeline: ``compress`` maps the amplitude in absolute dB (``power_floor_db`` to
    ``power_floor_db + power_range_db``) to [0, 1] through an S-curve; the peak of
    the image is put at the top of that range. The output is mapped back to an
    amplitude by inverting the same chain without eres, so that it is displayed
    and measured like the other methods."""

    def __init__(self, domain="bmode"):
        self.domain = domain
        from ginger.postprocessing.compress.compress import Compress
        from ginger.postprocessing.image_filter.eres_v2 import EresV2
        from ginger.us.default_intent import GetDefaultBModeIntent
        from ginger.us.modes.generate_modalities import GenerateModality
        from ginger.us.probe_heads import vermon_la2536

        probe = vermon_la2536.build()
        modality = GenerateModality(GetDefaultBModeIntent(probe.geometry), probe)
        filters = modality.pipeline_descriptors.descriptors[0].pipeline_b_descriptors
        self.params = [f.compress for f in filters.renderer.pre_sc_filters[::2]]
        self.params.insert(1, filters.renderer.pre_sc_filters[1].eres_v2)
        self.top_db = self.params[0].power_floor_db + self.params[0].power_range_db
        self.classes = Compress, EresV2, Compress
        self.chains = {}
        # Inverse of compress_2(compress(.)) without eres, on a fine dB ramp: the
        # ramp is cut to the strictly increasing part between floor and saturation.
        ramp_db = np.linspace(self.top_db - 120, self.top_db + 10, 4096)
        x = (10 ** (ramp_db[None] / 20)).astype(np.float32)
        ramp = self.stage(2, *x.shape).filter(self.stage(0, *x.shape).filter(x))[0]
        lo, hi = np.flatnonzero(ramp == ramp.min())[-1], np.flatnonzero(ramp == ramp.max())[0]
        self.ramp, first = np.unique(ramp[lo : hi + 1], return_index=True)
        self.ramp_db = ramp_db[lo : hi + 1][first]

    def stage(self, i, n_x, n_z):
        """Filter ``i`` of the chain for an ``(n_x, n_z)`` image (ginger's layout)."""
        if (i, n_x, n_z) not in self.chains:
            cls, param = self.classes[i], self.params[i]
            param = type(param).FromString(param.SerializeToString())
            shape = param.pyramid.shape if i == 1 else param.shape
            shape.n_x, shape.n_z = n_x, n_z
            self.chains[i, n_x, n_z] = cls(param)
        return self.chains[i, n_x, n_z]

    def __call__(self, amp, grid=None):
        """``amp``: ``(Z, X)`` amplitude tensor; ``grid``: ``(Z, X)`` size to run on."""
        size = tuple(amp.shape)
        if self.domain in ("bmode", "envelope"):
            peak = amp.max()
            x = to_log(amp, peak) if self.domain == "bmode" else amp / peak
            if grid is not None:
                x = torch.nn.functional.interpolate(
                    x[None, None], size=grid, mode="bilinear", antialias=True
                )[0, 0]
            x = x.T.contiguous().numpy().astype(np.float32)
            y = torch.from_numpy(self.stage(1, *x.shape).filter(x).T.copy())
            if grid is not None:
                y = torch.nn.functional.interpolate(y[None, None], size=size, mode="bicubic")[0, 0]
            if self.domain == "bmode":
                return from_log(y, peak)
            return peak * y.clamp_min(0)  # eres can undershoot below zero
        db = 20 * torch.log10(amp.clamp_min(1e-12) / amp.max()) + self.top_db
        if grid is not None:  # resample the log image, like the other methods' input
            db = torch.nn.functional.interpolate(
                db[None, None], size=grid, mode="bilinear", antialias=True
            )[0, 0]
        x = (10 ** (db / 20)).T.contiguous().numpy().astype(np.float32)
        for i in range(3):
            x = self.stage(i, *x.shape).filter(x)
        y = torch.from_numpy(x.T.copy())
        if grid is not None:
            y = torch.nn.functional.interpolate(y[None, None], size=size, mode="bicubic")[0, 0]
        out_db = torch.from_numpy(np.interp(y.numpy(), self.ramp, self.ramp_db)).float()
        return amp.max() * 10 ** ((out_db - self.top_db) / 20)


eres = Eres(ERES_DOMAIN) if ERES else None


def to_log(amp, peak):
    db = 20 * torch.log10(amp.clamp_min(1e-12) / peak)
    return ((db + DYNAMIC_RANGE_DB) / DYNAMIC_RANGE_DB).clamp(0, 1)


def from_log(x, peak):
    return peak * 10 ** ((x.clamp(0, 1) - 1) * DYNAMIC_RANGE_DB / 20)


def despeckle(img, pixel_size, verbose=False):
    """All methods on the complex ``(Z, X)`` image -> {name: amplitude ``(Z, X)``},
    as complex tensors with zero phase: ``dinv.utils.bmode`` takes the modulus of a
    complex image, but the Hilbert envelope along depth of a real one.
    ``pixel_size``: ``(dz, dx)`` in m."""
    amp = img.abs().to(device)
    peak = amp.max()
    out = {}
    with torch.no_grad():
        y = to_log(amp, peak)[None, None]
        sigma = LOG_SPECKLE_STD_DB / DYNAMIC_RANGE_DB
        for gain in LOG_GAINS:
            x = ram2(y, sigma=sigma * gain, verbose=verbose)[0, 0]
            out[f"RAM 2, {LOG_SPECKLE_STD_DB * gain:.1f} dB"] = from_log(x, peak)
        if bm3d is not None:
            out[f"BM3D, {LOG_SPECKLE_STD_DB:.1f} dB"] = from_log(bm3d(y, sigma=sigma)[0, 0], peak)
    if eres is not None:
        amp = amp.cpu()
        grid = [round(n * d * 1e3 / g) for n, d, g in zip(amp.shape, pixel_size, ERES_GRID_MM)]
        out[f"eres {ERES_DOMAIN}"] = eres(amp)
        out[f"eres {ERES_DOMAIN}, coarse"] = eres(amp, grid)  # on the ERES_GRID_MM grid
    return {k: torch.complex(v, torch.zeros_like(v)).cpu() for k, v in out.items()}


def effective_looks(img, manifest):
    """(mean / std)^2 of the intensity in the phantom speckle region (1 for fully
    developed speckle of a single coherent image)."""
    _, speckle = n2i_invitro.regions(manifest)
    intensity = img.abs().pow(2).double().numpy()[speckle]
    return intensity.mean() ** 2 / intensity.var()


# %%
# Plot every frame
# ----------------
def bmode(amp, peak):
    return dinv.utils.bmode(amp[None, None], amplitude_floor_db=-DYNAMIC_RANGE_DB, reference=peak, normalize=False)


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
        files = sorted(glob.glob(os.path.join(cache_dir, "invi*_*.npy")))[:N_FRAMES]
        print(f"{rel}: {len(files)} frames -> {out_dir}")
        for i, path in enumerate(files):
            cpwc = torch.from_numpy(np.load(path)).mean(0)  # (Z, X) complex
            peak = cpwc.abs().max()
            images = {f"CPWC, {manifest['n_angles']} transmits": cpwc}
            images |= despeckle(cpwc, manifest["pixel_size"], verbose=i == 0)
            dinv.utils.plot(
                {k: bmode(v, peak) for k, v in images.items()},
                rescale_mode="clip",
                vmin=-DYNAMIC_RANGE_DB,
                vmax=0.0,
                cmap="gray",
                extent=extent,
                aspect="equal",
                figsize=(3 * len(images), 6),
                suptitle=f"{rel}/{Path(path).stem}, {DYNAMIC_RANGE_DB:.0f} dB",
                save_fn=os.path.join(out_dir, Path(path).stem + "_despeckling.png"),
                show=False,
                close=True,
            )


# %%
# CIRS phantom
# ------------
invitro_path = os.path.join(CACHE_ROOT, n2i_invitro.FRAME)
if not os.path.exists(invitro_path):
    print(f"Phantom test skipped: {invitro_path} not cached (run the precompute script)")
else:
    planes, manifest = n2i_invitro.load_frame(CACHE_ROOT)
    cpwc = planes.mean(0)
    print(
        f"Phantom {n2i_invitro.FRAME}: effective looks of the CPWC intensity "
        f"{effective_looks(cpwc, manifest):.2f}"
    )
    images = {f"CPWC, {manifest['n_angles']} transmits": cpwc}
    images |= despeckle(cpwc, manifest["pixel_size"], verbose=True)
    results = {name: n2i_invitro.metrics(img, manifest) for name, img in images.items()}
    n2i_invitro.print_table(results)

    out_dir = os.path.join(OUT_DIR, os.path.dirname(n2i_invitro.FRAME))
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(n2i_invitro.FRAME).stem
    with open(os.path.join(out_dir, stem + "_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *next(iter(results.values()))])
        writer.writeheader()
        writer.writerows({"method": name, **m} for name, m in results.items())
    n2i_invitro.plot(
        images, manifest, os.path.join(out_dir, stem + "_despeckling.png"), DYNAMIC_RANGE_DB
    )
    print(f"  -> {out_dir}")
