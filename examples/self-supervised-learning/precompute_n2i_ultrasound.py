r"""
Noise2Inverse for ultrafast ultrasound -- stage 1: beamform every transmit
==========================================================================

Noise2Inverse :footcite:t:`hendriksen2020noise2inverse` splits the
*measurements* rather than the pixels: sub-reconstructions from disjoint subsets
of the plane-wave transmits have independent noise but the same anatomy, and a
network trained to map one to another learns to denoise.

This script beamforms each of the 87 plane-wave transmits of every acquisition
of the EPFL dataset, in vivo and in vitro (CIRS 054GS phantom), **separately** with
:class:`deepinv.physics.UltrasoundPlaneWave`, demodulates the RF images to
complex baseband (IQ) and caches the 87 single-angle IQ images per frame. The
adjoint and the demodulation are linear in the transmits, so the image of any
subset of transmits :math:`g` is exactly the mean of its cached single-angle
images,

.. math::
    x_g = \frac{1}{|g|} A_g^{\top} y_g = \frac{1}{|g|} \sum_{a \in g} A_a^{\top} y_a,

and ``train_n2i_deepinv.py`` is free to choose any partition of the transmits
without touching the raw data again (the plane-wave adjoint costs ~32 ms per
transmit per call, which would otherwise dominate training).

The RF images are sampled at an eighth of a wavelength axially to resolve the
carrier; once demodulated, the baseband keeps its information at a quarter of
that rate, so the IQ planes are decimated axially by ``DECIMATION`` and take half
the bytes of the RF ones (complex64 at :math:`\lambda/2`). Each frame is
normalized so that its full-aperture CPWC image (the mean of the 87 planes) has
unit RMS envelope. The cache mirrors the layout of the raw
data, ``<CACHE_ROOT>/volunteer_NNN/<region>/invivo_XXXXX.npy`` and
``<CACHE_ROOT>/invitro_CIRS054GS/<section>/invitro_XXXXX.npy``, so the training
script can assign whole volunteers or regions to training, validation and test.
The phantom is beamformed on the same grid as the in-vivo data (its acquisitions
share the probe and sequence), so the trained networks apply to it directly;
the training script only draws its training set from the in-vivo folders.
Frames already in the cache are skipped, so the script can be interrupted and
resumed.

Each cache folder also keeps a ``provenance.json`` so that the raw data can be
deleted once cached: for every frame, the raw file it comes from (path relative
to ``DATA_ROOT``, in the layout of ``download_epfl_ultrasound.py``), its size and
CRC32 (the checksum stored in the archive's central directory, so the file can
be identified and downloaded again with ``download_epfl_ultrasound.py --ids``),
the metadata stored in the ``.npz`` (volunteer and body region), and the two
scales applied to it, ``cache = DAS(rf / rf_scale) / iq_scale``, which give
back the absolute amplitude. Frames cached before this record existed are
backfilled from the raw file if it is still there (without ``iq_scale``), or
listed by name only otherwise. At the end of each folder the script says
whether all its raw files are cached and recorded, i.e. safe to delete.

Acquisition settings and field of view are identical to
``plug-and-play/demo_ultrasound_invivo_PnP.py``.
"""

import glob
import io
import json
import math
import os
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import deepinv as dinv

# %%
# Configuration
# -------------
DATA_ROOT = "/media/bobby/SSK SSD/epfl_ultrafast_ultrasound"
CACHE_ROOT = "/data/n2i_das_cache"
DATASET_URL = "https://datasets.epfl.ch/epfl_ultrafast_ultrasound"  # for provenance.json
# Folders to process; None = every volunteer region and both sections of the
# in-vitro phantom found in DATA_ROOT (as downloaded by
# download_epfl_ultrasound.py) or already in CACHE_ROOT.
FOLDERS = None
FORCE = False  # recompute frames already in the cache
LIMIT = None  # cap the number of frames per folder, e.g. 10 for a quick test
# Axial decimation of the demodulated IQ planes. RF is sampled at lambda/8; the
# transmit pulse (53% fractional bandwidth) fits in the band of lambda/2 sampling
# with the out-of-band energy below -27 dB on this dataset, even with the
# attenuation-induced downshift of the carrier. 87 complex64 planes of 468x234
# px = 76 MB per frame.
DECIMATION = 4

# Field of view (m). Deeper than the carotid window of the PnP demo, since
# abdominal targets sit further down.
Z_MIN, Z_MAX, X_HALF = 2e-3, 70e-3, 18e-3

# Probe geometry and transmit sequence of the EPFL LTS5 dataset (GE 9L-D linear
# array on a Verasonics scanner), from its acquisition metadata.
CENTER_FREQ = 5.3e6
FRAC_BW = 0.75 / math.sqrt(2)
N_ELEMENTS = 192
PITCH = 2.3e-4
ELEMENT_WIDTH = 6.1206151719371e-05
SAMPLING_FREQ = 20833333.333333332
SOUND_SPEED = 1540.0
N_ANGLES = 87
T0 = 4.272e-06 + (4.1e-07 - 1.92e-07)  # -time_axis[0] + (peak_time - lens_correction)

device = dinv.utils.get_device()

# %%
# Imaging operator
# ----------------
# The 87 steering angles in acquisition order: alternating +/- 0.38 deg steps
# from the widest angle inwards, ending at 0.
angles = math.radians(0.38) * torch.tensor(
    [k * sign for k in range(43, 0, -1) for sign in (-1, 1)] + [0]
)

# Pixel grid: a wavelength eighth axially, since the images are reconstructed in
# the RF domain and must resolve the carrier; a pitch and a half laterally.
wavelength = SOUND_SPEED / CENTER_FREQ
pixel_size = (wavelength / 8, PITCH / 1.5)
img_size = (int((Z_MAX - Z_MIN) / pixel_size[0]), int(2 * X_HALF / pixel_size[1]))

element_x = torch.arange(N_ELEMENTS) * PITCH + ELEMENT_WIDTH / 2
element_x = element_x - element_x.mean()
element_positions = torch.stack([element_x, torch.zeros_like(element_x)], dim=-1)

# Pulse-echo impulse response: a Gaussian-modulated cosine at the carrier.
sigma_t = math.sqrt(2 * math.log(2)) / (math.pi * FRAC_BW * CENTER_FREQ)
n_half = math.ceil(3.5 * sigma_t * SAMPLING_FREQ)
t = (np.arange(2 * n_half + 1) - n_half) / SAMPLING_FREQ
pulse = np.exp(-(t**2) / (2 * sigma_t**2)) * np.cos(2 * np.pi * CENTER_FREQ * t)
pulse = torch.as_tensor(pulse, dtype=torch.float32)

# Demodulation: along z, the RF carrier sits at the round-trip wavenumber
# 2/lambda, i.e. at bin ``carrier_bin`` of the axial FFT. The spectrum is shifted
# to baseband, cut to the central 1/DECIMATION (an ideal low-pass), and inverse
# transformed on the coarser grid. The factor 2 makes ``abs(iq)`` the envelope.
n_z = img_size[0] - img_size[0] % DECIMATION
n_z_iq = n_z // DECIMATION
carrier_bin = round(2 / wavelength * n_z * pixel_size[0])
img_size_iq = (n_z_iq, img_size[1])
pixel_size_iq = (pixel_size[0] * DECIMATION, pixel_size[1])

# %%
# Beamform, demodulate and cache, folder by folder
# ------------------------------------------------
# Per frame, an ``(n_angles, Z, X)`` complex64 array whose plane ``a`` is the
# demodulated single-angle image :math:`A_a^{\top} y_a`, scaled so that the mean
# over the planes (the full-aperture CPWC image) has unit RMS envelope.
def read_raw(path):
    """Load a raw ``.npz`` and its provenance record (without the scales)."""
    buf = Path(path).read_bytes()
    npz = np.load(io.BytesIO(buf))
    data = npz["data"]
    record = {
        "source": os.path.relpath(path, DATA_ROOT),
        "size": len(buf),
        "crc32": f"{zlib.crc32(buf):08x}",
        "metadata": {k: npz[k].tolist() for k in npz.files if k != "data"},
        "shape": list(data.shape),
        "dtype": str(data.dtype),
    }
    return data, record


def save_provenance(cache_dir, frames):
    """Write ``provenance.json`` atomically, so an interruption never corrupts it."""
    path = os.path.join(cache_dir, "provenance.json")
    with open(path + ".tmp", "w") as f:
        json.dump(
            {
                "dataset": DATASET_URL,
                "relation": "cache = DAS(rf / rf_scale) / iq_scale, rf = data[0] of source",
                "frames": dict(sorted(frames.items())),
            },
            f,
            indent=1,
        )
    os.replace(path + ".tmp", path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if FOLDERS is None:
    FOLDERS = sorted(
        {
            os.path.relpath(d, root)
            for root in (DATA_ROOT, CACHE_ROOT)
            for pattern in ("volunteer_*", "invitro_*")
            for d in glob.glob(os.path.join(root, pattern, "*"))
            if os.path.isdir(d)
        }
    )

for folder in FOLDERS:
    # invivo_XXXXX.npz or invitro_XXXXX.npz, same content and layout.
    all_files = sorted(glob.glob(os.path.join(DATA_ROOT, folder, "invi*_*.npz")))
    files = all_files[:LIMIT]
    cache_dir = os.path.join(CACHE_ROOT, folder)
    os.makedirs(cache_dir, exist_ok=True)
    prov_path = os.path.join(cache_dir, "provenance.json")
    provenance = {}
    if os.path.exists(prov_path):
        with open(prov_path) as f:
            provenance = json.load(f)["frames"]

    # Frames cached before provenance.json existed: record them from the raw
    # file if it is still there (iq_scale is then unknown), by name otherwise.
    raw_by_stem = {Path(p).stem: p for p in all_files}
    cached = sorted(Path(p).stem for p in glob.glob(os.path.join(cache_dir, "invi*_*.npy")))
    missing = [s for s in cached if s not in provenance]
    if missing:
        print(f"{folder}: recording the provenance of {len(missing)} cached frames")
    for i, stem in enumerate(missing):
        if stem in raw_by_stem:
            data, record = read_raw(raw_by_stem[stem])
            record["rf_scale"] = float(np.abs(data[0]).max())
            record["iq_scale"] = None
        else:
            source = os.path.join(folder, stem + ".npz")
            record = {"source": source, "size": None, "crc32": None}
        provenance[stem] = record | {"recorded": now(), "backfilled": True}
        if (i + 1) % 50 == 0 or i + 1 == len(missing):
            save_provenance(cache_dir, provenance)
            print(f"  [{i + 1}/{len(missing)}]", flush=True)

    if not files:
        print(f"{folder}: no raw data under {DATA_ROOT}, {len(cached)} frames cached")
        continue

    # A single-transmit operator; the steering angle is swapped in place per
    # transmit (cheap, as the receive delays do not depend on it).
    n_samples = int(np.load(files[0])["data"].shape[-1])
    physics = dinv.physics.UltrasoundPlaneWave(
        img_size=img_size,
        angles=angles[:1],
        element_positions=element_positions.to(device),
        n_samples=n_samples,
        sampling_frequency=SAMPLING_FREQ,
        sound_speed=SOUND_SPEED,
        pixel_size=pixel_size,
        pixel_origin=(Z_MIN, -X_HALF),
        t0=T0,
        pulse=pulse,
        f_number=1.5,
        receive_apod_window="hann",
        normalize=False,
        device=device,
    )

    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                "source": folder,
                "angles_deg": torch.rad2deg(angles).tolist(),
                "img_size": list(img_size_iq),
                "pixel_size": list(pixel_size_iq),
                "z_min": Z_MIN,
                "z_max": Z_MIN + n_z * pixel_size[0],
                "x_half": X_HALF,
                "n_angles": N_ANGLES,
                "n_samples": n_samples,
                "decimation": DECIMATION,
                "demodulation_wavenumber": 2 / wavelength,
                "dtype": "complex64",
            },
            f,
            indent=2,
        )

    print(
        f"{folder}: beamforming {len(files)} frames into {N_ANGLES} single-angle "
        f"IQ images of {img_size_iq[0]}x{img_size_iq[1]} px -> {cache_dir}"
    )
    t_start, n_done = time.time(), 0
    for i, path in enumerate(files):
        out_path = os.path.join(cache_dir, Path(path).stem + ".npy")
        if os.path.exists(out_path) and not FORCE:
            continue

        data, record = read_raw(path)
        rf = data[0].astype(np.float32)  # (n_angles, n_elements, n_samples)
        rf_scale = float(np.abs(rf).max())
        y = torch.as_tensor(rf / rf_scale)[None, None].to(device)

        planes = torch.empty((N_ANGLES, *img_size), device=device)
        with torch.no_grad():
            for a in range(N_ANGLES):
                physics.update_parameters(angles=angles[a : a + 1])
                planes[a] = physics.A_adjoint(y[:, :, a : a + 1].contiguous()).reshape(img_size)
        spectrum = torch.fft.fft(planes[:, :n_z], dim=1)
        spectrum = torch.fft.fftshift(torch.roll(spectrum, -carrier_bin, dims=1), dim=1)
        spectrum = spectrum[:, (n_z - n_z_iq) // 2 : (n_z + n_z_iq) // 2]
        iq = 2 * torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=1), dim=1) / DECIMATION
        iq_scale = iq.mean(0).abs().pow(2).mean().sqrt()
        iq /= iq_scale
        # Written under a temporary name, so an interrupted run never leaves a
        # truncated frame that a rerun would skip and training would load.
        with open(out_path + ".tmp", "wb") as f:
            np.save(f, iq.cpu().numpy().astype(np.complex64))
        os.replace(out_path + ".tmp", out_path)
        provenance[Path(path).stem] = record | {
            "rf_scale": rf_scale,
            "iq_scale": float(iq_scale),
            "recorded": now(),
        }
        save_provenance(cache_dir, provenance)

        n_done += 1
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            rate = (time.time() - t_start) / n_done
            remaining = rate * (len(files) - i - 1) / 60
            print(
                f"  [{i + 1}/{len(files)}] {rate:.1f} s/frame, {remaining:.0f} min left",
                flush=True,
            )

    cached = {Path(p).stem for p in glob.glob(os.path.join(cache_dir, "invi*_*.npy"))}
    pending = [s for s in raw_by_stem if s not in cached or not provenance.get(s, {}).get("crc32")]
    print(f"  cache ready: {len(cached)} frames")
    if pending:
        print(f"  NOT safe to delete the raw data yet: {len(pending)} raw files not cached/recorded")
    else:
        print(f"  all {len(raw_by_stem)} raw files cached and recorded: {os.path.join(DATA_ROOT, folder)} can be deleted")
