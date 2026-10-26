r"""
Noise2Inverse for ultrafast ultrasound -- stage 3: inference
============================================================

Denoise fast plane-wave acquisitions with a network trained by
``train_n2i_deepinv.py``. A fast acquisition is the sub-reconstruction of a few
steering angles ``ANGLES_DEG`` out of the 87 transmits, i.e. the mean of their
cached single-angle IQ images (see ``precompute_n2i_ultrasound.py``); the
network was trained on sub-reconstructions of ``87 / K`` transmits, so
``ANGLES_DEG`` should have about that many entries for the input noise level to
match.

For every frame of the requested folders, the script writes the denoised IQ
image and, for the first frames, a B-mode figure comparing the fast acquisition,
its Noise2Inverse denoising and the full-aperture CPWC reference.
"""

import glob
import json
import os
import time
from pathlib import Path

import matplotlib

# Figures are only written to files: without this, matplotlib picks an
# interactive backend whenever DISPLAY is set, e.g. under ssh -X, and draws
# every figure through the X tunnel.
matplotlib.use("Agg")

import numpy as np
import torch

import deepinv as dinv
from n2i_models import AttentionResUNet, AttentionUNet, PaddedArtifactRemoval, ResUNet

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
INPUT_DIRS = ["volunteer_005/carotid"]
# Network: "unet", "resunet", "attunet", "attresunet" or "drunet", as in the
# training script.
ARCH = "drunet"
OUT_DIR = f"./results/n2i_ultrasound_{ARCH}"
# Checkpoint of the trained network; None = ``ckp_best`` of the latest training
# run under SAVE_PATH.
CHECKPOINT = None
SAVE_PATH = f"./checkpoints/n2i_ultrasound_{ARCH}"

# Steering angles of the fast acquisition, in degrees; each is mapped to the
# nearest transmit of the sequence (0.38 deg steps between -16.3 and +16.3).
ANGLES_DEG = [-10.0, 0.0, 10.0]

# Network: must match the training script.
SCALES = 4  # unet only
CHANNELS = 32
RESIDUAL = False  # unet only
BLOCKS = 2  # drunet only
UPSAMPLE = "bilinear"  # ported nets only

N_FRAMES = 2  # cap on frames per folder (None = all)
N_FIGURES = 5  # B-mode figures per folder
DYNAMIC_RANGE_DB = 50.0

device = dinv.utils.get_device()

# %%
# Network
# -------
if CHECKPOINT is None:
    CHECKPOINT = sorted(glob.glob(os.path.join(SAVE_PATH, "*", "ckp_best.pth.tar")))[-1]
if ARCH in ("resunet", "attunet", "attresunet"):
    net_cls = {"resunet": ResUNet, "attunet": AttentionUNet, "attresunet": AttentionResUNet}
    denoiser = net_cls[ARCH](
        in_channels=2, out_channels=2, base_channels=CHANNELS, upsample_mode=UPSAMPLE
    )
elif ARCH == "unet":
    denoiser = dinv.models.UNet(
        in_channels=2,  # I and Q
        out_channels=2,
        scales=SCALES,
        channels_per_scale=[CHANNELS * 2**k for k in range(SCALES)],
        residual=RESIDUAL,
        batch_norm=False,
    )
elif ARCH == "drunet":
    denoiser = dinv.models.DRUNet(
        in_channels=2,
        out_channels=2,
        nc=[CHANNELS * 2**k for k in range(4)],
        nb=BLOCKS,
        pretrained=None,
    )
else:
    raise ValueError(f"Unknown ARCH {ARCH!r}")

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = PaddedArtifactRemoval(
    denoiser,
    mode="direct",
    device=device,
    scale_equivariant=checkpoint.get("scale_equivariant", False),
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
physics = dinv.physics.Denoising().to(device)
print(f"Loaded {CHECKPOINT} (epoch {checkpoint['epoch']})")


# %%
# Denoise every frame
# -------------------
bmode = lambda img: dinv.utils.bmode(img[None, None], amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=False)

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
        angles_deg = np.asarray(manifest["angles_deg"])
        ANGLES = [int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG]
        files = sorted(glob.glob(os.path.join(cache_dir, "invi*_*.npy")))[:N_FRAMES]
        print(
            f"{rel}: denoising {len(files)} frames from transmits {ANGLES} "
            f"({', '.join(f'{a:+.2f}' for a in angles_deg[ANGLES])} deg) -> {out_dir}"
        )

        t_start = time.time()
        for i, path in enumerate(files):
            planes = np.load(path, mmap_mode="r")  # (n_angles, Z, X) complex64
            fast = torch.from_numpy(planes[ANGLES].mean(0))  # (Z, X)

            # I/Q channels in (the model pads and crops back), then recombine.
            net_in = torch.view_as_real(fast).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                out = model(net_in, physics)[0].cpu()
            denoised = torch.complex(out[0], out[1])

            np.save(os.path.join(out_dir, Path(path).stem + ".npy"), denoised.numpy())

            if i < N_FIGURES:
                cpwc = torch.from_numpy(planes[:].mean(0))
                dinv.utils.plot(
                    {
                        f"DAS, {len(ANGLES)} transmits": bmode(fast),
                        f"Noise2Inverse, {len(ANGLES)} transmits": bmode(denoised),
                        f"CPWC, {manifest['n_angles']} transmits": bmode(cpwc),
                    },
                    rescale_mode="clip",
                    vmin=-DYNAMIC_RANGE_DB,
                    vmax=0.0,
                    cmap="gray",
                    extent=extent,
                    aspect="equal",
                    figsize=(12, 6),
                    suptitle=f"{rel}/{Path(path).stem}, {DYNAMIC_RANGE_DB:.0f} dB",
                    save_fn=os.path.join(out_dir, Path(path).stem + "_bmode.png"),
                    show=False,
                )
        print(f"  done, {(time.time() - t_start) / max(len(files), 1):.2f} s/frame")
