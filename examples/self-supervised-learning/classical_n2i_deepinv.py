r"""
Noise2Inverse for ultrafast ultrasound -- classical denoisers
=============================================================

Score BM3D and non-local means (NLM) as ``eval_n2i_deepinv.py`` scores the
networks: on the I and Q channels of the three-transmit DAS image, against the
held-out reconstruction (mean of the transmits not in the input), on the fast
and the interleaved inputs of every frame of the test volunteers.

Each method has one parameter, tuned like the networks' checkpoints on the
validation volunteer: BM3D's noise level ``sigma_psd`` and NLM's filter
strength ``h`` (``patch_size``, ``patch_distance`` fixed), each picked from a
grid by the mean PSNR of the fast input on ``N_TUNE`` frames per validation
folder. Images are normalised to unit RMS envelope by the precompute script,
so one value serves every frame.

The script prints, per test folder, the mean ± standard deviation of every
metric and the paired difference of each method from ``REFERENCE``, read from
the per-frame CSV of ``eval_n2i_deepinv.py`` (``REFERENCE_CSV_DIR``), and
writes its own per-frame CSV. The work is spread over ``N_WORKERS`` processes.
"""

import csv
import glob
import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process

import bm3d
import numpy as np
import torch
from scipy.stats import wilcoxon
from skimage.restoration import denoise_nl_means

import deepinv as dinv

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
VAL_DIRS = ["volunteer_005"]  # validation volunteer of train_n2i_deepinv.py (round 4)
EVAL_DIRS = ["volunteer_002", "volunteer_006", "volunteer_008"]  # test volunteers
N_TUNE = 8  # frames per validation folder for the tuning, evenly spaced
# Frames per test folder (None = all); the N2I_N_FRAMES environment variable
# overrides it ("all" or a number).
N_FRAMES = None
if (n_frames := os.environ.get("N2I_N_FRAMES")) is not None:
    N_FRAMES = None if n_frames == "all" else int(n_frames)
# Parameter grids, in units of the normalised IQ image (I and Q have an RMS of about 0.7).
GRIDS = {
    "BM3D": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
    "NLM": [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
}
NLM_PATCH, NLM_DISTANCE = 5, 6  # pixels
# Paired differences are taken from this network, whose per-frame metrics
# eval_n2i_deepinv.py wrote to REFERENCE_CSV_DIR/<folder>/metrics.csv.
REFERENCE = "DRUNet"
REFERENCE_CSV_DIR = "./results/n2i_ultrasound_metrics_test002_006_008"
OUT_DIR = "./results/n2i_ultrasound_classical_test002_006_008"
N_WORKERS = 12

# Inputs and metrics of eval_n2i_deepinv.py.
K = 29
GROUPS = [list(range(j, 87, K)) for j in range(K)]
ANGLES_DEG = [-10.0, 0.0, 10.0]
DYNAMIC_RANGE_DB = 50.0


# %%
# Denoisers and metrics
# ---------------------
def denoise(method, img, param):
    """Denoise the I and Q channels of a complex ``(Z, X)`` image separately."""
    if method == "BM3D":
        f = lambda c: bm3d.bm3d(c, sigma_psd=param)
    elif method == "NLM":
        f = lambda c: denoise_nl_means(c, h=param, patch_size=NLM_PATCH, patch_distance=NLM_DISTANCE, fast_mode=True)
    else:
        raise ValueError(method)
    return (f(img.real.astype(np.float64)) + 1j * f(img.imag.astype(np.float64))).astype(np.complex64)


ssim = dinv.metric.SSIM(max_pixel=1.0)
bmode01 = lambda img: dinv.utils.bmode(
    torch.from_numpy(img)[None, None], amplitude_floor_db=-DYNAMIC_RANGE_DB, normalize=True
)


def psnr(img, ref):
    return float(10 * np.log10(np.abs(ref).max() ** 2 / (np.abs(img - ref) ** 2).mean()))


def inputs(manifest):
    """Transmits of the fast and interleaved inputs, as in eval_n2i_deepinv.py."""
    angles_deg = np.asarray(manifest["angles_deg"])
    by_angle = lambda tx: sorted(tx, key=angles_deg.__getitem__)
    return {
        "fast": by_angle(int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG),
        "interleaved": by_angle(GROUPS[0]),
    }


def frames(folder, n=None):
    """``(relative folder, path, manifest)`` of the frames of every region of a volunteer."""
    out = []
    for manifest_path in sorted(glob.glob(os.path.join(CACHE_ROOT, folder, "**", "manifest.json"), recursive=True)):
        with open(manifest_path) as f:
            manifest = json.load(f)
        files = sorted(glob.glob(os.path.join(os.path.dirname(manifest_path), "invivo_*.npy")))
        if n is not None:
            files = [files[i] for i in np.linspace(0, len(files) - 1, n).round().astype(int)]
        rel = os.path.relpath(os.path.dirname(manifest_path), CACHE_ROOT)
        out += [(rel, path, manifest) for path in files]
    return out


def tune_frame(job):
    """PSNR of the fast input of one frame for every method and parameter."""
    _, path, manifest = job
    planes = np.load(path, mmap_mode="r")
    idx = inputs(manifest)["fast"]
    inp = np.asarray(planes[idx]).mean(0)
    ref = np.asarray(planes[[a for a in range(planes.shape[0]) if a not in idx]]).mean(0)
    return {(m, p): psnr(denoise(m, inp, p), ref) for m, grid in GRIDS.items() for p in grid}


def score_frame(job):
    """Metrics of DAS and every tuned method on both inputs of one frame."""
    (rel, path, manifest), params = job
    planes = np.load(path, mmap_mode="r")
    row = {"folder": rel, "frame": os.path.basename(path)}
    for setting, idx in inputs(manifest).items():
        inp = np.asarray(planes[idx]).mean(0)
        ref = np.asarray(planes[[a for a in range(planes.shape[0]) if a not in idx]]).mean(0)
        outs = {"DAS": inp} | {m: denoise(m, inp, p) for m, p in params.items()}
        for name, img in outs.items():
            row[f"{name} PSNR {setting}"] = psnr(img, ref)
            row[f"{name} SSIM {setting}"] = ssim(bmode01(img), bmode01(ref)).item()
    return row


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs(OUT_DIR, exist_ok=True)

    # %%
    # Tuning on the validation volunteer
    # ----------------------------------
    jobs = [job for folder in VAL_DIRS for job in frames(folder, N_TUNE)]
    with Pool(N_WORKERS) as pool:
        tuned = pool.map(tune_frame, jobs)
    print(f"Tuning on {len(jobs)} frames of {', '.join(VAL_DIRS)} (fast input): mean PSNR (dB)")
    params = {}
    for m, grid in GRIDS.items():
        means = {p: np.mean([t[(m, p)] for t in tuned]) for p in grid}
        params[m] = max(means, key=means.get)
        print(f"  {m:5s} " + "  ".join(f"{p:g}: {v:.2f}" for p, v in means.items()) + f"  -> {params[m]:g}")
        if params[m] in (grid[0], grid[-1]):
            print(f"  warning: the best {m} parameter is at the edge of its grid")

    # %%
    # Test volunteers
    # ---------------
    jobs = [(job, params) for folder in EVAL_DIRS for job in frames(folder, N_FRAMES)]
    with Pool(N_WORKERS) as pool:
        rows = pool.map(score_frame, jobs, chunksize=4)
    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = ["PSNR fast", "SSIM fast", "PSNR interleaved", "SSIM interleaved"]
    for rel in dict.fromkeys(r["folder"] for r in rows):
        sel = [r for r in rows if r["folder"] == rel]
        print(f"\n{rel}: {len(sel)} frames, " + ", ".join(f"{m} {p:g}" for m, p in params.items()))
        print(f"| Method | {' | '.join(columns)} |\n|---{'|---' * len(columns)}|")
        for name in ["DAS", *params]:
            cells = []
            for c in columns:
                v = np.array([r[f"{name} {c}"] for r in sel])
                cells.append(f"{v.mean():.{2 if 'PSNR' in c else 3}f} ± {v.std():.{2 if 'PSNR' in c else 3}f}")
            print(f"| {name} | {' | '.join(cells)} |")

        ref_csv = os.path.join(REFERENCE_CSV_DIR, rel, "metrics.csv")
        if not os.path.exists(ref_csv):
            print(f"  no {ref_csv}, no paired differences")
            continue
        with open(ref_csv) as f:
            reference = {r["frame"]: r for r in csv.DictReader(f)}
        if not all(r["frame"] in reference for r in sel):
            print(f"  {ref_csv} lacks some frames, no paired differences")
            continue
        print(f"Paired differences from {REFERENCE}: mean (frames where better, Wilcoxon p)")
        print(f"| Method | {' | '.join('Δ' + c for c in columns)} |\n|---{'|---' * len(columns)}|")
        for name in ["DAS", *params]:
            cells = []
            for c in columns:
                diff = np.array([r[f"{name} {c}"] - float(reference[r["frame"]][f"{REFERENCE} {c}"]) for r in sel])
                p = wilcoxon(diff).pvalue if np.any(diff) else 1.0
                unit = " dB" if "PSNR" in c else ""
                cells.append(f"{diff.mean():+.3f}{unit} ({np.mean(diff > 0) * 100:.0f}%, p={p:.1g})")
            print(f"| {name} | {' | '.join(cells)} |")
    print(f"\nPer-frame metrics written to {OUT_DIR}/metrics.csv")
