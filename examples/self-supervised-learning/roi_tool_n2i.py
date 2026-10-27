r"""
Noise2Inverse for ultrafast ultrasound -- in-vivo ROI tool
==========================================================

Hand-drawn regions of interest on in-vivo carotid frames, for the contrast,
CNR, gCNR and speckle-SNR metrics that have no ground truth otherwise.

The script has two modes.

- **Draw** (default): writes ``roi_tool.html``, a single self-contained page
  with the B-mode of ``FRAMES_PER_FOLDER`` frames of each folder of
  ``ROI_DIRS``, evenly spaced through the folder. Copy it to a machine with a
  browser (no server, no X forwarding needed), draw the ROIs on each frame and
  save them as a JSON file. The page shows the 87-transmit CPWC B-mode, and the
  three-transmit fast input on demand, both on the 468 x 234 IQ grid at its
  physical aspect ratio.
- **Check** (``N2I_ROI_JSON=/path/to/rois.json``): reads the saved ROIs,
  rasterises them with :func:`roi_masks` exactly as the metric scripts will,
  writes one overlay PNG per frame and prints the size and mean CPWC
  brightness of every ROI, so that a mis-drawn region shows up before it is
  used.

ROIs are polygons in the pixel coordinates of the IQ grid: ``x`` is the column
(lateral) and ``z`` the row (axial), pixel ``(r, c)`` covering
``[c, c + 1) x [r, r + 1)``; a pixel belongs to a ROI if its centre does. Labels:

- ``lumen``  vessel lumen (should be anechoic);
- ``wall``   vessel wall (intima-media and adventitia);
- ``tissue`` homogeneous soft tissue away from the vessel, for contrast
  against the lumen and for the speckle SNR.
"""

import base64
import glob
import io
import json
import os
from pathlib import Path

import matplotlib

# Figures are only written to files (see infer_n2i_deepinv.py).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

# %%
# Configuration
# -------------
CACHE_ROOT = "/data/n2i_das_cache"  # output of the precompute script
# Folders to draw on: the carotid frames of the test volunteers.
ROI_DIRS = ["volunteer_002/carotid", "volunteer_006/carotid", "volunteer_008/carotid"]
# Frames per folder, evenly spaced from the first to the last; the
# N2I_ROI_FRAMES environment variable overrides it.
FRAMES_PER_FOLDER = int(os.environ.get("N2I_ROI_FRAMES", 3))
LABELS = ["lumen", "wall", "tissue"]
# Steering angles of the fast acquisition, in degrees, as in eval_n2i_deepinv.py.
ANGLES_DEG = [-10.0, 0.0, 10.0]
DYNAMIC_RANGE_DB = 50.0
OUT_DIR = "./results/n2i_ultrasound_rois"
# Saved ROIs to check; unset = draw mode.
ROI_JSON = os.environ.get("N2I_ROI_JSON")


# %%
# ROI masks (also imported by the metric scripts)
# -----------------------------------------------
def roi_masks(roi_json):
    """Boolean masks of the ROIs saved by the page.

    :param str roi_json: path of the JSON file saved by ``roi_tool.html``.
    :return: dict ``{"<folder>/<frame stem>": {label: (Z, X) bool array}}``,
        the union of all the ROIs of a label on that frame; labels without any
        ROI are missing.
    """
    with open(roi_json) as f:
        saved = json.load(f)
    masks = {}
    for key, frame in saved["frames"].items():
        n_z, n_x = frame["img_size"]
        rows, cols = np.mgrid[:n_z, :n_x]
        centres = np.stack([cols.ravel() + 0.5, rows.ravel() + 0.5], axis=1)
        for roi in frame["rois"]:
            if len(roi["points"]) < 3:
                continue
            inside = MplPath(np.asarray(roi["points"], dtype=float)).contains_points(centres)
            label_masks = masks.setdefault(key, {})
            label_masks[roi["label"]] = label_masks.get(roi["label"], False) | inside.reshape(n_z, n_x)
    return masks


def bmode_db(img):
    """B-mode in dB relative to the image maximum, clipped to the display window."""
    amplitude = np.abs(img)
    db = 20 * np.log10(np.maximum(amplitude / amplitude.max(), 1e-12))
    return np.clip(db, -DYNAMIC_RANGE_DB, 0.0)


def png_base64(db):
    """Grey-level PNG of a B-mode (-DYNAMIC_RANGE_DB -> black, 0 dB -> white)."""
    buf = io.BytesIO()
    plt.imsave(buf, db, cmap="gray", vmin=-DYNAMIC_RANGE_DB, vmax=0.0, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def frame_files(folder):
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, folder, "invi*_*.npy")))
    idx = np.unique(np.linspace(0, len(files) - 1, FRAMES_PER_FOLDER).round().astype(int))
    return [files[i] for i in idx]


def fast_transmits(manifest):
    angles_deg = np.asarray(manifest["angles_deg"])
    return [int(np.abs(angles_deg - a).argmin()) for a in ANGLES_DEG]


def load_manifest(folder):
    with open(os.path.join(CACHE_ROOT, folder, "manifest.json")) as f:
        return json.load(f)


# %%
# Check mode
# ----------
def check(roi_json):
    """Overlay PNGs and per-ROI statistics of the saved ROIs."""
    masks = roi_masks(roi_json)
    with open(roi_json) as f:
        saved = json.load(f)
    colors = {"lumen": "tab:red", "wall": "tab:orange", "tissue": "tab:cyan"}
    for key, frame in saved["frames"].items():
        folder, stem = key.rsplit("/", 1)
        planes = np.load(os.path.join(CACHE_ROOT, folder, stem + ".npy"), mmap_mode="r")
        db = bmode_db(np.asarray(planes).mean(0))
        n_z, n_x = frame["img_size"]
        present = masks.get(key, {})
        missing = [label for label in LABELS if label not in present]
        print(f"{key}: {len(frame['rois'])} ROIs" + (f", no {', '.join(missing)}" if missing else ""))
        for label, mask in present.items():
            print(f"  {label:7s} {mask.sum():6d} px, mean CPWC B-mode {db[mask].mean():6.1f} dB")

        fig, ax = plt.subplots(figsize=(4, 7))
        ax.imshow(db, cmap="gray", vmin=-DYNAMIC_RANGE_DB, vmax=0.0, extent=[0, n_x, n_z, 0],
                  aspect=frame["pixel_size"][0] / frame["pixel_size"][1])
        for label, mask in present.items():
            ax.contour(np.arange(n_x) + 0.5, np.arange(n_z) + 0.5, mask, levels=[0.5],
                       colors=colors.get(label, "white"), linewidths=1)
        ax.set_title(f"{key}\n" + ", ".join(f"{l}: {colors.get(l)[4:]}" for l in present), fontsize=8)
        ax.set_axis_off()
        out = os.path.join(OUT_DIR, "check", key.replace("/", "_") + ".png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Overlays written to {os.path.join(OUT_DIR, 'check')}")


# %%
# Draw mode: build the page
# -------------------------
def build_page():
    """Write the self-contained drawing page."""
    frames = []
    for folder in ROI_DIRS:
        manifest = load_manifest(folder)
        tx = fast_transmits(manifest)
        for path in frame_files(folder):
            planes = np.load(path, mmap_mode="r")
            stem = Path(path).stem
            frames.append({
                "key": f"{folder}/{stem}",
                "img_size": manifest["img_size"],
                "pixel_size": manifest["pixel_size"],
                "z_min": manifest["z_min"],
                "x_half": manifest["x_half"],
                "cpwc": png_base64(bmode_db(np.asarray(planes).mean(0))),
                "fast": png_base64(bmode_db(np.asarray(planes[tx]).mean(0))),
            })
            print(f"{folder}/{stem}")

    page = Path(__file__).with_name("roi_tool_template.html").read_text()
    page = page.replace("/*__FRAMES__*/[]", json.dumps(frames)).replace(
        "/*__CONFIG__*/{}", json.dumps({"labels": LABELS, "dynamic_range_db": DYNAMIC_RANGE_DB})
    )
    out = os.path.join(OUT_DIR, "roi_tool.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"{len(frames)} frames -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    check(ROI_JSON) if ROI_JSON else build_page()
