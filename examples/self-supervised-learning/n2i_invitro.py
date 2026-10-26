r"""
In-vitro test for ``train_n2i_deepinv.py``
=========================================

Image-quality metrics on acquisition ``invitro_02162`` of the EPFL dataset, a
CIRS 054GS phantom frame used in the EPFL publications on this dataset. Beamformed
on the training grid by ``precompute_n2i_ultrasound.py``, it shows two cysts in
homogeneous speckle and no wire target:

- an anechoic cyst centred at (x, z) = (10.5, 40.5) mm, radius about 3.3 mm;
- a -6 dB hypoechoic cyst at (-1.25, 39.6) mm, radius about 3.25 mm.

The centres and radii were fitted on the 87-transmit CPWC image. For each cyst,
the inside region is a disk of ``R_IN`` and the background is the annulus
between ``R_BG`` at the same depth, clear of the cyst edge and of the other
cyst. The metrics are computed on the envelope ``|x|`` of the IQ image:

- contrast :math:`20 \log_{10}(\mu_\mathrm{in} / \mu_\mathrm{bg})` (dB);
- CNR :math:`|\mu_\mathrm{in} - \mu_\mathrm{bg}| / \sqrt{\sigma_\mathrm{in}^2 + \sigma_\mathrm{bg}^2}`;
- gCNR :footcite:t:`rodriguez2020generalized`, one minus the overlap of the
  inside and background envelope histograms. It is invariant to monotonic
  transforms, so it is the same on the envelope and on the B-mode;
- speckle SNR :math:`\mu / \sigma` in a homogeneous speckle region (1.91 for
  fully developed speckle, higher when the speckle is smoothed);
- speckle FWHM, the full width at half maximum of the envelope
  autocorrelation, axially and laterally, in the same region: a resolution
  proxy that grows if the network blurs.
"""

import json
import os

import matplotlib

# Figures are only written to files: without this, matplotlib picks an
# interactive backend whenever DISPLAY is set, e.g. under ssh -X, and draws
# every figure through the X tunnel.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import deepinv as dinv

FRAME = "invitro_CIRS054GS/054GS/invitro_02162.npy"  # relative to the cache root

# Cysts: centre (x, z) and radius, in mm.
CYSTS = {
    "anechoic": {"center": (10.5, 40.5), "radius": 3.3},
    "hypoechoic": {"center": (-1.25, 39.6), "radius": 3.25},
}
R_IN = 0.75  # inside disk, as a fraction of the cyst radius
R_BG = (1.4, 1.9)  # background annulus, as fractions of the cyst radius
# Homogeneous speckle region (x_min, x_max, z_min, z_max) in mm.
SPECKLE = (-14.0, 14.0, 10.0, 30.0)


def load_frame(cache_root):
    """Single-angle IQ planes ``(n_angles, Z, X)`` of the frame and its manifest."""
    path = os.path.join(cache_root, FRAME)
    with open(os.path.join(os.path.dirname(path), "manifest.json")) as f:
        manifest = json.load(f)
    return torch.from_numpy(np.load(path)), manifest


def _grid(manifest):
    """Pixel coordinates (x, z) in mm, as ``(Z, X)`` arrays."""
    (n_z, n_x), (dz, dx) = manifest["img_size"], manifest["pixel_size"]
    z = (manifest["z_min"] + np.arange(n_z) * dz) * 1e3
    x = (-manifest["x_half"] + np.arange(n_x) * dx) * 1e3
    zz, xx = np.meshgrid(z, x, indexing="ij")
    return xx, zz


def regions(manifest):
    """Boolean masks: ``{cyst: (inside, background)}`` and the speckle region."""
    xx, zz = _grid(manifest)
    masks = {}
    for name, cyst in CYSTS.items():
        (cx, cz), r = cyst["center"], cyst["radius"]
        d = np.hypot(xx - cx, zz - cz)
        masks[name] = (d < R_IN * r, (d > R_BG[0] * r) & (d < R_BG[1] * r))
    x0, x1, z0, z1 = SPECKLE
    speckle = (xx >= x0) & (xx <= x1) & (zz >= z0) & (zz <= z1)
    return masks, speckle


def gcnr(a, b, bins=256):
    """Generalized CNR: one minus the overlap of the histograms of ``a`` and ``b``."""
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), bins + 1)
    pa, _ = np.histogram(a, edges)
    pb, _ = np.histogram(b, edges)
    return 1 - np.minimum(pa / a.size, pb / b.size).sum()


def _fwhm(profile, step):
    """FWHM of a symmetric peak centred in ``profile``, linearly interpolated."""
    c = len(profile) // 2
    right = profile[c:]
    k = np.argmax(right < 0.5)  # first sample below half maximum
    half = k - 1 + (right[k - 1] - 0.5) / (right[k - 1] - right[k])
    return 2 * half * step


def speckle_fwhm(env, manifest, mask):
    """Axial and lateral FWHM (mm) of the envelope autocorrelation in ``mask``."""
    rows, cols = np.where(mask)
    patch = env[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    patch = patch - patch.mean()
    spec = np.fft.fft2(patch, s=(2 * patch.shape[0], 2 * patch.shape[1]))
    acf = np.fft.fftshift(np.fft.ifft2(np.abs(spec) ** 2).real)
    acf /= acf.max()
    c = np.array(acf.shape) // 2
    dz, dx = (p * 1e3 for p in manifest["pixel_size"])
    return _fwhm(acf[:, c[1]], dz), _fwhm(acf[c[0], :], dx)


def metrics(img, manifest):
    """All metrics of a complex ``(Z, X)`` IQ image, as a flat dict."""
    env = torch.as_tensor(img).abs().double().cpu().numpy()
    masks, speckle = regions(manifest)
    out = {}
    for name, (inside, bg) in masks.items():
        a, b = env[inside], env[bg]
        out[f"{name}_contrast_db"] = 20 * np.log10(a.mean() / b.mean())
        out[f"{name}_cnr"] = abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var())
        out[f"{name}_gcnr"] = gcnr(a, b)
    s = env[speckle]
    out["speckle_snr"] = s.mean() / s.std()
    out["speckle_fwhm_axial_mm"], out["speckle_fwhm_lateral_mm"] = speckle_fwhm(
        env, manifest, speckle
    )
    return out


def print_table(results):
    """Print ``{method: metrics(...)}`` as a table, one row per method."""
    cols = [
        ("anechoic_contrast_db", "An. C (dB)", "{:.1f}"),
        ("anechoic_cnr", "An. CNR", "{:.2f}"),
        ("anechoic_gcnr", "An. gCNR", "{:.3f}"),
        ("hypoechoic_contrast_db", "Hypo C (dB)", "{:.1f}"),
        ("hypoechoic_cnr", "Hypo CNR", "{:.2f}"),
        ("hypoechoic_gcnr", "Hypo gCNR", "{:.3f}"),
        ("speckle_snr", "sSNR", "{:.2f}"),
        ("speckle_fwhm_axial_mm", "FWHM ax (mm)", "{:.3f}"),
        ("speckle_fwhm_lateral_mm", "FWHM lat (mm)", "{:.3f}"),
    ]
    width = max(len(m) for m in results) + 2
    print(" " * width + " ".join(f"{label:>13}" for _, label, _ in cols))
    for method, m in results.items():
        print(f"{method:<{width}}" + " ".join(f"{fmt.format(m[k]):>13}" for k, _, fmt in cols))


def plot(images, manifest, save_fn, dynamic_range_db=50.0):
    """B-mode of each complex image of ``{title: image}``, with the regions drawn.

    Same layout as ``compare_n2i_deepinv.py`` (:func:`deepinv.utils.plot`, one
    column per image); the cyst disks and background annuli are outlined in
    orange, the speckle region in blue.
    """
    extent = [
        -manifest["x_half"] * 1e3,
        manifest["x_half"] * 1e3,
        manifest["z_max"] * 1e3,
        manifest["z_min"] * 1e3,
    ]
    bmode = lambda img: dinv.utils.bmode(
        torch.as_tensor(img)[None, None], amplitude_floor_db=-dynamic_range_db, normalize=False
    )
    fig, axs = dinv.utils.plot(
        {title: bmode(img) for title, img in images.items()},
        rescale_mode="clip",
        vmin=-dynamic_range_db,
        vmax=0.0,
        cmap="gray",
        extent=extent,
        aspect="equal",
        figsize=(3 * len(images), 6),
        suptitle=f"{FRAME}, {dynamic_range_db:.0f} dB",
        show=False,
        return_fig=True,
        return_axs=True,
    )
    for ax in axs[0]:
        for cyst in CYSTS.values():
            for r, style in [(R_IN, "-"), (R_BG[0], "--"), (R_BG[1], "--")]:
                ax.add_patch(
                    plt.Circle(
                        cyst["center"], r * cyst["radius"], fill=False, ec="C1", ls=style, lw=0.5
                    )
                )
        x0, x1, z0, z1 = SPECKLE
        ax.add_patch(plt.Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec="C0", lw=0.5))
    fig.savefig(save_fn, dpi=1200)  # the default of deepinv.utils.plot
    plt.close(fig)
