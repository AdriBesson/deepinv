# Self-Supervised Denoising of Few-Transmit Plane-Wave Ultrasound with Noise2Inverse over Transmit Partitions

*Draft manuscript, IEEE TUFFC / IEEE TMI format. Status: 2026-10-26. Items marked **[TODO]** must be resolved before submission.*

> **About this version.** It presents only the third training round (checkpoints `n2i_ultrasound_<arch>_test002`), the first with a test volunteer that is used neither for training nor for checkpoint selection, and with the whole 054GS phantom section held out. Rounds 1–2 had no independent test set and are no longer reported; their logs remain in `results/eval_round2.log` and `results/compare_round2.log`. All five networks finished round 3, and Tables I–III and Figs. 1–2 cover all of them (2026-10-26). Section V-D and the Discussion add an analysis of the dark artifacts on the phantom (2026-10-12). A fourth round (U-Net and DRUNet), with a larger test set and two changes aimed at these artifacts, is trained; its phantom results are in Section V-E (2026-10-14): the artifacts remain and follow the image orientation. Section V-F gives the final results (2026-10-23) of a multi-transmit variant, which sees the three transmits as separate channels, with and without vertical-flip augmentation; Section V-G those of a network trained for any number of transmits, with and without vertical flips. Table XI gives the in-vivo metrics of Table I on all test frames of volunteers 002, 006 and 008 for the round-4 networks and the multi-transmit DRUNets (2026-10-23, round-4 U-Net added 2026-10-26): the multi-transmit DRUNet without vertical flips is the best network of the study. Table XII scores the any-n networks on all test frames (2026-10-26); only the flip tests (Tables VII and X) remain on six frames per folder.

---

## Abstract

Ultrafast plane-wave imaging trades image quality for frame rate: coherent compounding of many steered transmits restores contrast and signal-to-noise ratio (SNR), but lowers the frame rate by the same factor. Supervised deep-learning methods that map few-transmit images to compounded ones need paired high-quality targets. Those targets are themselves compounded images, so the networks learn their residual noise and clutter, and they cannot be acquired when the tissue moves. We propose a self-supervised alternative based on Noise2Inverse (N2I). The 87 transmits of an in-vivo acquisition are partitioned into *K* = 29 interleaved sub-sequences of three plane waves, each spanning the full angular aperture. A network is trained to map the delay-and-sum (DAS) image of one sub-sequence to the mean of the DAS images of the other sub-sequences. Because the noise of disjoint transmits is independent while the insonified anatomy is shared, the minimizer of this loss is the conditional expectation of the compounded image given the three-transmit input, and no clean target is ever needed. Beamforming is linear in the transmits, so we cache the beamformed IQ images once, and training costs no more than supervised image-to-image training. We train encoder–decoder networks on 3805 frames (3694 in vivo from five volunteers, 111 phantom frames) and test them on a sixth volunteer that plays no part in training or checkpoint selection, against the independent reconstruction from the 84 transmits not used as input. From three transmits, five networks (U-Net, ResUNet, AttResUNet, DRUNet and MS-D) raise the IQ PSNR by 1.5–1.9 dB and the B-mode SSIM from 0.38 to 0.60–0.61 on carotid frames, and the PSNR by 3.7–4.0 dB on abdominal frames, the encoder–decoders at 7–16 ms per frame on a consumer GPU. The four encoder–decoders are within 0.3 dB of each other; DRUNet has the best PSNR and beats U-Net on every in-vivo metric, and MS-D, the network of the original N2I paper, is the weakest and the slowest. On a CIRS 054GS phantom section held out of training, four of the networks deepen the anechoic-cyst contrast from −23 dB (three transmits) to between −27 and −31 dB (CPWC: −37 dB), and U-Net, DRUNet and MS-D preserve the speckle statistics. On eight held-out phantom frames, however, every network darkens parts of the homogeneous phantom at depth, by up to 14–22 dB, DRUNet the least (at most 9 dB). The in-vivo metrics do not detect this. Rescaling the input shows that the networks key on absolute brightness: they have learned that deep, weak signal is noise, which is correct in vivo but not on the phantom. Making the networks exactly scale-equivariant and augmenting the training data with random depth-gain curves does not remove the artifacts: they then depend on the vertical orientation of the image, and vanish when the input is flipped upside down. Feeding the three transmits as separate channels, so that the network can see whether they agree, halves the dark area on the phantom (from 5.5% to 2.5%) and is the best network in vivo; adding vertical flips to the training data makes it insensitive to orientation for 0.2–0.4 dB of in-vivo NMSE, but the last dark patches, deep at the lateral edges, remain. A single DRUNet trained for any number of transmits (1–80) beats the three-transmit network in vivo on one transmit (by 0.9–1.9 dB) and on 11 (by 0.5–1.1 dB), but darkens 18–43% of the phantom from one to three transmits, with or without vertical flips. Against plain compounding scored on the same reference, three transmits through a network match the PSNR of 5–6.6 compounded transmits and, on carotid frames, the SSIM of 9–11.

**Index Terms** — Ultrafast ultrasound, plane-wave imaging, coherent compounding, self-supervised learning, Noise2Inverse, denoising, deep learning.

---

## I. Introduction

Plane-wave (PW) imaging insonifies the whole field of view with each transmit, so frame rates reach several kilohertz [1], [2]. A single unfocused transmit, however, yields low contrast and SNR. Coherent plane-wave compounding (CPWC) [1] averages the beamformed images of many steered transmits. This restores image quality, but the frame rate falls in proportion to the number of transmits. Applications that need the full frame rate, such as shear-wave elastography, ultrafast Doppler and functional ultrasound, therefore work with few-transmit images or trade temporal resolution against image quality.

Deep networks have been trained to predict a compounded image from a few transmits [3], [4]. These methods are supervised: the target is a CPWC image of the same scene. This has two drawbacks. (i) The target still contains residual noise, clutter and motion blur, and the network learns to reproduce them. (ii) Targets can only be formed where the scene is static over the whole compounding sequence, which excludes the moving tissue in which fast imaging matters most.

Self-supervised denoising removes the need for clean targets. Noise2Noise [5] showed that a network trained to map one noisy realization to another, independent realization of the same signal converges to the same estimator as one trained on clean targets. Noise2Inverse (N2I) [6] carried this to tomography. There, independent realizations are not acquired twice; they are created by splitting the *measurements* (the projection angles) into disjoint subsets and reconstructing each subset separately.

PW ultrasound has the same structure as tomography. Each transmit is a separate measurement, the DAS image is linear in the transmits, and the noise of different transmits is independent. We exploit this structure and make the following contributions:

1. We formulate Noise2Inverse for plane-wave ultrasound over a partition of the transmit sequence into interleaved sub-sequences. We show that the resulting estimator targets the expected compounded image rather than the clean reflectivity (Section II).
2. We describe an implementation that beamforms and demodulates every transmit once and caches the IQ images. Any sub-reconstruction is then read from the cache, so training costs no more than an image-to-image network (Section III).
3. We compare encoder–decoder architectures on in-vivo data with a volunteer-level split and an independent test volunteer, using an evaluation protocol that needs no ground truth and ranks the methods without bias (Sections IV–V).
4. We test the same networks on a tissue-mimicking phantom with standard ultrasound image-quality metrics (contrast, CNR, gCNR, speckle SNR and speckle size), which exposes an artifact that the in-vivo metrics do not reveal (Sections V-C, VI).

---

## II. Method

### A. Plane-wave imaging model

A PW sequence consists of *N* = 87 transmits steered at angles $\theta_a$, $a = 1, \dots, N$. Let $y_a$ denote the channel data of transmit $a$ and $A_a$ the linear operator that maps the reflectivity to $y_a$. The DAS image of transmit $a$ is the adjoint $A_a^\top y_a$. It is demodulated to complex baseband (IQ), which is also a linear operation. The IQ image obtained from a subset of transmits $g$ is therefore the mean of its single-angle images:

$$
x_g = \frac{1}{|g|}\sum_{a\in g} A_a^\top y_a ,
\qquad
x_{\mathrm{CPWC}} = x_{\{1,\dots,N\}} .
$$

### B. Noise2Inverse over transmit partitions

We partition the transmits into *K* disjoint groups $g_1,\dots,g_K$ and write $x_j := x_{g_j}$. We use the interleaved partition $g_j = \{j, j+K, j+2K, \dots\}$, with *K* = 29 groups of three transmits. The sequence alternates signs from the widest angle inwards, so every group samples the full angular aperture (for example, group 1 = {−16.3°, +11.0°, −5.3°}). Groups therefore differ mainly in noise rather than in the width of their point-spread function.

Each group image is written as $x_j = s_j + n_j$. Here $s_j = \mathbb{E}[x_j \mid \text{anatomy}]$ is the deterministic image of the scene through the three-transmit PSF of group $j$, and $n_j$ is noise. The noise includes electronic noise and any contribution that is not coherent across transmits. It is independent across disjoint groups and zero-mean. At each training step, an input group $j$ is drawn uniformly, and the network $f_\psi$ is trained to predict the mean of the other groups, $\bar{x}_{\setminus j} = \frac{1}{K-1}\sum_{k \neq j} x_k$:

$$
\mathcal{L}(\psi) = \mathbb{E}_{\text{frames}}\,\mathbb{E}_{j}\, \big\| f_\psi(x_j) - \bar{x}_{\setminus j} \big\|_2^2 .
$$

Because the noise of the target is independent of $x_j$, the minimizer is

$$
f^\star(x_j) = \mathbb{E}\!\left[ \tfrac{1}{K-1}\textstyle\sum_{k\neq j} s_k \;\middle|\; x_j \right],
$$

that is, the MMSE estimate of the *compound of the other groups' noiseless images* given the input. Mapping $x_j$ to a single other group $x_k$ has the same minimizer, but the noise of the mean of the other groups is $K-1$ times smaller, so its gradients are less noisy; this is the "X:1" strategy of Noise2Inverse [6]. Two consequences follow.

- **The target is a compound image, not the reflectivity.** Every group spans the full aperture, so the average of $s_k$ over $k$ is, up to the missing group, the noiseless CPWC image. Components of the three-transmit PSF that change from group to group, such as grating lobes and side-lobe clutter tied to specific angles, average out and are removed. Components common to all groups are kept.
- **This is the same estimator as supervised training on noiseless CPWC targets.** As in Noise2Noise [5], the loss differs from the supervised one only by the variance of the target noise, a constant. The approach needs neither a static scene over 87 transmits nor any clean data.

The splitting is done on the transmits, so this is Noise2Inverse [6] rather than pixel-level blind-spot self-supervision [7]. Speckle, which is spatially correlated noise, therefore raises no difficulty: speckle that is coherent across transmits is part of $s_j$ and is kept, while its incoherent part is treated as noise.

### C. Inference

At inference time, a *fast acquisition* of three transmits is beamformed and passed through $f_\psi$ once. No compounding sequence is needed.

---

## III. Implementation

### A. Data

We use the EPFL ultrafast ultrasound dataset [8]. It was acquired with a GE 9L-D linear array (192 elements, 0.23 mm pitch, 5.3 MHz center frequency) on a Verasonics scanner. Each frame holds 87 plane waves steered from −16.3° to +16.3° in 0.38° steps, sampled at 20.8 MHz. The in-vivo part has 20,140 acquisitions of nine volunteers (from 492 for volunteer 005 to 9217 for volunteer 001), and the in-vitro part has about 2180 acquisitions of a CIRS 054GS phantom, in two sections, "054GS" (cysts) and "FullSpeckle". Both parts use the same probe and sequence, so the phantom is beamformed on the in-vivo grid and the networks apply to it without change.

The dataset is distributed as one 60–330 GB zip archive per volunteer. We wrote a downloader that reads the archive's central directory with HTTP range requests and fetches only the selected acquisitions (`download_epfl_ultrasound.py`). It skips frames that are already downloaded or already in the beamformed cache. The cached frames are split as follows:

| Volunteer | Region | Frames | Use |
|---|---|---:|---|
| 001 | abdomen | 1008 | train |
| 003 | abdomen / breast | 35 / 679 | train |
| 004 | breast / carotid | 513 / 594 | train |
| 005 | abdomen | 353 | train |
| 007 | abdomen / breast | 449 / 63 | train |
| phantom | FullSpeckle | 111 | train |
| 005 | carotid | 100 | validation |
| 006 | carotid | 289 | validation |
| 002 | abdomen / carotid | 298 / 161 | **test** |
| phantom | 054GS | 568 | held out; `invitro_02162` is the in-vitro test frame |
| | **training frames** | **3805** | 3694 in vivo + 111 in vitro |

Frames of one acquisition are strongly correlated, so the split is made by volunteer and region. The test volunteer 002 is used neither for training nor for checkpoint selection. The phantom test frame `invitro_02162` is the frame used in the EPFL publications on this dataset; the whole 054GS section it belongs to is held out of training, since its other frames likely image the same phantom position.

### B. Beamforming and IQ cache

Each transmit is beamformed separately with the plane-wave DAS operator of the DeepInverse library [9] (`UltrasoundPlaneWave`). We use a *f*-number of 1.5 with Hann receive apodization, a Gaussian-modulated pulse model with 53% fractional bandwidth, and a speed of sound of 1540 m/s. The field of view spans 2–70 mm in depth and ±18 mm laterally. The RF image is sampled at λ/8 axially and at 2/3 of the pitch laterally.

Each RF image is demodulated in the axial Fourier domain. The spectrum is shifted by the round-trip wavenumber $2/\lambda$, low-pass filtered to its central quarter and decimated four times, which gives a 468 × 234 complex IQ image at λ/2 axial sampling. The out-of-band energy of the pulse is below −27 dB. Each frame is scaled so that its CPWC image has unit RMS envelope. The 87 planes of a frame are stored as one complex64 array (76 MB). Beamforming one transmit takes about 32 ms, and this cache is built once.

The cache sits on a hard disk, where averaging the three planes of two groups means reading six scattered parts of a 76 MB file (about 0.4 s per frame), which would leave the GPU waiting. Each frame is therefore condensed once more into its 29 group images and the mean of all 87 transmits (float32, 26 MB per frame). A training sample is then two reads from one file, and the target of Section II-B follows from the stored mean. The loss needs no forward or adjoint operator, so one training step costs the same as one step of image-to-image training; an epoch takes 3–5 min.

### C. Networks

The network takes the I and Q components as two real channels and returns two channels. We use it in direct mode: it predicts the image, not a residual. The encoder–decoder networks have 32 channels at the finest scale, doubling at each coarser scale:

| Network | Scales | Blocks | Skips | Norm. | Params |
|---|---|---|---|---|---:|
| U-Net [10] | 4 | double conv | concat | none | 2.14 M |
| ResUNet | 5 | residual (Conv-BN-ReLU-Conv-BN + 1×1 shortcut) | concat | BN | 8.99 M |
| AttResUNet [11] | 5 | residual + attention gates on skips | concat | BN | 9.08 M |
| DRUNet [12] | 4 | 2 residual blocks per scale, strided/transposed conv | additive | none | 4.25 M |
| MS-D [15] | 1 | 100 dilated 3×3 conv layers of width 1, dilations 1–10 | dense | none | 0.047 M |

ResUNet and AttResUNet upsample with bilinear interpolation followed by a convolution. DRUNet receives its usual noise-level map as an extra constant channel. MS-D, the mixed-scale dense network [15], is the network of the original Noise2Inverse paper [6]. It has no down-sampling: each of its layers convolves the input and all earlier layers at full resolution, and a final 1 × 1 convolution combines them. Each layer recomputes its input during the backward pass (gradient checkpointing), so memory grows linearly rather than quadratically with depth.

Inside the model, frames are reflect-padded to 480 × 256 (multiples of 32) and the output is cropped back, so the loss only covers the 468 × 234 frame.

### D. Training

We use Adam with a learning rate of $10^{-3}$, cosine annealing over 100 epochs, gradient clipping at 1, batches of 8 whole frames, float32 arithmetic, and lateral flips as the only augmentation. The flip is physically valid for a symmetric array and angle set. The validation loss is the N2I loss between fixed group pairs (1→2, 3→4, 5→6, 7→8), averaged over the 389 validation frames, on the 468 × 234 frame. It selects the best checkpoint and drives early stopping with a patience of 30 epochs. Interrupted runs are resumed with their optimizer, scheduler and loss histories, so the cosine schedule and the early-stopping counter continue where they stopped. All code is built on DeepInverse [9] and runs on a single 8 GB RTX 2070.

---

## IV. Evaluation Protocol

### A. In-vivo metrics

No clean in-vivo image exists. Using the full CPWC as the reference would also be biased, because it contains the input transmits and hence the input noise, which favors networks that pass the input through unchanged. We instead score each output $\hat{x}$ against the **held-out reconstruction** $x_{\bar g}$, the mean of the $N - |g|$ transmits that are *not* in the input. Its noise is independent of the input, so

$$
\mathbb{E}\|\hat{x} - x_{\bar g}\|^2 = \mathbb{E}\|\hat{x} - s_{\bar g}\|^2 + \mathbb{E}\|n_{\bar g}\|^2 .
$$

The second term is the same for every method. The ranking is therefore unbiased, but absolute PSNR values underestimate the true ones. We report, on the test volunteer 002, separately for its carotid (161 frames) and abdominal (298 frames) acquisitions:

- **PSNR (IQ)**: $10\log_{10}\big(\max|x_{\bar g}|^2 / \mathrm{MSE}(\hat{x}, x_{\bar g})\big)$ on the complex image;
- **SSIM (B-mode)**: SSIM between the log-compressed envelopes, each normalized to its maximum and clipped to a 50 dB window mapped to [0, 1];
- **N2I loss**: the MSE on the I/Q channels between the output for group 1 and the image of group 2 (lower is better);
- **Runtime**: median GPU time per 468 × 234 frame, batch 1, fp32, after warm-up.

We test two inputs:

- *Interleaved*: group 1, the training distribution.
- *Fast*: the three transmits nearest to {−10°, 0°, +10°}, a realistic fast sequence that does not match any training group.

Differences between networks are tested with a paired Wilcoxon signed-rank test over the frames of each region.

### B. In-vitro metrics

The phantom frame `invitro_02162` gives a known geometry. It shows two cysts in homogeneous speckle: an anechoic cyst centred at (x, z) = (10.5, 40.5) mm with a radius of about 3.3 mm, and a −6 dB hypoechoic cyst at (−1.25, 39.6) mm with a radius of about 3.25 mm. The centres and radii were fitted on the CPWC image. The frame has no wire target. For each cyst, the inside region is a disk of 0.75 radius, and the background is the annulus from 1.4 to 1.9 radii, at the same depth and clear of the other cyst. On the envelope $|x|$ of the IQ image, we compute:

- **Contrast**: $20\log_{10}(\mu_\mathrm{in}/\mu_\mathrm{bg})$, in dB;
- **CNR**: $|\mu_\mathrm{in}-\mu_\mathrm{bg}| / \sqrt{\sigma_\mathrm{in}^2+\sigma_\mathrm{bg}^2}$;
- **gCNR** [14]: one minus the overlap of the inside and background histograms. It is invariant to monotonic transforms, so it is the same on the envelope and on the B-mode;
- **Speckle SNR**: $\mu/\sigma$ in a homogeneous region (x ∈ [−14, 14] mm, z ∈ [10, 30] mm). It is 1.91 for fully developed speckle and larger when the speckle is smoothed;
- **Speckle FWHM**: the axial and lateral full width at half maximum of the envelope autocorrelation in the same region. This is a resolution proxy that grows if the network blurs.

The input is the same *fast* sequence {−10°, 0°, +10°} as in vivo.

---

## V. Results

### A. Training

| Network | Run | Epochs run | Best epoch | Val. loss |
|---|---|---:|---:|---:|
| U-Net | `n2i_ultrasound_unet_test002/26-10-07-21:10:34` | 100 | 79 | 0.1698 |
| ResUNet | `n2i_ultrasound_resunet_test002/26-10-08-19:39:52` | 100 | 85 | 0.1700 |
| AttResUNet | `n2i_ultrasound_attresunet_test002/26-10-08-09:33:54` | 100 | 75 | 0.1698 |
| DRUNet | `n2i_ultrasound_drunet_test002/26-10-07-19:13:15` | 100 | 85 | **0.1678** |
| MS-D | `n2i_ultrasound_msd_test002/26-10-09-09:28:31` | 71 (early stop) | 41 | 0.1779 |

- **No overfitting.** The validation losses of the four encoder–decoders kept decreasing until the cosine schedule brought the learning rate to zero, and their best checkpoints are late (epochs 75–85). Longer training could still help slightly.
- **The encoder–decoders end up equal on the validation loss** (0.1678–0.1700). ResUNet and AttResUNet reach a lower training loss (0.071, against 0.075 for DRUNet and 0.077 for U-Net) without a better validation loss.
- **MS-D is clearly worse and much slower.** Its validation loss plateaued from about epoch 22 at 0.178–0.183, 5–6% above the others, and early stopping ended the run at epoch 70 with the learning rate still at 1.9·10⁻⁴. With 47 k parameters, an epoch took about 25 min, against 3.5–4 min for ResUNet and AttResUNet on the same GPU.
- **DRUNet is unstable early.** Its training loss spiked to 4.3·10¹⁰ at epoch 2 (validation loss 0.349) and recovered within two epochs, in float32. **[TODO]** Investigate (learning-rate warm-up, lower learning rate).

### B. In-vivo results

**Table I.** Mean ± standard deviation over the frames of the test volunteer 002, against the held-out reconstruction (84 transmits). The DAS row is the unprocessed three-transmit input. Time: median per frame, batch 1, fp32, RTX 2070, all five networks in the same run (2026-10-26); medians vary by up to 20% between runs (the U-Net and DRUNet of the 2026-10-08 run took 8.6 and 12.9 ms, the DRUNets of Table XI 13.0 ms).

| Region | Method | PSNR, fast (dB) | SSIM, fast | PSNR, interleaved (dB) | SSIM, interleaved | N2I loss | Time (ms) |
|---|---|---|---|---|---|---:|---:|
| Carotid (161) | DAS, 3 tx | 35.48 ± 2.30 | 0.382 ± 0.077 | 33.20 ± 2.20 | 0.322 ± 0.073 | 0.2661 | — |
| | U-Net | 37.21 ± 2.16 | 0.599 ± 0.112 | 34.62 ± 2.19 | 0.571 ± 0.116 | 0.1906 | **7.3** |
| | ResUNet | 37.32 ± 2.23 | 0.607 ± 0.111 | 34.83 ± 2.25 | **0.580 ± 0.114** | 0.1966 | 13.2 |
| | AttResUNet | 37.07 ± 2.21 | **0.611 ± 0.110** | 34.54 ± 2.28 | **0.580 ± 0.114** | 0.1925 | 16.2 |
| | DRUNet | **37.33 ± 2.18** | 0.607 ± 0.109 | **34.89 ± 2.24** | 0.576 ± 0.113 | **0.1889** | 10.6 |
| | MS-D | 37.00 ± 2.19 | 0.595 ± 0.112 | 34.32 ± 2.16 | 0.564 ± 0.115 | 0.1962 | 77.4 |
| Abdomen (298) | DAS, 3 tx | 27.40 ± 3.12 | 0.290 ± 0.052 | 26.37 ± 2.87 | 0.246 ± 0.044 | 1.0194 | — |
| | U-Net | 31.41 ± 2.46 | 0.301 ± 0.063 | 30.00 ± 2.19 | 0.270 ± 0.063 | 0.6199 | **7.3** |
| | ResUNet | 31.32 ± 2.50 | 0.298 ± 0.065 | 30.06 ± 2.26 | 0.270 ± 0.066 | 0.6174 | 13.1 |
| | AttResUNet | 31.29 ± 2.50 | **0.309 ± 0.067** | 29.90 ± 2.23 | **0.277 ± 0.068** | **0.6084** | 16.1 |
| | DRUNet | **31.44 ± 2.48** | 0.307 ± 0.063 | **30.11 ± 2.22** | 0.275 ± 0.062 | 0.6175 | 10.5 |
| | MS-D | 31.07 ± 2.46 | 0.284 ± 0.062 | 29.68 ± 2.15 | 0.258 ± 0.059 | 0.6189 | 76.7 |

**Table II.** Paired differences from U-Net on volunteer 002: mean difference, fraction of frames on which the network beats U-Net (lower is better for the N2I loss), and Wilcoxon *p*-value.

| Region | Network | ΔPSNR fast | ΔSSIM fast | ΔPSNR interleaved | ΔSSIM interleaved | ΔN2I loss |
|---|---|---|---|---|---|---|
| Carotid | ResUNet | +0.11 dB (71%, p=2e-8) | +0.008 (73%, p=3e-12) | +0.21 dB (78%, p=2e-8) | +0.009 (73%, p=4e-15) | +0.0059 (22%, p=2e-13) |
| | AttResUNet | −0.14 dB (55%, p=0.9) | +0.011 (80%, p=1e-16) | −0.08 dB (54%, p=1) | +0.009 (76%, p=4e-14) | +0.0019 (42%, p=0.04) |
| | DRUNet | +0.12 dB (69%, p=3e-8) | +0.008 (83%, p=7e-19) | +0.27 dB (89%, p=2e-24) | +0.005 (73%, p=7e-11) | −0.0017 (65%, p=2e-7) |
| | MS-D | −0.21 dB (22%, p=5e-14) | −0.004 (26%, p=5e-9) | −0.30 dB (12%, p=4e-23) | −0.007 (22%, p=3e-14) | +0.0056 (20%, p=3e-15) |
| Abdomen | ResUNet | −0.09 dB (32%, p=3e-12) | −0.004 (36%, p=4e-10) | +0.06 dB (65%, p=2e-7) | +0.000 (50%, p=0.7) | −0.0024 (50%, p=0.3) |
| | AttResUNet | −0.12 dB (35%, p=4e-13) | +0.008 (73%, p=5e-22) | −0.10 dB (42%, p=3e-6) | +0.008 (71%, p=8e-21) | −0.0114 (80%, p=2e-32) |
| | DRUNet | +0.02 dB (57%, p=0.002) | +0.006 (81%, p=4e-32) | +0.11 dB (83%, p=2e-33) | +0.006 (85%, p=1e-33) | −0.0023 (59%, p=7e-4) |
| | MS-D | −0.34 dB (3%, p=1e-49) | −0.017 (3%, p=1e-48) | −0.32 dB (3%, p=2e-49) | −0.011 (11%, p=2e-40) | −0.0010 (43%, p=0.5) |

**Fig. 1.** B-mode images (50 dB) of the first test frame, carotid of volunteer 002: three-transmit DAS, the five networks and 87-transmit CPWC. Frames 10443–10446 are in the same folder.

![Fig. 1](results/n2i_ultrasound_comparison_test002/volunteer_002/carotid/invivo_10442_comparison.png)

Main observations:

- **The networks denoise substantially on an unseen volunteer.** On carotid frames, the PSNR rises by 1.5–1.9 dB on fast inputs and 1.1–1.7 dB on interleaved ones, the B-mode SSIM from 0.38 to 0.60–0.61 on fast inputs and from 0.32 to 0.56–0.58 on interleaved ones, and the N2I loss drops by 26–29%.
- **Abdomen: large PSNR gain, small SSIM gain.** The abdominal frames are much noisier (DAS N2I loss 1.02, against 0.27 for carotid). The networks gain 3.7–4.0 dB of PSNR and cut the N2I loss by 39–40%, but the B-mode SSIM moves only from 0.29 to 0.28–0.31. Most of the abdominal image is deep, low-SNR tissue, which the networks render dark (see below), so the gain in IQ error does not carry over to the B-mode structure.
- **The four encoder–decoders are within 0.3 dB of each other, and each wins somewhere.** DRUNet has the best PSNR on both regions and both inputs, and beats U-Net on every metric (57–89% of the frames, p ≤ 0.002). AttResUNet has the best SSIM (+0.008 to +0.011 over U-Net, on 71–80% of the frames) and the lowest abdominal N2I loss, but a lower PSNR than U-Net on three of the four columns. ResUNet is better than U-Net on carotid frames (+0.11 and +0.21 dB) and slightly worse on abdominal fast inputs (−0.09 dB). All these margins are much smaller than the gain over DAS and than the frame-to-frame spread (±2–3 dB).
- **MS-D is the weakest network.** It is below U-Net on every metric of both regions except the N2I loss, by 0.2–0.34 dB of PSNR, on 78–97% of the frames, in line with its higher validation loss (Section V-A).
- **The ranking holds on the other test volunteers.** On volunteers 008 (never seen in round 3) and 006 (round-3 validation set), MS-D is again last in every column, AttResUNet has the best SSIM in nine of ten columns, and DRUNet and ResUNet share the best PSNR within 0.08 dB (log `results/eval_test002_all.log`).
- **The networks generalize to non-interleaved sequences.** The *fast* input {−9.9°, 0°, +9.9°} was never a training group, yet the gains remain. Its absolute PSNR is higher than that of the interleaved group, whose widest angle, −16.3°, has lower SNR.
- **Runtime.** U-Net runs at 7.3 ms, DRUNet at 10.6 ms, ResUNet at 13 ms and AttResUNet at 16 ms per frame, i.e. 60–135 frames/s on an RTX 2070. This is cheaper than the extra 84 transmits of beamforming that CPWC would require. MS-D, with 100 sequential dilated layers, takes 77 ms (13 frames/s) despite its 47 k parameters.
- **The deep region is rendered dark.** In Fig. 1, all five networks render roughly the lower half of the image black, where CPWC still shows faint tissue (Section VI).

### C. In-vitro results

**Table III.** Phantom frame `invitro_02162` (CIRS 054GS section, held out of training), fast input {−10°, 0°, +10°}. Contrast in dB (more negative is better for both cysts), speckle FWHM in mm.

| Method | Anechoic C | Anechoic CNR | Anechoic gCNR | Hypo. C | Hypo. CNR | Hypo. gCNR | Speckle SNR | FWHM ax. | FWHM lat. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DAS, 3 tx | −23.1 | 1.62 | 0.969 | −6.0 | 0.84 | 0.490 | 1.82 | 0.438 | 0.453 |
| U-Net† | **−30.6** | 1.57 | 0.986 | −10.1 | 0.95 | 0.558 | 1.81 | 0.441 | 0.467 |
| ResUNet | −23.5 | 1.53 | 0.965 | −6.5 | 0.84 | 0.497 | 1.76 | 0.461 | 0.469 |
| AttResUNet | −28.1 | 1.65 | **0.988** | −6.3 | 0.84 | 0.506 | 1.79 | 0.452 | **0.435** |
| DRUNet | −27.0 | **1.68** | 0.985 | −6.1 | 0.85 | 0.507 | 1.82 | 0.442 | 0.446 |
| MS-D† | −27.6 | 1.61 | 0.983 | −7.3 | 0.87 | 0.525 | 1.82 | 0.443 | 0.476 |
| *CPWC, 87 tx* | *−37.2* | *1.74* | *0.998* | *−5.8* | *0.82* | *0.476* | *1.82* | *0.440* | *0.436* |

† The U-Net and MS-D hypoechoic values reflect dark artifacts below the cyst, not better imaging (Fig. 2).

**Fig. 2.** B-mode images (50 dB) of `invitro_02162`: three-transmit DAS, the five networks and CPWC. The orange circles show the cyst disks and background annuli; the blue rectangle shows the speckle region.

![Fig. 2](results/n2i_ultrasound_comparison_test002/invitro_CIRS054GS/054GS/invitro_02162_comparison.png)

Main observations:

- **Anechoic cyst: four networks recover part of the compounding gain.** Its contrast goes from −23.1 dB (DAS) to −27.0 dB (DRUNet), −27.6 dB (MS-D), −28.1 dB (AttResUNet) and −30.6 dB (U-Net), against −37.2 dB for CPWC, and its gCNR from 0.969 to 0.983–0.988 (CPWC 0.998). The interior of the cyst is filled with off-axis clutter and noise in the three-transmit image; this clutter changes from group to group, so N2I removes it. ResUNet is the exception: it leaves the cyst as filled as the input (−23.5 dB, gCNR 0.965, below DAS).
- **Hypoechoic cyst: no gain, but none is available.** Compounding itself does not improve it (DAS gCNR 0.490, CPWC 0.476): its detectability is limited by speckle, which is coherent across transmits. DRUNet keeps it at its CPWC contrast (−6.1 vs. −5.8 dB), ResUNet and AttResUNet close to it (−6.5 and −6.3 dB). The apparent gains of U-Net (−10.1 dB, gCNR 0.558) and MS-D (−7.3 dB, 0.525), beyond CPWC, are not detection gains: the cyst lies above one of their dark artifacts (Fig. 2).
- **Speckle is preserved by three networks.** With U-Net, DRUNet and MS-D, the speckle SNR stays at 1.81–1.82, close to the 1.91 of fully developed speckle, and the axial speckle size is unchanged (0.44 mm). This is the behavior predicted in Section II-B: on a phantom, speckle is coherent across transmits, so it is part of the target $s_j$, and the networks do not smooth it. This contrasts with in vivo, where the B-mode of the outputs looks smoother. ResUNet and, less so, AttResUNet do smooth the phantom speckle: speckle SNR 1.76 and 1.79, axial size 0.461 and 0.452 mm.
- **Lateral resolution.** U-Net, ResUNet and MS-D widen the lateral speckle size from 0.45 mm (DAS) to 0.47–0.48 mm, while DRUNet slightly narrows it (0.446 mm) and AttResUNet narrows it to the CPWC value (0.435 vs. 0.436 mm).
- **Dark artifacts.** In Fig. 2, U-Net darkens a blob below the hypoechoic cyst and a region deeper down; ResUNet and AttResUNet a smaller patch below the cysts, AttResUNet also a band at the top of the image; MS-D a large region below the cysts that extends to the bottom. The phantom is homogeneous there, and neither the DAS input nor CPWC shows these regions. DRUNet is the only network without one on this frame, and Section V-D finds smaller ones on other frames.

### D. Homogeneity of the phantom

To measure the dark artifacts, we compare the local brightness of each output with that of CPWC: the ratio of the two intensities $|x|^2$, each smoothed with a Gaussian of σ = 8 pixels (about 1.2 mm), in dB, minus its median over the frame. Eight frames of the held-out 054GS section are used: `invitro_02162` and seven frames spread over the section (00112–00598), with the fast input {−10°, 0°, +10°}.

**Table IV.** Local brightness against CPWC on eight held-out phantom frames: the darkest point and the fraction of the image more than 3 dB darker than CPWC, range over the frames (mean in parentheses).

| Method | Darkest point (dB) | Area < −3 dB |
|---|---|---|
| DAS, 3 tx | −0.5 to −1.0 | 0% |
| U-Net | −3.6 to −18.5 | 0.1–15% (7.0%) |
| ResUNet | −6.5 to −21.7 | 3.3–17% (10.4%) |
| AttResUNet | −4.1 to −14.0 | 0.7–21% (8.9%) |
| DRUNet | **−2.4 to −8.7** | **0–3.7% (1.3%)** |
| MS-D | −13.0 to −19.2 | 3.1–20% (11.0%) |

- **The input is not the cause.** The three-transmit DAS image matches CPWC to within 1 dB everywhere, so the dark regions are created by the networks.
- **Every network produces them; DRUNet the least.** Averaged over the frames, the bottom 40% of the image is 1.7–2.7 dB too dark for ResUNet and MS-D. AttResUNet also darkens a band near the surface.
- **The dark regions move from frame to frame**, so they depend on the content of the frame. A fixed-position cause, such as the padding or the edges of the receptive field, is ruled out.
- **The metrics of Tables I–III miss them** (Section VI).

### E. Round 4: scale equivariance and depth-gain augmentation

Round 4 retrains U-Net and DRUNet with the two changes motivated in Section VI: (i) the input is divided by its RMS and the output multiplied back, which makes every network exactly scale-equivariant; (ii) training pairs are multiplied by a random depth-gain curve (±10 dB at five depths, linear in dB in between, zero mean, the same for input and target). The split is by whole volunteers: training on 001, 003, 004 and 007 plus the FullSpeckle phantom section (3452 frames), validation on volunteer 005 (abdomen and carotid, 453 frames), test on volunteers 002, 006 and 008. The validation set now contains the noisier abdominal frames, so its losses cannot be compared with those of round 3.

**Table V.** Round-4 training (checkpoints `n2i_ultrasound_<arch>_test002_006_008`).

| Network | Run | Epochs run | Best epoch | Val. loss |
|---|---|---:|---:|---:|
| U-Net | `26-10-12-16:12:33` | 55 (early stop) | 25 | 0.2739 |
| DRUNet | `26-10-12-15:37:52` (resumed from `26-10-12-14:06:05`) | 100 | 74 | **0.2672** |

Scale equivariance holds exactly: scaling the input of four phantom frames by 0.25 to 4 scales the output by the same factor (relative error 0.000), and the dark area no longer changes with the gain (U-Net 21.1%, DRUNet 5.7% at every gain; round 3: 11–41%).

**Table VI.** Phantom, round 3 against round 4, fast input {−10°, 0°, +10°}. Cyst metrics and speckle on `invitro_02162` (as Table III); darkest point and area more than 3 dB darker than CPWC over the eight frames of Section V-D (as Table IV), and the mean over these frames of the deepest fifth of the image.

| Method | Anechoic C | Anechoic gCNR | Hypo. C | Hypo. gCNR | Speckle SNR | FWHM lat. | Darkest point (dB) | Area < −3 dB | Deepest fifth |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| DAS, 3 tx | −23.1 | 0.969 | −6.0 | 0.490 | 1.82 | 0.453 | −0.5 to −1.0 | 0% | +0.3 dB |
| U-Net, round 3 | −30.6 | 0.986 | −10.1 | 0.558 | 1.81 | 0.467 | −3.6 to −18.5 | 0.1–15% (7.0%) | |
| U-Net, round 4 | −31.7 | 0.989 | −7.4 | 0.478 | 1.79 | 0.481 | −11.9 to −19.0 | 11–25% (16.1%) | −4.9 dB |
| DRUNet, round 3 | −27.0 | 0.985 | −6.1 | 0.507 | 1.82 | 0.446 | −2.4 to −8.7 | 0–3.7% (1.3%) | |
| DRUNet, round 4 | −28.1 | 0.985 | −7.9 | 0.521 | 1.82 | 0.457 | −4.5 to −19.2 | 0.2–15% (5.5%) | −1.4 dB |
| *CPWC, 87 tx* | *−37.2* | *0.998* | *−5.8* | *0.476* | *1.82* | *0.436* | — | — | — |

**Fig. 3.** B-mode images (50 dB) of `invitro_02162` for the round-4 networks.

![Fig. 3](results/n2i_ultrasound_comparison_test002_006_008/invitro_CIRS054GS/054GS/invitro_02162_comparison.png)

- **The artifacts remain and are larger.** Over the eight frames, the dark area is 2–4 times that of round 3 (U-Net 16.1%, DRUNet 5.5% on average), and both networks reach −19 dB on some frame. The deepest fifth of the image is 4.9 dB (U-Net) and 1.4 dB (DRUNet) too dark on average. U-Net also darkens the deepest fifth of the FullSpeckle frames, which are in its training set, by 2.8 dB (round 3: less than 0.3 dB).
- **Both networks now darken the region below the cysts** (Fig. 3). The hypoechoic contrast, −7.4 and −7.9 dB against −5.8 dB for CPWC, is again an artifact, not a gain.
- **The in-vivo shrinkage is unchanged.** In the deepest two fifths of the in-vivo frames, where the input coherence is 0.2–0.4, both networks still reduce their input by 7.5–10.4 dB, as an MMSE estimator should (Section VI).

Removing the brightness cue therefore did not remove the artifacts. Two tests on `invitro_02162` show that the networks use the *position in the image* instead:

- *Cropping.* Removing the top 64 to 192 rows of the input leaves the U-Net dark area at 12–13% (11.3% on the full frame), and the dark patch stays below the cysts rather than at fixed rows. A fixed-position cause (padding, receptive-field edges) is again ruled out.
- *Vertical flip.* Flipping the input upside down, and the output back, removes the artifacts almost completely: the dark area falls from 11.3% to 1.0% for U-Net and from 1.6% to 0.0% for DRUNet, and the darkest point rises from −11.9 to −4.5 dB and from −5.4 to −1.6 dB.

**Table VII.** NMSE (dB) against the mean of the 84 transmits not in the input, six frames per folder spread over the folder, fast input; plain / vertically flipped input. The multi-transmit DRUNet is the epoch-6 checkpoint of an early version of Section V-F; Table X gives the final checkpoints.

| Folder | DAS, 3 tx | U-Net | DRUNet | DRUNet multi-tx |
|---|---:|---|---|---|
| 002 carotid | −8.09 | −9.41 / −7.77 | −9.57 / −8.53 | −9.55 / −8.49 |
| 002 abdomen | −1.64 | −5.91 / −3.76 | −5.92 / −3.71 | −5.87 / −2.75 |
| 008 abdomen | −1.49 | −5.96 / −3.53 | −6.13 / −3.58 | −5.93 / −2.53 |
| 006 carotid | −8.64 | −10.79 / −8.09 | −11.02 / −8.88 | −10.87 / −9.08 |
| Phantom 054GS (held out) | **−13.38** | −10.16 / −10.15 | −11.25 / −8.94 | −11.97 / −12.24 |

- **The networks rely on the vertical orientation in vivo.** Flipping the input costs 1.0–2.7 dB of NMSE. The networks have learned a prior tied to depth and to what lies above a region ("deep, or below a structure, means noise or shadow, so shrink"), which the in-vivo data reward.
- **On the phantom, the networks are worse than their input.** Both round-4 networks have a higher NMSE than the three-transmit DAS image (−10.2 and −11.3 dB against −13.4 dB): the dark regions cost more than the denoising gains.

### F. Multi-transmit input and vertical flips

A network fed the mean of the three transmits cannot tell coherent speckle from incoherent noise, which look alike locally, so it falls back on context. Whether the transmits agree is exactly the information it lacks: the input coherence is 0.97–0.99 at every depth on the phantom but 0.2–0.4 deep in vivo (Section VI). In a variant (`train_n2i_multitx_deepinv.py`), the network therefore receives the three transmits of a group as six channels (I and Q of each, sorted by steering angle) and still returns one IQ image. The target, the scale equivariance (RMS over the six channels), the depth-gain augmentation and the split are those of round 4. A lateral flip mirrors the steering angles, so it also reverses the transmit channels. Noise2Inverse still holds, since the input uses only the transmits of one group. The training and validation frames are rewritten once with the transmits of each group next to each other (77 MB per frame, about 300 GB); an epoch takes about 12 min, against 3–5 min for the group cache.

A second run adds a **vertical-flip augmentation**: half of the training pairs are flipped upside down, input and target together. This is not physically valid (attenuation and the depth-dependent PSF are not symmetric), but it removes the orientation cue of Section V-E directly. Both runs use DRUNet.

**Table VIII.** Training of the multi-transmit and any-n DRUNets (Section V-G). The validation loss of the any-n networks averages four input sizes and cannot be compared with the others.

| Network | Run | Epochs run | Best epoch | Val. loss |
|---|---|---:|---:|---:|
| DRUNet, round 4 (mean input, Table V) | `n2i_ultrasound_drunet_test002_006_008/26-10-12-15:37:52` | 100 | 74 | 0.2672 |
| DRUNet multi-tx | `n2i_ultrasound_drunet_multitx_test002_006_008/26-10-14-13:23:04` | 100 | 74 | 0.2632 |
| DRUNet multi-tx + vflip | `n2i_ultrasound_drunet_multitx_vflip_test002_006_008/26-10-15-11:17:32` | 82 (early stop) | 52 | **0.2623** |
| DRUNet any-n | `n2i_ultrasound_drunet_anyn_test002_006_008/26-10-17-10:29:37` | 100 | 87 | 0.0859 |
| DRUNet any-n + vflip | `n2i_ultrasound_drunet_anyn_vflip_test002_006_008/26-10-20-10:59:43` | 130 | 109 | 0.0871 |

All four runs show the early DRUNet loss spikes of rounds 3–4 and recover from them within an epoch: training loss 1.25 (multi-tx, epoch 0), 2.64 (multi-tx + vflip, epoch 0), 0.56 (any-n, epoch 9, against 0.23 around it) and 592 (any-n + vflip, epoch 1).

**Table IX.** Phantom, fast input {−10°, 0°, +10°}, final checkpoints. Cyst metrics on `invitro_02162` (as Table III); darkest point, area more than 3 dB darker than CPWC and deepest fifth over the eight frames of Section V-D (as Table IV), with the plain input and with the input flipped upside down (output flipped back).

| Method | Anechoic C | Anechoic gCNR | Hypo. C | Speckle SNR | Darkest point (dB) | Area < −3 dB | Deepest fifth | Area, flipped input |
|---|---:|---:|---:|---:|---|---|---:|---:|
| DAS, 3 tx | −23.1 | 0.969 | −6.0 | 1.82 | −0.5 to −1.0 | 0% | +0.3 dB | — |
| DRUNet, round 4 | −28.1 | 0.985 | −7.9 | 1.82 | −4.5 to −19.2 | 0.2–15% (5.5%) | −1.4 dB | 0.0% |
| DRUNet multi-tx | **−32.0** | 0.989 | −6.2 | 1.81 | −1.0 to −10.7 | 0–7.7% (**2.5%**) | −0.9 dB | 0.1% |
| DRUNet multi-tx + vflip | −30.3 | **0.992** | −6.2 | 1.82 | −0.9 to −13.4 | 0–8.6% (2.7%) | −0.9 dB | 2.6% |
| DRUNet any-n | −25.6 | 0.979 | −6.7 | 1.82 | −14.7 to −21.1 | 18–27% (23.6%) | −6.4 dB | 0.1% |
| DRUNet any-n + vflip | −24.2 | 0.966 | −6.7 | 1.78 | −10.0 to −18.6 | 12–34% (17.8%) | −4.3 dB | 24.4% |
| *CPWC, 87 tx* | *−37.2* | *0.998* | *−5.8* | *1.82* | — | — | — | — |

**Table X.** NMSE (dB) against the mean of the 84 transmits not in the input, six frames per folder taken at regular steps through the folder (not the frames of Table VII, hence slightly different values), final checkpoints; plain / vertically flipped input.

| Folder | DAS, 3 tx | DRUNet r4 | DRUNet multi-tx | multi-tx + vflip | DRUNet any-n | any-n + vflip |
|---|---:|---|---|---|---|---|
| 002 carotid | −8.17 | −9.63 / −8.56 | **−10.12** / −8.72 | −9.83 / −9.95 | −9.09 / −7.01 | −9.38 / −9.23 |
| 002 abdomen | −2.50 | −6.26 / −4.32 | **−6.29** / −3.71 | −6.12 / −6.21 | −6.25 / −3.66 | −5.86 / −5.86 |
| 006 carotid | −9.17 | −11.19 / −8.98 | **−11.47** / −9.61 | −11.26 / −11.40 | −10.88 / −7.23 | −10.85 / −10.82 |
| 008 abdomen | −3.24 | −6.98 / −4.88 | −6.97 / −4.38 | −6.75 / −6.81 | −6.87 / −4.05 | −6.45 / −6.49 |
| 008 back | −4.36 | −7.68 / −5.93 | **−7.70** / −5.41 | −7.39 / −7.52 | −7.49 / −4.72 | −7.35 / −7.29 |
| 008 carotid | −7.04 | −9.45 / −7.88 | **−9.64** / −7.79 | −9.26 / −9.35 | −9.08 / −6.57 | −8.92 / −8.85 |
| Phantom 054GS (held out) | **−14.62** | −12.93 / −8.50 | −13.96 / −12.88 | −14.08 / −14.01 | −12.21 / −12.63 | −12.31 / −11.42 |

**Table XI.** In-vivo metrics on every frame of the test volunteers (2272 frames), as Table I: mean ± standard deviation against the held-out reconstruction (84 transmits), round-4 U-Net and DRUNet (mean input) against the multi-transmit DRUNets. The DRUNets run in 13.0–13.3 ms per frame, the U-Net in 8.8–8.9 ms. In parentheses, the paired difference from the round-4 DRUNet in PSNR (dB) and the fraction of frames on which the network is better; every difference not marked n.s. has p < 0.03 (Wilcoxon).

| Region | Method | PSNR, fast (dB) | SSIM, fast | PSNR, interleaved (dB) | SSIM, interleaved | N2I loss |
|---|---|---|---|---|---|---:|
| 002 carotid (161) | DAS, 3 tx | 35.48 ± 2.30 | 0.382 ± 0.077 | 33.20 ± 2.20 | 0.322 ± 0.073 | 0.2661 |
| | U-Net, round 4 | 36.83 ± 2.22 (−0.27, 14%) | 0.592 ± 0.113 | 34.21 ± 2.27 (−0.30, 11%) | 0.566 ± 0.117 | 0.1937 |
| | DRUNet, round 4 | 37.10 ± 2.20 | 0.602 ± 0.110 | 34.50 ± 2.24 | 0.573 ± 0.114 | 0.1882 |
| | DRUNet multi-tx | **37.28 ± 2.22** (+0.18, 65%) | **0.613 ± 0.108** | **35.83 ± 2.25** (+1.33, 100%) | **0.588 ± 0.111** | 0.1794 |
| | multi-tx + vflip | 37.09 ± 2.22 (−0.01 n.s.) | 0.600 ± 0.111 | 34.99 ± 2.19 (+0.49, 94%) | 0.567 ± 0.114 | **0.1792** |
| 002 abdomen (298) | DAS, 3 tx | 27.40 ± 3.12 | 0.290 ± 0.052 | 26.37 ± 2.87 | 0.246 ± 0.044 | 1.0194 |
| | U-Net, round 4 | 31.34 ± 2.46 (−0.08, 20%) | 0.298 ± 0.061 | 29.89 ± 2.17 (−0.11, 14%) | 0.265 ± 0.061 | 0.6277 |
| | DRUNet, round 4 | 31.42 ± 2.45 | 0.309 ± 0.061 | 29.99 ± 2.19 | 0.279 ± 0.060 | 0.6173 |
| | DRUNet multi-tx | **31.49 ± 2.46** (+0.07, 63%) | **0.311 ± 0.063** | **30.64 ± 2.31** (+0.65, 99%) | **0.286 ± 0.062** | **0.6081** |
| | multi-tx + vflip | 31.26 ± 2.41 (−0.16, 19%) | 0.301 ± 0.060 | 30.07 ± 2.21 (+0.08, 63%) | 0.269 ± 0.059 | 0.6085 |
| 006 carotid (289) | DAS, 3 tx | 35.46 ± 2.76 | 0.421 ± 0.117 | 33.48 ± 2.49 | 0.355 ± 0.104 | 0.2237 |
| | U-Net, round 4 | 37.13 ± 2.34 (−0.25, 11%) | 0.672 ± 0.068 | 34.86 ± 2.20 (−0.39, 4%) | 0.651 ± 0.070 | 0.1633 |
| | DRUNet, round 4 | 37.38 ± 2.41 | 0.678 ± 0.069 | 35.25 ± 2.25 | 0.654 ± 0.070 | 0.1555 |
| | DRUNet multi-tx | **37.58 ± 2.40** (+0.20, 79%) | **0.681 ± 0.070** | **36.42 ± 2.31** (+1.18, 100%) | **0.662 ± 0.071** | 0.1545 |
| | multi-tx + vflip | 37.36 ± 2.39 (−0.02, 43%) | 0.673 ± 0.070 | 35.72 ± 2.22 (+0.47, 94%) | 0.648 ± 0.070 | **0.1534** |
| 008 abdomen (602) | DAS, 3 tx | 28.19 ± 3.11 | 0.289 ± 0.052 | 27.16 ± 2.79 | 0.247 ± 0.043 | 0.8813 |
| | U-Net, round 4 | 31.77 ± 2.65 (−0.10, 20%) | 0.298 ± 0.056 | 30.19 ± 2.26 (−0.15, 11%) | 0.267 ± 0.054 | 0.5729 |
| | DRUNet, round 4 | 31.87 ± 2.63 | **0.309 ± 0.056** | 30.34 ± 2.27 | 0.280 ± 0.053 | 0.5611 |
| | DRUNet multi-tx | **31.96 ± 2.59** (+0.09, 63%) | 0.308 ± 0.059 | **31.12 ± 2.32** (+0.77, 100%) | **0.285 ± 0.055** | **0.5400** |
| | multi-tx + vflip | 31.70 ± 2.51 (−0.17, 19%) | 0.293 ± 0.054 | 30.51 ± 2.21 (+0.17, 75%) | 0.265 ± 0.051 | 0.5419 |
| 008 back (542) | DAS, 3 tx | 28.84 ± 3.22 | 0.242 ± 0.062 | 27.82 ± 2.84 | 0.205 ± 0.051 | 0.8569 |
| | U-Net, round 4 | 32.34 ± 2.95 (−0.07, 29%) | 0.275 ± 0.083 | 30.68 ± 2.26 (−0.12, 23%) | 0.249 ± 0.080 | 0.5741 |
| | DRUNet, round 4 | 32.41 ± 2.88 | **0.286 ± 0.082** | 30.80 ± 2.24 | **0.263 ± 0.078** | 0.5563 |
| | DRUNet multi-tx | **32.47 ± 2.77** (+0.06, 52%) | 0.282 ± 0.083 | **31.56 ± 2.29** (+0.76, 100%) | **0.263 ± 0.079** | **0.5435** |
| | multi-tx + vflip | 32.28 ± 2.64 (−0.13, 26%) | 0.269 ± 0.078 | 31.08 ± 2.17 (+0.28, 83%) | 0.247 ± 0.074 | 0.5438 |
| 008 carotid (380) | DAS, 3 tx | 36.27 ± 3.26 | 0.448 ± 0.134 | 34.12 ± 2.97 | 0.378 ± 0.121 | 0.2996 |
| | U-Net, round 4 | 37.76 ± 2.87 (−0.21, 20%) | 0.691 ± 0.098 | 35.44 ± 2.72 (−0.24, 10%) | 0.667 ± 0.101 | 0.2185 |
| | DRUNet, round 4 | 37.96 ± 2.91 | 0.699 ± 0.099 | 35.68 ± 2.76 | 0.672 ± 0.101 | 0.2114 |
| | DRUNet multi-tx | **38.21 ± 2.96** (+0.25, 76%) | **0.708 ± 0.101** | **37.00 ± 2.77** (+1.32, 100%) | **0.686 ± 0.102** | **0.2004** |
| | multi-tx + vflip | 37.93 ± 2.89 (−0.04 n.s.) | 0.697 ± 0.102 | 36.20 ± 2.66 (+0.52, 91%) | 0.667 ± 0.103 | 0.2006 |

- **The multi-transmit input halves the phantom artifacts and is the best in vivo.** Over the eight frames, the dark area falls from 5.5% to 2.5% and the darkest point from −19.2 to −10.7 dB. On `invitro_02162` there is no dark patch below the cysts any more: the anechoic contrast reaches −32.0 dB (CPWC −37.2 dB) and the hypoechoic contrast −6.2 dB is no longer an artifact (CPWC −5.8 dB, round 4 −7.9 dB). In vivo it is the best network on five of six folders, by up to 0.5 dB, and its phantom NMSE (−14.0 dB) is close to that of its input (−14.6 dB), which no mean-input network reaches.
- **It still uses the orientation cue.** Flipping its input costs 1.4–2.6 dB in vivo, slightly more than for the round-4 network (1.1–2.2 dB), and on the phantom the flipped input removes the remaining dark regions (0.1%). The multi-transmit input therefore reduces the reliance on context but does not replace it.
- **The round-4 U-Net is below the round-4 DRUNet in every region and column** (Table XI): −0.07 to −0.27 dB of fast-input PSNR (better on 11–29% of the frames), −0.11 to −0.39 dB on the interleaved input, −0.004 to −0.013 of SSIM and a 0.006–0.018 higher N2I loss (p < 1e-16 everywhere), as in round 3 (Table I).
- **On all test frames, the multi-transmit input is the best in vivo** (Table XI). On the fast input it gains 0.06–0.25 dB of PSNR over the round-4 DRUNet (better on 52–79% of the frames, p ≤ 0.02 in every region) and matches or improves its SSIM, except on the back and 008 abdominal frames (−0.001 to −0.004). On the interleaved input, the training distribution, it gains 0.65–1.33 dB on 99–100% of the frames; the larger gain probably comes from the wider and uneven angles of that group (−16.3°, −5.3°, +11.0°), whose transmits differ more from each other than those at ±10° and 0°, so that keeping them apart helps more.
- **Vertical flips make the network orientation-invariant.** With the flip augmentation, flipping the input changes the in-vivo NMSE by at most 0.15 dB, and the phantom dark area is the same either way (2.7% and 2.6%). The cost in vivo is 0.2–0.4 dB of NMSE against the multi-transmit network without flips; on all test frames (Table XI) it loses 0.19–0.28 dB of PSNR and 0.008–0.015 of SSIM on the fast input, which brings it back to the level of the round-4 DRUNet (−0.01 to −0.17 dB) or slightly below it in SSIM, while its interleaved PSNR stays 0.1–0.5 dB above; on the phantom it has the best NMSE of all networks (−14.1 dB), still slightly worse than its input. The validation loss is marginally lower (0.2623 against 0.2632).
- **The remaining multi-transmit patches are not an orientation effect.** They are the same with and without the flip augmentation (frames `00274`, `00355` and `00598`; darkest −13.4 dB), and they lie deep at the lateral edges of the image: 70–85% of the dark pixels of `00274` and `00355` are in the outer 15% of the columns, in the deepest 25–40% of the depth. These are the regions that the steered plane waves cover only partly. The 192-element aperture (44 mm) is wider than the image (36 mm), but a wave steered by ±10° leaves the image edges below 22 mm depth, and at 70 mm only the central ±9.6 mm is covered by all three transmits; this partly covered region is 16% of the image. A test on the eight phantom frames confirms the link:
  - 86% (multi-tx) and 88% (multi-tx + vflip) of all dark pixels lie in the partly covered region. In the deepest 40% of the image, 16% of its partly covered pixels are dark, against 1.1–1.4% of the fully covered ones. The dark pixels of the round-4 mean-input network show no such preference (25% in the region; 8% and 11% dark rates).
  - In that region one transmit is about 10 dB weaker than the strongest (min/max local energy 0.1, against 0.7 where all three cover), and the mean pairwise correlation between the three transmits drops from 0.45–0.60 to 0.30–0.39. This is the cue the multi-transmit network reads.
  - The information is nevertheless there: the three-transmit mean stays coherent with the 84 other transmits (0.94–0.97, against 0.97–0.99 elsewhere), and the DAS input is not darker than CPWC there (+0.1 dB). Darkening the region is therefore an error of the network, not a correct response to missing signal.
  - It is a domain shift, not a consequence of the geometry. On eight frames of the FullSpeckle section, which is in the training set and has the same coverage signature (pairwise correlation 0.39 in the region), no network darkens anything (0.0%). The held-out frames that are darkened have the lowest pairwise correlation in the region (0.30 for `00274` and `00598`), although the relation is loose (`00193` at 0.34 is clean).
  - In vivo the region is not special: the pairwise correlation is 0.09–0.13 at depth whether all three transmits cover it or not, and the networks shrink both regions alike (−7.7 to −9.3 dB).

### G. Any number of transmits

The networks above are trained for one acquisition, three transmits at ±10° and 0°. Fed the DAS image of 1, 11 or 87 transmits instead, the round-4 DRUNet still gains 2.4–5.0 dB in vivo at one transmit, 1.7–4.1 dB at three and 1.0–1.7 dB at 11, but it under-denoises a single transmit and changes a full CPWC image by −10.3 to −15.9 dB of NMSE (Table XII, all test frames): it applies the shrinkage it learned for three transmits whatever the input. A variant (`train_n2i_anyn_deepinv.py`) trains one network for any acquisition. Its input is the mean of *n* transmits, with *n* drawn log-uniformly in [1, 80] and the transmits evenly spaced in angle over a random span of 10–32.6°; its target is the mean of all other transmits, which keeps the N2I loss valid for every *n*. DRUNet is told *n* through its noise-level channel, σ(*n*) = 0.1 √(3/*n*), which equals the fixed level of the other networks at *n* = 3. The split, scale equivariance and depth-gain augmentation are those of round 4; the training frames come from the transmit cache of Section V-F. Validation averages the loss over four fixed acquisitions of 1, 3, 11 and 43 transmits, so it is dominated by the noisiest inputs (the N2I loss of the identity spans 1.38 at one transmit to 0.004 at 43). An epoch takes about 33 min. A second run adds the vertical flips of Section V-F.

The test acquisitions are nested: 1 transmit (0°), 3 (±10°, 0°), 11 (evenly spaced over ±10°) and all 87. The NMSE reference is the mean of the 76 transmits outside the 11-transmit set, the same for every input; with 87 transmits no independent reference is left, and the table gives instead the NMSE of the output against its input (CPWC), i.e. how much the network still changes it.

**Table XII.** NMSE (dB) against the mean of the 76 transmits outside the 11-transmit set, mean over every frame of each folder (in vivo 161–602 frames, phantom section 1926 frames; 2026-10-26). Net = round-4 DRUNet (trained on 3 transmits), any-n and any-n + vflip as in Table VIII. Last three columns: NMSE of the output against the 87-transmit CPWC input (lower = changed less). Against the same reference, the three-transmit inputs give DAS / Net / any-n / +vflip −8.25 / −9.99 / −9.49 / −9.60 dB (002 carotid), −1.72 / −5.83 / −5.69 / −5.36 (002 abdomen), −9.02 / −11.03 / −10.75 / −10.70 (006 carotid), −2.53 / −6.31 / −6.20 / −5.86 (008 abdomen), −2.93 / −6.60 / −6.49 / −6.15 (008 back), −8.11 / −9.93 / −9.46 / −9.46 (008 carotid) and −14.33 / −12.31 / −12.04 / −12.23 (phantom); Table X gives them against the 84-transmit reference on six frames per folder.

| Folder | DAS 1 | Net 1 | any-n 1 | +vflip 1 | DAS 11 | Net 11 | any-n 11 | +vflip 11 | Net 87 | any-n 87 | +vflip 87 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 002 carotid | −0.62 | −3.05 | −4.20 | **−4.23** | −8.80 | −9.87 | **−10.67** | −10.62 | −14.64 | −14.75 | −14.36 |
| 002 abdomen | +4.05 | −0.93 | **−2.45** | −2.38 | −5.40 | −7.10 | **−8.01** | −7.87 | −10.27 | −14.88 | −14.74 |
| 006 carotid | −1.91 | −4.41 | −5.21 | **−5.24** | −10.35 | −11.32 | **−12.13** | −12.11 | −15.85 | −17.30 | −16.87 |
| 008 abdomen | +3.72 | −0.50 | **−2.27** | −2.19 | −5.64 | −7.29 | **−8.16** | −8.03 | −11.31 | −15.25 | −15.15 |
| 008 back | +3.36 | −0.45 | **−2.31** | −2.20 | −5.99 | −7.64 | **−8.49** | −8.29 | −12.23 | −15.84 | −15.93 |
| 008 carotid | −0.49 | −2.94 | −3.96 | **−3.99** | −8.80 | −9.90 | −10.57 | **−10.60** | −14.84 | −16.57 | −16.30 |
| Phantom 054GS | −4.83 | **−5.51** | −4.70 | −3.32 | −11.26 | −11.04 | **−13.04** | −12.52 | −15.44 | −15.44 | −17.08 |

**Table XIII.** Phantom against the number of transmits: darkest point, area more than 3 dB darker than CPWC, mean (max) and deepest fifth over the eight frames of Section V-D; anechoic and hypoechoic contrast on `invitro_02162` (CPWC −37.2 and −5.8 dB).

| Transmits | Method | Darkest (dB) | Area < −3 dB | Deepest fifth (dB) | Anechoic C | Hypo. C |
|---:|---|---:|---|---:|---:|---:|
| 1 | DAS | −3.1 | 0.0% (0.1%) | +1.7 | −19.6 | −5.7 |
| | Net (3 tx) | −20.3 | 3.9% (12.4%) | −0.0 | −25.7 | −7.2 |
| | any-n | −19.9 | 43.1% (45.8%) | −12.3 | −25.8 | −7.8 |
| | any-n + vflip | −13.5 | 31.9% (35.0%) | −4.1 | −26.1 | −8.8 |
| 3 | DAS | −1.0 | 0% | +0.3 | −23.1 | −6.0 |
| | Net (3 tx) | −19.2 | 5.5% (15.0%) | −1.4 | −28.1 | −7.9 |
| | any-n | −21.1 | 23.6% (27.4%) | −6.4 | −25.6 | −6.7 |
| | any-n + vflip | −18.6 | 17.8% (34.4%) | −4.3 | −24.2 | −6.7 |
| 11 | DAS | −1.2 | 0% | +0.6 | −29.8 | −5.7 |
| | Net (3 tx) | −19.0 | 2.8% (8.5%) | −0.3 | −33.2 | −7.0 |
| | any-n | −9.7 | **1.3%** (2.9%) | −1.3 | −29.6 | −5.8 |
| | any-n + vflip | −11.4 | 1.5% (3.8%) | −1.3 | −28.5 | −6.3 |
| 87 | Net (3 tx) | −19.3 | 5.0% (15.1%) | −1.4 | −35.8 | −8.1 |
| | any-n | −8.8 | 4.6% (6.0%) | −1.8 | −31.2 | −5.9 |
| | any-n + vflip | −7.1 | **2.0%** (3.7%) | −1.6 | −28.6 | −6.1 |

- **In vivo, one network serves every acquisition.** On all test frames (Table XII), the any-n network beats the three-transmit network by 0.8–1.9 dB on a single transmit (most on the abdominal and back frames, where the latter under-denoises), and by 0.7–0.9 dB on 11 transmits. On three transmits, the acquisition the other network was trained for, it loses 0.1–0.5 dB. On a CPWC image it changes its input 0.1–4.6 dB less, as it should for an almost noise-free input (least on the 002 carotid frames). The vertical flips cost at most 0.2 dB, except 0.3 dB on the abdominal and back frames at three transmits.
- **On the phantom, the any-n network is the worst at 1–3 transmits.** It darkens the deepest third of the image across its whole width: 43% of the image at one transmit and 24% at three, with the deepest fifth 12.3 and 6.4 dB too dark, against 3.9% and 5.5% for the three-transmit network. At one transmit its phantom NMSE over the whole section (−4.7 dB, −3.3 dB with flips) is worse than its input (−4.8 dB) and than the three-transmit network (−5.5 dB). On the phantom, this NMSE is dominated by angle-dependent differences rather than noise: three-transmit DAS is 3 dB closer to the reference than 11-transmit DAS (−14.3 against −11.3 dB), which averaging alone cannot explain. At 11 and 87 transmits the two any-n networks are the cleanest (1.3–4.6% dark area, darkest −7.1 to −11.4 dB, against about −19 dB for the three-transmit network). It also weakens the anechoic contrast of a CPWC input (−31.2 against −37.2 dB).
- **This dependence is orientation-driven only without flips.** Without flip augmentation, flipping the three-transmit input removes the dark band (0.1%) and costs 2.1–3.7 dB in vivo, the strongest orientation dependence of all networks (Table X). With the flips the network becomes orientation-invariant (in-vivo change below 0.2 dB), the dark area shrinks (32% at one transmit, 18% at three) but does not disappear, and it stays at depth whichever way the input is presented (24% with the flipped input). The any-n network therefore darkens weak deep signal on another cue than orientation. A likely one is the local amplitude relative to the noise level it is given: the scale equivariance normalizes the whole image, not each depth, and on the phantom the deep signal of a 1–3-transmit image is weak against σ(*n*). **[TODO]** Test this by scaling the noise-level input, and by measuring the phantom artifacts against *n* in finer steps between 3 and 11.

### H. Equivalent number of compounded transmits

How many compounded transmits is a network worth? Plain DAS compounding of *N* = 3, 5, 9, 15 and 29 transmits, evenly spaced in angle over the full ±16.3° aperture (nearest transmit; the sets are not nested), gives a quality-against-frame-rate curve. Every image, compounded or denoised, is scored against one common reference: the mean of the 50 transmits that belong to none of these sets and to neither network input. PSNR and SSIM are those of Table I, on every frame of the test volunteers (2272 frames). The equivalent *N* of a method is where the DAS curve reaches its score, interpolated linearly in log *N*. The networks are the round-4 U-Net and DRUNet and the multi-transmit DRUNet (Section V-F), all trained by N2I.

**Table XIV.** Equivalent number of compounded transmits, PSNR / SSIM, fast input {−10°, 0°, +10°}, all test frames. The DAS column places the unprocessed input on the same curve.

| Region | DAS, fast input | U-Net | DRUNet | DRUNet multi-tx |
|---|---:|---:|---:|---:|
| 002 carotid (161) | 4.3 / 4.5 | 4.9 / 9.4 | 5.0 / 9.8 | 5.1 / **10.2** |
| 002 abdomen (298) | 3.6 / 4.3 | 6.4 / 4.3 | 6.5 / 4.6 | **6.6** / 4.6 |
| 006 carotid (289) | 4.2 / 4.4 | 5.0 / 10.7 | 5.2 / 11.0 | **5.3** / **11.1** |
| 008 abdomen (602) | 3.7 / 4.3 | 6.0 / 4.3 | 6.1 / 4.5 | 6.1 / 4.5 |
| 008 back (542) | 3.8 / 4.3 | 5.9 / 4.7 | 6.0 / 4.9 | 5.9 / 4.8 |
| 008 carotid (380) | 4.3 / 4.6 | 5.0 / 10.7 | 5.1 / 11.0 | **5.2** / **11.4** |

**Fig. 4.** PSNR (top) and SSIM (bottom) against the number of compounded transmits, one panel per region; networks on the fast input as horizontal lines.

![Fig. 4a](results/n2i_ultrasound_cpwc_curve_test002_006_008/psnr_vs_n.png)
![Fig. 4b](results/n2i_ultrasound_cpwc_curve_test002_006_008/ssim_vs_n.png)

- **The three-transmit input is not the start of the curve.** The fast input (±10°, 0°) sits at *N* = 3.6–4.3 by PSNR, 1.4–3.7 dB above three transmits spread over ±16.3°: its narrower angles decorrelate the speckle less from the reference. The interleaved input (−16.3°, −5.3°, +11.0°) sits at *N* = 3.1–3.4. A network's gain is therefore best read against its own input.
- **By PSNR, a network is worth 5–6.6 transmits from three.** Against the fast input's own place on the curve, this is a factor of 1.2–1.3 on the carotid frames and 1.6–1.8 on the abdominal and back frames. The DAS curve rises by 14–18 dB from 3 to 29 transmits, much faster than the 9.9 dB of averaging independent noise, so its PSNR also measures how closely the angles of a set match those of the reference, not only noise. The equivalent *N* therefore holds for this compounding scheme, not for any acquisition of *N* transmits.
- **By SSIM, the gain depends on the tissue.** On the carotid frames the networks reach the SSIM of 9.4–11.4 compounded transmits, 2.1–2.5 times their input, which matches the visual gain of Fig. 1. On the abdominal and back frames, where deep signal is weak, the SSIM gain is small (4.3–4.9 against 4.3).
- **The architectures and inputs are close.** The networks differ by at most 0.3 transmits in PSNR and 0.8 in SSIM; the multi-transmit DRUNet is ahead on the carotid frames. On the interleaved input, the training distribution, it gains more (*N* = 4.3–5.5 against 3.8–5.0 for the mean-input DRUNet).

**Table XV.** Phantom `invitro_02162` against the number of compounded transmits (same sets as Table XIV) and networks on the fast input; in-vitro metrics of Table III.

| Method | An. C (dB) | An. gCNR | Hypo C (dB) | sSNR | FWHM lat (mm) |
|---|---:|---:|---:|---:|---:|
| DAS, 3 tx (±16.3°) | −21.2 | 0.949 | −6.0 | 1.83 | 0.341 |
| DAS, 5 tx | −23.9 | 0.972 | −5.9 | 1.82 | 0.394 |
| DAS, 9 tx | −26.7 | 0.984 | −5.9 | 1.82 | 0.417 |
| DAS, 15 tx | −30.1 | 0.991 | −5.8 | 1.82 | 0.425 |
| DAS, 29 tx | −33.3 | 0.995 | −5.8 | 1.82 | 0.431 |
| DAS, fast input | −23.1 | 0.969 | −6.0 | 1.82 | 0.453 |
| U-Net, round 4 | −31.7 | 0.989 | −7.4† | 1.79 | 0.481 |
| DRUNet, round 4 | −28.1 | 0.985 | −7.9† | 1.82 | 0.457 |
| DRUNet multi-tx | −32.0 | 0.989 | −6.2 | 1.81 | 0.446 |
| CPWC, 87 tx | −37.2 | 0.998 | −5.8 | 1.82 | 0.436 |

† Dark artifact below the cysts (Section V-E), not a contrast gain.

- **On the phantom, the anechoic contrast is worth more transmits.** The fast input matches about 4 compounded transmits; the round-4 DRUNet about 11, and the multi-transmit DRUNet, whose contrast is not an artifact, about 20 (between −30.1 and −33.3 dB). Its gCNR (0.989) is that of 9–15 transmits. The U-Net contrast is partly that of the dark patch below the cysts.
- Sparse wide-angle sets have a narrower lateral FWHM than CPWC (0.34 mm at three transmits), at the cost of stronger side lobes, so the resolution column cannot be read as quality.

The supervised networks of the next step (same architectures, data and budget, trained on the 87-transmit CPWC image) will be placed on the same curve. **[TODO]** Add them when trained.

---

## VI. Discussion

**Why the deep region is darkened in vivo.** This part is expected from an MSE-trained estimator. Where the input carries little information about the phase of the coherent signal, the conditional mean of a zero-mean complex speckle field shrinks towards zero, and its envelope is underestimated. We measured how much of the input is coherent with the target: the local magnitude of the normalized complex correlation between the three-transmit input and the mean of the other 84 transmits (smoothed as in Section V-D). In vivo it falls from 0.88–0.97 near the surface to 0.20–0.39 in the deepest two fifths of the image (volunteers 001–004, four frames each). All five networks reduce the amplitude of their input by 7.5–12 dB there, close to the factor of 0.2–0.4 that an MMSE estimator would apply. This is a property of the MMSE estimator, not of the self-supervision: a network trained with the same loss on noiseless CPWC targets would do the same.

**Why the phantom is darkened.** On the phantom, the same correlation stays at 0.97–0.99 at every depth, so no shrinkage is justified, and the artifacts of Section V-D do not follow it (pixel-wise correlation between the local correlation and the local gain of the network: −0.01 to +0.16, and 0.45 for DRUNet). The whole 054GS section is held out of training, so they are not a memorization of near-duplicate frames either; on frames of the FullSpeckle section, which is in the training set, U-Net and DRUNet change the brightness of their input by less than 0.3 dB, and the other networks by up to 2 dB. Two experiments point to a **learned dependence on absolute brightness**:

- *Rescaling the input.* A denoiser should satisfy $f(a x) = a f(x)$. None of the round-3 networks does: scaling the phantom input by 0.5 or 2 changes the rescaled output by 18–54%, and the area more than 3 dB too dark grows from 3–18% to 11–41% (four frames). The networks have only seen one brightness level, since every frame is normalized to unit CPWC RMS, and they use it: in vivo, deep and weak signal is mostly noise, so "weak at depth" has been learned as "set to zero".
- *DRUNet.* All convolutions of DRUNet are bias-free, which makes a ReLU network scale-equivariant [16], except that it also receives a constant noise-level channel. When this level is scaled with the input, DRUNet becomes exactly equivariant, and its output on `invitro_02162` stays within 2.8 dB of CPWC at every gain from 0.25 to 4. Being equivariant except for one channel probably explains why it shows the fewest artifacts.

This is a domain-shift artifact and a hallucination of hypoechoic structure, which is clinically worse than a uniform loss of brightness. The output is a good *estimate of the IQ image* in the MSE sense, but its local brightness in low-SNR or out-of-distribution regions should not be read as echogenicity. A per-pixel uncertainty map, e.g. the spread of the outputs over the 29 input groups, would flag such regions.

**Round 4: removing the brightness cue is not enough.** Two changes targeted this dependence. (i) Every network divides its input by its RMS and multiplies its output back, which makes it exactly scale-equivariant whatever its architecture; bias-free layers would not be enough with batch normalization or the sigmoid of the attention gates. This also removes the dependence on the CPWC normalization of the cache, which is not available for a real fast acquisition. (ii) The training pairs are multiplied by a random depth-gain curve, like the TGC of a scanner, so that brightness at a given depth no longer tells tissue from noise. Both work as intended (Section V-E): the networks are exactly equivariant, and their artifacts no longer change with the input gain. But the artifacts did not disappear; they grew. Without the brightness cue, the networks lean on another cue that is valid in vivo: the position of a region, in depth and below other structures. Cropping moves the dark patch with the content, and flipping the input upside down removes it, at a cost of 1–3 dB of in-vivo NMSE. The networks behave like the conditional mean they estimate: given only the mean of three transmits, any feature that predicts low coherence in vivo is used, and the phantom, whose coherence stays high at every depth, contradicts all of them. The fix therefore has to give the network the coherence itself rather than remove its proxies one by one.

**Multi-transmit input.** The three transmits of a fast acquisition carry this information: where they agree, the signal is coherent and should be kept. Feeding them as separate channels (Section V-F) costs nothing at inference and keeps the N2I loss valid. It helps: the phantom dark area is halved, the false hypoechoic contrast below the cysts disappears, the phantom NMSE comes within 0.7 dB of the input, and the network is also the best in vivo. But it does not remove the reliance on context: the network still loses 1.4–2.6 dB in vivo when its input is flipped, so even with the coherence in view, position remains a useful predictor of what the target keeps. The vertical-flip augmentation removes that cue at a small in-vivo cost (0.2–0.4 dB), which is the price of a prior that was valid in vivo. The patches it leaves are not orientation effects. They lie deep at the lateral edges, where the steered plane waves cover the image only partly and one transmit is about 10 dB weaker than the others. The network reads the resulting drop in agreement between transmits as noise, although the three-transmit mean is still coherent with the rest of the sequence there; on the phantom section seen in training, with the same geometry, it does not do so. The geometry is known, so the coverage of each transmit could be given to the network as extra input channels, or each transmit equalized by its illumination before it is fed in, so that partial coverage no longer looks like disagreement. On the data at hand, however, the flips buy nothing measurable: the phantom artifacts are the same with and without them, and the fast-input PSNR and SSIM drop back to those of the mean-input network (Table XI). The multi-transmit network without flips is therefore the best network of this study; the flip augmentation is a robustness option for data whose orientation or depth structure differs from the training set, such as other probes or a different depth range, which this dataset cannot test.

**One network for every acquisition.** A three-transmit network applies the same shrinkage to any input, so it under-denoises one transmit and still changes a CPWC image. Training on random numbers of transmits and telling DRUNet that number through its noise-level channel fixes this in vivo (Section V-G), at almost no cost on three transmits. On the phantom, however, the any-n network darkens the deep third of the image at 1–3 transmits far more than the three-transmit network, and vertical flips only reduce this. Its training loss is dominated by the noisiest inputs, where deep in-vivo signal is almost pure noise and aggressive shrinkage pays off, and on a mean image it has, like the round-4 networks, no way to tell weak coherent signal from noise. Combining both variants, a network fed the individual transmits of an acquisition of any size, would give it that information; the input size then varies, which a permutation-invariant pooling over transmits, or a fixed number of channels holding sub-means, could handle.

**Speckle: in vivo vs. in vitro.** On the phantom, U-Net, DRUNet and MS-D keep speckle intact (speckle SNR and size unchanged; ResUNet and AttResUNet smooth it slightly, Table III), as Section II-B predicts for speckle that is coherent across transmits. The smoother look of the in-vivo outputs therefore comes from the part of the in-vivo speckle that is *not* coherent across the 87 transmits, which tissue motion over the 87 transmits and the angle-dependence of the speckle pattern can explain. **[TODO]** Check this, e.g. by measuring the speckle SNR in a homogeneous in-vivo region. This is consistent with N2I removing exactly what differs between groups.

**What the metrics miss.** The in-vivo PSNR/SSIM of Table I and the N2I validation loss miss the dark artifacts of Fig. 2, because they are averaged out by global metrics. The cyst metrics of Table III miss them when they lie outside the cyst regions, and distort them, possibly into an apparent gain, when they cover the regions (U-Net, hypoechoic cyst). Checkpoint selection on the validation loss alone can therefore pick a network with this artifact. The homogeneity measure of Section V-D, over several frames, should therefore be part of the in-vitro evaluation.

**What N2I cannot remove.** The target is the expected compound image, so anything coherent across all groups survives. This includes the part of the PSF shared by every group, reverberation, and off-axis clutter from strong scatterers. N2I thus reaches CPWC-like, not beyond-CPWC, image quality. To go beyond CPWC, we also implemented Sparse2Inverse [13], which projects the network output back onto the channel data of held-out transmits. A preliminary analysis on this dataset showed that the linear single-scattering operator explains only 0.6–5.5% of a transmit's channel-data energy, depending on which image is projected. A measurement-domain loss would therefore sit on a large floor, and we leave it to future work.

**Temporal consistency.** Fast imaging is only worth its cost if the frames can be used *as a sequence*: Doppler and functional imaging read the phase of the IQ signal from frame to frame, and elastography and motion estimation track the speckle. The networks see one frame at a time and know nothing of its neighbours, so the quality of a single frame does not guarantee a stable sequence. N2I bears on this in three ways.

- *Static tissue should become more stable, not less.* The estimator is a deterministic function of the input, and its target keeps what is coherent across transmits (Section II-B). Where the tissue is static, consecutive outputs therefore differ only through the input noise that the network lets through, which is less than in the three-transmit DAS images. The speckle that tracking relies on is part of the target, and on the phantom it is preserved (Table III).
- *Shrunk regions carry no usable phase.* Where the network shrinks its input towards zero (the deep in-vivo region, Section VI), the remaining signal is small and its phase is set by the input noise. Doppler or phase-based displacement estimates there would be no better than on the input, while the B-mode suggests a clean, dark region. Such regions should be masked, e.g. with the input coherence of Section VI or the uncertainty map proposed above.
- *Hallucinations may flicker.* The dark regions of Section V-D change between frames of the held-out phantom section. Those frames are spread over the section rather than consecutive, but if the regions also change between consecutive frames, a static phantom would show dark regions appearing and disappearing. That would be a temporal artifact that no single-frame metric can detect.

The dataset does not allow a direct test at the target frame rate. Each acquisition holds one 87-transmit sequence, a three-transmit "fast" input is cut out of it, and its three transmits are spread over a large part of the sequence (the acquisition order alternates the angles from the widest inwards, so ±10° are transmits 35 and 36 of 87 and 0° is the last one). It is therefore not an acquisition at 29 times the frame rate. Within one acquisition, however, the 29 groups are 29 images of the *same* anatomy with independent noise, which is what a static scene imaged 29 times would give. **[TODO]** (i) Denoise all 29 groups of a frame and measure, per pixel, the spread of the outputs and the phase difference between them (the lag-one autocorrelation used by Doppler estimators), against the same quantities for DAS and for 29-transmit compounds. For a static scene the phase difference should be zero, so its spread is the phase noise floor of a Doppler estimate. (ii) On the phantom, check whether the dark regions of Section V-D appear for some groups and not others. (iii) Estimate the displacement between consecutive in-vivo acquisitions (carotid wall motion) by speckle tracking on DAS, network and CPWC images, and compare the motion fields. The time between consecutive acquisitions of the dataset has to be checked first. If the networks turn out to be unstable over time, the fixes are to give them several frames as input or to add a consistency term between the outputs of different groups of the same frame, which the partition provides without any extra data.

**Architecture.** The four encoder–decoders reach the same validation loss within 1.3%, a difference small compared with the effect of self-supervision itself, which agrees with observations in image denoising, where training data and loss dominate architecture. The extra capacity of ResUNet and AttResUNet (9 M parameters) only lowers the training loss. MS-D, the network of the original N2I paper, is 5–6% worse on the validation loss, 6–7× slower to train, and has the strongest phantom artifacts. The test metrics agree (Tables I–II): the encoder–decoders are within 0.3 dB of PSNR and 0.012 of SSIM of each other, each leads on some metric (DRUNet on PSNR, AttResUNet on SSIM), and MS-D is last on almost every metric while being 7–10× slower at inference. The extra capacity of ResUNet and AttResUNet buys no PSNR, and on the phantom it smooths the speckle and, for ResUNet, leaves the anechoic cyst unimproved. On the in-vivo metrics alone, the 2 M-parameter U-Net would be a sensible default for real-time use. The phantom changes that conclusion: DRUNet has by far the fewest dark artifacts, and it is also slightly better than U-Net on every in-vivo metric, at 1.5× its runtime.

### Limitations **[TODO before submission]**

1. **One test volunteer.** Volunteer 002 is the only test volunteer of round 3. Volunteers 006 (carotid, 289 frames) and 008 (abdomen, back and carotid, 1524 frames) are now cached and form the test set of round 4 with 002. Volunteer 006 was a validation volunteer in round 3, so round-3 results on it are slightly optimistic; 008 was never seen. All round-4 networks are scored on every frame of the three test volunteers (Tables XI and XII); only the flip tests use six frames per folder (Tables VII and X).
2. **Single seed.** Each network is trained once. Retrain with several seeds and report the spread across seeds.
3. **One partition size.** Only *K* = 29 (three transmits) was studied. Sweep *K* ∈ {87, 43, 29, 15, 9} to map the frame-rate/quality trade-off, and compare with supervised training on CPWC targets and with classical denoisers (BM3D, non-local means).
4. **In-vitro evaluation is thin.** (a) Table III uses a single phantom frame, so it has no confidence interval; Section V-D uses eight, and the other frames of the held-out 054GS section could all be used. (b) No wire target, so resolution is only measured through the speckle size; download a section with wire targets if the dataset has one. (c) The homogeneity measure of Section V-D is computed by `artifacts_n2i_deepinv.py`, separately from Table III.
5. **In-vivo ultrasound metrics.** Add contrast, CNR and gCNR on vessel lumen/wall ROIs in the carotid frames.
6. **Temporal behavior.** Frame-to-frame consistency is not measured yet; the Discussion (*Temporal consistency*) gives the expected behavior and three tests that the dataset allows. A direct test at the target frame rate needs a true fast acquisition, which the dataset does not contain.

---

## VII. Conclusion

Splitting the transmits of a plane-wave sequence into disjoint, aperture-spanning groups provides Noise2Inverse training pairs for free in any ultrafast acquisition. A network trained this way on in-vivo data alone, without a single clean image, turns a three-transmit acquisition of an unseen volunteer into an image close to the 84-transmit compound. It gains 1.7–1.9 dB PSNR and 0.22 SSIM on carotid frames and 4.0 dB PSNR on abdominal frames against an independent reference, in 7–16 ms per frame. Architecture matters little: four encoder–decoders end within 0.3 dB, DRUNet slightly but consistently ahead of U-Net; MS-D, the network of the original N2I paper, is the weakest. On a tissue-mimicking phantom held out of training, the networks recover part of the compounding gain in anechoic-cyst contrast while preserving the speckle statistics. The same test reveals dark artifacts in every network, absent from the input and invisible to the in-vivo metrics and to the validation loss. They come from contextual cues that predict low coherence in vivo: first absolute brightness, and, once the networks are made scale-equivariant, the vertical position in the image. Phantom tests should therefore accompany in-vivo metrics when such networks are evaluated. Giving the network the coherence of the transmits directly, by feeding them as separate channels, halves the artifacts and gives the best in-vivo results, and vertical-flip augmentation then removes the orientation cue at a small in-vivo cost; a single network for any number of transmits works in vivo but brings back the artifacts at few transmits. Measured against plain compounding, three denoised transmits are worth 5–6.6 compounded ones by PSNR and, on carotid frames, 9–11 by SSIM.

---

## References

**[TODO]** Check every entry against the published version.

[1] G. Montaldo, M. Tanter, J. Bercoff, N. Benech, and M. Fink, "Coherent plane-wave compounding for very high frame rate ultrasonography and transient elastography," *IEEE Trans. Ultrason., Ferroelectr., Freq. Control*, vol. 56, no. 3, pp. 489–506, 2009.

[2] M. Tanter and M. Fink, "Ultrafast imaging in biomedical ultrasound," *IEEE Trans. Ultrason., Ferroelectr., Freq. Control*, vol. 61, no. 1, pp. 102–119, 2014.

[3] M. Gasse, F. Millioz, E. Roux, D. Garcia, H. Liebgott, and D. Friboulet, "High-quality plane wave compounding using convolutional neural networks," *IEEE Trans. Ultrason., Ferroelectr., Freq. Control*, vol. 64, no. 10, pp. 1637–1639, 2017.

[4] D. Perdios, M. Vonlanthen, F. Martinez, M. Arditi, and J.-P. Thiran, "CNN-based image reconstruction method for ultrafast ultrasound imaging," *IEEE Trans. Ultrason., Ferroelectr., Freq. Control*, vol. 69, no. 4, pp. 1154–1168, 2022.

[5] J. Lehtinen *et al.*, "Noise2Noise: Learning image restoration without clean data," in *Proc. ICML*, 2018.

[6] A. A. Hendriksen, D. M. Pelt, and K. J. Batenburg, "Noise2Inverse: Self-supervised deep convolutional denoising for tomography," *IEEE Trans. Comput. Imaging*, vol. 6, pp. 1320–1335, 2020.

[7] J. Batson and L. Royer, "Noise2Self: Blind denoising by self-supervision," in *Proc. ICML*, 2019.

[8] EPFL ultrafast ultrasound dataset (in-vivo, GE 9L-D, Verasonics). **[TODO: dataset citation/DOI; likely the same source as [4].]**

[9] J. Tachella *et al.*, "DeepInverse: A Python package for solving imaging inverse problems with deep learning," arXiv:2505.20160, 2025.

[10] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. MICCAI*, 2015.

[11] O. Oktay *et al.*, "Attention U-Net: Learning where to look for the pancreas," in *Proc. MIDL*, 2018.

[12] K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, and R. Timofte, "Plug-and-play image restoration with deep denoiser prior," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 10, pp. 6360–6376, 2022.

[13] N. Gruber *et al.*, "Sparse2Inverse: Self-supervised inversion of sparse-view CT data," arXiv:2402.16921, 2024.

[14] A. Rodriguez-Molares *et al.*, "The generalized contrast-to-noise ratio: A formal definition for lesion detectability," *IEEE Trans. Ultrason., Ferroelectr., Freq. Control*, vol. 67, no. 4, pp. 745–759, 2020.

[15] D. M. Pelt and J. A. Sethian, "A mixed-scale dense convolutional neural network for image analysis," *Proc. Natl. Acad. Sci. USA*, vol. 115, no. 2, pp. 254–259, 2018.

[16] S. Mohan, Z. Kadkhodaie, E. P. Simoncelli, and C. Fernandez-Granda, "Robust and interpretable blind image denoising via bias-free convolutional neural networks," in *Proc. ICLR*, 2020.

---

### Reproducibility

The downloader is `download_epfl_ultrasound.py` at the repository root; all other scripts are in `examples/self-supervised-learning/`:

| Step | Script |
|---|---|
| Selective download of EPFL acquisitions (in vivo and `--invitro`) | `download_epfl_ultrasound.py` |
| Per-transmit beamforming and IQ cache (in vivo and phantom) | `precompute_n2i_ultrasound.py` |
| Training (`N2I_ARCH` ∈ unet, resunet, attunet, attresunet, drunet, msd), with group cache and in-vitro test after training | `train_n2i_deepinv.py` |
| Training with the transmits as separate channels (Section V-F), with transmit cache; vertical flips on by default (`N2I_VFLIP=0` disables them, also in the other training scripts) | `train_n2i_multitx_deepinv.py` |
| Training on the mean of any number of transmits (Section V-G), from the transmit cache; vertical flips with `N2I_VFLIP=1` | `train_n2i_anyn_deepinv.py` |
| Round-4 DRUNet on DAS of 1, 3, 11 and 87 transmits | `nangles_n2i_deepinv.py` |
| Three-transmit DRUNet against the any-n networks on 1, 3, 11 and 87 transmits (Tables XII–XIII) | `compare_n2i_anyn_deepinv.py` |
| DAS of 3–29 compounded transmits against the networks, equivalent *N* (Tables XIV–XV, Fig. 4); env `N2I_N_FRAMES` (default all) | `cpwc_curve_n2i_deepinv.py` |
| In-vitro metrics on `invitro_02162` (Table III) | `n2i_invitro.py` |
| Inference with one network | `infer_n2i_deepinv.py` |
| B-mode comparison of all networks and in-vitro table (Figs. 1–3, Tables III and VI); `INPUTS` or the `N2I_INPUTS` environment variable selects mean-input networks, multi-transmit ones or both | `compare_n2i_deepinv.py` |
| In-vivo metrics against the held-out reconstruction, paired tests (Tables I–II and XI); `INPUTS` (env `N2I_INPUTS`) selects mean-input, multi-transmit and vflip networks, env `N2I_RUN_SUFFIX` the split, `N2I_ARCHS` the networks | `eval_n2i_deepinv.py` |
| Phantom homogeneity, input rescaling and input coherence (Tables IV and VI, Sections V-E and VI) | `artifacts_n2i_deepinv.py` |

The evaluation scripts use `RUN_SUFFIX = "_test002"` (round 3); the training script now uses `"_test002_006_008"` (round 4), and `eval_n2i_deepinv.py` and `compare_n2i_deepinv.py` now also score volunteers 006 and 008, each region separately. Tables I–III and Figs. 1–2 were recomputed on 2026-10-26 from the checkpoints listed in Section V-A, all five networks in one run (the U-Net and DRUNet values reproduce those of 2026-10-08 exactly). Section V-D (Table IV) and the measurements of the Discussion come from `artifacts_n2i_deepinv.py` (output in `results/n2i_ultrasound_artifacts_test002/`, log in `results/artifacts_test002.log`). The per-frame metrics are in `results/n2i_ultrasound_metrics_test002/`, the figures in `results/n2i_ultrasound_comparison_test002/`, and the logs in `results/eval_test002_all.log` and `results/compare_test002_all.log` (the 2026-10-08 logs with U-Net and DRUNet only remain as `results/eval_test002.log` and `results/compare_test002.log`). The phantom comparison takes the round-3 split from the `N2I_SPLIT_SUFFIX=_test002` environment variable. Round 4 (Section V-E, `RUN_SUFFIX = "_test002_006_008"`): figures and phantom tables in `results/n2i_ultrasound_comparison_test002_006_008/` and, for the multi-transmit networks, `results/n2i_ultrasound_comparison_multitx_test002_006_008/`; homogeneity, rescaling and coherence in `results/n2i_ultrasound_artifacts_test002_006_008/` (log `results/artifacts_test002_006_008.log`). **[TODO]** The cropping and flipping tests and Table VII come from scratch scripts and `artifacts_n2i_deepinv.py` does not support multi-transmit checkpoints yet; add both to the repository.

Sections V-F and V-G (2026-10-23): Tables XII–XIII and the any-n figures come from `compare_n2i_anyn_deepinv.py`; Table XII was rerun on every test frame on 2026-10-26 with `N2I_N_FRAMES=all` (output in `results/n2i_ultrasound_anyn_vs_mean_test002_006_008_all/`, log `results/anyn_vs_mean_all_test002_006_008.log`; the six-frame run remains in `results/n2i_ultrasound_anyn_vs_mean_test002_006_008/` and `results/anyn_vs_mean_test002_006_008.log`). The phantom metrics of Table XIII do not depend on this setting and are identical in both runs. Tables IX–X and the location of the dark patches come from a scratch script (not in the repository) that evaluates the five DRUNets of Table VIII with plain and flipped input; its log is `results/flip_mtx_anyn_test002_006_008.log`. The coverage test of Section V-F (partly covered region, inter-transmit correlation, FullSpeckle control) is a second scratch script; log `results/edge_coverage_test002_006_008.log`. The B-mode figures in `results/n2i_ultrasound_comparison_multitx_vflip_test002_006_008/` and `results/n2i_ultrasound_comparison_mean_vs_anyn_vflip_test002_006_008/` were regenerated from the final checkpoints on 2026-10-26 (logs `results/compare_multitx_vflip_test002_006_008.log` and `results/compare_mean_vs_anyn_vflip_test002_006_008.log`); their phantom metrics match Tables IX and XIII. **[TODO]** Add the flip test to `artifacts_n2i_deepinv.py`, with multi-transmit support.

Table XI (2026-10-23): `N2I_ARCHS=drunet N2I_INPUTS=mean,transmits,transmits_vflip N2I_RUN_SUFFIX=_test002_006_008 python eval_n2i_deepinv.py`, per-frame metrics in `results/n2i_ultrasound_metrics_mean_vs_multitx_vs_multitx_vflip_test002_006_008/`, log `results/eval_mtx_test002_006_008.log`. Its U-Net rows (2026-10-26): `N2I_ARCHS=unet,drunet N2I_REFERENCE=DRUNet N2I_RUN_SUFFIX=_test002_006_008 python eval_n2i_deepinv.py`, per-frame metrics in `results/n2i_ultrasound_metrics_test002_006_008/`, log `results/eval_r4_test002_006_008.log`; its DRUNet values reproduce those of the 2026-10-23 run exactly.

Section V-H (2026-10-26): `python cpwc_curve_n2i_deepinv.py`, per-folder metrics and `equivalent_n.csv` in `results/n2i_ultrasound_cpwc_curve_test002_006_008/`, log `results/cpwc_curve_test002_006_008.log`. Checkpoints with the suffix `_sup` (supervised, `N2I_TARGET=cpwc`) are added automatically when present.
