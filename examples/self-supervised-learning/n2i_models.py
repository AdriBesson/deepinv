r"""
U-Net variants for ``train_n2i_deepinv.py``
==========================================

Ported unchanged from ``playground/users/adrien/ai_for_image_quality/model.py``
so that the Noise2Inverse results here are comparable with the earlier ones:

- :class:`ResUNet`: a 5-scale U-Net whose blocks are all residual
  (Conv-BN-ReLU-Conv-BN with a 1x1 shortcut, ReLU after the sum), max-pool
  down, bilinear-upsample + conv up, concatenating skip connections.
- :class:`AttentionUNet`: the plain double-conv U-Net with the attention gates
  of :footcite:t:`oktay2018attention` on the skip connections: the decoder
  feature at each scale gates the encoder feature before concatenation.
- :class:`AttentionResUNet`: residual blocks and attention gates together.

- :class:`MSDNet`: the mixed-scale dense network of :footcite:t:`pelt2018mixed`,
  the network of the Noise2Inverse paper :footcite:t:`hendriksen2020noise2inverse`
  (not part of the ported file).

They differ from :class:`deepinv.models.DRUNet` (4 scales, several Conv-ReLU-Conv
blocks per scale without batch norm, strided/transposed convolutions, additive
skips, noise-level channel), which the training script also offers.

The only change from the original file is that ``forward`` accepts and ignores
the ``sigma`` argument that :class:`deepinv.models.ArtifactRemoval` passes to
its backbone.

- :class:`PaddedArtifactRemoval`: the wrapper the training, evaluation and
  inference scripts put around every backbone (padding, and optionally exact
  scale equivariance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import deepinv as dinv


def pad_to_multiple(x, multiple=32):
    """Reflect-pad a ``(B, C, Z, X)`` batch up to a multiple of ``multiple``, for the U-Nets."""
    z, w = x.shape[-2:]
    pad_z, pad_w = (multiple - z % multiple) % multiple, (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_z), mode="reflect")


class PaddedArtifactRemoval(dinv.models.ArtifactRemoval):
    """:class:`deepinv.models.ArtifactRemoval` that pads its input and crops its output.

    The U-Nets need sizes divisible by 32, so the 468x234 frames are
    reflect-padded to 480x256 inside the model and the output is cropped back.
    The loss then only sees the real frame, not the 11% of mirrored pixels, and
    the trainer's validation loss has the definition of eval_n2i_deepinv.py. It
    holds the same parameters as ArtifactRemoval, so the checkpoints load in
    either.

    With ``scale_equivariant``, each input is divided by its own RMS before the
    backbone and the output multiplied back, so that ``f(a y) = a f(y)`` exactly
    for any backbone and any ``a > 0``: the network only sees relative
    brightness. Bias-free backbones would give the same property but not with
    batch norm or the sigmoid of the attention gates; for DRUNet it amounts to a
    noise-level channel proportional to the input level. It has no parameters,
    so the flag is stored next to the weights in the checkpoint
    (``checkpoint["scale_equivariant"]``) and must be set when loading it.
    """

    def __init__(self, *args, scale_equivariant=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_equivariant = scale_equivariant

    def forward(self, y, physics, **kwargs):
        z, x = y.shape[-2:]
        s = 1.0
        if self.scale_equivariant:
            s = y.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
        return super().forward(pad_to_multiple(y / s), physics, **kwargs)[..., :z, :x] * s


class ConvBlock(nn.Module):
    """
    Standard Double Conv Block: (Conv -> BN -> ReLU) x 2
    """

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=3,
                      padding=1,
                      padding_mode='reflect'), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch,
                      out_ch,
                      kernel_size=3,
                      padding=1,
                      padding_mode='reflect'), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """
    Upscaling Block
    
    IMPROVEMENTS:
    - Uses bilinear interpolation + Conv instead of ConvTranspose2d (optional).
    - Eliminates 'checkerboard' artifacts in the reconstruction.
    """

    def __init__(self, in_ch, out_ch, upsample_mode='bilinear'):
        super(UpConv, self).__init__()
        if upsample_mode == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode='bilinear',
                            align_corners=True),
                nn.Conv2d(in_ch,
                          out_ch,
                          kernel_size=3,
                          padding=1,
                          padding_mode='reflect'), nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False))
        elif upsample_mode == 'conv_transpose':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False))
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

    def forward(self, x):
        return self.up(x)


class AttentionGate(nn.Module):
    """
    Attention Gate:
    Filters the skip connection (x) using the gating signal (g) from the decoder.
    This suppresses noise in irrelevant regions (e.g. anechoic cysts).
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # W_g: Transform gating signal (from deep decoder layer)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        # W_x: Transform skip connection (from shallow encoder layer)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        # psi: Combine and compute attention map (0 to 1)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Add them (broadcasting handles size mismatch if standard U-Net shapes are used)
        psi = self.relu(g1 + x1)

        # Get attention coefficients (Batch, 1, H, W)
        psi = self.psi(psi)

        # Multiply original skip connection by the attention map
        return x * psi


class AttentionUNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=2,
                 base_channels=64,
                 upsample_mode='bilinear'):
        """
        Args:
            in_channels (int): Number of input channels (e.g. 6, 12, 24).
            out_channels (int): Number of output channels (usually 2 for Real/Imag).
            base_channels (int): Filters in the first layer. 64 is standard, 32 is faster.
            upsample_mode (str): 'bilinear' or 'conv_transpose'.
        """
        super(AttentionUNet, self).__init__()

        # Define filter sizes based on the base
        f1 = base_channels  # 64
        f2 = base_channels * 2  # 128
        f3 = base_channels * 4  # 256
        f4 = base_channels * 8  # 512
        f5 = base_channels * 16  # 1024

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, f1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(f1, f2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(f2, f3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(f3, f4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(f4, f5)

        # --- Decoder with Attention ---

        # Up 4
        self.up4 = UpConv(f5, f4, upsample_mode=upsample_mode)
        self.att4 = AttentionGate(F_g=f4, F_l=f4, F_int=f3)
        self.dec4 = ConvBlock(
            f5, f4)  # After cat: f4(skip) + f4(up) -> f4 features

        # Up 3
        self.up3 = UpConv(f4, f3, upsample_mode=upsample_mode)
        self.att3 = AttentionGate(F_g=f3, F_l=f3, F_int=f2)
        self.dec3 = ConvBlock(f4, f3)

        # Up 2
        self.up2 = UpConv(f3, f2, upsample_mode=upsample_mode)
        self.att2 = AttentionGate(F_g=f2, F_l=f2, F_int=f1)
        self.dec2 = ConvBlock(f3, f2)

        # Up 1
        self.up1 = UpConv(f2, f1, upsample_mode=upsample_mode)
        self.att1 = AttentionGate(F_g=f1, F_l=f1, F_int=f1 // 2)
        self.dec1 = ConvBlock(f2, f1)

        # --- Final Output ---
        self.final = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x, sigma=None, **kwargs):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder

        # Block 4
        u4 = self.up4(b)
        x4 = self.att4(g=u4, x=e4)
        d4 = self.dec4(torch.cat((x4, u4), dim=1))

        # Block 3
        u3 = self.up3(d4)
        x3 = self.att3(g=u3, x=e3)
        d3 = self.dec3(torch.cat((x3, u3), dim=1))

        # Block 2
        u2 = self.up2(d3)
        x2 = self.att2(g=u2, x=e2)
        d2 = self.dec2(torch.cat((x2, u2), dim=1))

        # Block 1
        u1 = self.up1(d2)
        x1 = self.att1(g=u1, x=e1)
        d1 = self.dec1(torch.cat((x1, u1), dim=1))

        out = self.final(d1)
        return out




class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection. 
    If input and output channels differ, applies a 1x1 conv to the shortcut.
    """

    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size=3,
                      padding=1,
                      padding_mode='reflect'), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch,
                      out_ch,
                      kernel_size=3,
                      padding=1,
                      padding_mode='reflect'), nn.BatchNorm2d(out_ch))

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_ch))

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # f(x) + x
        return self.relu(self.conv_block(x) + self.shortcut(x))


class ResUNet(nn.Module):
    """
    A U-Net architecture built entirely out of Residual Blocks.
    Often converges faster and handles deeper architectures better than a standard U-Net.
    """

    def __init__(self,
                 in_channels,
                 out_channels=2,
                 base_channels=64,
                 upsample_mode='bilinear'):
        super(ResUNet, self).__init__()

        f1, f2, f3, f4, f5 = [base_channels * (2**i) for i in range(5)]

        # --- Encoder ---
        self.enc1 = ResidualBlock(in_channels, f1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ResidualBlock(f1, f2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ResidualBlock(f2, f3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ResidualBlock(f3, f4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(f4, f5)

        # --- Decoder ---
        # Note: After concatenation, the channel size is (Deep_Channels + Shallow_Channels)
        self.up4 = UpConv(f5, f4, upsample_mode=upsample_mode)
        self.dec4 = ResidualBlock(
            f5, f4)  # f4(skip) + f4(up) = f5 -> conv down to f4

        self.up3 = UpConv(f4, f3, upsample_mode=upsample_mode)
        self.dec3 = ResidualBlock(
            f4, f3)  # f3(skip) + f3(up) = f4 -> conv down to f3

        self.up2 = UpConv(f3, f2, upsample_mode=upsample_mode)
        self.dec2 = ResidualBlock(
            f3, f2)  # f2(skip) + f2(up) = f3 -> conv down to f2

        self.up1 = UpConv(f2, f1, upsample_mode=upsample_mode)
        self.dec1 = ResidualBlock(
            f2, f1)  # f1(skip) + f1(up) = f2 -> conv down to f1

        # --- Final Output ---
        self.final = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x, sigma=None, **kwargs):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.dec4(torch.cat((self.up4(b), e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))

        return self.final(d1)



class AttentionResUNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=2,
                 base_channels=64,
                 upsample_mode='bilinear'):
        super(AttentionResUNet, self).__init__()

        f = [base_channels * (2**i)
             for i in range(5)]  # [64, 128, 256, 512, 1024]

        # --- Encoder (Using Residual Blocks) ---
        self.enc1 = ResidualBlock(in_channels, f[0])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(f[0], f[1])
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(f[1], f[2])
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualBlock(f[2], f[3])
        self.pool4 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(f[3], f[4])

        # --- Decoder with Attention Gates ---
        self.up4 = UpConv(f[4], f[3], upsample_mode=upsample_mode)
        self.att4 = AttentionGate(F_g=f[3], F_l=f[3], F_int=f[2])
        self.dec4 = ResidualBlock(
            f[4], f[3])  # Cat(att, up) = f3+f3 = f4 -> output f3

        self.up3 = UpConv(f[3], f[2], upsample_mode=upsample_mode)
        self.att3 = AttentionGate(F_g=f[2], F_l=f[2], F_int=f[1])
        self.dec3 = ResidualBlock(f[3], f[2])

        self.up2 = UpConv(f[2], f[1], upsample_mode=upsample_mode)
        self.att2 = AttentionGate(F_g=f[1], F_l=f[1], F_int=f[0])
        self.dec2 = ResidualBlock(f[2], f[1])

        self.up1 = UpConv(f[1], f[0], upsample_mode=upsample_mode)
        self.att1 = AttentionGate(F_g=f[0], F_l=f[0], F_int=f[0] // 2)
        self.dec1 = ResidualBlock(f[1], f[0])

        self.final = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x, sigma=None, **kwargs):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with Gated Skip Connections
        u4 = self.up4(b)
        x4 = self.att4(g=u4, x=e4)
        d4 = self.dec4(torch.cat((x4, u4), dim=1))

        u3 = self.up3(d4)
        x3 = self.att3(g=u3, x=e3)
        d3 = self.dec3(torch.cat((x3, u3), dim=1))

        u2 = self.up2(d3)
        x2 = self.att2(g=u2, x=e2)
        d2 = self.dec2(torch.cat((x2, u2), dim=1))

        u1 = self.up1(d2)
        x1 = self.att1(g=u1, x=e1)
        d1 = self.dec1(torch.cat((x1, u1), dim=1))

        return self.final(d1)


def _msd_layer(conv, *features):
    """One MS-D layer: dilated 3x3 convolution of all earlier feature maps, then ReLU."""
    return F.relu(conv(torch.cat(features, dim=1)))


class MSDNet(nn.Module):
    r"""Mixed-scale dense network :footcite:t:`pelt2018mixed`.

    No down-sampling: ``depth`` layers of ``width`` channels at full resolution,
    each a 3x3 convolution of the input and of *every* earlier layer, dilated by
    ``1, 2, ..., max_dilation, 1, 2, ...`` so that the scales mix throughout the
    network. A final 1x1 convolution maps all the feature maps to the output.
    With the defaults of the Noise2Inverse paper (100 layers of width 1,
    dilations 1-10) it has about 47k parameters.

    Concatenating the earlier maps at every layer would store a copy of all of
    them per layer for the backward pass, quadratic in ``depth`` (about 17 GB for
    a batch of eight 480x256 frames). With ``checkpointing`` each layer keeps only
    its own output and redoes the concatenation in the backward pass instead,
    so the memory is linear in ``depth`` at the cost of a second forward pass.

    Weights follow the paper: zero-mean normal with variance ``2 / fan_in``, zero
    biases.
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        depth=100,
        width=1,
        max_dilation=10,
        checkpointing=True,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        self.layers = nn.ModuleList()
        for i in range(depth):
            d = i % max_dilation + 1
            conv = nn.Conv2d(
                in_channels + i * width,
                width,
                kernel_size=3,
                padding=d,
                dilation=d,
                padding_mode="reflect",
            )
            nn.init.normal_(conv.weight, std=(2 / (9 * conv.in_channels)) ** 0.5)
            nn.init.zeros_(conv.bias)
            self.layers.append(conv)
        self.final = nn.Conv2d(in_channels + depth * width, out_channels, kernel_size=1)

    def forward(self, x, sigma=None, **kwargs):
        features = [x]
        for conv in self.layers:
            if self.checkpointing and torch.is_grad_enabled():
                f = checkpoint(_msd_layer, conv, *features, use_reentrant=False)
            else:
                f = _msd_layer(conv, *features)
            features.append(f)
        return self.final(torch.cat(features, dim=1))

