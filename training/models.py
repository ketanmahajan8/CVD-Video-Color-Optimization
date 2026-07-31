# D:\CVD_GAN\training\models.py
from __future__ import annotations
import torch
import torch.nn as nn


# ---------------------------
# Helpers
# ---------------------------
def conv_norm_lrelu(in_c, out_c, k=4, s=2, p=1, norm=True):
    layers = [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def deconv_norm_relu(in_c, out_c, k=4, s=2, p=1, norm=True, dropout=0.0):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=True))
    layers.append(nn.ReLU(inplace=True))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------
# Generator: UNet (pix2pix)
# Input:  (B, 6, 256, 256) -> RGB + one-hot type maps
# Output: (B, 3, 256, 256) -> enhanced/precompensated RGB in [0,1] via sigmoid
# ---------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, base=64):
        super().__init__()

        # Encoder
        self.d1 = conv_norm_lrelu(in_channels, base, norm=False)      # 128
        self.d2 = conv_norm_lrelu(base, base * 2)                     # 64
        self.d3 = conv_norm_lrelu(base * 2, base * 4)                 # 32
        self.d4 = conv_norm_lrelu(base * 4, base * 8)                 # 16
        self.d5 = conv_norm_lrelu(base * 8, base * 8)                 # 8
        self.d6 = conv_norm_lrelu(base * 8, base * 8)                 # 4
        self.d7 = conv_norm_lrelu(base * 8, base * 8)                 # 2
        self.d8 = conv_norm_lrelu(base * 8, base * 8, norm=False)     # 1

        # Decoder
        self.u1 = deconv_norm_relu(base * 8, base * 8, dropout=0.5)   # 2
        self.u2 = deconv_norm_relu(base * 16, base * 8, dropout=0.5)  # 4
        self.u3 = deconv_norm_relu(base * 16, base * 8, dropout=0.5)  # 8
        self.u4 = deconv_norm_relu(base * 16, base * 8)               # 16
        self.u5 = deconv_norm_relu(base * 16, base * 4)               # 32
        self.u6 = deconv_norm_relu(base * 8, base * 2)                # 64
        self.u7 = deconv_norm_relu(base * 4, base)                    # 128

        self.u8 = nn.ConvTranspose2d(base * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.out_act = nn.Sigmoid()  # keep in [0,1]

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))

        u8 = self.u8(torch.cat([u7, d1], dim=1))
        return self.out_act(u8)


# ---------------------------
# Discriminator: PatchGAN (pix2pix)
# D takes concat([x_cond, y]) where x_cond is (RGB+type maps), y is RGB
# Total channels = 6 + 3 = 9
# ---------------------------
class PatchDiscriminator(nn.Module):
    # CHANGED DEFAULT: in_channels=9 (matches the actual concat in forward)
    def __init__(self, in_channels=9, base=64):
        super().__init__()
        self.c1 = conv_norm_lrelu(in_channels, base, norm=False)          # 128
        self.c2 = conv_norm_lrelu(base, base * 2)                         # 64
        self.c3 = conv_norm_lrelu(base * 2, base * 4)                     # 32
        self.c4 = conv_norm_lrelu(base * 4, base * 8, s=1)                # 31
        self.out = nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x_cond, y):
        inp = torch.cat([x_cond, y], dim=1)  # (B,9,H,W)

        # Safety: crash with clear message if channels mismatch
        expected = self.c1[0].in_channels
        got = inp.shape[1]
        if got != expected:
            raise RuntimeError(f"PatchDiscriminator channel mismatch: expected {expected}, got {got}")

        h = self.c1(inp)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        return self.out(h)