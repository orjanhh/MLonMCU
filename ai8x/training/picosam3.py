###################################################################################################
# PicoSAM3 - Tiny U-Net for binary segmentation on MAX78000
#
# Architecture following L^3U-net paper (Okman et al. 2022):
#   - Data folding (alpha=2) for parallel processor utilization
#   - Concatenation for skip connections (not eltwise add)
#   - Channel counts kept <= 64 throughout for clean MAX78000 deployment
#
# Input:  3 x 80 x 80
# After fold:  12 x 40 x 40
# Output: 1 x 20 x 20  binary mask
###################################################################################################

import torch
from torch import nn
import ai8x


class PicoSAM3(nn.Module):
    def __init__(
        self,
        num_classes=1,
        dimensions=(80, 80),
        num_channels=3,
        bias=True,
        fold_factor=2,
        **kwargs,
    ):
        super().__init__()
        self.fold_factor = fold_factor
        in_ch = num_channels * fold_factor * fold_factor  # 12 channels post-fold

        # =====================================================================
        # Encoder
        # =====================================================================

        # 12x40x40 -> 16x40x40
        self.enc1_conv = ai8x.FusedConv2dBNReLU(
            in_channels=in_ch, out_channels=16, kernel_size=3,
            padding=1, stride=1, bias=bias, batchnorm="NoAffine", **kwargs,
        )
        # 16x40x40 -> 32x20x20
        self.enc2_pool_conv = ai8x.FusedMaxPoolConv2dBNReLU(
            in_channels=16, out_channels=32, kernel_size=3,
            padding=1, pool_size=2, pool_stride=2, bias=bias,
            batchnorm="NoAffine", **kwargs,
        )
        # 32x20x20 -> 32x10x10  (reduced from 48 to fit concat within 64)
        self.enc3_pool_conv = ai8x.FusedMaxPoolConv2dBNReLU(
            in_channels=32, out_channels=32, kernel_size=3,
            padding=1, pool_size=2, pool_stride=2, bias=bias,
            batchnorm="NoAffine", **kwargs,
        )

        # =====================================================================
        # Bottleneck
        # =====================================================================
        # 32x10x10 -> 48x5x5  (reduced from 64)
        self.bottleneck_pool_conv = ai8x.FusedMaxPoolConv2dBNReLU(
            in_channels=32, out_channels=48, kernel_size=3,
            padding=1, pool_size=2, pool_stride=2, bias=bias,
            batchnorm="NoAffine", **kwargs,
        )

        # =====================================================================
        # Decoder with CONCATENATION skip connections (paper-style)
        # =====================================================================

        # 48x5x5 -> 32x10x10
        self.dec3_up = ai8x.ConvTranspose2d(
            in_channels=48, out_channels=32, kernel_size=3,
            stride=2, padding=1, bias=bias, **kwargs,
        )
        # After concat with enc3 (32 ch): 64-channel input
        # 64x10x10 -> 32x10x10
        self.dec3_conv = ai8x.FusedConv2dBNReLU(
            in_channels=64, out_channels=32, kernel_size=3,
            padding=1, stride=1, bias=bias, batchnorm="NoAffine", **kwargs,
        )

        # 32x10x10 -> 32x20x20
        self.dec2_up = ai8x.ConvTranspose2d(
            in_channels=32, out_channels=32, kernel_size=3,
            stride=2, padding=1, bias=bias, **kwargs,
        )
        # After concat with enc2 (32 ch): 64-channel input
        # 64x20x20 -> 32x20x20
        self.dec2_conv = ai8x.FusedConv2dBNReLU(
            in_channels=64, out_channels=32, kernel_size=3,
            padding=1, stride=1, bias=bias, batchnorm="NoAffine", **kwargs,
        )

        # =====================================================================
        # Output head: 32x20x20 -> 1x20x20
        # =====================================================================
        self.output_conv = ai8x.Conv2d(
            in_channels=32, out_channels=num_classes, kernel_size=1,
            padding=0, stride=1, bias=True, wide=True, **kwargs,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def fold_input(self, x):
        """Spatial -> channel folding. (B, C, H, W) -> (B, C*f^2, H/f, W/f)"""
        f = self.fold_factor
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // f, f, W // f, f)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * f * f, H // f, W // f)
        return x

    def forward(self, x):
        # Fold: (B, 3, 80, 80) -> (B, 12, 40, 40)
        x = self.fold_input(x)

        # Encoder
        enc1 = self.enc1_conv(x)            # (B, 16, 40, 40)
        enc2 = self.enc2_pool_conv(enc1)    # (B, 32, 20, 20)
        enc3 = self.enc3_pool_conv(enc2)    # (B, 32, 10, 10)

        # Bottleneck
        bot = self.bottleneck_pool_conv(enc3)  # (B, 48, 5, 5)

        # Decoder with concat skips (paper-style)
        d3 = self.dec3_up(bot)              # (B, 32, 10, 10)
        d3 = torch.cat([d3, enc3], dim=1)   # (B, 64, 10, 10)
        d3 = self.dec3_conv(d3)             # (B, 32, 10, 10)

        d2 = self.dec2_up(d3)               # (B, 32, 20, 20)
        d2 = torch.cat([d2, enc2], dim=1)   # (B, 64, 20, 20)
        d2 = self.dec2_conv(d2)             # (B, 32, 20, 20)

        # Output
        out = self.output_conv(d2)          # (B, 1, 20, 20)
        return out


def picosam3(pretrained=False, **kwargs):
    assert not pretrained
    kwargs['num_classes'] = 1
    return PicoSAM3(**kwargs)


models = [
    {"name": "picosam3", "min_input": 1, "dim": 2}
]