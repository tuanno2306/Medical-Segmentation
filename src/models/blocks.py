"""Reusable building blocks for U-Net and Pretrained U-Net architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double 3x3 convolution block with BatchNorm, ReLU, and optional Dropout2d."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """U-Net decoder stage: upsample, align & concatenate skip connection, then ConvBlock."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        up_channels = in_channels // 2
        if bilinear:
            self.up: nn.Module = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, up_channels, kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, up_channels, kernel_size=2, stride=2
            )

        self.conv = ConvBlock(
            up_channels + skip_channels, out_channels, dropout=dropout
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
