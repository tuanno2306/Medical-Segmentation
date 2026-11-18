import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DoubleConv


class UNet(nn.Module):
    """U-Net for binary segmentation; returns one logit per pixel."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        self.output = nn.Conv2d(c, out_channels, kernel_size=1)

    @staticmethod
    def _merge(upsampled, skip):
        # Max pooling can round odd spatial dimensions down.
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat((skip, upsampled), dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.dec4(self._merge(self.up4(x), e4))
        x = self.dec3(self._merge(self.up3(x), e3))
        x = self.dec2(self._merge(self.up2(x), e2))
        x = self.dec1(self._merge(self.up1(x), e1))
        return self.output(x)
