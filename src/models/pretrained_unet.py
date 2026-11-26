"""U-Net architecture with a pretrained ResNet34 encoder."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

from src.models.blocks import ConvBlock, DecoderBlock


class ResNet34UNet(nn.Module):
    """U-Net segmentation model backed by a ResNet34 encoder.

    Encoder stages (for input H x W):
        - stage0 (conv1 + bn1 + relu): (B, 64, H/2, W/2)
        - stage1 (maxpool + layer1):   (B, 64, H/4, W/4)
        - stage2 (layer2):             (B, 128, H/8, W/8)
        - stage3 (layer3):             (B, 256, H/16, W/16)
        - stage4 (layer4):             (B, 512, H/32, W/32)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        if in_channels != 3:
            backbone.conv1 = self._adapt_first_conv(
                backbone.conv1, in_channels, pretrained=pretrained
            )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.decoder4 = DecoderBlock(512, 256, 256, bilinear=bilinear, dropout=dropout)
        self.decoder3 = DecoderBlock(256, 128, 128, bilinear=bilinear, dropout=dropout)
        self.decoder2 = DecoderBlock(128, 64, 64, bilinear=bilinear, dropout=dropout)
        self.decoder1 = DecoderBlock(64, 64, 32, bilinear=bilinear, dropout=dropout)

        self.final_conv = ConvBlock(32, 32, dropout=dropout)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        if freeze_encoder:
            self.freeze_encoder()

    @staticmethod
    def _adapt_first_conv(
        old_conv: nn.Conv2d, in_channels: int, pretrained: bool
    ) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        return new_conv

    def _encoder_modules(self) -> tuple[nn.Module, ...]:
        return (self.stem, self.encoder1, self.encoder2, self.encoder3, self.encoder4)

    def _decoder_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.final_conv,
            self.head,
        )

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters so only the decoder is updated."""
        for module in self._encoder_modules():
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters for full fine-tuning."""
        for module in self._encoder_modules():
            for param in module.parameters():
                param.requires_grad = True

    def get_param_groups(
        self, encoder_lr: float, decoder_lr: float
    ) -> list[dict[str, Any]]:
        """Return optimizer parameter groups with differential learning rates."""
        encoder_params = [
            p
            for m in self._encoder_modules()
            for p in m.parameters()
            if p.requires_grad
        ]
        decoder_params = [
            p
            for m in self._decoder_modules()
            for p in m.parameters()
            if p.requires_grad
        ]
        groups: list[dict[str, Any]] = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        if decoder_params:
            groups.append({"params": decoder_params, "lr": decoder_lr})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        s0 = self.stem(x)
        s1 = self.encoder1(self.maxpool(s0))
        s2 = self.encoder2(s1)
        s3 = self.encoder3(s2)
        s4 = self.encoder4(s3)

        d4 = self.decoder4(s4, s3)
        d3 = self.decoder3(d4, s2)
        d2 = self.decoder2(d3, s1)
        d1 = self.decoder1(d2, s0)

        out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        out = self.final_conv(out)
        return self.head(out)
