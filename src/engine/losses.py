"""Loss functions for binary medical image segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss computed over raw logits."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted combination of BCEWithLogitsLoss and Soft Dice Loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )
