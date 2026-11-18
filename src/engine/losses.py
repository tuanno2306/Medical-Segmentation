import torch
import torch.nn as nn


class BCETverskyLoss(nn.Module):
    """Weighted BCE and soft Tversky loss, averaged over images.

    Alpha weights false positives; beta weights false negatives. Inputs are
    logits and binary targets with shape (N, 1, H, W).
    """

    def __init__(self, alpha=0.3, beta=0.7, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta == 0:
            raise ValueError("alpha and beta must be nonnegative and not both zero")
        if not 0 <= bce_weight <= 1 or smooth <= 0:
            raise ValueError("bce_weight must be in [0, 1] and smooth must be positive")
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("logits and targets must have identical (N, 1, H, W) shapes")
        probabilities = torch.sigmoid(logits)
        dims = (1, 2, 3)
        true_positive = (probabilities * targets).sum(dim=dims)
        false_positive = (probabilities * (1 - targets)).sum(dim=dims)
        false_negative = ((1 - probabilities) * targets).sum(dim=dims)
        tversky = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive
            + self.beta * false_negative + self.smooth
        )
        return self.bce_weight * self.bce(logits, targets) + (
            1 - self.bce_weight
        ) * (1 - tversky).mean()


def build_loss(name="bce_tversky", **kwargs):
    """Build the default BCE + Tversky loss or the BCE-only baseline."""
    if name == "bce_tversky":
        return BCETverskyLoss(**kwargs)
    if name == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    raise ValueError(f"Unknown loss: {name}")
