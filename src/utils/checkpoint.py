"""Checkpoint saving and loading helpers."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    path: str | Path,
) -> Path:
    """Save model and optimizer state to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        target,
    )
    return target


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model (and optionally optimizer) state from disk."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
