"""Evaluation engine for segmentation models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics.segmentation_metrics import (
    compute_confusion_counts,
    metrics_from_counts,
)


class Evaluator:
    """Computes loss and dataset-level segmentation metrics over a DataLoader."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(
        self, dataloader: DataLoader
    ) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        """Evaluate model on dataloader, returning (metrics_dict, sample_batch)."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        tp = fp = fn = tn = 0.0
        sample_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            b_tp, b_fp, b_fn, b_tn = compute_confusion_counts(
                logits, masks, threshold=self.threshold
            )
            tp += b_tp
            fp += b_fp
            fn += b_fn
            tn += b_tn

            if sample_batch is None:
                preds = torch.sigmoid(logits)
                sample_batch = (images.cpu(), masks.cpu(), preds.cpu())

        avg_loss = total_loss / max(total_samples, 1)
        metrics = metrics_from_counts(tp, fp, fn, tn)
        metrics["loss"] = avg_loss
        return metrics, sample_batch
