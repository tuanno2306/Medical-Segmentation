"""Binary segmentation metrics (Dice, IoU, Precision, Recall, Accuracy)."""

import torch


def compute_confusion_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    """Compute TP, FP, FN, TN counts from raw logits and binary target masks."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets_bin = (targets >= 0.5).float()

    tp = float((preds * targets_bin).sum().item())
    fp = float((preds * (1.0 - targets_bin)).sum().item())
    fn = float(((1.0 - preds) * targets_bin).sum().item())
    tn = float(((1.0 - preds) * (1.0 - targets_bin)).sum().item())
    return tp, fp, fn, tn


def metrics_from_counts(
    tp: float, fp: float, fn: float, tn: float, eps: float = 1e-7
) -> dict[str, float]:
    """Compute segmentation metrics from confusion counts."""
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Compute Dice, IoU, Precision, Recall, and Accuracy for a batch."""
    tp, fp, fn, tn = compute_confusion_counts(logits, targets, threshold=threshold)
    return metrics_from_counts(tp, fp, fn, tn, eps=eps)
