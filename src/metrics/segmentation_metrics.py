import torch


@torch.no_grad()
def segmentation_scores(logits, targets, threshold=0.5):
    """Return per-image Dice and IoU for binary masks in (N, 1, H, W).

    A prediction and target that are both empty receive a score of 1.
    Aggregate the returned scores over images, not over batches.
    """
    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits and targets must have identical (N, 1, H, W) shapes")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    predictions = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    dims = (1, 2, 3)
    intersection = (predictions & truth).sum(dim=dims).float()
    predicted_area = predictions.sum(dim=dims).float()
    target_area = truth.sum(dim=dims).float()
    total = predicted_area + target_area
    union = total - intersection
    dice = torch.where(total > 0, 2 * intersection / total.clamp_min(1), 1.0)
    iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
    return dice, iou
