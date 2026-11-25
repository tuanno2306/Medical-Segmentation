import torch

from src.metrics.segmentation_metrics import segmentation_scores


def run_epoch(model, loader, criterion, device, optimizer=None, threshold=0.5):
    """Train if optimizer is provided, otherwise validate; average per image."""
    training = optimizer is not None
    model.train(training)
    loss_sum = dice_sum = iou_sum = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            dice, iou = segmentation_scores(logits.detach(), masks, threshold)
            batch_size = images.size(0)
            loss_sum += loss.item() * batch_size
            dice_sum += dice.sum().item()
            iou_sum += iou.sum().item()
            count += batch_size
    if not count:
        raise ValueError("Dataset is empty")
    return {"loss": loss_sum / count, "dice": dice_sum / count, "iou": iou_sum / count}
