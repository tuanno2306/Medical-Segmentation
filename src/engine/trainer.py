"""Training engine with W&B metric logging and best-checkpoint tracking."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.evaluator import Evaluator
from src.metrics.segmentation_metrics import (
    compute_confusion_counts,
    metrics_from_counts,
)
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import WandbLogger


class Trainer:
    """Runs epoch-based training and validation with optional W&B logging."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        logger: WandbLogger | None = None,
        checkpoint_dir: str | Path = "outputs/checkpoints",
        log_image_every_n_epochs: int = 5,
        early_stopping_patience: int | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.evaluator = Evaluator(model=model, criterion=criterion, device=device)

    def _train_one_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        tp = fp = fn = tn = 0.0

        for images, masks in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            with torch.no_grad():
                b_tp, b_fp, b_fn, b_tn = compute_confusion_counts(logits, masks)
                tp += b_tp
                fp += b_fp
                fn += b_fn
                tn += b_tn

        metrics = metrics_from_counts(tp, fp, fn, tn)
        metrics["loss"] = total_loss / max(total_samples, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Path:
        """Train for `epochs`, returning the path to the best validation checkpoint."""
        best_val_dice = -1.0
        epochs_without_improvement = 0
        best_ckpt_path = self.checkpoint_dir / "best_model.pth"

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(train_loader)
            val_metrics, val_samples = self.evaluator.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[-1]["lr"]
            log_payload = {
                "epoch": float(epoch),
                "lr": float(current_lr),
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
            if self.logger is not None:
                self.logger.log_metrics(log_payload, step=epoch)

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                epochs_without_improvement = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    best_score=best_val_dice,
                    path=best_ckpt_path,
                )
                if self.logger is not None:
                    self.logger.log_best_val(
                        {"dice": best_val_dice, "loss": val_metrics["loss"]},
                        step=epoch,
                    )
            else:
                epochs_without_improvement += 1

            if (
                self.logger is not None
                and val_samples is not None
                and self.log_image_every_n_epochs > 0
                and (epoch % self.log_image_every_n_epochs == 0 or epoch == epochs)
            ):
                imgs, msks, preds = val_samples
                self.logger.log_predictions(imgs, msks, preds, step=epoch)

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f}"
            )

            if (
                self.early_stopping_patience is not None
                and self.early_stopping_patience > 0
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no val_dice improvement for {self.early_stopping_patience} epochs)."
                )
                break

        return best_ckpt_path
