"""Weights & Biases experiment logger tailored to project BTL_DL_2026."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import wandb


class WandbLogger:
    """Wrapper around wandb for logging training, validation, test metrics, and mask overlays."""

    def __init__(
        self,
        entity: str = "noct2306-uet",
        project: str = "BTL_DL_2026",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
        mode: str | None = None,
    ) -> None:
        self.enabled = enabled and (mode != "disabled")
        self.run = None

        if not self.enabled:
            return

        init_kwargs: dict[str, Any] = {
            "entity": entity,
            "project": project,
            "name": run_name,
            "config": config or {},
            "tags": tags or [],
        }
        if mode is not None:
            init_kwargs["mode"] = mode

        self.run = wandb.init(**init_kwargs)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics for the current epoch or step."""
        if not self.enabled or self.run is None:
            return
        wandb.log(metrics, step=step)

    def log_best_val(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log best validation metrics under the `best_val/` namespace and run summary."""
        if not self.enabled or self.run is None:
            return
        prefixed = {
            (k if k.startswith("best_val/") else f"best_val/{k}"): float(v)
            for k, v in metrics.items()
        }
        wandb.log(prefixed, step=step)
        for k, v in prefixed.items():
            self.run.summary[k] = v

    def log_test_metrics(self, metrics: dict[str, float]) -> None:
        """Log test evaluation metrics under `test/` (loss, dice, iou, precision, recall, accuracy)."""
        if not self.enabled or self.run is None:
            return
        prefixed = {
            (k if k.startswith("test/") else f"test/{k}"): float(v)
            for k, v in metrics.items()
        }
        wandb.log(prefixed)
        for k, v in prefixed.items():
            self.run.summary[k] = v

    def log_predictions(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        step: int | None = None,
        max_samples: int = 4,
        key: str = "plots/predictions",
    ) -> None:
        """Log segmentation mask overlays (ground truth vs prediction) to W&B."""
        if not self.enabled or self.run is None:
            return

        n = min(images.shape[0], max_samples)
        wandb_images: list[wandb.Image] = []

        # ImageNet denormalization stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(3, 1, 1)

        for idx in range(n):
            img = (images[idx] * std + mean).clamp(0.0, 1.0)
            img_np = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

            gt_np = (masks[idx].squeeze(0).detach().cpu().numpy() > 0.5).astype(np.uint8)
            pred_np = (preds[idx].squeeze(0).detach().cpu().numpy() > 0.5).astype(np.uint8)

            wandb_images.append(
                wandb.Image(
                    img_np,
                    masks={
                        "ground_truth": {
                            "mask_data": gt_np,
                            "class_labels": {0: "background", 1: "polyp"},
                        },
                        "prediction": {
                            "mask_data": pred_np,
                            "class_labels": {0: "background", 1: "polyp"},
                        },
                    },
                )
            )

        wandb.log({key: wandb_images}, step=step)

    def finish(self) -> None:
        """Finish the active W&B run."""
        if self.enabled and self.run is not None:
            wandb.finish()
            self.run = None
