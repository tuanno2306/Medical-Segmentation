"""Training entrypoint for ResNet34-UNet with Weights & Biases logging."""

import argparse
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.datasets.polyp_dataset import PolypDataset
from src.datasets.split_dataset import split_image_mask_pairs
from src.datasets.transform import SegmentationTransform
from src.engine.evaluator import Evaluator
from src.engine.losses import build_loss
from src.engine.trainer import Trainer
from src.models import ResNet34UNet
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import WandbLogger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet34-UNet with W&B logging")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet34_unet.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--run-name", type=str, default=None, help="Override W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.run_name is not None:
        cfg["wandb"]["run_name"] = args.run_name
    if args.no_wandb:
        cfg["wandb"]["enabled"] = False

    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = split_image_mask_pairs(
        images_dir=cfg["data"]["images_dir"],
        masks_dir=cfg["data"]["masks_dir"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=cfg["data"]["seed"],
    )

    image_size = cfg["data"]["image_size"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_ds = PolypDataset(
        splits["train"], transform=SegmentationTransform(image_size=image_size, is_train=True)
    )
    val_ds = PolypDataset(
        splits["val"], transform=SegmentationTransform(image_size=image_size, is_train=False)
    )
    test_ds = PolypDataset(
        splits["test"], transform=SegmentationTransform(image_size=image_size, is_train=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model_cfg = cfg["model"]
    model = ResNet34UNet(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        freeze_encoder=model_cfg["freeze_encoder"],
        bilinear=model_cfg["bilinear"],
        dropout=model_cfg["dropout"],
    ).to(device)

    train_cfg = cfg["training"]
    param_groups = model.get_param_groups(
        encoder_lr=train_cfg["encoder_lr"],
        decoder_lr=train_cfg["decoder_lr"],
    )
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )
    criterion = build_loss(
        loss_name=train_cfg.get("loss", "bce"),
        bce_weight=train_cfg.get("bce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
    )

    wandb_cfg = cfg["wandb"]
    logger = WandbLogger(
        entity=wandb_cfg["entity"],
        project=wandb_cfg["project"],
        run_name=wandb_cfg["run_name"],
        config=cfg,
        tags=wandb_cfg.get("tags", []),
        enabled=wandb_cfg["enabled"],
    )

    try:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            logger=logger,
            checkpoint_dir=Path(train_cfg["checkpoint_dir"]),
            log_image_every_n_epochs=(
                wandb_cfg.get("log_image_every_n_epochs", 5)
                if wandb_cfg.get("log_images", True)
                else 0
            ),
            early_stopping_patience=train_cfg.get("early_stopping_patience"),
        )
        best_ckpt_path = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_cfg["epochs"],
        )

        load_checkpoint(best_ckpt_path, model, map_location=device)
        evaluator = Evaluator(model=model, criterion=criterion, device=device)
        test_metrics, test_samples = evaluator.evaluate(test_loader)

        logger.log_test_metrics(test_metrics)
        if test_samples is not None and wandb_cfg.get("log_images", True):
            imgs, msks, preds = test_samples
            logger.log_predictions(
                imgs, msks, preds, key="plots/test_predictions"
            )

        print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
