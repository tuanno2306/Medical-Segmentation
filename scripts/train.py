"""Train a binary U-Net. Run from the project root: python -m scripts.train."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.polyp_dataset import PolypDataset
from src.engine.losses import build_loss
from src.engine.trainer import run_epoch
from src.models.unet import UNet


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Kvasir-SEG")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/Kvasir-SEG"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["bce_tversky", "bce"], default="bce_tversky")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="medical-segmentation")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16 or args.base_channels < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, channels and learning rate must be positive; image size >= 16; workers >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    datasets = {
        split: PolypDataset(args.data_dir, args.split_dir / f"{split}.txt", (args.image_size, args.image_size))
        for split in ("train", "val")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"),
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
        for split, dataset in datasets.items()
    }
    model = UNet(base_channels=args.base_channels).to(device)
    loss_kwargs = {"alpha": args.alpha, "beta": args.beta, "bce_weight": args.bce_weight} if args.loss == "bce_tversky" else {}
    criterion = build_loss(args.loss, **loss_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("Install wandb to use --wandb: pip install wandb") from exc
        run = wandb.init(project=args.wandb_project, name=args.run_name, config=config)

    print(f"Device: {device}; train: {len(datasets['train'])}; val: {len(datasets['val'])}", flush=True)
    best_dice = -1.0
    try:
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val = run_epoch(model, loaders["val"], criterion, device)
            metrics = {**{f"train/{k}": v for k, v in train.items()},
                       **{f"val/{k}": v for k, v in val.items()}}
            print(f"Epoch {epoch}/{args.epochs} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            if run is not None:
                run.log(metrics, step=epoch)
            improved = val["dice"] > best_dice
            best_dice = max(best_dice, val["dice"])
            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(), "best_val_dice": best_dice,
                          "config": config}
            torch.save(checkpoint, args.output_dir / "last.pt")
            if improved:
                torch.save(checkpoint, args.output_dir / "best.pt")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
