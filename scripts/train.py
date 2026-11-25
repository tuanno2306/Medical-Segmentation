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
from src.models import build_model


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("configs/comparison.json"))
    config_args, _ = config_parser.parse_known_args()
    defaults = json.loads(config_args.config.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Train U-Net on Kvasir-SEG", parents=[config_parser])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--loss", choices=["bce_tversky", "bce"])
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--encoder")
    parser.add_argument("--run-name", default=None)
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = f"unet-{args.encoder}-{args.loss}-s{args.seed}"
    if args.output_dir is None:
        args.output_dir = Path("outputs") / args.run_name
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
        split: PolypDataset(args.data_dir, args.split_dir / f"{split}.txt", (args.image_size, args.image_size),
                           rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)
        for split in ("train", "val")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"),
                          generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers, pin_memory=(device.type == "cuda"))
        for split, dataset in datasets.items()
    }
    model = build_model(config).to(device)
    loss_kwargs = {"alpha": args.alpha, "beta": args.beta, "bce_weight": args.bce_weight} if args.loss == "bce_tversky" else {}
    criterion = build_loss(args.loss, **loss_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=tuple(args.adam_betas),
                                 eps=args.adam_eps, weight_decay=args.weight_decay)
    run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("Install wandb to use --wandb: pip install wandb") from exc
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.run_name, config=config)

    print(f"Device: {device}; train: {len(datasets['train'])}; val: {len(datasets['val'])}", flush=True)
    best_dice = -1.0
    try:
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, loaders["train"], criterion, device, optimizer, threshold=args.threshold)
            val = run_epoch(model, loaders["val"], criterion, device, threshold=args.threshold)
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
