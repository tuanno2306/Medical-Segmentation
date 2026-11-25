"""Evaluate the Kaggle U-Net checkpoint on the saved test split."""

import argparse
import json
from pathlib import Path

import matplotlib
import torch
from torch.utils.data import DataLoader

from src.datasets.polyp_dataset import PolypDataset
from src.metrics.segmentation_metrics import segmentation_scores
from src.models import build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_examples(dataset, predictions, scores, path):
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx]["dice"])
    low_count = min(5, (len(ranked) + 1) // 2)
    high_count = min(5, len(ranked) - low_count)
    examples = [
        (f"Lowest #{rank}", idx)
        for rank, idx in enumerate(ranked[:low_count], start=1)
    ]
    examples.extend(
        (f"Highest #{rank}", idx)
        for rank, idx in enumerate(reversed(ranked[len(ranked) - high_count:]), start=1)
    )
    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 3.4 * len(examples)),
                             squeeze=False, layout="constrained")
    for (label, idx), row in zip(examples, axes):
        image, target = dataset[idx]
        if dataset.rgb_mean is not None:
            image = (image * dataset.rgb_std + dataset.rgb_mean).clamp(0, 1)
        score = scores[idx]
        row[0].imshow(image.permute(1, 2, 0))
        row[0].set_title(f"{label} Dice\n{score['filename']}", fontsize=9)
        row[1].imshow(target[0], cmap="gray", vmin=0, vmax=1)
        row[1].set_title("Ground truth")
        row[2].imshow(predictions[idx], cmap="gray", vmin=0, vmax=1)
        row[2].set_title(f"Prediction | Dice {score['dice']:.3f}, IoU {score['iou']:.3f}")
        for ax in row:
            ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/Kvasir-SEG"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/test.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test-unet"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch size and CPU threads must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    size = config["image_size"]
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    dataset = PolypDataset(args.data_dir, args.split_file, (size, size),
                           rgb_mean=config.get("rgb_mean"), rgb_std=config.get("rgb_std"))
    if not len(dataset):
        raise ValueError(f"Empty split: {args.split_file}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    scores, predictions = [], []
    print(f"Evaluating {len(dataset)} images on {device} at {size}x{size}", flush=True)

    threshold = config.get("threshold", 0.5)
    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            dice, iou = segmentation_scores(logits, masks.to(device), threshold)
            predictions.extend((logits.sigmoid() >= threshold)[:, 0].cpu().numpy())
            for dice_score, iou_score in zip(dice.tolist(), iou.tolist()):
                scores.append({
                    "filename": dataset.filenames[len(scores)],
                    "dice": dice_score,
                    "iou": iou_score,
                })
            if len(scores) == len(dataset) or len(scores) % 20 == 0:
                print(f"Evaluated {len(scores)}/{len(dataset)} images", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "best_val_dice": checkpoint["best_val_dice"],
        "split": str(args.split_file),
        "image_size": size,
        "threshold": threshold,
        "count": len(scores),
        "mean_dice": sum(item["dice"] for item in scores) / len(scores),
        "mean_iou": sum(item["iou"] for item in scores) / len(scores),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({**summary, "per_image": scores}, indent=2) + "\n",
        encoding="utf-8",
    )
    save_examples(dataset, predictions, scores, args.output_dir / "examples.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
