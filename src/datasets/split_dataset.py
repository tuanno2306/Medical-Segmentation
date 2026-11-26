"""Deterministic train/val/test splitting for paired image and mask directories."""

import random
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def split_image_mask_pairs(
    images_dir: str | Path,
    masks_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    """Match image and mask files by stem and split into train/val/test lists."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    img_root = Path(images_dir)
    mask_root = Path(masks_dir)

    mask_by_stem = {
        p.stem: p
        for p in sorted(mask_root.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    }

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(img_root.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        mask_path = mask_by_stem.get(img_path.stem)
        if mask_path is not None:
            pairs.append((img_path, mask_path))

    if not pairs:
        raise FileNotFoundError(
            f"No matching image/mask pairs found in {img_root} and {mask_root}"
        )

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }
