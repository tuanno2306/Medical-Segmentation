"""PyTorch Dataset for polyp segmentation."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transform import SegmentationTransform


class PolypDataset(Dataset):
    """Loads RGB images and binary grayscale masks from (image_path, mask_path) pairs."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        transform: SegmentationTransform | None = None,
    ) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            return self.transform(image, mask)

        default_tf = SegmentationTransform(is_train=False)
        return default_tf(image, mask)
