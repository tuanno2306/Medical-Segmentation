from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PolypDataset(Dataset):
    def __init__(self, data_dir, split_file, image_size=(256, 256)):
        """Load paired Kvasir-SEG images and binary masks for a saved split.

        Args:
            data_dir: Kvasir-SEG directory containing images/ and masks/.
            split_file: Text file listing one image filename per line.
            image_size: Target size as (width, height).
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        self.image_size = image_size

        split_path = Path(split_file)
        self.filenames = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image_path = self.image_dir / filename
        mask_path = self.mask_dir / filename

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            image = np.array(image, dtype=np.float32) / 255.0

        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")
            mask = mask.resize(self.image_size, Image.Resampling.NEAREST)
            mask = np.array(mask)
            mask = (mask >= 128).astype(np.float32)

        # Ảnh: (cao, rộng, kênh) -> (kênh, cao, rộng)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Mask: (cao, rộng) -> (1, cao, rộng)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
