"""Joint image and mask transformations for polyp segmentation."""

import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegmentationTransform:
    """Resize, optionally flip, and normalize (image, mask) pairs."""

    def __init__(self, image_size: int = 256, is_train: bool = True) -> None:
        self.image_size = image_size
        self.is_train = is_train

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = [self.image_size, self.image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        if self.is_train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = (TF.to_tensor(mask) >= 0.5).float()
        return image_tensor, mask_tensor
