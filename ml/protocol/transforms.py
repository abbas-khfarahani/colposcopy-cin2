"""Conservative 224×224 colposcopy transforms (design-doc default)."""

from __future__ import annotations

from typing import Literal

import torchvision.transforms as T
from torchvision.transforms import functional as TF

from ml.data.transforms import IMAGENET_MEAN, IMAGENET_STD

AugmentName = Literal["conservative", "ssl", "eval", "shift"]


def build_protocol_transforms(
    kind: AugmentName = "eval",
    image_size: int = 224,
    shift_frac: float = 0.0,
) -> T.Compose:
    """Eval is deterministic. Train uses medically conservative geometry/color."""
    normalize = [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if kind == "eval":
        ops: list = [T.Resize((image_size, image_size)), *normalize]
        return T.Compose(ops)

    if kind == "shift":
        dx = int(round(shift_frac * image_size))
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.Lambda(lambda im: TF.affine(im, angle=0, translate=(dx, 0), scale=1.0, shear=0)),
                *normalize,
            ]
        )

    if kind == "ssl":
        return T.Compose(
            [
                T.RandomResizedCrop(
                    image_size, scale=(0.70, 1.00), ratio=(0.90, 1.10)
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.15),
                *normalize,
            ]
        )

    # conservative supervised (design document)
    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size, scale=(0.85, 1.00), ratio=(0.90, 1.10)
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06, hue=0.0),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.10),
            *normalize,
        ]
    )
