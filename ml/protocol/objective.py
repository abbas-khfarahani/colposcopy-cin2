"""Patient-normalized BCE, focal loss, and balanced samplers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


def patient_pos_weight(persons: pd.DataFrame) -> float:
    """Fold-specific negative/positive *person* ratio."""
    y = persons["person_label"].to_numpy(dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def image_patient_weights(images: pd.DataFrame) -> pd.Series:
    """Each image gets 1 / n_images(person) so patients contribute equally."""
    counts = images.groupby("person_id")["person_id"].transform("count").astype(float)
    weights = 1.0 / counts.clip(lower=1.0)
    return weights.reset_index(drop=True)


def balanced_patient_sampler(images: pd.DataFrame) -> WeightedRandomSampler:
    """Oversample so each *person* is equal and classes are balanced."""
    frame = images.reset_index(drop=True)
    n_pos_p = int(frame.drop_duplicates("person_id")["person_label"].sum())
    n_neg_p = int(frame["person_id"].nunique() - n_pos_p)
    n_pos_p = max(n_pos_p, 1)
    n_neg_p = max(n_neg_p, 1)
    counts = frame.groupby("person_id")["person_id"].transform("count").astype(float)
    class_w = np.where(frame["person_label"].to_numpy() == 1, 1.0 / n_pos_p, 1.0 / n_neg_p)
    w = torch.as_tensor(class_w / counts.to_numpy(), dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(frame), replacement=True)


class WeightedBCEWithLogits(nn.Module):
    """BCE with logits, optional pos_weight, per-sample weights, optional focal."""

    def __init__(
        self,
        pos_weight: float = 1.0,
        gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pw = logits.new_tensor([self.pos_weight])
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        if self.gamma > 0:
            p = torch.sigmoid(logits)
            pt = torch.where(targets >= 0.5, p, 1.0 - p)
            bce = ((1.0 - pt) ** self.gamma) * bce
        if sample_weight is None:
            return bce.mean()
        w = sample_weight.to(dtype=bce.dtype, device=bce.device)
        return (bce * w).sum() / w.sum().clamp_min(1e-8)
