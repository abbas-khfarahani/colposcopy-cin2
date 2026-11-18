"""Shared training utilities: device, loaders, optimizer, AMP, checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from ml.protocol.constants import BACKBONE_LR_MULT, HEAD_PARAM_KEYS, LR, WEIGHT_DECAY, WARMUP_EPOCHS
from ml.utils.seed import set_seed


def resolve_device(name: str = "cuda") -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn,
    sampler: Sampler | None = None,
    drop_last: bool = False,
    pin_memory: bool | None = None,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
        persistent_workers=False,
    )


def param_groups(
    model: nn.Module,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    backbone_lr_mult: float = BACKBONE_LR_MULT,
) -> list[dict[str, Any]]:
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in HEAD_PARAM_KEYS):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if backbone_params:
        groups.append(
            {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay}
        )
    if head_params:
        groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
    if not groups:
        groups.append(
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
    return groups


def build_optimizer(model: nn.Module, lr: float = LR, weight_decay: float = WEIGHT_DECAY) -> torch.optim.AdamW:
    groups = [g for g in param_groups(model, lr=lr, weight_decay=weight_decay) if g["params"]]
    return torch.optim.AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = WARMUP_EPOCHS,
) -> torch.optim.lr_scheduler.LRScheduler:
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def iterate_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif k == "bags":
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def seed_all(seed: int) -> None:
    set_seed(int(seed))
