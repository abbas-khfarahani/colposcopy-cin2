"""Patient-bag MIL trainer (person-level loss and early stopping)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.protocol.constants import (
    AMP,
    IMAGE_SIZE,
    MAX_EPOCHS,
    MIL_ATTN_HIDDEN,
    MIL_BATCH_PATIENTS,
    MIL_DROPOUT,
    MIL_EMBED_DIM,
    MIN_EPOCHS,
    PATIENCE,
)
from ml.protocol.datasets import PatientBagDataset, collate_bags
from ml.protocol.encoders import build_protocol_model, load_body_weights
from ml.protocol.io_utils import save_json
from ml.protocol.mil import GatedAttentionMIL
from ml.protocol.objective import WeightedBCEWithLogits, patient_pos_weight
from ml.protocol.train_common import (
    build_optimizer,
    build_scheduler,
    format_duration,
    iterate_device,
    load_checkpoint,
    make_loader,
    resolve_device,
    save_checkpoint,
    seed_all,
)
from ml.protocol.transforms import build_protocol_transforms


def _safe_auroc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


@torch.no_grad()
def predict_bags(model: GatedAttentionMIL, loader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = iterate_device(batch, device)
        logits = model(batch["bags"])
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        logit_np = logits.detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        for meta, yi, lg, pr in zip(batch["meta"], y, logit_np, prob):
            rows.append(
                {
                    "person_id": meta["person_id"],
                    "label": int(yi),
                    "person_label": int(yi),
                    "logit": float(lg),
                    "probability": float(pr),
                    "image_count": int(meta["n_images"]),
                    "hpv_bucket": meta.get("hpv_bucket"),
                    "cyto_bucket": meta.get("cyto_bucket"),
                    "age": meta.get("age"),
                }
            )
    return pd.DataFrame(rows)


def train_mil_fold(
    *,
    train_images: pd.DataFrame,
    val_images: pd.DataFrame,
    dataset_root: Path,
    out_dir: Path,
    backbone: str,
    model_type: str,
    center_prior: bool,
    loss_name: str,
    seed: int,
    device_name: str,
    num_workers: int,
    ssl_body: dict | None = None,
    balanced_sampler: bool = False,
    epochs: int = MAX_EPOCHS,
    smoke: bool = False,
) -> dict[str, Any]:
    seed_all(seed)
    device = resolve_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = build_protocol_transforms("conservative", IMAGE_SIZE)
    eval_tf = build_protocol_transforms("eval", IMAGE_SIZE)
    train_ds = PatientBagDataset(train_images, dataset_root, train_tf)
    val_ds = PatientBagDataset(val_images, dataset_root, eval_tf)
    sampler = None
    if balanced_sampler:
        y = np.asarray([int(g["person_label"].max()) for g in train_ds.groups])
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = max(int((y == 0).sum()), 1)
        w = np.where(y == 1, 1.0 / n_pos, 1.0 / n_neg)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True
        )
    train_loader = make_loader(
        train_ds,
        batch_size=MIL_BATCH_PATIENTS,
        shuffle=sampler is None,
        num_workers=num_workers,
        collate_fn=collate_bags,
        sampler=sampler,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=MIL_BATCH_PATIENTS,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bags,
    )

    pos_w = 1.0 if balanced_sampler else patient_pos_weight(train_images.drop_duplicates("person_id"))
    gamma = 2.0 if loss_name == "focal" else 0.0
    criterion = WeightedBCEWithLogits(pos_weight=pos_w, gamma=gamma)

    encoder = build_protocol_model(
        backbone=backbone,
        model_type=model_type,
        center_prior=center_prior,
    )
    if ssl_body:
        load_body_weights(encoder, ssl_body)
    model = GatedAttentionMIL(
        encoder,
        embed_dim=MIL_EMBED_DIM,
        attn_hidden=MIL_ATTN_HIDDEN,
        dropout=MIL_DROPOUT,
    ).to(device)

    use_amp = AMP and device.type == "cuda"
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, epochs=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_path = out_dir / "best.pt"
    best_monitor = -1.0
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    max_epochs = 1 if smoke else epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        running, n = 0.0, 0
        epoch_t0 = time.perf_counter()
        for batch in train_loader:
            batch = iterate_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["bags"])
                loss = criterion(logits, batch["y"])
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            bs = int(batch["y"].size(0))
            running += float(loss.item()) * bs
            n += bs
        scheduler.step()

        val_frame = predict_bags(model, val_loader, device)
        val_auroc = _safe_auroc(val_frame["label"], val_frame["probability"])
        val_auprc = (
            float(average_precision_score(val_frame["label"], val_frame["probability"]))
            if val_frame["label"].nunique() > 1
            else float("nan")
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_person_auroc": val_auroc,
            "val_person_auprc": val_auprc,
            "lr": float(optimizer.param_groups[-1]["lr"]),
            "seconds": time.perf_counter() - epoch_t0,
        }
        history.append(row)
        print(
            f"  epoch {epoch:03d}/{max_epochs}  loss={row['train_loss']:.4f}  "
            f"val_person_auroc={val_auroc:.4f}  ({row['seconds']:.1f}s)  "
            f"elapsed={format_duration(time.perf_counter() - t0)}",
            flush=True,
        )
        monitor = val_auroc if np.isfinite(val_auroc) else -1.0
        if monitor > best_monitor + 1e-6:
            best_monitor = float(monitor)
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_person_auroc": best_monitor,
                    "backbone": backbone,
                    "model_type": model_type,
                    "center_prior": center_prior,
                    "kind": "mil",
                    "pos_weight": pos_w,
                },
            )
        else:
            stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                print(f"  early stop @ {epoch}  best={best_monitor:.4f} epoch={best_epoch}", flush=True)
                break
        if smoke:
            break

    if best_path.is_file():
        ckpt = load_checkpoint(best_path, device)
        model.load_state_dict(ckpt["model_state"])
    val_best = predict_bags(model, val_loader, device)
    summary = {
        "best_epoch": best_epoch,
        "best_val_person_auroc": best_monitor,
        "pos_weight": pos_w,
        "n_train_persons": int(len(train_ds)),
        "n_val_persons": int(len(val_ds)),
        "history": history,
        "seconds_total": time.perf_counter() - t0,
    }
    save_json(out_dir / "history.json", history)
    save_json(out_dir / "summary.json", summary)
    val_best.to_csv(out_dir / "val_person_predictions.csv", index=False)
    return {"summary": summary, "model": model, "val_predictions": val_best, "device": device}


def predict_bag_frame(
    model: GatedAttentionMIL,
    images: pd.DataFrame,
    dataset_root: Path,
    device: torch.device,
    num_workers: int,
) -> pd.DataFrame:
    ds = PatientBagDataset(images, dataset_root, build_protocol_transforms("eval", IMAGE_SIZE))
    loader = make_loader(
        ds,
        batch_size=MIL_BATCH_PATIENTS,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bags,
    )
    return predict_bags(model, loader, device)
