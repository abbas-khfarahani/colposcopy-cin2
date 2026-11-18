"""Image-level trainer with patient-normalized loss and person-AUROC early stop."""

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
    BATCH_SIZE,
    IMAGE_SIZE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
)
from ml.protocol.datasets import AceticImageDataset, collate_images
from ml.protocol.encoders import ProtocolImageModel, build_protocol_model, load_body_weights
from ml.protocol.io_utils import save_json
from ml.protocol.objective import WeightedBCEWithLogits, image_patient_weights, patient_pos_weight
from ml.protocol.pooling import aggregate_person_scores
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


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s))


@torch.no_grad()
def predict_images(
    model: ProtocolImageModel,
    loader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = iterate_device(batch, device)
        logits = model(batch["x"])
        if isinstance(logits, tuple):
            logits = logits[0]
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        logit_np = logits.detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        for meta, yi, lg, pr in zip(batch["meta"], y, logit_np, prob):
            rows.append(
                {
                    "person_id": meta["person_id"],
                    "folder_id": meta["folder_id"],
                    "file": meta["file"],
                    "image_path_rel": meta["image_path_rel"],
                    "label": int(yi),
                    "person_label": int(meta["person_label"]),
                    "logit": float(lg),
                    "probability": float(pr),
                    "hpv_bucket": meta.get("hpv_bucket"),
                    "cyto_bucket": meta.get("cyto_bucket"),
                    "age": meta.get("age"),
                }
            )
    return pd.DataFrame(rows)


def _person_auroc_from_frame(frame: pd.DataFrame, reduce: str = "max") -> float:
    y, s, _, _ = aggregate_person_scores(
        frame["person_id"],
        frame["label"].to_numpy(),
        frame["probability"].to_numpy(),
        reduce=reduce,
        person_labels=frame["person_label"].to_numpy(),
    )
    return _safe_auroc(y, s)


def train_image_fold(
    *,
    train_images: pd.DataFrame,
    val_images: pd.DataFrame,
    dataset_root: Path,
    out_dir: Path,
    backbone: str,
    model_type: str,
    center_prior: bool,
    loss_name: str,
    use_patient_weights: bool,
    sampler,
    pooling: str,
    seed: int,
    device_name: str,
    num_workers: int,
    ssl_body: dict | None = None,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    smoke: bool = False,
) -> dict[str, Any]:
    seed_all(seed)
    device = resolve_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = build_protocol_transforms("conservative", IMAGE_SIZE)
    eval_tf = build_protocol_transforms("eval", IMAGE_SIZE)
    weights = image_patient_weights(train_images) if use_patient_weights else None
    train_ds = AceticImageDataset(train_images, dataset_root, train_tf, sample_weights=weights)
    val_ds = AceticImageDataset(val_images, dataset_root, eval_tf)

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        num_workers=num_workers,
        collate_fn=collate_images,
        sampler=sampler,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
    )

    persons_train = train_images.drop_duplicates("person_id")
    pos_w = 1.0 if loss_name == "focal" and sampler is not None else patient_pos_weight(persons_train)
    if sampler is not None and loss_name != "focal":
        pos_w = 1.0
    gamma = 2.0 if loss_name == "focal" else 0.0
    criterion = WeightedBCEWithLogits(pos_weight=pos_w, gamma=gamma)

    model = build_protocol_model(
        backbone=backbone,
        model_type=model_type,
        center_prior=center_prior,
    ).to(device)
    if ssl_body:
        load_body_weights(model, ssl_body)

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
                logits = model(batch["x"])
                loss = criterion(
                    logits,
                    batch["y"],
                    sample_weight=batch["w"] if use_patient_weights else None,
                )
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

        val_frame = predict_images(model, val_loader, device)
        val_person = _person_auroc_from_frame(val_frame, reduce=pooling)
        val_image = _safe_auroc(val_frame["label"], val_frame["probability"])
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_image_auroc": val_image,
            "val_person_auroc": val_person,
            "val_person_auprc": _safe_auprc(
                *aggregate_person_scores(
                    val_frame["person_id"],
                    val_frame["label"].to_numpy(),
                    val_frame["probability"].to_numpy(),
                    reduce=pooling,
                    person_labels=val_frame["person_label"].to_numpy(),
                )[:2]
            ),
            "lr": float(optimizer.param_groups[-1]["lr"]),
            "seconds": time.perf_counter() - epoch_t0,
        }
        history.append(row)
        elapsed = time.perf_counter() - t0
        print(
            f"  epoch {epoch:03d}/{max_epochs}  loss={row['train_loss']:.4f}  "
            f"val_person_auroc={val_person:.4f}  val_img_auroc={val_image:.4f}  "
            f"({row['seconds']:.1f}s) elapsed={format_duration(elapsed)}",
            flush=True,
        )

        monitor = val_person if np.isfinite(val_person) else -1.0
        improved = monitor > best_monitor + 1e-6
        if improved:
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
                    "pos_weight": pos_w,
                    "pooling": pooling,
                },
            )
        else:
            stale += 1
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                print(
                    f"  early stop @ {epoch}  best_person_auroc={best_monitor:.4f} "
                    f"epoch={best_epoch}",
                    flush=True,
                )
                break
        if smoke:
            break

    if best_path.is_file():
        ckpt = load_checkpoint(best_path, device)
        model.load_state_dict(ckpt["model_state"])
    val_best = predict_images(model, val_loader, device)
    summary = {
        "best_epoch": best_epoch,
        "best_val_person_auroc": best_monitor,
        "pos_weight": pos_w,
        "loss": loss_name,
        "n_train_images": int(len(train_ds)),
        "n_val_images": int(len(val_ds)),
        "history": history,
        "seconds_total": time.perf_counter() - t0,
    }
    save_json(out_dir / "history.json", history)
    save_json(out_dir / "summary.json", summary)
    val_best.to_csv(out_dir / "val_image_predictions.csv", index=False)
    return {"summary": summary, "model": model, "val_predictions": val_best, "device": device}


def predict_image_frame(
    model: ProtocolImageModel,
    images: pd.DataFrame,
    dataset_root: Path,
    device: torch.device,
    num_workers: int,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    ds = AceticImageDataset(images, dataset_root, build_protocol_transforms("eval", IMAGE_SIZE))
    loader = make_loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
    )
    return predict_images(model, loader, device)
