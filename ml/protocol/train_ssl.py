"""Fold-safe contrastive pretraining."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ml.protocol.constants import IMAGE_SIZE, SSL_BATCH_SIZE, SSL_EPOCHS, SSL_LR, SSL_TEMPERATURE
from ml.protocol.datasets import SSLPairDataset, collate_ssl
from ml.protocol.io_utils import save_json
from ml.protocol.ssl import SSLResNet, body_state_from_ssl, nt_xent
from ml.protocol.train_common import (
    build_scheduler,
    format_duration,
    make_loader,
    resolve_device,
    save_checkpoint,
    seed_all,
)
from ml.protocol.transforms import build_protocol_transforms


def train_ssl_fold(
    *,
    train_images: pd.DataFrame,
    dataset_root: Path,
    out_dir: Path,
    patient_aware: bool,
    seed: int,
    device_name: str,
    num_workers: int,
    epochs: int = SSL_EPOCHS,
    smoke: bool = False,
) -> dict[str, Any]:
    seed_all(seed)
    device = resolve_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = build_protocol_transforms("ssl", IMAGE_SIZE)
    ds = SSLPairDataset(train_images, dataset_root, tf, patient_aware=patient_aware)
    loader = make_loader(
        ds,
        batch_size=min(SSL_BATCH_SIZE, max(2, len(ds))),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ssl,
        drop_last=len(ds) >= SSL_BATCH_SIZE * 2,
    )
    model = SSLResNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, epochs=epochs, warmup_epochs=5)
    # Encoder AMP is fine; NT-Xent is computed in fp32 outside autocast.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    max_epochs = 1 if smoke else epochs
    start_epoch = 1

    # Mid-fold resume: prefer last.pt; else best.pt if SSL was interrupted (no summary.json).
    resume_path: Path | None = None
    if not smoke and not (out_dir / "summary.json").is_file():
        if last_path.is_file():
            resume_path = last_path
        elif best_path.is_file():
            resume_path = best_path
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        done = int(ckpt.get("epoch") or 0)
        if done >= 1 and done < max_epochs:
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
            if ckpt.get("optimizer_state") is not None:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if scaler is not None and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            best_loss = float(ckpt.get("best_loss", best_loss))
            if isinstance(ckpt.get("history"), list) and ckpt["history"]:
                history = list(ckpt["history"])
            elif (out_dir / "history.json").is_file():
                import json

                history = json.loads((out_dir / "history.json").read_text())
            start_epoch = done + 1
            print(
                f"  ssl resume from {resume_path.name} epoch={done} → next={start_epoch}/{max_epochs} "
                f"best_loss={best_loss:.4f}",
                flush=True,
            )

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        running, n = 0.0, 0
        epoch_t0 = time.perf_counter()
        for v1, v2, pids in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z1 = model(v1)
                z2 = model(v2)
            loss = nt_xent(
                z1,
                z2,
                temperature=SSL_TEMPERATURE,
                person_ids=pids if patient_aware else None,
            )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            bs = int(v1.size(0))
            running += float(loss.item()) * bs
            n += bs
        scheduler.step()
        avg = running / max(n, 1)
        history.append({"epoch": epoch, "loss": avg, "seconds": time.perf_counter() - epoch_t0})
        print(
            f"  ssl epoch {epoch:03d}/{max_epochs}  loss={avg:.4f}  "
            f"({history[-1]['seconds']:.1f}s) elapsed={format_duration(time.perf_counter() - t0)}",
            flush=True,
        )
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "body_state": body_state_from_ssl(model),
            "patient_aware": patient_aware,
            "best_loss": best_loss if avg >= best_loss else avg,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "history": history,
        }
        save_checkpoint(last_path, payload)
        save_json(out_dir / "history.json", history)
        if avg < best_loss or not best_path.is_file():
            best_loss = avg
            payload["best_loss"] = best_loss
            save_checkpoint(best_path, payload)
        if smoke:
            break

    if not best_path.is_file():
        save_checkpoint(
            best_path,
            {
                "epoch": 0,
                "model_state": model.state_dict(),
                "body_state": body_state_from_ssl(model),
                "patient_aware": patient_aware,
                "best_loss": best_loss,
            },
        )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    summary = {
        "best_loss": best_loss,
        "patient_aware": patient_aware,
        "n_images": int(len(ds)),
        "history": history,
        "seconds_total": time.perf_counter() - t0,
    }
    save_json(out_dir / "history.json", history)
    save_json(out_dir / "summary.json", summary)
    return {"summary": summary, "body_state": ckpt["body_state"]}
