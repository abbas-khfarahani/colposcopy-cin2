#!/usr/bin/env python3
"""Fuse acetic person scores + multistate case scores + clinical."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.boost import load_ckpt, make_loader, predict_loader, save_json  # noqa: E402
from ml.data import (  # noqa: E402
    ColposcopyMultistateDataset,
    build_folder_clinical_map,
    build_transforms,
    missing_clinical_vector,
)
from ml.models import build_model  # noqa: E402
from ml.train import collate_batch, resolve_device  # noqa: E402
from ml.utils import (  # noqa: E402
    aggregate_person_level,
    choose_threshold,
    compute_binary_metrics,
    default_my_dataset_root,
    default_splits_dir,
    set_seed,
)


def load_ms_ckpt(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    cfg.setdefault("my_dataset_root", str(default_my_dataset_root()))
    cfg.setdefault("splits_dir", str(default_splits_dir()))
    model = build_model(
        backbone=str(cfg.get("backbone", "resnet50")),
        pretrained=False,
        model_type="multistate",
        dropout=float(cfg.get("dropout", 0.25)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def acetic_person(ckpt: Path, device):
    model, cfg = load_ckpt(ckpt, device)
    tf = build_transforms("val", int(cfg.get("image_size", 224)), False)
    out = {}
    for split in ("train", "val", "test"):
        s, y, m = predict_loader(model, make_loader(split, cfg, tf), device)
        py, ps, ids = aggregate_person_level(
            [mm["person_id"] for mm in m], y, s, "max"
        )
        folder = {}
        for mm in m:
            folder.setdefault(mm["person_id"], mm["folder_id"])
        out[split] = {"y": py, "score": ps, "ids": list(ids), "folder": folder}
    return out, cfg


@torch.no_grad()
def multistate_person(ckpt: Path, device):
    model, cfg = load_ms_ckpt(ckpt, device)
    tf = build_transforms("val", int(cfg.get("image_size", 224)), False)
    out = {}
    for split in ("train", "val", "test"):
        ds = ColposcopyMultistateDataset(
            split=split,
            my_dataset_root=cfg["my_dataset_root"],
            transform=tf,
            max_acetic=int(cfg.get("max_acetic", 3)),
            include_green=bool(cfg.get("include_green", True)),
            include_lugol=bool(cfg.get("include_lugol", True)),
        )
        loader = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_batch,
        )
        scores, labels, metas = [], [], []
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            type_ids = batch["type_ids"].to(device)
            logits = model(x, mask, type_ids)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            scores.extend(np.atleast_1d(prob).tolist())
            labels.extend(batch["y"].numpy().tolist())
            metas.extend(batch["meta"])
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int)
        ids = [m["person_id"] for m in metas]
        py, ps, pids = aggregate_person_level(ids, y, s, "max")
        out[split] = {"y": py, "score": ps, "ids": list(pids)}
    return out, cfg


def clin_vec(pid, folder_map, clinical_map):
    folder = folder_map.get(pid, pid)
    entry = clinical_map.get(folder) or clinical_map.get(pid)
    return (entry["vector"] if entry else missing_clinical_vector()).astype(np.float32)


def matrix(acetic_split, ms_split, clinical_map, mode: str):
    a = {i: (y, s) for i, y, s in zip(acetic_split["ids"], acetic_split["y"], acetic_split["score"])}
    ms = {}
    if ms_split is not None:
        ms = {i: s for i, s in zip(ms_split["ids"], ms_split["score"])}
    ids = sorted(a.keys())
    y_out, X = [], []
    for pid in ids:
        ay, ascore = a[pid]
        row = []
        if mode in ("acetic_clin", "acetic_ms_clin", "acetic_ms"):
            row.append(float(ascore))
        if mode in ("acetic_ms_clin", "acetic_ms"):
            row.append(float(ms.get(pid, ascore)))
        if mode in ("acetic_clin", "acetic_ms_clin", "clin_only"):
            row.extend(clin_vec(pid, acetic_split["folder"], clinical_map).tolist())
        if mode == "clin_only":
            pass
        y_out.append(int(ay))
        X.append(row)
    return np.asarray(y_out, dtype=int), np.asarray(X, dtype=np.float32), ids


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, tag):
    sc = StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    results = {}
    for C in [0.1, 0.25, 0.5, 1.0, 2.0]:
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", C=C, solver="lbfgs"
        )
        clf.fit(Xtr_s, ytr)
        pva = clf.predict_proba(Xva_s)[:, 1]
        pte = clf.predict_proba(Xte_s)[:, 1]
        thr = choose_threshold(yva, pva, "youden")
        vm = compute_binary_metrics(yva, pva, thr).to_dict()
        tm = compute_binary_metrics(yte, pte, thr).to_dict()
        results[f"C{C}"] = {"val": vm, "test": tm, "threshold": thr}
        print(
            f"[{tag} C={C}] val={vm['auroc']:.4f} test={tm['auroc']:.4f} "
            f"sens={tm['sensitivity']:.3f}"
        )
    return results


def main():
    set_seed(42)
    device = resolve_device("cuda")
    out = Path("ml/runs/boost_fuse_acetic_multistate")
    out.mkdir(parents=True, exist_ok=True)

    acetic, cfg = acetic_person(Path("ml/runs/acetic_resnet50_stable/best.pt"), device)
    clinical_map = build_folder_clinical_map(cfg["my_dataset_root"])
    ms_ckpt = Path("ml/runs/multistate_resnet50_stable_v2/best.pt")
    ms = None
    if ms_ckpt.is_file():
        ms, _ = multistate_person(ms_ckpt, device)

    packs = {}
    for mode in ("clin_only", "acetic_clin", "acetic_ms", "acetic_ms_clin"):
        if mode.startswith("acetic_ms") and ms is None:
            continue
        ytr, Xtr, _ = matrix(acetic["train"], ms["train"] if ms else None, clinical_map, mode)
        yva, Xva, _ = matrix(acetic["val"], ms["val"] if ms else None, clinical_map, mode)
        yte, Xte, _ = matrix(acetic["test"], ms["test"] if ms else None, clinical_map, mode)
        packs[mode] = fit_eval(Xtr, ytr, Xva, yva, Xte, yte, mode)

    save_json(out / "summary.json", packs)
    # highlight C=0.5 test scores
    print("\n==== C=0.5 TEST person AUROC ====")
    for mode, res in packs.items():
        t = res["C0.5"]["test"]
        print(f"  {mode}: {t['auroc']:.4f}  sens={t['sensitivity']:.3f}")


if __name__ == "__main__":
    main()
