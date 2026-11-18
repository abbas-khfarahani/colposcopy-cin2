#!/usr/bin/env python3
"""Richer late stacking: image stats + clinical, C tuned on val."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.boost import load_ckpt, make_loader, predict_loader, predict_tta, save_json  # noqa: E402
from ml.data import build_folder_clinical_map, build_transforms, missing_clinical_vector  # noqa: E402
from ml.train import resolve_device  # noqa: E402
from ml.utils import choose_threshold, compute_binary_metrics, set_seed  # noqa: E402


def person_rich_features(metas, scores, labels, clinical_map):
    buckets: dict[str, list] = defaultdict(list)
    for m, s, y in zip(metas, scores, labels):
        buckets[m["person_id"]].append((float(s), int(y), m["folder_id"]))
    ids, y_out, X = [], [], []
    for pid in sorted(buckets.keys()):
        pairs = buckets[pid]
        ss = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        folder = pairs[0][2]
        entry = clinical_map.get(folder) or clinical_map.get(pid)
        vec = entry["vector"] if entry else missing_clinical_vector()
        feat = np.concatenate(
            [
                np.asarray(
                    [max(ss), float(np.mean(ss)), float(np.std(ss) if len(ss) > 1 else 0.0), len(ss)],
                    dtype=np.float32,
                ),
                vec.astype(np.float32),
            ]
        )
        ids.append(pid)
        y_out.append(max(ys))
        X.append(feat)
    return np.asarray(y_out, dtype=int), np.asarray(X, dtype=np.float32), ids


def main():
    ckpt = Path("ml/runs/acetic_resnet50_stable/best.pt")
    out = Path("ml/runs/boost_rich_stack_resnet50_stable")
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cuda")
    model, cfg = load_ckpt(ckpt, device)
    set_seed(42)
    clinical_map = build_folder_clinical_map(cfg["my_dataset_root"])
    tf = build_transforms("val", int(cfg.get("image_size", 224)), False)

    def split_pred(split):
        return predict_loader(model, make_loader(split, cfg, tf), device)

    tr_s, tr_y, tr_m = split_pred("train")
    va_s, va_y, va_m = split_pred("val")
    te_s, te_y, te_m = split_pred("test")

    y_tr, X_tr, _ = person_rich_features(tr_m, tr_s, tr_y, clinical_map)
    y_va, X_va, _ = person_rich_features(va_m, va_s, va_y, clinical_map)
    y_te, X_te, ids = person_rich_features(te_m, te_s, te_y, clinical_map)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    best = None
    for C in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", C=C, solver="lbfgs"
        )
        clf.fit(X_tr_s, y_tr)
        va_prob = clf.predict_proba(X_va_s)[:, 1]
        thr = choose_threshold(y_va, va_prob, "youden")
        va_mtr = compute_binary_metrics(y_va, va_prob, thr)
        if best is None or va_mtr.auroc > best["val_auroc"]:
            te_prob = clf.predict_proba(X_te_s)[:, 1]
            te_mtr = compute_binary_metrics(y_te, te_prob, thr)
            best = {
                "C": C,
                "threshold": thr,
                "val_auroc": va_mtr.auroc,
                "val": va_mtr.to_dict(),
                "test": te_mtr.to_dict(),
                "test_scores": te_prob.tolist(),
                "test_ids": ids,
            }
            print(f"C={C} val_auroc={va_mtr.auroc:.4f} test_auroc={te_mtr.auroc:.4f} sens={te_mtr.sensitivity:.3f}")

    save_json(out / "summary.json", {
        "method": "rich_late_stack",
        "checkpoint": str(ckpt),
        "best": {k: v for k, v in best.items() if k not in {"test_scores"}},
        "test_person": best["test"],
    })
    print("BEST test person AUROC", best["test"]["auroc"])


if __name__ == "__main__":
    main()
