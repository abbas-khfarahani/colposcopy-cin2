#!/usr/bin/env python3
"""Weighted ensemble of difflr family + stable, then late clinical stack."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.boost import load_ckpt, make_loader, predict_loader, save_json, person_matrix  # noqa: E402
from ml.data import build_folder_clinical_map, build_transforms  # noqa: E402
from ml.train import resolve_device  # noqa: E402
from ml.utils import (  # noqa: E402
    aggregate_person_level,
    choose_threshold,
    compute_binary_metrics,
    set_seed,
)


def main() -> None:
    device = resolve_device("cuda")
    set_seed(42)
    ckpts = []
    for p in [
        "ml/runs/acetic_resnet50_difflr/best.pt",
        "ml/runs/acetic_resnet50_difflr_seed7/best.pt",
        "ml/runs/acetic_resnet50_difflr_seed13/best.pt",
        "ml/runs/acetic_resnet50_difflr_seed21/best.pt",
        "ml/runs/acetic_resnet50_stable/best.pt",
    ]:
        if Path(p).is_file():
            ckpts.append(Path(p))
    print("ckpts", ckpts)
    models = []
    cfg = None
    for p in ckpts:
        m, c = load_ckpt(p, device)
        models.append(m)
        cfg = c
    assert cfg is not None
    tf = build_transforms("val", 224, False)
    cmap = build_folder_clinical_map(cfg["my_dataset_root"])
    weights = []
    for p, m in zip(ckpts, models):
        s, y, meta = predict_loader(m, make_loader("val", cfg, tf), device)
        py, ps, _ = aggregate_person_level(
            [mm["person_id"] for mm in meta], y, s, "max"
        )
        a = compute_binary_metrics(
            py, ps, choose_threshold(py, ps, "youden")
        ).auroc
        weights.append(max(a - 0.5, 0.01))
        print(f"val person AUROC {p.parent.name}: {a:.4f}")
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    def ens(split: str):
        acc = None
        y = None
        meta = None
        for wi, m in zip(w, models):
            s, yy, mm = predict_loader(m, make_loader(split, cfg, tf), device)
            acc = wi * s if acc is None else acc + wi * s
            y, meta = yy, mm
        return acc, y, meta

    tr, va, te = ens("train"), ens("val"), ens("test")
    ytr, _, _, Xtr = person_matrix(tr[2], tr[0], tr[1], cmap)
    yva, _, _, Xva = person_matrix(va[2], va[0], va[1], cmap)
    yte, _, _, Xte = person_matrix(te[2], te[0], te[1], cmap)
    sc = StandardScaler()
    Xtrs = sc.fit_transform(Xtr)
    Xvas = sc.transform(Xva)
    Xtes = sc.transform(Xte)
    clf = LogisticRegression(
        max_iter=3000, class_weight="balanced", C=0.5, solver="lbfgs"
    )
    clf.fit(Xtrs, ytr)
    pva = clf.predict_proba(Xvas)[:, 1]
    pte = clf.predict_proba(Xtes)[:, 1]
    thr = choose_threshold(yva, pva, "youden")
    summary = {
        "method": "weighted_difflr_family_late_stack",
        "ckpts": [str(p) for p in ckpts],
        "weights": w.tolist(),
        "val": compute_binary_metrics(yva, pva, thr).to_dict(),
        "test": compute_binary_metrics(yte, pte, thr).to_dict(),
        "threshold": thr,
    }
    # also sens70
    best = None
    for t in np.unique(pva):
        vm = compute_binary_metrics(yva, pva, float(t))
        if vm.sensitivity >= 0.70:
            score = vm.sensitivity + vm.specificity - 1
            if best is None or score > best[0]:
                best = (score, float(t))
    if best is not None:
        thr_s = best[1]
        summary["sens70"] = {
            "threshold": thr_s,
            "val": compute_binary_metrics(yva, pva, thr_s).to_dict(),
            "test": compute_binary_metrics(yte, pte, thr_s).to_dict(),
        }
    out = Path("ml/runs/boost_ensemble_difflr_family")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "summary.json", summary)
    print(
        "FAMILY ens+stack test",
        summary["test"]["auroc"],
        "sens",
        summary["test"]["sensitivity"],
    )
    if "sens70" in summary:
        print(
            "FAMILY sens70 test",
            summary["sens70"]["test"]["auroc"],
            "sens",
            summary["sens70"]["test"]["sensitivity"],
            "spec",
            summary["sens70"]["test"]["specificity"],
        )


if __name__ == "__main__":
    main()
