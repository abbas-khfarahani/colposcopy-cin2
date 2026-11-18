#!/usr/bin/env python3
"""Ensemble attention ResNet (224) + attn-difflr with late clinical stack."""

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
from ml.utils import choose_threshold, compute_binary_metrics, set_seed  # noqa: E402


def sens70_threshold(yva, pva):
    best = None
    for thr in np.unique(pva):
        vm = compute_binary_metrics(yva, pva, float(thr))
        if vm.sensitivity >= 0.70:
            score = vm.sensitivity + vm.specificity - 1
            if best is None or score > best[0]:
                best = (score, float(thr))
    if best is None:
        return float(np.quantile(pva[yva == 1], 0.30))
    return best[1]


def main() -> None:
    device = resolve_device("cuda")
    set_seed(42)
    ckpts = [
        Path("ml/runs/acetic_resnet50_attn/best.pt"),
        Path("ml/runs/acetic_resnet50_attn_difflr/best.pt"),
    ]
    models = []
    cfg = None
    for p in ckpts:
        m, c = load_ckpt(p, device)
        models.append(m)
        cfg = c
    assert cfg is not None
    cmap = build_folder_clinical_map(cfg["my_dataset_root"])
    tf = build_transforms("val", int(cfg.get("image_size", 224)), False)

    def ens(split: str, w=(0.5, 0.5)):
        acc = None
        y = meta = None
        for wi, m in zip(w, models):
            s, yy, mm = predict_loader(m, make_loader(split, cfg, tf), device)
            acc = wi * s if acc is None else acc + wi * s
            y, meta = yy, mm
        return acc, y, meta

    def stack_and_eval(w, tag: str):
        tr, va, te = ens("train", w), ens("val", w), ens("test", w)
        ytr, _, _, Xtr = person_matrix(tr[2], tr[0], tr[1], cmap)
        yva, _, _, Xva = person_matrix(va[2], va[0], va[1], cmap)
        yte, _, _, Xte = person_matrix(te[2], te[0], te[1], cmap)
        sc = StandardScaler()
        Xtrs = sc.fit_transform(Xtr)
        Xvas = sc.transform(Xva)
        Xtes = sc.transform(Xte)
        out = {}
        for C in [0.25, 0.5, 1.0, 2.0]:
            clf = LogisticRegression(
                max_iter=3000, class_weight="balanced", C=C, solver="lbfgs"
            )
            clf.fit(Xtrs, ytr)
            pva = clf.predict_proba(Xvas)[:, 1]
            pte = clf.predict_proba(Xtes)[:, 1]
            thr_y = choose_threshold(yva, pva, "youden")
            thr_s = sens70_threshold(yva, pva)
            packs = {
                "youden": {
                    "thr": thr_y,
                    "val": compute_binary_metrics(yva, pva, thr_y).to_dict(),
                    "test": compute_binary_metrics(yte, pte, thr_y).to_dict(),
                },
                "sens70": {
                    "thr": thr_s,
                    "val": compute_binary_metrics(yva, pva, thr_s).to_dict(),
                    "test": compute_binary_metrics(yte, pte, thr_s).to_dict(),
                },
            }
            yt = packs["youden"]["test"]
            st = packs["sens70"]["test"]
            print(
                f"[{tag} C={C}] AUROC test={yt['auroc']:.4f} "
                f"youden_sens={yt['sensitivity']:.3f} "
                f"sens70_sens={st['sensitivity']:.3f} "
                f"sens70_spec={st['specificity']:.3f}"
            )
            out[f"C{C}"] = packs
        return out

    results = {}
    for w, tag in [
        ((0.5, 0.5), "50_50"),
        ((0.6, 0.4), "60_40"),
        ((0.7, 0.3), "70_30"),
        ((1.0, 0.0), "attn_only"),
        ((0.0, 1.0), "attn_difflr_only"),
    ]:
        results[tag] = stack_and_eval(w, tag)

    out = Path("ml/runs/boost_ensemble_attn_difflr")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "summary.json", results)

    print("\n==== SUMMARY C=0.5 test AUROC ====")
    for tag, res in results.items():
        t = res["C0.5"]["youden"]["test"]
        s = res["C0.5"]["sens70"]["test"]
        print(
            f"  {tag}: auroc={t['auroc']:.4f} youden_sens={t['sensitivity']:.3f} | "
            f"sens70_sens={s['sensitivity']:.3f} sens70_spec={s['specificity']:.3f}"
        )


if __name__ == "__main__":
    main()
