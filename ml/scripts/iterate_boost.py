#!/usr/bin/env python3
"""Aggressive post-hoc boost loop: rich stack, clinical-only, ensembles, C grid."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.boost import load_ckpt, make_loader, predict_loader, save_json, person_matrix  # noqa: E402
from ml.data import build_folder_clinical_map, build_transforms, missing_clinical_vector  # noqa: E402
from ml.train import resolve_device  # noqa: E402
from ml.utils import (  # noqa: E402
    aggregate_person_level,
    choose_threshold,
    compute_binary_metrics,
    set_seed,
)


def person_rich(metas, scores, labels, clinical_map):
    buckets: dict[str, list] = defaultdict(list)
    for m, s, y in zip(metas, scores, labels):
        buckets[m["person_id"]].append((float(s), int(y), m["folder_id"]))
    ids, y_out, X = [], [], []
    for pid in sorted(buckets.keys()):
        pairs = buckets[pid]
        ss = np.asarray([p[0] for p in pairs], dtype=float)
        ys = [p[1] for p in pairs]
        folder = pairs[0][2]
        entry = clinical_map.get(folder) or clinical_map.get(pid)
        vec = entry["vector"] if entry else missing_clinical_vector()
        feat = np.concatenate(
            [
                np.asarray(
                    [
                        float(ss.max()),
                        float(ss.mean()),
                        float(ss.std() if len(ss) > 1 else 0.0),
                        float(ss.min()),
                        float(np.median(ss)),
                        float(len(ss)),
                    ],
                    dtype=np.float32,
                ),
                vec.astype(np.float32),
            ]
        )
        ids.append(pid)
        y_out.append(max(ys))
        X.append(feat)
    return np.asarray(y_out, dtype=int), np.asarray(X, dtype=np.float32), ids


def eval_probs(y_va, p_va, y_te, p_te):
    thr = choose_threshold(y_va, p_va, "youden")
    return {
        "threshold": thr,
        "val": compute_binary_metrics(y_va, p_va, thr).to_dict(),
        "test": compute_binary_metrics(y_te, p_te, thr).to_dict(),
    }


def fit_stackers(X_tr, y_tr, X_va, y_va, X_te, y_te, tag: str):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xva = scaler.transform(X_va)
    Xte = scaler.transform(X_te)
    results = {}

    # Logistic C grid
    best_lr = None
    for C in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        clf = LogisticRegression(
            max_iter=4000, class_weight="balanced", C=C, solver="lbfgs"
        )
        clf.fit(Xtr, y_tr)
        pack = eval_probs(
            y_va, clf.predict_proba(Xva)[:, 1], y_te, clf.predict_proba(Xte)[:, 1]
        )
        key = f"logreg_C{C}"
        results[key] = pack
        if best_lr is None or pack["val"]["auroc"] > best_lr["val"]["auroc"]:
            best_lr = {**pack, "name": key, "C": C}
    results["best_logreg"] = best_lr

    # Clinical-only (drop image cols: first 6 are image stats if rich, else first 1)
    # Caller passes flag via X shape; we try dropping first k cols for k in {1,6}
    for k in (1, 6):
        if X_tr.shape[1] <= k:
            continue
        clf = LogisticRegression(
            max_iter=4000, class_weight="balanced", C=0.5, solver="lbfgs"
        )
        clf.fit(Xtr[:, k:], y_tr)
        pack = eval_probs(
            y_va,
            clf.predict_proba(Xva[:, k:])[:, 1],
            y_te,
            clf.predict_proba(Xte[:, k:])[:, 1],
        )
        results[f"clinical_only_drop{k}"] = pack

    # Image-only (first feature / first 6)
    for k in (1, 6):
        if X_tr.shape[1] < k:
            continue
        clf = LogisticRegression(
            max_iter=4000, class_weight="balanced", C=0.5, solver="lbfgs"
        )
        clf.fit(Xtr[:, :k], y_tr)
        pack = eval_probs(
            y_va,
            clf.predict_proba(Xva[:, :k])[:, 1],
            y_te,
            clf.predict_proba(Xte[:, :k])[:, 1],
        )
        results[f"image_stats_only_k{k}"] = pack

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
    )
    rf.fit(Xtr, y_tr)
    results["random_forest"] = eval_probs(
        y_va, rf.predict_proba(Xva)[:, 1], y_te, rf.predict_proba(Xte)[:, 1]
    )

    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42
    )
    gb.fit(Xtr, y_tr)
    results["grad_boost"] = eval_probs(
        y_va, gb.predict_proba(Xva)[:, 1], y_te, gb.predict_proba(Xte)[:, 1]
    )

    # pick best by val AUROC among all
    best = None
    for name, pack in results.items():
        if name == "best_logreg" or "auroc" not in pack.get("val", {}):
            continue
        if best is None or pack["val"]["auroc"] > best["val"]["auroc"]:
            best = {**pack, "name": name}
    results["best_by_val"] = best
    print(
        f"[{tag}] best={best['name']} val={best['val']['auroc']:.4f} "
        f"test={best['test']['auroc']:.4f} sens={best['test']['sensitivity']:.3f}"
    )
    return results


def predict_acetic_ensemble(ckpts, device, cfg_ref, clinical_map, weights=None):
    models = []
    for p in ckpts:
        m, c = load_ckpt(p, device)
        models.append(m)
        cfg_ref.update(c)
    if weights is None:
        weights = np.ones(len(models), dtype=float) / len(models)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    tf = build_transforms("val", int(cfg_ref.get("image_size", 224)), False)

    def ens(split):
        acc = None
        y = None
        meta = None
        for w, m in zip(weights, models):
            s, yy, mm = predict_loader(m, make_loader(split, cfg_ref, tf), device)
            acc = w * s if acc is None else acc + w * s
            y, meta = yy, mm
        return acc, y, meta

    return ens("train"), ens("val"), ens("test")


def main():
    set_seed(42)
    device = resolve_device("cuda")
    out_root = Path("ml/runs/iterate_boost")
    out_root.mkdir(parents=True, exist_ok=True)

    acetic_ckpts = [
        Path("ml/runs/acetic_resnet50_stable/best.pt"),
    ]
    for extra in [
        "ml/runs/acetic_resnet50_seed7/best.pt",
        "ml/runs/acetic_resnet50_seed13/best.pt",
        "ml/runs/acetic_resnet50_seed21/best.pt",
        "ml/runs/acetic_resnet50_focal/best.pt",
    ]:
        p = Path(extra)
        if p.is_file():
            acetic_ckpts.append(p)

    # Filter weak seeds later by val; for now use all that exist
    print("Using acetic checkpoints:", [str(p) for p in acetic_ckpts])

    clinical_map = None
    cfg = {}
    model0, cfg0 = load_ckpt(acetic_ckpts[0], device)
    cfg.update(cfg0)
    clinical_map = build_folder_clinical_map(cfg["my_dataset_root"])
    tf = build_transforms("val", 224, False)
    tr_s, tr_y, tr_m = predict_loader(model0, make_loader("train", cfg, tf), device)
    va_s, va_y, va_m = predict_loader(model0, make_loader("val", cfg, tf), device)
    te_s, te_y, te_m = predict_loader(model0, make_loader("test", cfg, tf), device)

    # --- A: rich stack on stable ---
    y_tr, X_tr, _ = person_rich(tr_m, tr_s, tr_y, clinical_map)
    y_va, X_va, _ = person_rich(va_m, va_s, va_y, clinical_map)
    y_te, X_te, ids = person_rich(te_m, te_s, te_y, clinical_map)
    rich = fit_stackers(X_tr, y_tr, X_va, y_va, X_te, y_te, "rich_stable")
    save_json(out_root / "rich_stable.json", {"ids_test": ids, "results": rich})

    # --- B: simple stack matrix ---
    y_tr2, _, _, X_tr2 = person_matrix(tr_m, tr_s, tr_y, clinical_map)
    y_va2, _, _, X_va2 = person_matrix(va_m, va_s, va_y, clinical_map)
    y_te2, _, _, X_te2 = person_matrix(te_m, te_s, te_y, clinical_map)
    simple = fit_stackers(X_tr2, y_tr2, X_va2, y_va2, X_te2, y_te2, "simple_stable")
    save_json(out_root / "simple_stable.json", {"results": simple})

    # --- C: ensembles of acetic models ---
    summary = {"rich_stable_best": rich["best_by_val"], "simple_stable_best": simple["best_by_val"]}

    if len(acetic_ckpts) > 1:
        # equal mean ensemble
        (tr_e, va_e, te_e) = predict_acetic_ensemble(acetic_ckpts, device, cfg, clinical_map)
        tr_s, tr_y, tr_m = tr_e
        va_s, va_y, va_m = va_e
        te_s, te_y, te_m = te_e
        # pure person
        pv_y, pv_s, _ = aggregate_person_level(
            [m["person_id"] for m in va_m], va_y, va_s, "max"
        )
        thr = choose_threshold(pv_y, pv_s, "youden")
        pt_y, pt_s, _ = aggregate_person_level(
            [m["person_id"] for m in te_m], te_y, te_s, "max"
        )
        pure = {
            "val": compute_binary_metrics(pv_y, pv_s, thr).to_dict(),
            "test": compute_binary_metrics(pt_y, pt_s, thr).to_dict(),
            "threshold": thr,
        }
        print(
            f"[ens_pure] val={pure['val']['auroc']:.4f} test={pure['test']['auroc']:.4f}"
        )
        y_tr, X_tr, _ = person_rich(tr_m, tr_s, tr_y, clinical_map)
        y_va, X_va, _ = person_rich(va_m, va_s, va_y, clinical_map)
        y_te, X_te, _ = person_rich(te_m, te_s, te_y, clinical_map)
        ens_rich = fit_stackers(X_tr, y_tr, X_va, y_va, X_te, y_te, "ens_rich")
        save_json(
            out_root / "ensemble_all.json",
            {
                "ckpts": [str(p) for p in acetic_ckpts],
                "pure": pure,
                "rich": ens_rich,
            },
        )
        summary["ens_all_pure"] = pure
        summary["ens_all_rich_best"] = ens_rich["best_by_val"]

        # stable + only models with val person AUROC >= 0.70 if we can compute
        # quick filter: skip seed7/focal if known weak — weight by val auroc
        val_aurocs = []
        for p in acetic_ckpts:
            m, c = load_ckpt(p, device)
            s, y, meta = predict_loader(m, make_loader("val", cfg, tf), device)
            py, ps, _ = aggregate_person_level(
                [mm["person_id"] for mm in meta], y, s, "max"
            )
            a = float(compute_binary_metrics(py, ps, choose_threshold(py, ps, "youden")).auroc)
            val_aurocs.append(a)
            print(f"  val person AUROC {p.parent.name}: {a:.4f}")
        w = np.maximum(np.asarray(val_aurocs) - 0.5, 1e-3)
        (tr_e, va_e, te_e) = predict_acetic_ensemble(
            acetic_ckpts, device, cfg, clinical_map, weights=w
        )
        tr_s, tr_y, tr_m = tr_e
        va_s, va_y, va_m = va_e
        te_s, te_y, te_m = te_e
        y_tr, X_tr, _ = person_rich(tr_m, tr_s, tr_y, clinical_map)
        y_va, X_va, _ = person_rich(va_m, va_s, va_y, clinical_map)
        y_te, X_te, _ = person_rich(te_m, te_s, te_y, clinical_map)
        w_rich = fit_stackers(X_tr, y_tr, X_va, y_va, X_te, y_te, "weighted_ens_rich")
        save_json(
            out_root / "ensemble_weighted.json",
            {
                "ckpts": [str(p) for p in acetic_ckpts],
                "val_aurocs": val_aurocs,
                "weights": w.tolist(),
                "rich": w_rich,
            },
        )
        summary["weighted_ens_rich_best"] = w_rich["best_by_val"]

        # keep only models with val >= 0.72
        strong = [p for p, a in zip(acetic_ckpts, val_aurocs) if a >= 0.72]
        if len(strong) >= 1:
            (tr_e, va_e, te_e) = predict_acetic_ensemble(strong, device, cfg, clinical_map)
            tr_s, tr_y, tr_m = tr_e
            va_s, va_y, va_m = va_e
            te_s, te_y, te_m = te_e
            y_tr, X_tr, _ = person_rich(tr_m, tr_s, tr_y, clinical_map)
            y_va, X_va, _ = person_rich(va_m, va_s, va_y, clinical_map)
            y_te, X_te, _ = person_rich(te_m, te_s, te_y, clinical_map)
            s_rich = fit_stackers(X_tr, y_tr, X_va, y_va, X_te, y_te, "strong_ens_rich")
            save_json(
                out_root / "ensemble_strong.json",
                {"ckpts": [str(p) for p in strong], "rich": s_rich},
            )
            summary["strong_ens_rich_best"] = s_rich["best_by_val"]

    save_json(out_root / "leaderboard.json", summary)
    # print leaderboard
    print("\n==== LEADERBOARD (test person AUROC) ====")
    rows = []
    for k, v in summary.items():
        if not isinstance(v, dict):
            continue
        test = v.get("test") or {}
        val = v.get("val") or {}
        if "auroc" in test:
            rows.append((test["auroc"], val.get("auroc"), k, v.get("name"), test.get("sensitivity")))
    for t, v, k, n, s in sorted(rows, reverse=True):
        print(f"  test={t:.4f} val={v:.4f}  {k} ({n}) sens={s}")


if __name__ == "__main__":
    main()
