#!/usr/bin/env python3
"""Thesis-grade evaluation for the locked primary CIN2+ pipeline.

Primary model: 50/50 ensemble (acetic ResNet50 stable + difflr) + late clinical stack.
Variant: difflr alone + late clinical stack.

Writes under ml/runs/thesis_eval/ — never into dataset folders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.boost import load_ckpt, make_loader, predict_loader, person_matrix  # noqa: E402
from ml.data import build_folder_clinical_map, build_transforms  # noqa: E402
from ml.train import resolve_device  # noqa: E402
from ml.utils import (  # noqa: E402
    choose_threshold,
    compute_binary_metrics,
    set_seed,
)


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            vals.append(float(metric_fn(yb, sb)))
        except Exception:
            continue
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "n_boot": int(arr.size),
    }


def expected_calibration_error(y: np.ndarray, s: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (s >= bins[i]) & (s < bins[i + 1] if i < n_bins - 1 else s <= bins[i + 1])
        if not np.any(m):
            continue
        ece += (m.mean()) * abs(y[m].mean() - s[m].mean())
    return float(ece)


def threshold_for_min_sens(y: np.ndarray, s: np.ndarray, min_sens: float = 0.70) -> float:
    best = None
    for thr in np.unique(s):
        m = compute_binary_metrics(y, s, float(thr))
        if m.sensitivity >= min_sens:
            j = m.sensitivity + m.specificity - 1.0
            if best is None or j > best[0]:
                best = (j, float(thr))
    if best is None:
        # keep ~min_sens of positives
        pos = s[y == 1]
        if pos.size == 0:
            return 0.5
        q = max(0.0, 1.0 - min_sens)
        return float(np.quantile(pos, q))
    return best[1]


def fit_stack(Xtr, ytr, Xva, Xte, C: float = 0.5):
    sc = StandardScaler()
    Xtrs = sc.fit_transform(Xtr)
    Xvas = sc.transform(Xva)
    Xtes = sc.transform(Xte)
    clf = LogisticRegression(
        max_iter=4000, class_weight="balanced", C=C, solver="lbfgs"
    )
    clf.fit(Xtrs, ytr)
    return (
        clf.predict_proba(Xvas)[:, 1],
        clf.predict_proba(Xtes)[:, 1],
        sc,
        clf,
    )


def ensemble_scores(models, cfg, device, tf, split: str, weights: list[float]):
    acc = None
    y = None
    meta = None
    for w, m in zip(weights, models):
        s, yy, mm = predict_loader(m, make_loader(split, cfg, tf), device)
        acc = w * s if acc is None else acc + w * s
        y, meta = yy, mm
    return acc, y, meta


def clinical_rows(ids, folder_map, clinical_map):
    rows = []
    for pid in ids:
        folder = folder_map.get(pid, pid)
        entry = clinical_map.get(folder) or clinical_map.get(pid) or {}
        rows.append(
            {
                "person_id": pid,
                "age": entry.get("age"),
                "hpv_bucket": entry.get("hpv_bucket", "missing"),
                "cyto_bucket": entry.get("cyto_bucket", "missing"),
                "raw_hpv": entry.get("raw_hpv"),
                "raw_cyto": entry.get("raw_cyto"),
            }
        )
    return rows


def analyze_errors(ids, y, s, thr, folder_map, clinical_map, n_img_map):
    clin = {r["person_id"]: r for r in clinical_rows(ids, folder_map, clinical_map)}
    pred = (s >= thr).astype(int)
    out = {"fn": [], "fp": [], "tp": [], "tn": []}
    for i, pid in enumerate(ids):
        yt, yp = int(y[i]), int(pred[i])
        key = "tp" if yt == 1 and yp == 1 else (
            "fn" if yt == 1 and yp == 0 else (
                "fp" if yt == 0 and yp == 1 else "tn"
            )
        )
        row = {
            **clin[pid],
            "y_true": yt,
            "score": float(s[i]),
            "n_images": int(n_img_map.get(pid, 0)),
        }
        out[key].append(row)
    # sort FN/FP by score distance to threshold
    out["fn"].sort(key=lambda r: r["score"])
    out["fp"].sort(key=lambda r: -r["score"])
    return out


def summarize_subgroup(errors: dict, key: str) -> dict[str, Any]:
    rows = errors["fn"] + errors["tp"]  # among positives
    if not rows:
        return {}
    from collections import Counter

    return {
        "n_pos": len(rows),
        "n_fn": len(errors["fn"]),
        "fn_hpv": dict(Counter(r["hpv_bucket"] for r in errors["fn"])),
        "fn_cyto": dict(Counter(r["cyto_bucket"] for r in errors["fn"])),
        "fn_ages": [r["age"] for r in errors["fn"]],
        "fn_n_images": [r["n_images"] for r in errors["fn"]],
        "tp_hpv": dict(Counter(r["hpv_bucket"] for r in errors["tp"])),
    }


def eval_model_pack(
    name: str,
    models,
    weights,
    cfg,
    device,
    clinical_map,
    C: float = 0.5,
) -> dict[str, Any]:
    tf = build_transforms("val", int(cfg.get("image_size", 224)), False)
    tr = ensemble_scores(models, cfg, device, tf, "train", weights)
    va = ensemble_scores(models, cfg, device, tf, "val", weights)
    te = ensemble_scores(models, cfg, device, tf, "test", weights)

    ytr, _, ids_tr, Xtr = person_matrix(tr[2], tr[0], tr[1], clinical_map)
    yva, s_img_va, ids_va, Xva = person_matrix(va[2], va[0], va[1], clinical_map)
    yte, s_img_te, ids_te, Xte = person_matrix(te[2], te[0], te[1], clinical_map)

    pva, pte, _, clf = fit_stack(Xtr, ytr, Xva, Xte, C=C)

    folder_map = {}
    n_img = {}
    for m in te[2]:
        folder_map.setdefault(m["person_id"], m["folder_id"])
        n_img[m["person_id"]] = n_img.get(m["person_id"], 0) + 1

    thr_youden = choose_threshold(yva, pva, "youden")
    thr_sens70 = threshold_for_min_sens(yva, pva, 0.70)
    thr_f1 = choose_threshold(yva, pva, "f1")

    def pack_at(thr: float) -> dict:
        return {
            "threshold": thr,
            "val": compute_binary_metrics(yva, pva, thr).to_dict(),
            "test": compute_binary_metrics(yte, pte, thr).to_dict(),
        }

    # Bootstrap on stacked test scores (AUROC/AUPRC independent of thr)
    ci_auroc = bootstrap_ci(yte, pte, lambda yy, ss: roc_auc_score(yy, ss))
    ci_auprc = bootstrap_ci(
        yte,
        pte,
        lambda yy, ss: float(
            __import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(
                yy, ss
            )
        ),
    )

    # Bootstrap sens/spec at fixed thr_sens70 (operating point locked on val)
    def sens_at(yy, ss):
        return compute_binary_metrics(yy, ss, thr_sens70).sensitivity

    def spec_at(yy, ss):
        return compute_binary_metrics(yy, ss, thr_sens70).specificity

    ci_sens = bootstrap_ci(yte, pte, sens_at)
    ci_spec = bootstrap_ci(yte, pte, spec_at)

    # Calibration on test stacked probs
    try:
        frac_pos, mean_pred = calibration_curve(yte, pte, n_bins=8, strategy="quantile")
        calib = {
            "fraction_positives": frac_pos.tolist(),
            "mean_predicted": mean_pred.tolist(),
        }
    except Exception as e:
        calib = {"error": str(e)}

    brier = float(brier_score_loss(yte, pte))
    ece = expected_calibration_error(yte, pte, n_bins=8)

    errors_youden = analyze_errors(
        ids_te, yte, pte, thr_youden, folder_map, clinical_map, n_img
    )
    errors_sens70 = analyze_errors(
        ids_te, yte, pte, thr_sens70, folder_map, clinical_map, n_img
    )

    # Image-only baseline person AUROC for reference
    img_only_test = compute_binary_metrics(
        yte, s_img_te, choose_threshold(yva, s_img_va, "youden")
    ).to_dict()

    return {
        "name": name,
        "C": C,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
        "n_test_pos": int(yte.sum()),
        "n_test_neg": int((yte == 0).sum()),
        "image_only_test_person": img_only_test,
        "operating_points": {
            "youden": pack_at(thr_youden),
            "sens70": pack_at(thr_sens70),
            "f1": pack_at(thr_f1),
        },
        "bootstrap_test": {
            "auroc": ci_auroc,
            "auprc": ci_auprc,
            "sensitivity_at_sens70_thr": ci_sens,
            "specificity_at_sens70_thr": ci_spec,
        },
        "calibration": {
            "brier": brier,
            "ece_8bin": ece,
            "reliability": calib,
        },
        "error_analysis": {
            "youden": {
                "fn": errors_youden["fn"],
                "fp": errors_youden["fp"][:10],
                "summary": summarize_subgroup(errors_youden, "youden"),
            },
            "sens70": {
                "fn": errors_sens70["fn"],
                "fp": errors_sens70["fp"][:10],
                "summary": summarize_subgroup(errors_sens70, "sens70"),
            },
        },
        "test_person_ids": list(ids_te),
        "test_scores": pte.tolist(),
        "test_labels": yte.astype(int).tolist(),
        "coef_abs_top": _coef_note(clf),
    }


def _coef_note(clf: LogisticRegression) -> list[dict[str, float]]:
    # Feature order: [image_score] + clinical (age_z, age_miss, 4 hpv, 7 cyto)
    names = [
        "image_score",
        "age_z",
        "age_missing",
        "hpv_neg",
        "hpv_hr16_18",
        "hpv_other_hr",
        "hpv_missing",
        "cyto_nilm",
        "cyto_ascus",
        "cyto_lsil",
        "cyto_asch",
        "cyto_hsil_plus",
        "cyto_other",
        "cyto_missing",
    ]
    coef = clf.coef_.ravel()
    rows = [{"feature": n, "coef": float(c)} for n, c in zip(names, coef)]
    rows.sort(key=lambda r: abs(r["coef"]), reverse=True)
    return rows


def write_markdown(report: dict, path: Path) -> None:
    primary = report["models"]["primary_50_50"]
    variant = report["models"]["variant_difflr"]
    lines = []
    lines.append("# Thesis evaluation report — CIN2+ (person-level)")
    lines.append("")
    lines.append("## Locked primary model")
    lines.append("")
    lines.append(
        "- **Primary:** 50/50 ensemble of `acetic_resnet50_stable` + `acetic_resnet50_difflr`, "
        "then late logistic stack (image score + HPV/cytology/age), C=0.5."
    )
    lines.append(
        "- **Variant (max AUROC):** difflr alone + same late stack."
    )
    lines.append(
        "- Splits: frozen person-level seed 42. Thresholds chosen on **validation only**."
    )
    lines.append("")
    lines.append("## Primary results (test, n=70 persons)")
    lines.append("")
    b = primary["bootstrap_test"]
    op = primary["operating_points"]["sens70"]["test"]
    oy = primary["operating_points"]["youden"]["test"]
    lines.append(
        f"- **AUROC:** {oy['auroc']:.3f} "
        f"(bootstrap 95% CI {b['auroc']['lo']:.3f}–{b['auroc']['hi']:.3f})"
    )
    lines.append(
        f"- **AUPRC:** {oy['auprc']:.3f} "
        f"(95% CI {b['auprc']['lo']:.3f}–{b['auprc']['hi']:.3f})"
    )
    lines.append(
        f"- **Recommended OP (val sens≥0.70):** sens={op['sensitivity']:.3f}, "
        f"spec={op['specificity']:.3f}, PPV={op['ppv']:.3f}, NPV={op['npv']:.3f}, "
        f"thr={primary['operating_points']['sens70']['threshold']:.3f}"
    )
    lines.append(
        f"- **Youden OP:** sens={oy['sensitivity']:.3f}, spec={oy['specificity']:.3f}, "
        f"thr={primary['operating_points']['youden']['threshold']:.3f}"
    )
    lines.append(
        f"- **Calibration:** Brier={primary['calibration']['brier']:.3f}, "
        f"ECE={primary['calibration']['ece_8bin']:.3f}"
    )
    lines.append("")
    lines.append("## Variant (difflr + stack)")
    lines.append("")
    vb = variant["bootstrap_test"]
    voy = variant["operating_points"]["youden"]["test"]
    vop = variant["operating_points"]["sens70"]["test"]
    lines.append(
        f"- AUROC {voy['auroc']:.3f} (95% CI {vb['auroc']['lo']:.3f}–{vb['auroc']['hi']:.3f})"
    )
    lines.append(
        f"- Youden: sens={voy['sensitivity']:.3f}, spec={voy['specificity']:.3f} "
        f"(often over-specific — prefer sens≥0.70 OP)"
    )
    lines.append(
        f"- Sens≥0.70 OP: sens={vop['sensitivity']:.3f}, spec={vop['specificity']:.3f}"
    )
    lines.append("")
    lines.append("## False negatives at recommended OP (primary)")
    lines.append("")
    fns = primary["error_analysis"]["sens70"]["fn"]
    if not fns:
        lines.append("- None.")
    else:
        lines.append("| person_id | score | age | HPV | cytology | n_images |")
        lines.append("|-----------|-------|-----|-----|----------|----------|")
        for r in fns:
            lines.append(
                f"| {r['person_id']} | {r['score']:.3f} | {r['age']} | "
                f"{r['hpv_bucket']} | {r['cyto_bucket']} | {r['n_images']} |"
            )
    lines.append("")
    lines.append("## Stack coefficients (primary, absolute importance)")
    lines.append("")
    for row in primary["coef_abs_top"][:8]:
        lines.append(f"- `{row['feature']}`: {row['coef']:+.3f}")
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python ml/scripts/thesis_eval.py")
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_plots(report: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    primary = report["models"]["primary_50_50"]
    y = np.asarray(primary["test_labels"])
    s = np.asarray(primary["test_scores"])

    # ROC
    from sklearn.metrics import roc_curve, precision_recall_curve

    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUROC={roc_auc_score(y,s):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Primary model — test ROC (person)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_primary.png", dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"AUPRC={primary['operating_points']['youden']['test']['auprc']:.3f}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Primary model — test PR (person)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_primary.png", dpi=150)
    plt.close(fig)

    rel = primary["calibration"].get("reliability") or {}
    if "mean_predicted" in rel:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(rel["mean_predicted"], rel["fraction_positives"], marker="o")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction CIN2+")
        ax.set_title(
            f"Calibration (Brier={primary['calibration']['brier']:.3f}, "
            f"ECE={primary['calibration']['ece_8bin']:.3f})"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_primary.png", dpi=150)
        plt.close(fig)


def main() -> None:
    set_seed(42)
    device = resolve_device("cuda")
    out_dir = Path("ml/runs/thesis_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    stable_p = Path("ml/runs/acetic_resnet50_stable/best.pt")
    difflr_p = Path("ml/runs/acetic_resnet50_difflr/best.pt")
    m_stable, cfg = load_ckpt(stable_p, device)
    m_difflr, _ = load_ckpt(difflr_p, device)
    clinical_map = build_folder_clinical_map(cfg["my_dataset_root"])

    print("Evaluating primary 50/50 ...")
    primary = eval_model_pack(
        "primary_50_50",
        [m_stable, m_difflr],
        [0.5, 0.5],
        cfg,
        device,
        clinical_map,
    )
    print(
        f"  AUROC={primary['operating_points']['youden']['test']['auroc']:.4f} "
        f"CI=[{primary['bootstrap_test']['auroc']['lo']:.3f},"
        f"{primary['bootstrap_test']['auroc']['hi']:.3f}] "
        f"sens70={primary['operating_points']['sens70']['test']['sensitivity']:.3f}"
    )

    print("Evaluating variant difflr ...")
    variant = eval_model_pack(
        "variant_difflr",
        [m_difflr],
        [1.0],
        cfg,
        device,
        clinical_map,
    )
    print(
        f"  AUROC={variant['operating_points']['youden']['test']['auroc']:.4f} "
        f"CI=[{variant['bootstrap_test']['auroc']['lo']:.3f},"
        f"{variant['bootstrap_test']['auroc']['hi']:.3f}]"
    )

    report = {
        "protocol": {
            "task": "binary_CIN2plus_vs_CIN1minus",
            "unit_of_analysis": "person",
            "split_seed": 42,
            "stack_C": 0.5,
            "threshold_policy_recommended": "val_sensitivity_ge_0.70_then_max_Youden",
            "primary_checkpoints": [str(stable_p), str(difflr_p)],
            "primary_weights": [0.5, 0.5],
        },
        "models": {
            "primary_50_50": primary,
            "variant_difflr": variant,
        },
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_dir / "REPORT.md")
    try_plots(report, out_dir)

    # Freeze pointer
    primary_md = Path("ml/runs/PRIMARY_MODEL.md")
    primary_md.write_text(
        "\n".join(
            [
                "# Primary model (frozen for thesis)",
                "",
                "**Do not keep swapping backbones.** This is the official pipeline.",
                "",
                "1. Image encoders (equal mean):",
                "   - `ml/runs/acetic_resnet50_stable/best.pt`",
                "   - `ml/runs/acetic_resnet50_difflr/best.pt` (`backbone_lr_mult=0.1`)",
                "2. Person score = max over acetic images, then average the two models.",
                "3. Late stack: logistic regression on `[image_score] + clinical`, C=0.5,",
                "   fit on train; choose threshold on val with **sensitivity ≥ 0.70**.",
                "",
                "Full metrics, CIs, calibration, FN list:",
                "`ml/runs/thesis_eval/REPORT.md`",
                "",
                "Reproduce:",
                "```bash",
                "python ml/scripts/thesis_eval.py",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("Wrote", out_dir / "REPORT.md")
    print("Wrote", primary_md)


if __name__ == "__main__":
    main()
