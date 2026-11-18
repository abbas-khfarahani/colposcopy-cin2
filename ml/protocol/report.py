"""Collect OOF tables into thesis-ready markdown, HTML, and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.protocol.constants import PROTOCOL_VERSION
from ml.protocol.figures import (
    save_calibration,
    save_dca,
    save_fold_strip,
    save_leaderboard,
    save_pr_overlay,
    save_roc_overlay,
)
from ml.protocol.io_utils import load_json, save_csv, save_json, utc_now, write_text
from ml.protocol.metrics import auroc_fn, paired_bootstrap_delta, summarize_patient_predictions


def _fmt_ci(ci: dict[str, Any]) -> str:
    if not ci or ci.get("n_boot", 0) == 0:
        return "—"
    return f"{ci['mean']:.3f} ({ci['lo']:.3f}–{ci['hi']:.3f})"


def write_experiment_report(exp_dir: Path, spec_id: str, name: str) -> dict[str, Any]:
    oof_path = exp_dir / "oof_predictions.csv"
    if not oof_path.is_file():
        return {}
    oof = pd.read_csv(oof_path)
    y = oof["label"].to_numpy()
    p = oof["calibrated_probability"].to_numpy()
    z = oof["logit"].to_numpy()
    # Per-fold thresholds already applied; use 0.5 on the already-binarized cols
    # for OP metrics from stored predicted_class, and threshold-independent AUROC.
    thr70 = 0.5
    # Reconstruct a representative threshold display from fold files if present.
    metrics = summarize_patient_predictions(
        y,
        oof["probability"].to_numpy(),
        p,
        z,
        thr70=float(oof["calibrated_probability"].median() * 0 + 0.5),  # placeholder
        thr80=0.5,
    )
    # Override operating-point confusion using stored predictions.
    from sklearn.metrics import confusion_matrix

    def _from_pred(col: str) -> dict[str, Any]:
        pred = oof[col].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        n_pos = int(tp + fn)
        n_neg = int(tn + fp)
        return {
            "sensitivity": float(tp / n_pos) if n_pos else 0.0,
            "specificity": float(tn / n_neg) if n_neg else 0.0,
            "ppv": float(tp / (tp + fp)) if (tp + fp) else 0.0,
            "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    metrics["op70"].update(_from_pred("predicted_class_op70"))
    metrics["op80"].update(_from_pred("predicted_class_op80"))
    save_json(exp_dir / "metrics.json", metrics)

    fig_dir = exp_dir / "figures"
    save_calibration(y, p, fig_dir / "calibration.png", f"{spec_id} calibration")
    save_dca(metrics["decision_curve"], fig_dir / "dca.png", f"{spec_id} decision curve")
    save_roc_overlay({name: metrics["roc"]}, fig_dir / "roc.png", f"{spec_id} ROC")
    return metrics


def write_global_report(root: Path, n_boot: int = 5000) -> None:
    exp_root = root / "experiments"
    if not exp_root.is_dir():
        return
    rows = []
    fold_rows = []
    oofs: dict[str, pd.DataFrame] = {}
    curves = {}
    for exp_dir in sorted(exp_root.glob("*")):
        for seed_dir in sorted(exp_dir.glob("seed*")):
            metrics_path = seed_dir / "metrics.json"
            oof_path = seed_dir / "oof_predictions.csv"
            if not metrics_path.is_file() or not oof_path.is_file():
                continue
            metrics = load_json(metrics_path)
            oof = pd.read_csv(oof_path)
            key = f"{exp_dir.name}/{seed_dir.name}"
            oofs[key] = oof
            disc = metrics["discrimination"]
            rows.append(
                {
                    "experiment_id": exp_dir.name,
                    "seed": seed_dir.name,
                    "name": key,
                    "n": int(len(oof)),
                    "auroc": disc["auroc"],
                    "auroc_lo": disc["auroc_ci"]["lo"],
                    "auroc_hi": disc["auroc_ci"]["hi"],
                    "auprc": disc["auprc"],
                    "sens70": metrics["op70"]["sensitivity"],
                    "spec70": metrics["op70"]["specificity"],
                    "sens80": metrics["op80"]["sensitivity"],
                    "spec80": metrics["op80"]["specificity"],
                    "brier": metrics["calibration"]["brier"],
                    "ece": metrics["calibration"]["ece"],
                    "slope": metrics["calibration"]["slope"],
                }
            )
            curves[key] = metrics["roc"]
            if "outer_fold" in oof.columns:
                for fold, g in oof.groupby("outer_fold"):
                    if g["label"].nunique() < 2:
                        continue
                    from sklearn.metrics import roc_auc_score

                    fold_rows.append(
                        {
                            "experiment_id": exp_dir.name,
                            "seed": seed_dir.name,
                            "fold": int(fold),
                            "auroc": float(roc_auc_score(g["label"], g["calibrated_probability"])),
                        }
                    )

    if not rows:
        return
    table = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    cmp_dir = root / "comparisons"
    save_csv(cmp_dir / "leaderboard.csv", table)
    if oofs:
        save_csv(cmp_dir / "oof_predictions_all.csv", pd.concat(oofs.values(), ignore_index=True))
    save_leaderboard(table[table["seed"] == "seed42"] if (table["seed"] == "seed42").any() else table, cmp_dir / "figures" / "leaderboard.png")
    save_roc_overlay(curves, cmp_dir / "figures" / "roc_overlay.png", "OOF ROC overlay")
    save_pr_overlay(oofs, cmp_dir / "figures" / "pr_overlay.png", "OOF precision–recall")
    if fold_rows:
        folds = pd.DataFrame(fold_rows)
        save_csv(cmp_dir / "fold_auroc.csv", folds)
        save_fold_strip(folds[folds["seed"] == "seed42"] if (folds["seed"] == "seed42").any() else folds, cmp_dir / "figures" / "fold_auroc.png")

    # Paired Δ vs B1
    paired = []
    b1_key = next((k for k in oofs if k.startswith("B1/") and k.endswith("seed42")), None)
    if b1_key is not None:
        base = oofs[b1_key].sort_values("person_id")
        for key, df in oofs.items():
            if key == b1_key:
                continue
            other = df.sort_values("person_id")
            if list(base["person_id"]) != list(other["person_id"]):
                merged = base.merge(other, on="person_id", suffixes=("_a", "_b"))
                y = merged["label_a"].to_numpy()
                a = merged["calibrated_probability_a"].to_numpy()
                b = merged["calibrated_probability_b"].to_numpy()
            else:
                y = base["label"].to_numpy()
                a = base["calibrated_probability"].to_numpy()
                b = other["calibrated_probability"].to_numpy()
            delta = paired_bootstrap_delta(y, a, b, auroc_fn, n_boot=n_boot)
            paired.append({"vs": "B1/seed42", "model": key, **delta})
        save_json(cmp_dir / "paired_bootstrap.json", paired)

    lines = [
        "# Patient-level grouped-CV protocol — results",
        "",
        f"Generated: `{utc_now()}`  ",
        f"Protocol version: `{PROTOCOL_VERSION}`  ",
        "",
        "Primary evidence is **out-of-fold patient predictions** from person-grouped",
        "StratifiedGroupKFold. The original 70-patient holdout is **not** used for",
        "model selection.",
        "",
        "## Leaderboard (OOF patient metrics)",
        "",
        "| Model | AUROC (95% CI) | AUPRC | Sens@OP70 | Spec@OP70 | Sens@OP80 | Spec@OP80 | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['experiment_id']} `{r['seed']}` | "
            f"{r['auroc']:.3f} ({r['auroc_lo']:.3f}–{r['auroc_hi']:.3f}) | "
            f"{r['auprc']:.3f} | {r['sens70']:.3f} | {r['spec70']:.3f} | "
            f"{r['sens80']:.3f} | {r['spec80']:.3f} | {r['brier']:.3f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **AUROC / AUPRC** are threshold-independent.",
        "- **OP70** is the thesis-continuity rule: validation sensitivity ≥ 0.70, then Youden.",
        "- **OP80** is a pre-specified research operating point, not a clinical guideline.",
        "- Calibration (Brier, slope, ECE) is computed on patient probabilities after",
        "  temperature scaling fit on inner-val only.",
        "- **B12** is nested selection among frozen B0 / B1 / B4 / B11: each outer",
        "  fold picks on inner-val AUROC only, then emits that model's outer-test",
        "  scores. It is not a new architecture. Architecture search is frozen.",
        "",
        "Figures: `comparisons/figures/`. Per-experiment OOF tables live under",
        "`experiments/<id>/seed<seed>/oof_predictions.csv`.",
        "",
    ]
    nested_root = exp_root / "B12"
    if nested_root.is_dir():
        lines += [
            "## Nested selection (B12)",
            "",
            "Each outer fold picks among frozen **B0 / B1 / B4 / B11** using inner-val",
            "AUROC only (simplicity tie-break in that order). Outer-test scores of the",
            "chosen model are concatenated. Missing candidates (e.g. B11 on seeds 73/101)",
            "are skipped for that seed.",
            "",
        ]
        for seed_dir in sorted(nested_root.glob("seed*")):
            summ_path = seed_dir / "fold_summaries.json"
            if not summ_path.is_file():
                continue
            rows = load_json(summ_path)
            keys = []
            for row in rows:
                for k in (row.get("inner_val_auroc") or {}):
                    if k not in keys:
                        keys.append(k)
            hdr = "| Fold | Choice | " + " | ".join(keys) + " |"
            sep = "|---|:---:|" + "|".join(["---:" for _ in keys]) + "|"
            lines += [f"### `{seed_dir.name}`", "", hdr, sep]
            for row in rows:
                vals = row.get("inner_val_auroc") or {}
                cells = []
                for k in keys:
                    v = vals.get(k)
                    cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "—")
                lines.append(f"| {row.get('fold')} | {row.get('choice')} | " + " | ".join(cells) + " |")
            lines.append("")
    if paired:
        lines += ["## Paired ΔAUROC vs B1 (ResNet50 GAP)", "", "| Model | ΔAUROC | 95% CI | excludes 0? |", "|---|---:|---:|:---:|"]
        for p in paired:
            lines.append(
                f"| {p['model']} | {p['delta_mean']:+.3f} | "
                f"{p['lo']:+.3f}–{p['hi']:+.3f} | {'yes' if p.get('ci_excludes_zero') else 'no'} |"
            )
        lines.append("")
    write_text(cmp_dir / "REPORT.md", "\n".join(lines))
    write_text(root / "REPORT.md", "\n".join(lines))

    html = [
        "<!doctype html><meta charset='utf-8'><title>CIN2+ protocol results</title>",
        "<style>body{font-family:Georgia,serif;margin:32px auto;max-width:980px;color:#1a1a1a}",
        "table{border-collapse:collapse;width:100%;font-size:14px} td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}",
        "th{background:#f4f6f8} img{max-width:100%;margin:12px 0;border:1px solid #eee}",
        "code{background:#f4f6f8;padding:1px 4px}</style>",
        "<h1>Patient-level CIN2+ protocol</h1>",
        f"<p>Updated {utc_now()}. Version {PROTOCOL_VERSION}.</p>",
        "<p><a href='comparisons/REPORT.md'>Markdown report</a> · ",
        "<a href='comparisons/leaderboard.csv'>leaderboard.csv</a></p>",
        "<h2>Leaderboard</h2><table><tr>",
        "<th>Model</th><th>AUROC</th><th>AUPRC</th><th>Sens70</th><th>Spec70</th><th>Brier</th></tr>",
    ]
    for _, r in table.iterrows():
        html.append(
            f"<tr><td>{r['experiment_id']} {r['seed']}</td>"
            f"<td>{r['auroc']:.3f} ({r['auroc_lo']:.3f}–{r['auroc_hi']:.3f})</td>"
            f"<td>{r['auprc']:.3f}</td><td>{r['sens70']:.3f}</td>"
            f"<td>{r['spec70']:.3f}</td><td>{r['brier']:.3f}</td></tr>"
        )
    html += [
        "</table>",
        "<h2>Figures</h2>",
        "<img src='comparisons/figures/leaderboard.png' alt='leaderboard'>",
        "<img src='comparisons/figures/roc_overlay.png' alt='roc'>",
        "<img src='comparisons/figures/pr_overlay.png' alt='pr'>",
        "<img src='comparisons/figures/fold_auroc.png' alt='folds'>",
    ]
    write_text(root / "index.html", "\n".join(html))


def write_model_card(root: Path, chosen: str) -> None:
    text = f"""# MODEL CARD — protocol `{PROTOCOL_VERSION}`

## Intended use
Research prototype for **patient-level CIN2+ risk ranking** from acetic-acid
colposcopy images, with optional HPV/cytology/age fusion. Not a diagnostic device.
Not for screening deployment without external validation.

## Chosen configuration
`{chosen}`

Selected from grouped OOF predictions using the pre-specified hierarchy:
patient AUROC, then AUPRC, then OP70 sensitivity/specificity, then simplicity.
B12 (nested selection among frozen B0/B1/B4/B11) is an estimate of the
selection rule and is excluded from this winner pick. Architecture search
is frozen; do not add new CNN/MIL/SSL IDs.

## Limitations
- Internal 462-person cohort; no external site.
- ~18 CIN2+ persons per 70-person fold-equivalent; operating points are noisy.
- Attention maps are error-analysis tools, not causal lesion proof.
- Domain shift (device/time) is expected; Yuan et al. show large drops on new cameras.
- Nested B12 still uses inner-val both for CNN early-stopping and for model choice
  (mildly optimistic for image models). Outer test is never used to pick.

## Data
See `setup/DATA_CARD.md`. Person-grouped splits in `setup/split_manifest_seed*.csv`.
"""
    write_text(root / "MODEL_CARD.md", text)
