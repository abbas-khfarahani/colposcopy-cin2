"""Thesis-quality figures written under ml/runs/protocol/comparisons/figures/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )


def save_roc_overlay(curves: dict[str, dict[str, list[float]]], path: Path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", c="#9aa0a6", lw=1, label="Chance")
    for name, pts in curves.items():
        ax.plot(pts["fpr"], pts["tpr"], lw=2, label=name)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_pr_overlay(frames: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    from sklearn.metrics import precision_recall_curve

    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for name, df in frames.items():
        y = df["label"].to_numpy(dtype=int)
        s = df["calibrated_probability"].to_numpy(dtype=float)
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, lw=2, label=name)
    prevalence = next(iter(frames.values()))["label"].mean()
    ax.axhline(prevalence, ls="--", c="#9aa0a6", lw=1, label=f"Prevalence ({prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_calibration(y: np.ndarray, p: np.ndarray, path: Path, title: str, n_bins: int = 10) -> None:
    from sklearn.calibration import calibration_curve

    _style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", c="#9aa0a6", lw=1)
    try:
        frac, meanp = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
        ax.plot(meanp, frac, marker="o", lw=2)
    except Exception:
        pass
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_dca(dca: dict[str, Any], path: Path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    t = np.asarray(dca["threshold"])
    ax.plot(t, dca["net_benefit_model"], lw=2, label="Model")
    ax.plot(t, dca["net_benefit_treat_all"], lw=1.5, ls="--", label="Treat all")
    ax.plot(t, dca["net_benefit_treat_none"], lw=1.5, ls=":", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_leaderboard(table: pd.DataFrame, path: Path) -> None:
    _style()
    df = table.sort_values("auroc", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.42 * len(df) + 1.5)))
    y = np.arange(len(df))
    ax.barh(y, df["auroc"], color="#1f4e79", height=0.7)
    if "auroc_lo" in df.columns:
        xerr = np.vstack(
            [
                df["auroc"] - df["auroc_lo"],
                df["auroc_hi"] - df["auroc"],
            ]
        )
        ax.errorbar(df["auroc"], y, xerr=xerr, fmt="none", ecolor="#333", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["experiment_id"] + "  " + df["name"])
    ax.set_xlabel("Patient AUROC (OOF)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Protocol leaderboard")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_fold_strip(fold_table: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    experiments = list(fold_table["experiment_id"].unique())
    for i, exp in enumerate(experiments):
        vals = fold_table.loc[fold_table["experiment_id"] == exp, "auroc"].to_numpy()
        ax.scatter(np.full_like(vals, i, dtype=float) + np.linspace(-0.12, 0.12, len(vals)), vals, s=28)
        ax.plot([i - 0.2, i + 0.2], [np.nanmean(vals)] * 2, c="#1f4e79", lw=2)
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, rotation=30, ha="right")
    ax.set_ylabel("Fold patient AUROC")
    ax.set_title("OOF AUROC by outer fold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
