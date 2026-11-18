"""Patient-level discrimination, calibration, DCA, and bootstrap CIs."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)
    if np.unique(y).size < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y, scores))
        auprc = float(average_precision_score(y, scores))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)
    sens = _safe_div(tp, n_pos)
    spec = _safe_div(tn, n_neg)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": 0.5 * (sens + spec),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "brier": float(brier_score_loss(y, np.clip(scores, 0, 1))),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(y.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": _safe_div(n_pos, int(y.size)),
    }


def expected_calibration_error(y: np.ndarray, s: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(int)
    s = np.clip(np.asarray(s).astype(float), 0, 1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        hi = bins[i + 1]
        m = (s >= bins[i]) & (s < hi if i < n_bins - 1 else s <= hi)
        if not np.any(m):
            continue
        ece += float(m.mean()) * abs(float(y[m].mean()) - float(s[m].mean()))
    return float(ece)


def calibration_slope_intercept(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """Logistic calibration: logit(y) ~ intercept + slope * logit_hat."""
    y = np.asarray(y).astype(int)
    z = np.asarray(logits).astype(float).reshape(-1, 1)
    if np.unique(y).size < 2:
        return {"intercept": float("nan"), "slope": float("nan")}
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    try:
        clf.fit(z, y)
        return {"intercept": float(clf.intercept_[0]), "slope": float(clf.coef_[0, 0])}
    except Exception:
        clf = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
        clf.fit(z, y)
        return {"intercept": float(clf.intercept_[0]), "slope": float(clf.coef_[0, 0])}


def decision_curve(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores).astype(float)
    n = len(y)
    prevalence = float(y.mean()) if n else 0.0
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.80, 16)
    nb_model, nb_all, nb_none = [], [], []
    for pt in thresholds:
        pred = s >= pt
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        nb = (tp / n) - (fp / n) * (pt / max(1.0 - pt, 1e-8))
        nb_model.append(float(nb))
        nb_all.append(float(prevalence - (1.0 - prevalence) * (pt / max(1.0 - pt, 1e-8))))
        nb_none.append(0.0)
    return {
        "threshold": thresholds.astype(float).tolist(),
        "net_benefit_model": nb_model,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": nb_none,
    }


def roc_points(y: np.ndarray, scores: np.ndarray) -> dict[str, list[float]]:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores).astype(float)
    if np.unique(y).size < 2:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}
    fpr, tpr, _ = roc_curve(y, s)
    return {"fpr": fpr.astype(float).tolist(), "tpr": tpr.astype(float).tolist()}


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    s = np.asarray(scores).astype(float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, sb = y[idx], s[idx]
        if np.unique(yb).size < 2:
            continue
        try:
            vals.append(float(fn(yb, sb)))
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


def paired_bootstrap_delta(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Δ = metric(B) − metric(A) on the same resampled patients."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    a = np.asarray(scores_a).astype(float)
    b = np.asarray(scores_b).astype(float)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        try:
            deltas.append(float(fn(yb, b[idx]) - fn(yb, a[idx])))
        except Exception:
            continue
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        return {"delta_mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    return {
        "delta_mean": float(arr.mean()),
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "n_boot": int(arr.size),
        "ci_excludes_zero": bool(np.quantile(arr, alpha / 2) > 0 or np.quantile(arr, 1 - alpha / 2) < 0),
    }


def auroc_fn(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def auprc_fn(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def summarize_patient_predictions(
    y: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    logits: np.ndarray,
    thr70: float,
    thr80: float,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    m70 = binary_metrics(y, calibrated_prob, thr70)
    m80 = binary_metrics(y, calibrated_prob, thr80)
    slope = calibration_slope_intercept(y, logits)
    out = {
        "discrimination": {
            "auroc": m70["auroc"],
            "auprc": m70["auprc"],
            "auroc_ci": bootstrap_ci(y, calibrated_prob, auroc_fn, n_boot=n_boot, seed=seed),
            "auprc_ci": bootstrap_ci(y, calibrated_prob, auprc_fn, n_boot=n_boot, seed=seed),
        },
        "op70": m70,
        "op80": m80,
        "calibration": {
            "brier": m70["brier"],
            "ece": expected_calibration_error(y, calibrated_prob),
            **slope,
        },
        "decision_curve": decision_curve(y, calibrated_prob),
        "roc": roc_points(y, calibrated_prob),
    }
    return out
