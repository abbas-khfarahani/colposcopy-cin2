"""Temperature scaling and validation-locked operating points."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


def _as_1d(values) -> np.ndarray:
    """Column extractors can return a (n, k) array if names are duplicated."""
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def temperature_scale(
    logits: np.ndarray,
    y: np.ndarray,
    max_iter: int = 200,
) -> float:
    """Fit a scalar temperature on inner-val logits by minimizing NLL."""
    y_t = torch.tensor(_as_1d(y).astype(np.float32))
    z = torch.tensor(_as_1d(logits).astype(np.float32))
    log_t = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        temp = log_t.exp().clamp(min=1e-3, max=100.0)
        loss = F.binary_cross_entropy_with_logits(z / temp, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(min=1e-3, max=100.0).detach())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = _as_1d(logits).astype(float) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _youden(y: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    pred = (scores >= threshold).astype(int)
    pos = y == 1
    neg = y == 0
    sens = pred[pos].mean() if pos.any() else 0.0
    spec = (1 - pred[neg]).mean() if neg.any() else 0.0
    return float(sens + spec - 1.0)


def choose_threshold_min_sens(
    y: np.ndarray,
    scores: np.ndarray,
    min_sens: float = 0.70,
) -> dict[str, Any]:
    """Among thresholds with sensitivity >= min_sens, maximize Youden.

    If none qualify, take the threshold with maximum sensitivity (flagged).
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    if y.size == 0 or np.unique(y).size < 2:
        return {"threshold": 0.5, "qualified": False, "strategy": "fallback_0.5"}

    fpr, tpr, thr = roc_curve(y, scores)
    valid = np.isfinite(thr)
    fpr, tpr, thr = fpr[valid], tpr[valid], thr[valid]
    if thr.size == 0:
        return {"threshold": 0.5, "qualified": False, "strategy": "fallback_0.5"}

    qualified = tpr >= float(min_sens)
    if qualified.any():
        youden = tpr[qualified] - fpr[qualified]
        pick = int(np.argmax(youden))
        idx = np.flatnonzero(qualified)[pick]
        return {
            "threshold": float(thr[idx]),
            "qualified": True,
            "strategy": f"sens>={min_sens:.2f}_then_youden",
            "val_sensitivity": float(tpr[idx]),
            "val_specificity": float(1.0 - fpr[idx]),
        }

    idx = int(np.argmax(tpr))
    return {
        "threshold": float(thr[idx]),
        "qualified": False,
        "strategy": f"max_sensitivity_unconstrained (no OP reached {min_sens:.2f})",
        "val_sensitivity": float(tpr[idx]),
        "val_specificity": float(1.0 - fpr[idx]),
    }


def choose_threshold_max_spec_at_sens(
    y: np.ndarray,
    scores: np.ndarray,
    min_sens: float = 0.80,
) -> dict[str, Any]:
    """Maximum specificity subject to sensitivity >= min_sens."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    if y.size == 0 or np.unique(y).size < 2:
        return {"threshold": 0.5, "qualified": False, "strategy": "fallback_0.5"}
    fpr, tpr, thr = roc_curve(y, scores)
    valid = np.isfinite(thr)
    fpr, tpr, thr = fpr[valid], tpr[valid], thr[valid]
    qualified = tpr >= float(min_sens)
    if qualified.any():
        spec = 1.0 - fpr[qualified]
        pick = int(np.argmax(spec))
        idx = np.flatnonzero(qualified)[pick]
        return {
            "threshold": float(thr[idx]),
            "qualified": True,
            "strategy": f"max_spec_s.t._sens>={min_sens:.2f}",
            "val_sensitivity": float(tpr[idx]),
            "val_specificity": float(1.0 - fpr[idx]),
        }
    idx = int(np.argmax(tpr))
    return {
        "threshold": float(thr[idx]),
        "qualified": False,
        "strategy": f"max_sensitivity_unconstrained (no OP reached {min_sens:.2f})",
        "val_sensitivity": float(tpr[idx]),
        "val_specificity": float(1.0 - fpr[idx]),
    }


def apply_fold_postprocess(val_person, test_person):
    """Fit temperature + OP70/OP80 on inner-val; apply to outer-test."""
    val_person = val_person.loc[:, ~val_person.columns.duplicated()]
    test_person = test_person.loc[:, ~test_person.columns.duplicated()]
    y_val = _as_1d(val_person["label"]).astype(int)
    z_val = _as_1d(val_person["logit"]).astype(float)
    z_test = _as_1d(test_person["logit"]).astype(float)
    temperature = temperature_scale(z_val, y_val)
    val = val_person.copy()
    test = test_person.copy()
    val["calibrated_probability"] = apply_temperature(z_val, temperature)
    test["calibrated_probability"] = apply_temperature(z_test, temperature)
    op70 = choose_threshold_min_sens(y_val, val["calibrated_probability"].to_numpy(), 0.70)
    op80 = choose_threshold_max_spec_at_sens(
        y_val, val["calibrated_probability"].to_numpy(), 0.80
    )
    val["predicted_class_op70"] = (val["calibrated_probability"] >= op70["threshold"]).astype(int)
    val["predicted_class_op80"] = (val["calibrated_probability"] >= op80["threshold"]).astype(int)
    test["predicted_class_op70"] = (test["calibrated_probability"] >= op70["threshold"]).astype(int)
    test["predicted_class_op80"] = (test["calibrated_probability"] >= op80["threshold"]).astype(int)
    info = {"temperature": temperature, "op70": op70, "op80": op80}
    return val, test, info
