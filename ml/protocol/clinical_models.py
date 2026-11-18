"""Fold-safe clinical-only logistic regression and late image+clinical stack."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ml.protocol.constants import CLINICAL_FEATURE_NAMES, CYTO_BUCKETS, HPV_BUCKETS, STACK_C_GRID


def _one_hot(bucket: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if bucket == c else 0.0 for c in choices]


def fit_age_stats(persons: pd.DataFrame) -> tuple[float, float]:
    ages = pd.to_numeric(persons["age"], errors="coerce")
    mean = float(ages.mean(skipna=True)) if ages.notna().any() else 35.0
    std = float(ages.std(skipna=True)) if ages.notna().sum() > 1 else 1.0
    if std < 1e-6:
        std = 1.0
    return mean, std


def clinical_vector(row: pd.Series, age_mean: float, age_std: float) -> np.ndarray:
    age_raw = row.get("age")
    missing = 1.0 if age_raw is None or (isinstance(age_raw, float) and np.isnan(age_raw)) else 0.0
    age_z = 0.0 if missing else (float(age_raw) - age_mean) / age_std
    hpv = str(row.get("hpv_bucket") or "missing")
    cyto = str(row.get("cyto_bucket") or "missing")
    vec = [age_z, missing] + _one_hot(hpv, HPV_BUCKETS) + _one_hot(cyto, CYTO_BUCKETS)
    return np.asarray(vec, dtype=np.float32)


def clinical_matrix(frame: pd.DataFrame, age_mean: float, age_std: float) -> np.ndarray:
    return np.stack([clinical_vector(row, age_mean, age_std) for _, row in frame.iterrows()], axis=0)


def _fit_lr(x: np.ndarray, y: np.ndarray, c: float) -> LogisticRegression:
    clf = LogisticRegression(
        C=float(c),
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(x, y)
    return clf


def select_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: Iterable[float] = STACK_C_GRID,
    refit_with_val: bool = True,
) -> tuple[LogisticRegression, float, dict[str, Any]]:
    best_c = float(list(c_grid)[0])
    best_auc = -1.0
    scores = {}
    for c in c_grid:
        clf = _fit_lr(x_train, y_train, c)
        if len(np.unique(y_val)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_val, clf.predict_proba(x_val)[:, 1]))
        scores[str(c)] = auc
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_c = float(c)
    if refit_with_val:
        x_all = np.concatenate([x_train, x_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)
        clf = _fit_lr(x_all, y_all, best_c)
    else:
        clf = _fit_lr(x_train, y_train, best_c)
    return clf, best_c, {"val_auroc_by_C": scores, "best_C": best_c, "best_val_auroc": best_auc}


def predict_logistic(clf: LogisticRegression, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logit = clf.decision_function(x)
    prob = clf.predict_proba(x)[:, 1]
    return logit.astype(float), prob.astype(float)


def fit_score_scaler(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    mean = float(v.mean())
    std = float(v.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


def apply_score_scaler(values: np.ndarray, mean: float, std: float, clip: float = 5.0) -> np.ndarray:
    z = (np.asarray(values, dtype=float) - mean) / std
    return np.clip(z, -float(clip), clip)


def stack_design_matrix(
    frame: pd.DataFrame,
    age_mean: float,
    age_std: float,
    image_score_col: str = "logit",
) -> np.ndarray:
    clin = clinical_matrix(frame, age_mean, age_std)
    score = frame[image_score_col].to_numpy(dtype=float).reshape(-1, 1)
    return np.concatenate([score, clin], axis=1)


FEATURE_NAMES_STACK: tuple[str, ...] = ("image_logit",) + CLINICAL_FEATURE_NAMES


def coefficient_table(clf: LogisticRegression, names: tuple[str, ...]) -> dict[str, float]:
    coef = clf.coef_.ravel()
    out = {n: float(c) for n, c in zip(names, coef)}
    out["intercept"] = float(clf.intercept_.ravel()[0])
    return out
