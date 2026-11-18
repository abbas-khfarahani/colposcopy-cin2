"""Binary classification metrics for CIN2+ screening (image- and person-level)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class MetricsBundle:
    auroc: float
    auprc: float
    accuracy: float
    f1: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    n: int
    n_pos: int
    n_neg: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strategy: str = "youden",
) -> float:
    """Pick an operating point on validation scores (youden | f1)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5

    strategy = strategy.lower().strip()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    if thresholds.size == 0:
        return 0.5

    if strategy == "f1":
        best_t, best_f1 = 0.5, -1.0
        for t in thresholds:
            pred = (y_score >= t).astype(int)
            score = f1_score(y_true, pred, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_t = float(t)
        return best_t

    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> MetricsBundle:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    if np.unique(y_true).size < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)

    return MetricsBundle(
        auroc=auroc,
        auprc=auprc,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        sensitivity=_safe_div(tp, n_pos),
        specificity=_safe_div(tn, n_neg),
        ppv=_safe_div(tp, tp + fp),
        npv=_safe_div(tn, tn + fn),
        threshold=float(threshold),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        n=int(y_true.size),
        n_pos=n_pos,
        n_neg=n_neg,
    )


def aggregate_person_level(
    person_ids: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    score_reduce: str = "max",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collapse image predictions to one score/label per person.

    Label = max image label (person is CIN2+ if any image is).
    Score = max or mean of image scores (default max: screening-friendly).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    buckets: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for pid, y, s in zip(person_ids, y_true, y_score):
        buckets[str(pid)].append((int(y), float(s)))

    ids: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    reduce = score_reduce.lower().strip()
    for pid in sorted(buckets.keys()):
        pairs = buckets[pid]
        ys = [p[0] for p in pairs]
        ss = [p[1] for p in pairs]
        ids.append(pid)
        labels.append(max(ys))
        if reduce == "mean":
            scores.append(float(np.mean(ss)))
        else:
            scores.append(float(np.max(ss)))
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), ids
