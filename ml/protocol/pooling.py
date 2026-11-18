"""Person-level score aggregation: max, mean, LSE."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def logsumexp_pool(scores: np.ndarray, tau: float) -> float:
    s = np.asarray(scores, dtype=float)
    tau = float(tau)
    m = np.max(s / tau)
    return float(tau * (m + np.log(np.sum(np.exp(s / tau - m)))))


def aggregate_person_scores(
    person_ids: Iterable[str],
    y_true: np.ndarray,
    scores: np.ndarray,
    reduce: str = "max",
    tau: float = 0.5,
    person_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Collapse image scores to one score per person.

    `y_true` may be image labels; `person_labels` overrides the person target
    when provided. Returns labels, scores, ids, image_counts.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    buckets: dict[str, list[tuple[int, float]]] = defaultdict(list)
    person_y: dict[str, int] = {}
    for i, pid in enumerate(person_ids):
        pid = str(pid)
        buckets[pid].append((int(y_true[i]), float(scores[i])))
        if person_labels is not None:
            person_y[pid] = int(person_labels[i])

    reduce = reduce.lower().strip()
    ids: list[str] = []
    labels: list[int] = []
    out: list[float] = []
    counts: list[int] = []
    for pid in sorted(buckets):
        pairs = buckets[pid]
        ys = [p[0] for p in pairs]
        ss = np.asarray([p[1] for p in pairs], dtype=float)
        ids.append(pid)
        labels.append(person_y.get(pid, max(ys)))
        counts.append(len(ss))
        if reduce == "mean":
            out.append(float(ss.mean()))
        elif reduce.startswith("lse"):
            out.append(logsumexp_pool(ss, tau=tau))
        else:
            out.append(float(ss.max()))
    return (
        np.asarray(labels, dtype=int),
        np.asarray(out, dtype=float),
        ids,
        np.asarray(counts, dtype=int),
    )


def persons_from_image_frame(frame: "pd.DataFrame", reduce: str = "max", tau: float = 0.5) -> "pd.DataFrame":
    """One row per person from image-level logit/probability predictions."""
    import pandas as pd

    rows = []
    reduce = reduce.lower().strip()
    for pid, grp in frame.groupby("person_id", sort=True):
        logits = grp["logit"].to_numpy(dtype=float)
        probs = grp["probability"].to_numpy(dtype=float)
        if reduce == "mean":
            logit = float(logits.mean())
            prob = float(probs.mean())
        elif reduce.startswith("lse"):
            logit = logsumexp_pool(logits, tau=tau)
            prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40))))
        else:
            logit = float(logits.max())
            prob = float(probs.max())
        r0 = grp.iloc[0]
        rows.append(
            {
                "person_id": str(pid),
                "label": int(r0["person_label"]),
                "person_label": int(r0["person_label"]),
                "logit": logit,
                "probability": prob,
                "image_count": int(len(grp)),
                "hpv_bucket": r0.get("hpv_bucket"),
                "cyto_bucket": r0.get("cyto_bucket"),
                "age": r0.get("age"),
            }
        )
    return pd.DataFrame(rows)
