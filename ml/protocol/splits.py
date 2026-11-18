"""Person-grouped stratified CV splits and leakage audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ml.protocol.cohort import Cohort
from ml.protocol.constants import MIN_POSITIVES_PER_FOLD, N_INNER_VAL_FOLDS, N_OUTER_FOLDS
from ml.protocol.io_utils import save_csv, save_json


def _person_arrays(persons: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = persons["person_id"].to_numpy()
    y = persons["person_label"].to_numpy(dtype=int)
    return ids, y, ids.copy()


def make_outer_splits(
    persons: pd.DataFrame,
    n_splits: int = N_OUTER_FOLDS,
    seed: int = 42,
) -> pd.DataFrame:
    ids, y, groups = _person_arrays(persons)
    n_pos = int(y.sum())
    if n_splits > n_pos:
        raise ValueError(f"Need at least {n_splits} positives to make {n_splits} folds")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = {}
    fold_pos: dict[int, int] = {}
    dummy_x = np.zeros((len(ids), 1))
    for fold, (_, test_idx) in enumerate(cv.split(dummy_x, y, groups)):
        pos = int(y[test_idx].sum())
        fold_pos[fold] = pos
        for i in test_idx:
            fold_of[str(ids[i])] = int(fold)
    if min(fold_pos.values()) < MIN_POSITIVES_PER_FOLD and n_splits > 4:
        # Deterministic fallback from labels only (never from model scores).
        return make_outer_splits(persons, n_splits=4, seed=seed)

    out = persons[["person_id", "person_label", "n_images", "legacy_split"]].copy()
    out["outer_fold"] = out["person_id"].map(fold_of).astype(int)
    out["split_seed"] = int(seed)
    out["n_outer_folds"] = int(n_splits)
    return out


def assign_inner_val(
    outer: pd.DataFrame,
    outer_fold: int,
    n_inner: int = N_INNER_VAL_FOLDS,
    seed: int = 42,
) -> pd.DataFrame:
    """Mark inner-val vs inner-train inside one outer training set."""
    train = outer[outer["outer_fold"] != outer_fold].copy()
    if len(train) < n_inner:
        raise RuntimeError("Not enough training persons for inner split")
    ids, y, groups = _person_arrays(train)
    cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    dummy_x = np.zeros((len(ids), 1))
    _, val_idx = next(iter(cv.split(dummy_x, y, groups)))
    val_ids = set(ids[val_idx].astype(str))
    role = []
    for _, row in outer.iterrows():
        pid = str(row["person_id"])
        if int(row["outer_fold"]) == int(outer_fold):
            role.append("outer_test")
        elif pid in val_ids:
            role.append("inner_val")
        else:
            role.append("inner_train")
    tagged = outer.copy()
    tagged["role"] = role
    tagged["eval_outer_fold"] = int(outer_fold)
    return tagged


def audit_splits(outer: pd.DataFrame, images: pd.DataFrame) -> dict[str, Any]:
    """Hard-fail checks. Returns a report dict; raises on leakage."""
    errors: list[str] = []
    by_person = outer.groupby("person_id")["outer_fold"].nunique()
    leaked_persons = by_person[by_person > 1]
    if len(leaked_persons):
        errors.append(f"persons in multiple outer folds: {list(leaked_persons.index)}")

    merged = images[["person_id"]].merge(
        outer[["person_id", "outer_fold"]], on="person_id", how="left"
    )
    if merged["outer_fold"].isna().any():
        missing = merged.loc[merged["outer_fold"].isna(), "person_id"].unique()
        errors.append(f"images with no fold: {missing[:10]}")
    img_folds = merged.groupby("person_id")["outer_fold"].nunique()
    leaked_img = img_folds[img_folds > 1]
    if len(leaked_img):
        errors.append(f"images of one person span folds: {list(leaked_img.index)}")

    first_ids = [
        pid for pid in outer["person_id"].astype(str) if "__" in pid
    ]
    # Canonical IDs should already be stripped; flag any leftover suffix.
    if first_ids:
        errors.append(f"non-canonical person_id values remain: {first_ids[:10]}")

    pos = (
        outer.groupby("outer_fold")["person_label"]
        .agg(["size", "sum"])
        .rename(columns={"size": "n", "sum": "n_pos"})
    )
    pos["n_neg"] = pos["n"] - pos["n_pos"]
    if int(pos["n_pos"].min()) < MIN_POSITIVES_PER_FOLD:
        errors.append(
            f"a fold has <{MIN_POSITIVES_PER_FOLD} positives:\n{pos.to_string()}"
        )

    report = {
        "ok": not errors,
        "errors": errors,
        "n_persons": int(len(outer)),
        "n_folds": int(outer["outer_fold"].nunique()),
        "fold_counts": pos.astype(int).to_dict(orient="index"),
        "legacy_by_fold": outer.groupby("outer_fold")["legacy_split"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict(),
    }
    if errors:
        raise RuntimeError("Split leakage audit failed:\n- " + "\n- ".join(errors))
    return report


def write_split_manifest(
    outer: pd.DataFrame,
    images: pd.DataFrame,
    dest_dir,
    seed: int,
) -> dict[str, Any]:
    audit = audit_splits(outer, images)
    save_csv(dest_dir / f"split_manifest_seed{seed}.csv", outer)
    img_m = images[["person_id", "folder_id", "file", "image_path_rel", "label_cin2plus"]].copy()
    img_m = img_m.merge(outer[["person_id", "outer_fold", "split_seed"]], on="person_id")
    save_csv(dest_dir / f"image_fold_manifest_seed{seed}.csv", img_m)
    save_json(dest_dir / f"leakage_audit_seed{seed}.json", audit)
    return audit


def person_ids_for_role(tagged: pd.DataFrame, role: str) -> list[str]:
    return tagged.loc[tagged["role"] == role, "person_id"].astype(str).tolist()


def filter_images(images: pd.DataFrame, person_ids: list[str]) -> pd.DataFrame:
    allow = set(person_ids)
    return images[images["person_id"].astype(str).isin(allow)].reset_index(drop=True)
