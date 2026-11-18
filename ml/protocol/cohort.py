"""Load the cleaned acetic cohort (images + person labels + clinical)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.data.clinical import build_folder_clinical_map
from ml.protocol.constants import DATASET_ROOT, LEGACY_SPLITS_DIR
from ml.protocol.io_utils import save_csv, save_json, write_text


@dataclass(frozen=True)
class Cohort:
    images: pd.DataFrame
    persons: pd.DataFrame
    dataset_root: Path

    @property
    def n_images(self) -> int:
        return int(len(self.images))

    @property
    def n_persons(self) -> int:
        return int(len(self.persons))

    @property
    def n_positives(self) -> int:
        return int(self.persons["person_label"].sum())


def _canonical_person_id(value: str) -> str:
    return str(value).strip().split("__", 1)[0]


def load_cohort(
    dataset_root: Path | None = None,
    splits_dir: Path | None = None,
) -> Cohort:
    root = Path(dataset_root) if dataset_root else DATASET_ROOT
    splits = Path(splits_dir) if splits_dir else LEGACY_SPLITS_DIR
    images_path = splits / "images_with_split.csv"
    persons_path = splits / "persons_with_split.csv"
    if not images_path.is_file():
        raise FileNotFoundError(images_path)

    images = pd.read_csv(images_path)
    images["person_id"] = images["person_id"].map(_canonical_person_id)
    images["folder_id"] = images["folder_id"].astype(str)
    images["label_cin2plus"] = images["label_cin2plus"].astype(int)
    images["legacy_split"] = images["split"].astype(str)
    images["image_path_rel"] = images["image_path_rel"].astype(str)

    clinical_map = build_folder_clinical_map(root)
    hpv, cyto, age = [], [], []
    for _, row in images.iterrows():
        entry = clinical_map.get(str(row["folder_id"])) or clinical_map.get(
            str(row["person_id"])
        )
        if entry is None:
            hpv.append("missing")
            cyto.append("missing")
            age.append(None)
        else:
            hpv.append(entry["hpv_bucket"])
            cyto.append(entry["cyto_bucket"])
            age.append(entry.get("age"))
    images["hpv_bucket"] = hpv
    images["cyto_bucket"] = cyto
    images["age"] = age

    if persons_path.is_file():
        persons_src = pd.read_csv(persons_path)
        persons_src["person_id"] = persons_src["person_id"].map(_canonical_person_id)
        person_label = {
            str(r["person_id"]): int(r["person_any_cin2plus"])
            for _, r in persons_src.iterrows()
        }
        legacy_split = {
            str(r["person_id"]): str(r["split"]) for _, r in persons_src.iterrows()
        }
    else:
        person_label = (
            images.groupby("person_id")["label_cin2plus"].max().astype(int).to_dict()
        )
        legacy_split = (
            images.groupby("person_id")["legacy_split"].agg(lambda s: s.iloc[0]).to_dict()
        )

    images["person_label"] = images["person_id"].map(person_label).astype(int)

    person_rows: list[dict[str, Any]] = []
    for pid, grp in images.groupby("person_id", sort=True):
        # Most-complete clinical row: prefer a non-missing age, then HPV/cyto.
        ranked = grp.copy()
        ranked["_score"] = (
            ranked["age"].notna().astype(int) * 2
            + (ranked["hpv_bucket"] != "missing").astype(int)
            + (ranked["cyto_bucket"] != "missing").astype(int)
        )
        best = ranked.sort_values("_score", ascending=False).iloc[0]
        person_rows.append(
            {
                "person_id": pid,
                "person_label": int(person_label[pid]),
                "n_images": int(len(grp)),
                "n_folders": int(grp["folder_id"].nunique()),
                "legacy_split": legacy_split.get(pid, str(best["legacy_split"])),
                "age": None if pd.isna(best["age"]) else float(best["age"]),
                "hpv_bucket": str(best["hpv_bucket"]),
                "cyto_bucket": str(best["cyto_bucket"]),
                "folder_ids": "|".join(sorted(grp["folder_id"].unique())),
            }
        )
    persons = pd.DataFrame(person_rows)
    return Cohort(images=images, persons=persons, dataset_root=root)


def write_data_card(cohort: Cohort, path: Path) -> dict[str, Any]:
    persons = cohort.persons
    images = cohort.images
    n1 = int((persons["n_images"] == 1).sum())
    n_multi = int((persons["n_images"] > 1).sum())
    mixed = int(
        (
            images.groupby("person_id")["label_cin2plus"].nunique()
            > 1
        ).sum()
    )
    stats = {
        "n_persons": cohort.n_persons,
        "n_images": cohort.n_images,
        "n_positive_persons": cohort.n_positives,
        "n_negative_persons": cohort.n_persons - cohort.n_positives,
        "positive_prevalence_persons": float(cohort.n_positives / cohort.n_persons),
        "images_per_person_mean": float(persons["n_images"].mean()),
        "persons_with_1_image": n1,
        "persons_with_gt1_image": n_multi,
        "persons_with_mixed_image_labels": mixed,
        "legacy_split_persons": persons["legacy_split"].value_counts().to_dict(),
        "hpv_bucket_persons": persons["hpv_bucket"].value_counts().to_dict(),
        "cyto_bucket_persons": persons["cyto_bucket"].value_counts().to_dict(),
        "label_unit": "person_any_cin2plus (histology-linked CIN2+)",
        "image_modality": "acetic acid colposcopy (type A)",
        "dataset_root": str(cohort.dataset_root),
    }
    save_json(path.with_suffix(".json"), stats)
    lines = [
        "# DATA CARD — Patient-level CIN2+ acetic cohort",
        "",
        "This card describes the **cleaned** acetic training cohort used by the",
        "grouped cross-validation protocol. It is generated from the frozen",
        "`images_with_split.csv` / `persons_with_split.csv` tables.",
        "",
        "## Counts",
        "",
        f"- Persons: **{stats['n_persons']}**",
        f"- Acetic images: **{stats['n_images']}**",
        f"- CIN2+ persons: **{stats['n_positive_persons']}** "
        f"({100 * stats['positive_prevalence_persons']:.1f}%)",
        f"- CIN1− persons: **{stats['n_negative_persons']}**",
        f"- Mean acetic images / person: **{stats['images_per_person_mean']:.2f}**",
        f"- Persons with exactly 1 image: **{n1}**",
        f"- Persons with >1 image: **{n_multi}**",
        f"- Persons with mixed image-level labels: **{mixed}**",
        "",
        "## Label",
        "",
        "- Gold standard: histopathology linked in clinical metadata.",
        "- Person label = CIN2+ if any linked histology for that person is CIN2+.",
        "- Prediction unit: **person**, never a single image.",
        "",
        "## Grouping",
        "",
        "- `person_id` is canonical (suffix `__first` is stripped).",
        "- Every folder/visit of a person stays in the same CV fold.",
        "",
        "## Legacy split (not used for model selection)",
        "",
        "The original 70/15/15 person split is retained only as a diagnostic",
        "column (`legacy_split`). Protocol evidence is grouped OOF.",
        "",
        f"- Legacy persons: `{stats['legacy_split_persons']}`",
        "",
        "## Clinical covariates",
        "",
        f"- HPV buckets: `{stats['hpv_bucket_persons']}`",
        f"- Cytology buckets: `{stats['cyto_bucket_persons']}`",
        "",
        "Age mean/std are **never** estimated on evaluation patients; they are",
        "fit inside each training fold.",
        "",
    ]
    write_text(path, "\n".join(lines))
    save_csv(path.parent / "cohort_persons.csv", persons)
    keep_img = [
        "person_id",
        "folder_id",
        "file",
        "image_path_rel",
        "label_cin2plus",
        "person_label",
        "legacy_split",
        "hpv_bucket",
        "cyto_bucket",
        "age",
        "grade",
    ]
    cols = [c for c in keep_img if c in images.columns]
    save_csv(path.parent / "cohort_images.csv", images[cols])
    return stats
