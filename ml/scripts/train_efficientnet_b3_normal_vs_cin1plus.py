#!/usr/bin/env python3
"""Train EfficientNet-B3 for Normal vs CIN1+ using the frozen B3 settings.

This is a new entry point. It does not edit ml/train.py, the CIN2+ dataset
code, or the acetic_efficientnet_b3_stable run.

The optimizer, image size, augmentations, loss, early stopping, and threshold
rule are the same as ml/configs/acetic_efficientnet_b3_stable.json. The only
change is the case label:

  0 = Normal
  1 = CIN1+   (CIN1, CIN2, CIN3, and Cancer/AIS)

Each acetic image keeps the grade of its case. A person is CIN1+ when the
highest image label is CIN1+.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.data.dataset import ColposcopyAceticDataset  # noqa: E402
from ml.data.transforms import build_transforms  # noqa: E402
from ml.train import load_config, run_training  # noqa: E402
import ml.train as train_mod  # noqa: E402
from ml.utils.paths import ML_ROOT  # noqa: E402

NEGATIVE_GRADES = frozenset({"Normal"})
POSITIVE_GRADES = frozenset({"CIN1", "CIN2", "CIN3", "Cancer_AIS"})
LABEL_NAMES = {0: "Normal", 1: "CIN1+"}


def label_from_grade(grade: str) -> int:
    """Map a case grade to the new binary target."""
    name = (grade or "").strip()
    if name in NEGATIVE_GRADES:
        return 0
    if name in POSITIVE_GRADES:
        return 1
    raise ValueError(
        f"Grade {grade!r} is neither Normal nor CIN1+ "
        f"({sorted(POSITIVE_GRADES)})."
    )


class NormalVsCin1PlusDataset(ColposcopyAceticDataset):
    """Same acetic splits as the CIN2+ model, with Normal vs CIN1+ labels."""

    def labels(self) -> list[int]:
        return [label_from_grade(row.get("grade", "")) for row in self.rows]

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        label = float(label_from_grade(self.rows[index].get("grade", "")))
        if len(item) == 4:
            image, _old, clinical, meta = item
        else:
            image, _old, meta = item
            clinical = None
        meta = dict(meta)
        meta["label_name"] = LABEL_NAMES[int(label)]
        meta["task"] = "normal_vs_cin1plus"
        if clinical is None:
            return image, label, meta
        return image, label, clinical, meta


def _count_split(dataset: NormalVsCin1PlusDataset) -> dict[str, Any]:
    grades = Counter((row.get("grade") or "").strip() for row in dataset.rows)
    labels = dataset.labels()
    person_label: dict[str, int] = {}
    person_grades: dict[str, set[str]] = {}
    for row, label in zip(dataset.rows, labels):
        person_id = row["person_id"]
        person_label[person_id] = max(person_label.get(person_id, 0), label)
        person_grades.setdefault(person_id, set()).add((row.get("grade") or "").strip())
    return {
        "n_images": len(dataset),
        "n_image_normal": labels.count(0),
        "n_image_cin1plus": labels.count(1),
        "n_persons": len(person_label),
        "n_person_normal": sum(v == 0 for v in person_label.values()),
        "n_person_cin1plus": sum(v == 1 for v in person_label.values()),
        "image_grades": dict(sorted(grades.items())),
        "persons_with_mixed_grades": sorted(
            pid for pid, grades_set in person_grades.items() if len(grades_set) > 1
        ),
    }


def make_loader(split: str, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    """Same loader as the B3 run, with the remapped case labels."""
    transform = build_transforms(
        split="train" if split == "train" else "val",
        image_size=int(cfg["image_size"]),
        augment=bool(cfg.get("train_augment", True)) and split == "train",
        augment_level=str(cfg.get("augment_level", "light")),
    )
    dataset = NormalVsCin1PlusDataset(
        split=split,
        splits_dir=cfg["splits_dir"],
        my_dataset_root=cfg["my_dataset_root"],
        transform=transform,
        include_clinical=str(cfg.get("model_type", "image")).lower() == "multimodal",
    )
    use_cuda = torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith(
        "cuda"
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=use_cuda,
        collate_fn=train_mod.collate_batch,
        drop_last=False,
    )


def label_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    audit = {
        "task": "normal_vs_cin1plus",
        "negative_class": "Normal",
        "positive_class": "CIN1+",
        "positive_grades": sorted(POSITIVE_GRADES),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = NormalVsCin1PlusDataset(
            split=split,
            splits_dir=cfg["splits_dir"],
            my_dataset_root=cfg["my_dataset_root"],
            transform=None,
        )
        audit["splits"][split] = _count_split(dataset)
    return audit


def _enrich_predictions(run_dir: Path, cfg: dict[str, Any]) -> None:
    """Add the case grade and Normal/CIN1+ name onto the saved test rows."""
    pred_path = run_dir / "test_predictions.json"
    if not pred_path.is_file():
        return
    dataset = NormalVsCin1PlusDataset(
        split="test",
        splits_dir=cfg["splits_dir"],
        my_dataset_root=cfg["my_dataset_root"],
        transform=None,
    )
    by_key = {
        (row["person_id"], row["file"]): row for row in dataset.rows
    }
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    for row in predictions:
        source = by_key.get((row.get("person_id"), row.get("file")), {})
        grade = (source.get("grade") or "").strip()
        row["grade"] = grade
        row["label_name"] = LABEL_NAMES[int(row["label"])]
    pred_path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ML_ROOT / "configs" / "acetic_efficientnet_b3_normal_vs_cin1plus.json",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="acetic_efficientnet_b3_normal_vs_cin1plus",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print Normal vs CIN1+ counts and exit without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.device is not None:
        cfg["device"] = args.device
    cfg["task"] = "binary_normal_vs_cin1plus"
    cfg["negative_class"] = "Normal"
    cfg["positive_class"] = "CIN1+"
    cfg["negative_grades"] = sorted(NEGATIVE_GRADES)
    cfg["positive_grades"] = sorted(POSITIVE_GRADES)

    audit = label_audit(cfg)
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit:
        return

    # Route the existing B3 training loop through the new labels.
    train_mod.make_loader = make_loader
    run_dir = ML_ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "label_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    run_training(cfg, run_dir)
    _enrich_predictions(run_dir, cfg)


if __name__ == "__main__":
    main()
