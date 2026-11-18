#!/usr/bin/env python3
"""Rerun the five best CIN2+ models on Normal vs CIN1+, across learning rates.

Each model keeps the settings in its own config (backbone, image size, batch
size, loss, early stopping). The case label and the learning rate are the only
changes. Finished runs are skipped, including the EfficientNet-B3 result at
1e-4.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.data.dataset import ColposcopyAceticDataset  # noqa: E402
from ml.data.multistate_dataset import ColposcopyMultistateDataset  # noqa: E402
from ml.data.transforms import build_transforms  # noqa: E402
from ml.train import load_config, run_training  # noqa: E402
import ml.train as train_mod  # noqa: E402
from ml.utils.paths import ML_ROOT  # noqa: E402

NEGATIVE_GRADES = frozenset({"Normal"})
POSITIVE_GRADES = frozenset({"CIN1", "CIN2", "CIN3", "Cancer_AIS"})
LABEL_NAMES = {0: "Normal", 1: "CIN1+"}
SEVERITY = {"Normal": 0, "CIN1": 1, "CIN2": 2, "CIN3": 3, "Cancer_AIS": 4}

# Ranked by holdout test person AUROC on the old CIN1− vs CIN2+ task.
TOP5: list[tuple[str, str]] = [
    ("efficientnet_b3", "acetic_efficientnet_b3_stable.json"),
    ("resnet50_384", "acetic_resnet50_stable_384.json"),
    ("multistate_efficientnet_b0", "multistate_efficientnet_b0_stable.json"),
    ("convnext_tiny", "acetic_convnext_tiny_stable.json"),
    ("resnet50_attn_cam", "acetic_resnet50_attn_cam.json"),
]
DEFAULT_LRS = (5e-5, 1e-4, 3e-4)
EXISTING_B3 = ML_ROOT / "runs" / "acetic_efficientnet_b3_normal_vs_cin1plus"


def label_from_grade(grade: str) -> int:
    name = (grade or "").strip()
    if name in NEGATIVE_GRADES:
        return 0
    if name in POSITIVE_GRADES:
        return 1
    raise ValueError(f"Grade {grade!r} is neither Normal nor CIN1+.")


class NormalVsCin1PlusDataset(ColposcopyAceticDataset):
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


class NormalVsCin1PlusMultistateDataset(ColposcopyMultistateDataset):
    def labels(self) -> list[int]:
        return [label_from_grade(row.get("grade", "")) for row in self.rows]

    def __getitem__(self, index: int):
        images, mask, type_ids, _old, meta = super().__getitem__(index)
        label = float(label_from_grade(self.rows[index].get("grade", "")))
        meta = dict(meta)
        meta["label_name"] = LABEL_NAMES[int(label)]
        meta["task"] = "normal_vs_cin1plus"
        return images, mask, type_ids, label, meta


def make_loader(split: str, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    transform = build_transforms(
        split="train" if split == "train" else "val",
        image_size=int(cfg["image_size"]),
        augment=bool(cfg.get("train_augment", True)) and split == "train",
        augment_level=str(cfg.get("augment_level", "light")),
    )
    model_type = str(cfg.get("model_type", "image")).lower()
    if model_type == "multistate":
        dataset: Any = NormalVsCin1PlusMultistateDataset(
            split=split,
            my_dataset_root=cfg["my_dataset_root"],
            transform=transform,
            max_acetic=int(cfg.get("max_acetic", 3)),
            include_green=bool(cfg.get("include_green", True)),
            include_lugol=bool(cfg.get("include_lugol", True)),
        )
    else:
        dataset = NormalVsCin1PlusDataset(
            split=split,
            splits_dir=cfg["splits_dir"],
            my_dataset_root=cfg["my_dataset_root"],
            transform=transform,
            include_clinical=model_type == "multimodal",
        )
    use_cuda = torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith(
        "cuda"
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=use_cuda,
        collate_fn=train_mod.collate_batch,
        drop_last=False,
    )


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def run_dir_for(model_name: str, lr: float) -> Path:
    if model_name == "efficientnet_b3" and abs(lr - 1e-4) < 1e-12 and (
        EXISTING_B3 / "summary.json"
    ).is_file():
        return EXISTING_B3
    root = ML_ROOT / "runs" / "normal_cin1plus_top5"
    return root / f"{model_name}_lr{lr_tag(lr)}"


def _person_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["person_id"])].append(row)
    people = []
    for person_id, items in grouped.items():
        worst = max(items, key=lambda row: SEVERITY.get(str(row.get("grade")), -1))
        people.append(
            {
                "person_id": person_id,
                "grade": worst.get("grade"),
                "label": max(int(row["label"]) for row in items),
                "score": max(float(row["score"]) for row in items),
            }
        )
    return people


def grade_snapshot(run_dir: Path, threshold: float) -> dict[str, Any]:
    pred_path = run_dir / "test_predictions.json"
    if not pred_path.is_file():
        return {}
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    people = _person_rows(predictions)
    out: dict[str, Any] = {}
    for grade in ("Normal", "CIN1", "CIN2", "CIN3", "Cancer_AIS"):
        group = [row for row in people if row.get("grade") == grade]
        if not group:
            continue
        called = sum(1 for row in group if row["score"] >= threshold)
        out[grade] = {
            "n_persons": len(group),
            "called_cin1plus": called,
            "recall": called / len(group),
            "mean_score": sum(row["score"] for row in group) / len(group),
        }
    return out


def result_row(model_name: str, lr: float, run_dir: Path, status: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "lr": lr,
        "run_dir": str(run_dir),
        "status": status,
    }
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return row
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test_person = summary["test_metrics"]["person"]
    val_person = summary["val_metrics"]["person"]
    threshold = float(summary["threshold"])
    row.update(
        {
            "best_epoch": summary.get("best_epoch"),
            "threshold": threshold,
            "val_person_auroc": val_person["auroc"],
            "test_person_auroc": test_person["auroc"],
            "test_person_sensitivity": test_person["sensitivity"],
            "test_person_specificity": test_person["specificity"],
            "test_image_auroc": summary["test_metrics"]["image"]["auroc"],
            "by_grade": grade_snapshot(run_dir, threshold),
        }
    )
    return row


def print_board(rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        rows,
        key=lambda row: float(row.get("test_person_auroc") or -1),
        reverse=True,
    )
    print("\n=== Normal vs CIN1+  top-5 x learning rate ===", flush=True)
    print(
        f"{'model':28} {'lr':8} {'test AUROC':>10} {'sens':>6} {'spec':>6} {'CIN1 recall':>12}",
        flush=True,
    )
    for row in ranked:
        cin1 = (row.get("by_grade") or {}).get("CIN1") or {}
        recall = cin1.get("recall")
        recall_txt = f"{recall:.3f}" if isinstance(recall, float) else "-"
        auroc = row.get("test_person_auroc")
        auroc_txt = f"{auroc:.3f}" if isinstance(auroc, float) else "-"
        sens = row.get("test_person_sensitivity")
        spec = row.get("test_person_specificity")
        print(
            f"{row['model']:28} {lr_tag(float(row['lr'])):8} {auroc_txt:>10} "
            f"{sens if sens is None else f'{sens:.3f}':>6} "
            f"{spec if spec is None else f'{spec:.3f}':>6} {recall_txt:>12}  {row['status']}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lrs", type=str, default="5e-5,1e-4,3e-4")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lrs = [float(part) for part in args.lrs.split(",") if part.strip()]
    train_mod.make_loader = make_loader
    board_dir = ML_ROOT / "runs" / "normal_cin1plus_top5"
    board_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for model_name, config_name in TOP5:
        base = load_config(ML_ROOT / "configs" / config_name)
        for lr in lrs:
            run_dir = run_dir_for(model_name, lr)
            if (run_dir / "summary.json").is_file():
                print(f"skip existing {model_name} lr={lr_tag(lr)} -> {run_dir}", flush=True)
                rows.append(result_row(model_name, lr, run_dir, "skipped_existing"))
                continue
            cfg = dict(base)
            cfg["lr"] = lr
            cfg["num_workers"] = args.num_workers
            cfg["device"] = args.device
            cfg["experiment_name"] = f"normal_cin1plus_{model_name}_lr{lr_tag(lr)}"
            cfg["task"] = "binary_normal_vs_cin1plus"
            cfg["negative_class"] = "Normal"
            cfg["positive_class"] = "CIN1+"
            cfg["negative_grades"] = sorted(NEGATIVE_GRADES)
            cfg["positive_grades"] = sorted(POSITIVE_GRADES)
            if args.epochs is not None:
                cfg["epochs"] = args.epochs
            run_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"\n=== train {model_name} lr={lr_tag(lr)} "
                f"backbone={cfg.get('backbone')} image_size={cfg.get('image_size')} "
                f"model_type={cfg.get('model_type')} ===",
                flush=True,
            )
            try:
                run_training(cfg, run_dir)
                rows.append(result_row(model_name, lr, run_dir, "trained"))
            except Exception as exc:  # one failed setting should not stop the sweep
                traceback.print_exc()
                (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                rows.append(result_row(model_name, lr, run_dir, f"failed: {exc}"))
            (board_dir / "comparison.json").write_text(
                json.dumps(rows, indent=2),
                encoding="utf-8",
            )
            print_board(rows)

    (board_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print_board(rows)
    print(f"\nWrote {board_dir / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
