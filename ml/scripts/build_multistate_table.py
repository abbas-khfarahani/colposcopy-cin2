#!/usr/bin/env python3
"""Build case-level multistate table (acetic + green + Lugol) under ml/data/tables/.

Reads My_Dataset patients + primary gold cases + frozen person splits.
Does not write into dataset folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.utils.paths import ML_ROOT, default_my_dataset_root, default_splits_dir

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SKIP_PREFIX = ("._", "dsc")


def _classify_file(name: str) -> tuple[str, int] | None:
    low = name.lower()
    if name.startswith("._") or low.startswith("dsc"):
        return None
    if Path(name).suffix.lower() not in IMG_EXT:
        return None
    m = re.search(r"(?i)_A(\d*)\.(?:jpe?g|png|tiff?)$", name)
    if m:
        n = m.group(1)
        return "A", int(n) if n else 1
    m = re.search(r"(?i)_G(\d*)\.(?:jpe?g|png|tiff?)$", name)
    if m:
        n = m.group(1)
        return "G", int(n) if n else 1
    m = re.search(r"(?i)_L(\d*)\.(?:jpe?g|png|tiff?)$", name)
    if m:
        n = m.group(1)
        return "L", int(n) if n else 1
    return None


def _load_person_splits(splits_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}_person_ids.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            pid = line.strip()
            if pid:
                out[pid] = split
    return out


def _folder_from_case(root: Path, row: dict[str, str]) -> Path:
    rel = (row.get("acetic_paths_rel") or "").replace("|", ";").split(";")[0].strip()
    if rel:
        return (root / rel).parent
    fid = str(row["folder_id"]).strip()
    return root / "patients" / fid


def build(max_acetic: int = 3) -> dict:
    root = default_my_dataset_root()
    splits_dir = default_splits_dir()
    person_split = _load_person_splits(splits_dir)
    cases_csv = root / "primary_training_table" / "primary_acetic_cases.csv"
    cases = list(csv.DictReader(cases_csv.open(encoding="utf-8")))

    out_rows: list[dict] = []
    skipped = []
    stats = Counter()

    for row in cases:
        folder = _folder_from_case(root, row)
        if not folder.is_dir():
            skipped.append({"folder_id": row["folder_id"], "reason": "missing_folder"})
            continue

        views: dict[str, list[tuple[int, str]]] = {"A": [], "G": [], "L": []}
        for p in folder.iterdir():
            if not p.is_file():
                continue
            c = _classify_file(p.name)
            if c is None:
                continue
            kind, idx = c
            views[kind].append((idx, p.name))
            # Recurse one level if images live in a nested same-name folder.
            if not any(views.values()):
                for sub in folder.iterdir():
                    if sub.is_dir() and not sub.name.startswith("."):
                        for p in sub.iterdir():
                            if not p.is_file():
                                continue
                            c = _classify_file(p.name)
                            if c is None:
                                continue
                            kind, idx = c
                            views[kind].append((idx, p.name))
                        if any(views.values()):
                            folder = sub
                            break
            for k in views:
                views[k] = sorted(views[k], key=lambda t: t[0])

        if not views["A"]:
            skipped.append({"folder_id": row["folder_id"], "reason": "no_acetic"})
            continue

        acetic_files = [fn for _, fn in views["A"][:max_acetic]]
        green_file = views["G"][0][1] if views["G"] else ""
        lugol_file = views["L"][0][1] if views["L"] else ""

        person_id = str(row["person_id"]).strip()
        split = person_split.get(person_id)
        if split is None:
            # fallback: folder name as person
            split = person_split.get(folder.name)
        if split is None:
            skipped.append({"folder_id": row["folder_id"], "reason": "no_split"})
            continue

        n_a, n_g, n_l = len(views["A"]), len(views["G"]), len(views["L"])
        stats[f"n_acetic_{n_a}"] += 1
        stats[f"has_green_{int(n_g > 0)}"] += 1
        stats[f"has_lugol_{int(n_l > 0)}"] += 1
        stats[f"split_{split}"] += 1

        try:
            folder_rel = str(folder.resolve().relative_to(root.resolve()))
        except ValueError:
            folder_rel = f"patients/{folder.name}"

        out_rows.append(
            {
                "person_id": person_id,
                "folder_id": str(row["folder_id"]).strip(),
                "image_folder_id": folder.name,
                "split": split,
                "label_cin2plus": int(float(row["label_cin2plus"])),
                "grade": row.get("grade", ""),
                "label_source": row.get("label_source", ""),
                "n_acetic": n_a,
                "n_green": n_g,
                "n_lugol": n_l,
                "acetic_files": ";".join(acetic_files),
                "green_file": green_file,
                "lugol_file": lugol_file,
                "folder_rel": folder_rel.replace("\\", "/"),
            }
        )

    out_dir = ML_ROOT / "data" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "multistate_cases.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    summary = {
        "n_cases": len(out_rows),
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "stats": dict(stats),
        "max_acetic": max_acetic,
        "out_csv": str(out_csv),
        "pos": sum(r["label_cin2plus"] for r in out_rows),
        "neg": sum(1 - r["label_cin2plus"] for r in out_rows),
        "by_split": {
            s: {
                "n": sum(1 for r in out_rows if r["split"] == s),
                "pos": sum(r["label_cin2plus"] for r in out_rows if r["split"] == s),
            }
            for s in ("train", "val", "test")
        },
    }
    summary_path = out_dir / "multistate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-acetic", type=int, default=3)
    args = p.parse_args()
    build(max_acetic=args.max_acetic)


if __name__ == "__main__":
    main()
