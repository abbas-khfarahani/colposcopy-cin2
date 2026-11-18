#!/usr/bin/env python3
"""Export train positives for lesion marking + optional pseudo-masks from attn model.

Annotation format (JSONL), one object per line:
  {"image_path_rel": "...", "person_id": "...", "boxes": [[x1,y1,x2,y2], ...]}
Coordinates are in [0,1] relative to the full image (after any on-disk crop).

Usage:
  # list positives to mark manually
  python ml/scripts/export_lesion_annotation_template.py

  # also dump soft pseudo-masks from current attention model (for weak CAM loss)
  python ml/scripts/export_lesion_annotation_template.py --pseudo-from ml/runs/acetic_resnet50_attn/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.boost import load_ckpt  # noqa: E402
from ml.data import ColposcopyAceticDataset, build_transforms  # noqa: E402
from ml.models.classifier import SpatialAttentionClassifier  # noqa: E402
from ml.train import resolve_device  # noqa: E402
from ml.utils import default_splits_dir, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", default="ml/data/lesion_annotations.jsonl")
    ap.add_argument("--pseudo-from", default=None, help="attn checkpoint for soft masks")
    ap.add_argument("--pseudo-dir", default="ml/data/lesion_pseudo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-frac", type=float, default=0.15, help="keep top frac of attn as mask")
    args = ap.parse_args()

    set_seed(42)
    splits = default_splits_dir()
    ds = ColposcopyAceticDataset(
        split="train",
        splits_dir=str(splits),
        transform=None,
    )

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(ds)):
        sample = ds[i]
        _img, y, meta = sample[0], int(sample[1]), sample[-1]
        if y != 1:
            continue
        rows.append(
            {
                "image_path_rel": meta.get("image_path_rel") or meta.get("path"),
                "person_id": str(meta["person_id"]),
                "folder_id": str(meta.get("folder_id", "")),
                "file": str(meta.get("file", "")),
                "boxes": [],  # fill manually: [[x1,y1,x2,y2], ...] in 0-1
                "note": "draw 1+ boxes on acetowhite / lesion",
            }
        )
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} positive templates → {out_jsonl}")

    if not args.pseudo_from:
        return

    device = resolve_device(args.device)
    model, cfg = load_ckpt(Path(args.pseudo_from), device)
    if not isinstance(model, SpatialAttentionClassifier):
        raise SystemExit("pseudo-from must be an image_attn checkpoint")
    model.eval()
    tf = build_transforms("val", int(cfg.get("image_size", 224)), False)
    ds_t = ColposcopyAceticDataset(
        split="train",
        splits_dir=cfg["splits_dir"],
        my_dataset_root=cfg["my_dataset_root"],
        transform=tf,
    )
    pseudo_dir = Path(args.pseudo_dir)
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for i in range(len(ds_t)):
        image, y, meta = ds_t[i][0], int(ds_t[i][1]), ds_t[i][-1]
        if y != 1:
            continue
        x = image.unsqueeze(0).to(device)
        with torch.no_grad():
            _logits, attn = model(x, return_attn=True)
        attn = F.interpolate(attn, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        a = attn.detach().cpu().numpy()
        a = a - a.min()
        if a.max() > 1e-8:
            a = a / a.max()
        thr = float(np.quantile(a, 1.0 - args.top_frac))
        mask = (a >= thr).astype(np.float32)
        key = str(meta.get("image_path_rel") or meta.get("file") or i).replace("/", "__")
        np_path = (pseudo_dir / f"{key}.npy").resolve()
        np.save(np_path, mask)
        try:
            mask_rel = str(np_path.relative_to(_PROJECT_ROOT.resolve()))
        except ValueError:
            mask_rel = str(np_path)
        index.append(
            {
                "image_path_rel": meta.get("image_path_rel"),
                "person_id": str(meta["person_id"]),
                "file": str(meta.get("file") or ""),
                "mask_path": mask_rel,
            }
        )
    (pseudo_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {len(index)} soft pseudo-masks → {pseudo_dir}")


if __name__ == "__main__":
    main()
