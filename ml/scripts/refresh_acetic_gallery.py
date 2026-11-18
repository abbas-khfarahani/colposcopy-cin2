#!/usr/bin/env python3
"""Refresh acetic overlays in feature_table_full with the current detector.

Does not re-run green / Lugol / zoom. Not gold. Not for training.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from demo_auto_lesion_masks import open_bgr as open_bgr_a  # noqa: E402
from demo_auto_lesion_masks import tissue_masks  # noqa: E402
from paint_feature_masks import acetic_views, cap, save_jpg, write_html, write_views  # noqa: E402
from run_full_cohort_features import (  # noqa: E402
    CSV_OUT,
    JSONL,
    KEYS,
    ON_IMAGE,
    OUT,
    PAD,
    SAMPLE_CSV,
    THUMBS,
    case_html,
    dump_csv,
    files_from_disk,
    load_done,
    measure_view,
    write_thumb,
)

DRAW_IDX = {14, 109, 116, 165}


def load_sample() -> list[dict]:
    rows = list(csv.DictReader(SAMPLE_CSV.open(newline="", encoding="utf-8")))
    rows.sort(key=lambda r: int(r["idx"]))
    return rows


def refresh_one(s: dict) -> dict:
    idx = int(s["idx"])
    folder = ON_IMAGE / f"{idx:0{PAD}d}"
    folder.mkdir(parents=True, exist_ok=True)
    ap = Path(s["path"])
    bgr = cap(open_bgr_a(ap))
    masks = tissue_masks(bgr)
    views = acetic_views(bgr, masks=masks)
    write_views(folder, "a", views)
    if idx in DRAW_IDX:
        save_jpg(folder / "a_photo_lg.jpg", cap(open_bgr_a(ap), 1024), quality=90)
        save_jpg(folder / "a_aceto_lg.jpg", cap(views["aceto"], 1024), quality=90)
        save_jpg(folder / "a_density_lg.jpg", cap(views["density"], 1024), quality=90)
    metrics = measure_view(
        masks, bgr, "aceto", {"glare": masks["glare"], "dense": masks.get("dense"), "vessels": None}
    )
    metrics["missing"] = 0
    metrics["present"] = 1
    return metrics


def main() -> None:
    t0 = time.time()
    sample = load_sample()
    sample = sorted(sample, key=lambda s: (0 if int(s["idx"]) in DRAW_IDX else 1, int(s["idx"])))
    done = load_done()
    html_cases = []
    n = len(sample)
    print(f"refresh acetic n={n} → {OUT}", flush=True)

    for i, s in enumerate(sample, 1):
        idx = int(s["idx"])
        try:
            am = refresh_one(s)
        except Exception as e:
            print(f"  FAIL {idx:03d} {s['file']} {e}", flush=True)
            am = None
        rec = done.get(idx, {"idx": idx, "person_id": s["person_id"], "acetic_file": s["file"], "grade": s["grade"]})
        rec["idx"] = idx
        rec["gold"] = False
        if am is not None:
            rec["A_present"] = 1
            rec["A_missing"] = 0
            for k in KEYS:
                rec[f"A_{k}"] = am.get(k, 0)
        done[idx] = rec
        try:
            a_files, g_files, l_files, z_files, a_ok, g_ok, l_ok, z_ok = files_from_disk(idx)
            write_thumb(ON_IMAGE / f"{idx:0{PAD}d}", a_files, g_files, l_files, z_files, THUMBS / f"{idx:0{PAD}d}.jpg")
        except Exception:
            pass
        if i % 20 == 0 or idx in DRAW_IDX:
            elapsed = (time.time() - t0) / 60
            afrac = rec.get("A_area_frac", 0)
            print(f"  {i:3d}/{n} {idx:03d} {s['grade']:11s} {s['file']:22s} A={afrac:.3f}  {elapsed:.1f} min", flush=True)

    sample_by_idx = load_sample()
    html_cases = []
    for s in sample_by_idx:
        idx = int(s["idx"])
        a_files, g_files, l_files, z_files, a_ok, g_ok, l_ok, z_ok = files_from_disk(idx)
        html_cases.append(case_html(idx, s, a_files, g_files, l_files, z_files, a_ok, g_ok, l_ok, z_ok))
    rows = [done[int(s["idx"])] for s in sample_by_idx if int(s["idx"]) in done]
    JSONL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    dump_csv(rows)
    write_html(html_cases, dest=OUT / "on_image.html", page_size=20)

    slim = []
    for s in sample_by_idx:
        rec = done[int(s["idx"])]
        idx = int(s["idx"])
        slim.append(
            {
                "idx": idx,
                "file": s["file"],
                "grade": s["grade"],
                "plus": int(s["cin2plus"]),
                "draw": idx in DRAW_IDX,
                "a_frac": float(rec.get("A_area_frac") or 0),
                "folder": f"on_image/{idx:0{PAD}d}",
            }
        )
    (OUT / "acetic_cases.json").write_text(json.dumps(slim), encoding="utf-8")
    print(f"done n={len(rows)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
