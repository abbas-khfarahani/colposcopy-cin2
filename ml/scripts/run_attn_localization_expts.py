#!/usr/bin/env python3
"""Train spatial-attention ResNet variants and regenerate localization heatmaps.

Runs:
  1) acetic_resnet50_attn       @224
  2) acetic_resnet50_attn_384   @384  (tests whether higher res helps with attention)
  3) acetic_resnet50_attn_difflr @224
Then heatmaps for each + a short comparison JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.train import format_duration, resolve_device  # noqa: E402

JOBS = [
    {
        "name": "acetic_resnet50_attn",
        "train": [
            "ml/train.py",
            "--config",
            "ml/configs/acetic_resnet50_attn.json",
            "--run-name",
            "acetic_resnet50_attn",
        ],
        "heatmap_out": "ml/runs/heatmaps_attn_224",
    },
    {
        "name": "acetic_resnet50_attn_384",
        "train": [
            "ml/train.py",
            "--config",
            "ml/configs/acetic_resnet50_attn_384.json",
            "--run-name",
            "acetic_resnet50_attn_384",
        ],
        "heatmap_out": "ml/runs/heatmaps_attn_384",
    },
    {
        "name": "acetic_resnet50_attn_difflr",
        "train": [
            "ml/train.py",
            "--config",
            "ml/configs/acetic_resnet50_attn_difflr.json",
            "--run-name",
            "acetic_resnet50_attn_difflr",
        ],
        "heatmap_out": "ml/runs/heatmaps_attn_difflr",
    },
]


def run(cmd: list[str]) -> int:
    full = [sys.executable, "-u", *cmd]
    print(">>>", " ".join(full), flush=True)
    return subprocess.call(full, cwd=str(_PROJECT_ROOT))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-heatmaps", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"device={device}  jobs={len(JOBS)}", flush=True)
    t0 = time.perf_counter()
    results = []

    for job in JOBS:
        name = job["name"]
        entry = {"name": name}
        if not args.skip_train:
            t1 = time.perf_counter()
            train_cmd = [
                *job["train"],
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
            ]
            code = run(train_cmd)
            entry["train_code"] = code
            entry["train_sec"] = time.perf_counter() - t1
            if code != 0:
                print(f"FAIL train {name} code={code}", flush=True)
                results.append(entry)
                continue
        if not args.skip_heatmaps:
            t1 = time.perf_counter()
            hm = [
                "ml/scripts/generate_heatmaps_attn.py",
                "--checkpoint",
                f"ml/runs/{name}/best.pt",
                "--out-dir",
                job["heatmap_out"],
                "--device",
                args.device,
            ]
            code = run(hm)
            entry["heatmap_code"] = code
            entry["heatmap_sec"] = time.perf_counter() - t1
        # metrics
        summary_path = _PROJECT_ROOT / "ml" / "runs" / name / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            tm = s.get("test_metrics") or {}
            person = tm.get("person") or {}
            image = tm.get("image") or {}
            entry["test_person_auroc"] = person.get("auroc")
            entry["test_image_auroc"] = image.get("auroc")
            entry["n_train"] = s.get("n_train")
        hm_sum = _PROJECT_ROOT / job["heatmap_out"] / "summary.json"
        if hm_sum.exists():
            entry["heatmap_counts"] = json.loads(hm_sum.read_text(encoding="utf-8")).get("counts")
        results.append(entry)
        print(
            f"<<< {name} person_auroc={entry.get('test_person_auroc')} "
            f"counts={entry.get('heatmap_counts')}",
            flush=True,
        )

    # baseline reference from locked stable (already trained)
    baseline = {}
    bp = _PROJECT_ROOT / "ml" / "runs" / "acetic_resnet50_stable" / "summary.json"
    if bp.exists():
        s = json.loads(bp.read_text(encoding="utf-8"))
        tm = s.get("test_metrics") or {}
        baseline = {
            "run": "acetic_resnet50_stable",
            "test_person_auroc": (tm.get("person") or {}).get("auroc"),
            "test_image_auroc": (tm.get("image") or {}).get("auroc"),
        }

    out = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed": format_duration(time.perf_counter() - t0),
        "baseline_gap": baseline,
        "results": results,
        "note": (
            "Spatial attention + center prior. Compare heatmaps_attn_* vs "
            "heatmaps_primary for localization quality. 384 tests whether "
            "higher resolution helps when attention can use the extra pixels."
        ),
    }
    out_path = _PROJECT_ROOT / "ml" / "runs" / "attn_localization_comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
