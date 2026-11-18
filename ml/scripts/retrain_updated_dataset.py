#!/usr/bin/env python3
"""Retrain all ML models after My_Dataset cleanup/update.

Writes new weights + summaries under ml/runs/ (overwrites named runs).
Prints per-job and overall elapsed / ETA remaining.
"""

from __future__ import annotations

import argparse
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
from ml.utils.paths import ML_ROOT  # noqa: E402

# Ordered by priority: primary pair first, then main ablations, then extras.
TRAIN_JOBS: list[dict[str, str]] = [
    # Primary thesis encoders
    {"config": "acetic_resnet50_stable.json", "run": "acetic_resnet50_stable"},
    {"config": "acetic_resnet50_difflr.json", "run": "acetic_resnet50_difflr"},
    # Spatial attention (acetic only; attn_cam needs a later pseudo-mask step)
    {"config": "acetic_resnet50_attn.json", "run": "acetic_resnet50_attn"},
    {"config": "acetic_resnet50_attn_384.json", "run": "acetic_resnet50_attn_384"},
    {"config": "acetic_resnet50_attn_difflr.json", "run": "acetic_resnet50_attn_difflr"},
    # Other acetic backbones / recipes
    {"config": "acetic_efficientnet_b0_v2.json", "run": "acetic_efficientnet_b0_v2"},
    {"config": "acetic_efficientnet_b3_stable.json", "run": "acetic_efficientnet_b3_stable"},
    {"config": "acetic_densenet121_stable.json", "run": "acetic_densenet121_stable"},
    {"config": "acetic_convnext_tiny_stable.json", "run": "acetic_convnext_tiny_stable"},
    {"config": "acetic_swin_t_stable.json", "run": "acetic_swin_t_stable"},
    {"config": "acetic_resnet50_focal.json", "run": "acetic_resnet50_focal"},
    {"config": "acetic_resnet50_v2.json", "run": "acetic_resnet50_v2"},
    {"config": "acetic_resnet50_stable_384.json", "run": "acetic_resnet50_stable_384"},
    {"config": "acetic_resnet50_stable_es12.json", "run": "acetic_resnet50_stable_es12"},
    {"config": "acetic_resnet50_person_es.json", "run": "acetic_resnet50_person_es"},
    {"config": "acetic_cin2plus_baseline.json", "run": "acetic_cin2plus_resnet50_baseline"},
    # Seed variants
    {"config": "acetic_resnet50_seed13.json", "run": "acetic_resnet50_seed13"},
    {"config": "acetic_resnet50_seed21.json", "run": "acetic_resnet50_seed21"},
    {"config": "acetic_resnet50_stable.json", "run": "acetic_resnet50_seed7", "seed": "7"},
    {"config": "acetic_resnet50_difflr_seed7.json", "run": "acetic_resnet50_difflr_seed7"},
    {"config": "acetic_resnet50_difflr_seed13.json", "run": "acetic_resnet50_difflr_seed13"},
    {"config": "acetic_resnet50_difflr_seed21.json", "run": "acetic_resnet50_difflr_seed21"},
    # Early clinical fusion
    {"config": "acetic_clinical_resnet50_stable.json", "run": "acetic_clinical_resnet50_stable"},
    {"config": "acetic_clinical_efficientnet_b0.json", "run": "acetic_clinical_efficientnet_b0"},
    # Multistate
    {"config": "multistate_resnet50_stable.json", "run": "multistate_resnet50_stable_v2"},
    {"config": "multistate_efficientnet_b0_stable.json", "run": "multistate_efficientnet_b0_stable_v2"},
    {"config": "multistate_acetic_green_resnet50.json", "run": "multistate_acetic_green_resnet50"},
    {"config": "multistate_agl_freeze_resnet50.json", "run": "multistate_agl_freeze_resnet50"},
]

POST_JOBS: list[dict[str, str]] = [
    {
        "name": "boost_stack_stable",
        "cmd": (
            "python -m ml.boost --mode stack "
            "--checkpoint ml/runs/acetic_resnet50_stable/best.pt "
            "--out-dir ml/runs/boost_stack_resnet50_stable"
        ),
    },
    {
        "name": "boost_stack_difflr",
        "cmd": (
            "python -m ml.boost --mode stack "
            "--checkpoint ml/runs/acetic_resnet50_difflr/best.pt "
            "--out-dir ml/runs/boost_stack_resnet50_difflr"
        ),
    },
    {
        "name": "boost_tta_stable",
        "cmd": (
            "python -m ml.boost --mode tta "
            "--checkpoint ml/runs/acetic_resnet50_stable/best.pt "
            "--out-dir ml/runs/boost_tta_resnet50_stable"
        ),
    },
    {
        "name": "boost_stack_tta_stable",
        "cmd": (
            "python -m ml.boost --mode stack --tta "
            "--checkpoint ml/runs/acetic_resnet50_stable/best.pt "
            "--out-dir ml/runs/boost_stack_tta_resnet50_stable"
        ),
    },
    {"name": "ensemble_stable_difflr", "cmd": "python ml/scripts/ensemble_stable_difflr.py"},
    {"name": "ensemble_difflr_family", "cmd": "python ml/scripts/ensemble_difflr_family.py"},
    {"name": "ensemble_attn_difflr", "cmd": "python ml/scripts/ensemble_attn_difflr.py"},
    {"name": "compare_runs", "cmd": "python ml/compare_runs.py"},
    {"name": "thesis_eval", "cmd": "python ml/scripts/thesis_eval.py"},
    {"name": "heatmaps_primary", "cmd": "python ml/scripts/generate_heatmaps.py"},
    {
        "name": "heatmaps_attn",
        "cmd": (
            "python ml/scripts/generate_heatmaps_attn.py "
            "--checkpoint ml/runs/acetic_resnet50_attn/best.pt "
            "--out-dir ml/runs/heatmaps_attn_224"
        ),
    },
    {
        "name": "attn_cam_pipeline",
        "cmd": "bash ml/scripts/run_attn_cam_pipeline.sh",
    },
]


def run_cmd(cmd: list[str] | str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cmd, str):
        shell_cmd = cmd
        printable = cmd
    else:
        shell_cmd = None
        printable = " ".join(cmd)
    print(f"\n>>> {printable}", flush=True)
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        if shell_cmd is not None:
            proc = subprocess.Popen(
                shell_cmd,
                shell=True,
                cwd=str(_PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device (falls back to cpu if cuda unavailable)",
    )
    parser.add_argument(
        "--only-primary",
        action="store_true",
        help="Only retrain stable + difflr + post-hoc primary stack/ensemble",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip CNN training; only run boost/eval jobs",
    )
    parser.add_argument(
        "--skip-post",
        action="store_true",
        help="Skip boost/ensemble/thesis/heatmap jobs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers (default: 2 on cpu, 4 on cuda)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    num_workers = args.num_workers
    if num_workers is None:
        # WSL /dev/shm is often tiny (e.g. 64MB); multi-worker DataLoaders OOM there.
        num_workers = 2 if device.type == "cuda" else 0

    train_jobs = TRAIN_JOBS
    post_jobs = POST_JOBS
    if args.only_primary:
        train_jobs = [j for j in TRAIN_JOBS if j["run"] in {
            "acetic_resnet50_stable",
            "acetic_resnet50_difflr",
        }]
        post_jobs = [j for j in POST_JOBS if j["name"] in {
            "boost_stack_stable",
            "boost_stack_difflr",
            "ensemble_stable_difflr",
            "compare_runs",
            "thesis_eval",
            "heatmaps_primary",
        }]

    status_path = ML_ROOT / "runs" / "retrain_status.json"
    log_dir = ML_ROOT / "runs" / "retrain_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    if not args.skip_train:
        jobs.append({"kind": "prep", "name": "build_multistate_table"})
        for j in train_jobs:
            jobs.append({"kind": "train", **j})
    if not args.skip_post:
        for j in post_jobs:
            jobs.append({"kind": "post", **j})

    print("=" * 72)
    print("Retrain on updated My_Dataset")
    print(f"device={device}  num_workers={num_workers}  jobs={len(jobs)}")
    if device.type != "cuda":
        print(
            "WARNING: CUDA unavailable (GPU blocked). Training on CPU — "
            "expect ~1–3 min/epoch for ResNet50; full suite may take many hours."
        )
    print("=" * 72, flush=True)

    t0 = time.perf_counter()
    durations: list[float] = []
    results: list[dict] = []

    def write_status(current: str | None = None) -> None:
        done = len(durations)
        avg = sum(durations) / len(durations) if durations else None
        remaining = len(jobs) - done
        eta = (avg * remaining) if avg is not None else None
        payload = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "current_job": current,
            "completed": done,
            "total": len(jobs),
            "elapsed_sec": time.perf_counter() - t0,
            "elapsed": format_duration(time.perf_counter() - t0),
            "eta_remaining_sec": eta,
            "eta_remaining": format_duration(eta) if eta is not None else None,
            "results": results,
        }
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for idx, job in enumerate(jobs, start=1):
        name = job.get("run") or job.get("name") or "job"
        write_status(current=name)
        print(
            f"\n[{idx}/{len(jobs)}] {name}  "
            f"elapsed={format_duration(time.perf_counter() - t0)}  "
            f"ETA~{format_duration((sum(durations)/len(durations))*(len(jobs)-idx+1)) if durations else 'computing...'}",
            flush=True,
        )
        jt0 = time.perf_counter()
        code = 0
        if job["kind"] == "prep":
            code = run_cmd(
                [sys.executable, "ml/scripts/build_multistate_table.py"],
                log_dir / "build_multistate_table.log",
            )
        elif job["kind"] == "train":
            cfg = ML_ROOT / "configs" / job["config"]
            cmd = [
                sys.executable,
                "ml/train.py",
                "--config",
                str(cfg),
                "--run-name",
                job["run"],
                "--device",
                device.type,
                "--num-workers",
                str(num_workers),
            ]
            if job.get("seed"):
                cmd.extend(["--seed", str(job["seed"])])
            code = run_cmd(cmd, log_dir / f"train_{job['run']}.log")
        else:
            code = run_cmd(job["cmd"], log_dir / f"post_{job['name']}.log")

        dt = time.perf_counter() - jt0
        durations.append(dt)
        results.append(
            {
                "name": name,
                "kind": job["kind"],
                "exit_code": code,
                "seconds": dt,
                "duration": format_duration(dt),
            }
        )
        remaining = len(jobs) - idx
        avg = sum(durations) / len(durations)
        print(
            f"<<< done {name} code={code} in {format_duration(dt)}  "
            f"overall elapsed={format_duration(time.perf_counter() - t0)}  "
            f"ETA remaining~{format_duration(avg * remaining)}",
            flush=True,
        )
        write_status(current=None)
        if code != 0:
            print(f"ERROR: job failed ({name}). Continuing with remaining jobs.", flush=True)

    write_status(current="finished")
    print("\n" + "=" * 72)
    print(f"All jobs finished in {format_duration(time.perf_counter() - t0)}")
    print(f"Status: {status_path}")
    failed = [r for r in results if r["exit_code"] != 0]
    if failed:
        print(f"Failed jobs ({len(failed)}): " + ", ".join(r["name"] for r in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
