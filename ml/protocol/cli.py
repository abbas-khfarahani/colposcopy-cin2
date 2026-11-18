#!/usr/bin/env python3
"""CLI for the leakage-safe patient-level CIN2+ protocol.

Examples
--------
python -m ml.protocol smoke
python -m ml.protocol run --all --device cuda
python -m ml.protocol run --experiments B0,B1,B4 --device cuda
python -m ml.protocol report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.protocol.constants import RUNS_DIR  # noqa: E402
from ml.protocol.pipeline import ProtocolRunner  # noqa: E402
from ml.protocol.report import write_global_report  # noqa: E402
from ml.protocol.specs import LADDER  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run setup + experiments")
    run.add_argument("--all", action="store_true", help="Run B0–B12")
    run.add_argument("--experiments", type=str, default="", help="Comma-separated IDs, e.g. B0,B1,B4")
    run.add_argument("--device", default="cuda")
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--force", action="store_true")
    run.add_argument("--out", type=Path, default=RUNS_DIR)
    run.add_argument("--seeds", type=str, default="42", help="Outer split seeds, comma-separated")

    smoke = sub.add_parser("smoke", help="One-fold, one-epoch sanity check")
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--out", type=Path, default=RUNS_DIR / "_smoke")

    sub.add_parser("report", help="Rebuild markdown/HTML from existing OOF files")
    lst = sub.add_parser("list", help="List experiment IDs")

    args = p.parse_args(argv)

    if args.cmd == "list":
        for spec in LADDER:
            print(f"{spec.id:4}  {spec.kind:10}  {spec.name:28}  {spec.description}")
        return

    if args.cmd == "report":
        write_global_report(RUNS_DIR)
        print(f"Wrote {RUNS_DIR / 'REPORT.md'} and {RUNS_DIR / 'index.html'}")
        return

    if args.cmd == "smoke":
        runner = ProtocolRunner(
            root=args.out,
            device=args.device,
            num_workers=0,
            force=True,
            smoke=True,
            split_seeds=(42,),
        )
        runner.run_all(ids=["B0", "B1"])
        print("smoke ok")
        return

    ids = None
    if args.experiments:
        ids = [x.strip() for x in args.experiments.split(",") if x.strip()]
    elif not args.all:
        p.error("pass --all or --experiments B0,B1,...")
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    runner = ProtocolRunner(
        root=args.out,
        device=args.device,
        num_workers=args.num_workers,
        force=args.force,
        smoke=False,
        split_seeds=seeds,
    )
    runner.run_all(ids=ids)


if __name__ == "__main__":
    main()
