#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured experiment phases end-to-end")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--phases", type=str, default="1,2,3", help="Comma-separated set from: 1,2,3,4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")
    parser.add_argument("--exp9-cost-json", type=str, default="", help="Optional MOT cost JSON if phase 3 exp9 is included")
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    valid = {"1", "2", "3", "4"}
    invalid = [p for p in phases if p not in valid]
    if invalid:
        raise ValueError(f"Unsupported phase values: {invalid}. Allowed: {sorted(valid)}")

    for phase in phases:
        if phase == "1":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_one.run_phase1",
                    args.data_dir,
                    str(out_root / "phase_one"),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--dropout",
                    str(args.dropout),
                    "--seed",
                    str(args.seed),
                    "--save-every",
                    str(args.save_every),
                ]
            )
            continue

        if phase == "2":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_two.run_phase2",
                    args.data_dir,
                    str(out_root / "phase_two"),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--dropout",
                    str(args.dropout),
                    "--seed",
                    str(args.seed),
                    "--save-every",
                    str(args.save_every),
                ]
            )
            continue

        if phase == "3":
            cmd = [
                sys.executable,
                "-m",
                "phase_three.run_phase3",
                args.data_dir,
                str(out_root / "phase_three"),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--dropout",
                str(args.dropout),
                "--seed",
                str(args.seed),
                "--save-every",
                str(args.save_every),
            ]
            if args.exp9_cost_json:
                cmd.extend(["--only", "exp6,exp7,exp8,exp9", "--exp9-cost-json", args.exp9_cost_json])
            run_cmd(cmd)
            continue

        if phase == "4":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_four.run_phase4",
                    args.data_dir,
                    str(out_root / "phase_four"),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--dropout",
                    str(args.dropout),
                    "--seed",
                    str(args.seed),
                    "--save-every",
                    str(args.save_every),
                ]
            )

    print(f"Run complete. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
