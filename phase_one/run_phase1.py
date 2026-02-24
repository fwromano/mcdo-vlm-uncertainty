#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from phase_one.common import list_images, sample_paths, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 experiments from paper_outline_v3")
    parser.add_argument("data_dir", type=str, help="ImageNet val (or other image root)")
    parser.add_argument("out_dir", type=str, help="Output root for phase one")
    parser.add_argument(
        "--only",
        type=str,
        default="exp0,exp0b,exp4,exp5",
        help="Comma-separated subset of experiments: exp0,exp0b,exp4,exp5",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--exp0-num-images", type=int, default=500)
    parser.add_argument("--exp0-models", type=str, default="clip_b32,siglip2_b16,siglip2_so400m")
    parser.add_argument("--exp0-trials", type=int, default=10)
    parser.add_argument("--exp0-passes", type=str, default="4,16,64")

    parser.add_argument("--exp0b-num-images", type=int, default=500)
    parser.add_argument("--exp0b-trials", type=int, default=5)
    parser.add_argument("--exp0b-passes", type=int, default=64)
    parser.add_argument("--exp0b-model", type=str, default="clip_b32")

    parser.add_argument("--exp4-num-images", type=int, default=500)
    parser.add_argument("--exp4-trials", type=int, default=10)
    parser.add_argument("--exp4-passes", type=int, default=64)
    parser.add_argument("--exp4-models", type=str, default="clip_b32,siglip2_b16")

    parser.add_argument("--exp5-num-images", type=int, default=5000)
    parser.add_argument("--exp5-trials", type=int, default=1)
    parser.add_argument("--exp5-passes", type=int, default=64)
    parser.add_argument("--exp5-models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--exp5-class-map", type=str, default="")
    parser.add_argument("--exp5-templates", type=str, default="a photo of a {}|a {}|an image of a {}")

    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = {x.strip() for x in args.only.split(",") if x.strip()}

    all_paths = list_images(args.data_dir)

    def make_manifest(count: int, seed: int, filename: str) -> Path:
        path = out_root / filename
        save_manifest(sample_paths(all_paths, count, seed), str(path))
        return path

    manifest_exp0 = make_manifest(args.exp0_num_images, args.seed, "manifest_exp0.json")
    manifest_exp0b = make_manifest(args.exp0b_num_images, args.seed, "manifest_exp0b.json")
    manifest_exp4 = make_manifest(args.exp4_num_images, args.seed, "manifest_exp4.json")
    manifest_exp5 = make_manifest(args.exp5_num_images, args.seed + 1, "manifest_exp5.json")

    if "exp0" in selected:
        run_cmd(
            [
                sys.executable,
                "-m",
                "phase_one.exp0_nested_mc",
                args.data_dir,
                str(out_root / "exp0_nested_mc"),
                "--manifest",
                str(manifest_exp0),
                "--models",
                args.exp0_models,
                "--dropout",
                str(args.dropout),
                "--trials",
                str(args.exp0_trials),
                "--passes",
                args.exp0_passes,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
        )

    if "exp0b" in selected:
        run_cmd(
            [
                sys.executable,
                "-m",
                "phase_one.exp0b_norm_geometry",
                args.data_dir,
                str(out_root / "exp0b_norm_geometry"),
                "--manifest",
                str(manifest_exp0b),
                "--model",
                args.exp0b_model,
                "--dropout",
                str(args.dropout),
                "--passes",
                str(args.exp0b_passes),
                "--trials",
                str(args.exp0b_trials),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
        )

    if "exp4" in selected:
        run_cmd(
            [
                sys.executable,
                "-m",
                "phase_one.exp4_subset_recipe",
                args.data_dir,
                str(out_root / "exp4_subset_recipe"),
                "--manifest",
                str(manifest_exp4),
                "--models",
                args.exp4_models,
                "--dropout",
                str(args.dropout),
                "--passes",
                str(args.exp4_passes),
                "--trials",
                str(args.exp4_trials),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
        )

    if "exp5" in selected:
        cmd = [
            sys.executable,
            "-m",
            "phase_one.exp5_subset_ambiguity",
            args.data_dir,
            str(out_root / "exp5_subset_ambiguity"),
            "--manifest",
            str(manifest_exp5),
            "--models",
            args.exp5_models,
            "--templates",
            args.exp5_templates,
            "--dropout",
            str(args.dropout),
            "--passes",
            str(args.exp5_passes),
            "--trials",
            str(args.exp5_trials),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        if args.exp5_class_map:
            cmd.extend(["--class-map", args.exp5_class_map])
        run_cmd(cmd)

    print(f"Phase 1 run completed. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
