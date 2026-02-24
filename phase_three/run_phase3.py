#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from phase_one.common import list_images, sample_paths, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 experiments from paper_outline_v3")
    parser.add_argument("data_dir", type=str, help="Image root for experiments 6-8")
    parser.add_argument("out_dir", type=str, help="Output root for phase three")
    parser.add_argument(
        "--only",
        type=str,
        default="exp6,exp7,exp8",
        help="Comma-separated subset/order: exp6,exp7,exp8,exp9",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")

    parser.add_argument("--exp6-models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--exp6-num-images", type=int, default=10000)
    parser.add_argument("--exp6-passes", type=int, default=64)
    parser.add_argument("--exp6-trials", type=int, default=1)
    parser.add_argument("--exp6-quantile", type=float, default=0.10)
    parser.add_argument("--exp6-class-map", type=str, default="")
    parser.add_argument("--exp6-templates", type=str, default="a photo of a {}|a {}|an image of a {}")
    parser.add_argument("--exp6-laplace-lambda", type=float, default=1.0)

    parser.add_argument("--exp7-models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--exp7-num-images", type=int, default=500)
    parser.add_argument("--exp7-passes", type=int, default=64)
    parser.add_argument("--exp7-trials", type=int, default=3)
    parser.add_argument("--exp7-quantile", type=float, default=0.10)
    parser.add_argument("--exp7-class-map", type=str, default="")
    parser.add_argument("--exp7-templates", type=str, default="a photo of a {}|a {}|an image of a {}")
    parser.add_argument("--exp7-jpeg-qualities", type=str, default="100,80,60,40,20")
    parser.add_argument("--exp7-blur-sigmas", type=str, default="0,1,2,4")
    parser.add_argument("--exp7-occlusion-ratios", type=str, default="0,0.1,0.25")

    parser.add_argument("--exp8-models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--exp8-num-images", type=int, default=500)
    parser.add_argument("--exp8-passes", type=int, default=64)
    parser.add_argument("--exp8-trials", type=int, default=3)
    parser.add_argument("--exp8-quantile", type=float, default=0.10)
    parser.add_argument("--exp8-prompts", type=str, default="")

    parser.add_argument("--exp9-cost-json", type=str, default="")
    parser.add_argument("--exp9-alpha", type=float, default=1.0)
    parser.add_argument("--exp9-beta", type=float, default=1.0)
    parser.add_argument("--exp9-epsilon", type=float, default=1e-6)
    parser.add_argument("--exp9-uncertainty-key", type=str, default="uncertainty")
    parser.add_argument("--exp9-oracle-key", type=str, default="oracle_uncertainty")
    parser.add_argument("--exp9-laplace-key", type=str, default="laplace_uncertainty")
    parser.add_argument("--exp9-max-frames", type=int, default=0)
    parser.add_argument("--exp9-modes", type=str, default="baseline,adaptive,oracle,laplace")

    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = [x.strip() for x in args.only.split(",") if x.strip()]
    known = {"exp6", "exp7", "exp8", "exp9"}
    invalid = [x for x in selected if x not in known]
    if invalid:
        raise ValueError(f"Unknown experiment(s) in --only: {invalid}. Known: {sorted(known)}")

    all_paths = list_images(args.data_dir)

    def make_manifest(count: int, seed: int, filename: str) -> Path:
        path = out_root / filename
        save_manifest(sample_paths(all_paths, count, seed), str(path))
        return path

    manifest_exp6 = make_manifest(args.exp6_num_images, args.seed, "manifest_exp6.json")
    manifest_exp7 = make_manifest(args.exp7_num_images, args.seed, "manifest_exp7.json")
    manifest_exp8 = make_manifest(args.exp8_num_images, args.seed, "manifest_exp8.json")

    for exp_name in selected:
        if exp_name == "exp6":
            cmd = [
                sys.executable,
                "-m",
                "phase_three.exp6_laplace_comparison",
                args.data_dir,
                str(out_root / "exp6_laplace_comparison"),
                "--manifest",
                str(manifest_exp6),
                "--models",
                args.exp6_models,
                "--templates",
                args.exp6_templates,
                "--dropout",
                str(args.dropout),
                "--passes",
                str(args.exp6_passes),
                "--trials",
                str(args.exp6_trials),
                "--quantile",
                str(args.exp6_quantile),
                "--laplace-lambda",
                str(args.exp6_laplace_lambda),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--save-every",
                str(args.save_every),
            ]
            if args.exp6_class_map:
                cmd.extend(["--class-map", args.exp6_class_map])
            run_cmd(cmd)
            continue

        if exp_name == "exp7":
            cmd = [
                sys.executable,
                "-m",
                "phase_three.exp7_aleatoric_epistemic",
                args.data_dir,
                str(out_root / "exp7_aleatoric_epistemic"),
                "--manifest",
                str(manifest_exp7),
                "--models",
                args.exp7_models,
                "--templates",
                args.exp7_templates,
                "--dropout",
                str(args.dropout),
                "--passes",
                str(args.exp7_passes),
                "--trials",
                str(args.exp7_trials),
                "--quantile",
                str(args.exp7_quantile),
                "--jpeg-qualities",
                args.exp7_jpeg_qualities,
                "--blur-sigmas",
                args.exp7_blur_sigmas,
                "--occlusion-ratios",
                args.exp7_occlusion_ratios,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--save-every",
                str(args.save_every),
            ]
            if args.exp7_class_map:
                cmd.extend(["--class-map", args.exp7_class_map])
            run_cmd(cmd)
            continue

        if exp_name == "exp8":
            cmd = [
                sys.executable,
                "-m",
                "phase_three.exp8_semantic_space",
                args.data_dir,
                str(out_root / "exp8_semantic_space"),
                "--manifest",
                str(manifest_exp8),
                "--models",
                args.exp8_models,
                "--dropout",
                str(args.dropout),
                "--passes",
                str(args.exp8_passes),
                "--trials",
                str(args.exp8_trials),
                "--quantile",
                str(args.exp8_quantile),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--save-every",
                str(args.save_every),
            ]
            if args.exp8_prompts:
                cmd.extend(["--prompts", args.exp8_prompts])
            run_cmd(cmd)
            continue

        if exp_name == "exp9":
            if not args.exp9_cost_json:
                raise ValueError("exp9 selected but --exp9-cost-json is missing")
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_three.exp9_mot_adaptive_demo",
                    args.exp9_cost_json,
                    str(out_root / "exp9_mot_adaptive_demo"),
                    "--alpha",
                    str(args.exp9_alpha),
                    "--beta",
                    str(args.exp9_beta),
                    "--epsilon",
                    str(args.exp9_epsilon),
                    "--uncertainty-key",
                    args.exp9_uncertainty_key,
                    "--oracle-key",
                    args.exp9_oracle_key,
                    "--laplace-key",
                    args.exp9_laplace_key,
                    "--max-frames",
                    str(args.exp9_max_frames),
                    "--modes",
                    args.exp9_modes,
                ]
            )

    print(f"Phase 3 run completed. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
