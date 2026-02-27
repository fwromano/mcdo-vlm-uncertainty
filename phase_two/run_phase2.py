#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from phase_one.common import list_images, sample_paths, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 experiments from paper_outline_v3")
    parser.add_argument("data_dir", type=str, help="Image root for classification-side experiments")
    parser.add_argument("out_dir", type=str, help="Output root for phase two")
    parser.add_argument(
        "--only",
        type=str,
        default="exp1,exp3,exp5,exp2,exp4",
        help="Comma-separated subset/order of experiments: exp1,exp2,exp3,exp4,exp5,exp6",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1, help="Save partial artifacts every N completed trials")

    parser.add_argument("--exp1-models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--exp1-num-images", type=int, default=1000)
    parser.add_argument("--exp1-p-values", type=str, default="0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--exp1-passes", type=int, default=64)
    parser.add_argument("--exp1-trials", type=int, default=3)

    parser.add_argument("--exp2-models", type=str, default="clip_b32,siglip2_so400m")
    parser.add_argument("--exp2-num-natural", type=int, default=10)
    parser.add_argument("--exp2-num-each-synth", type=int, default=10)
    parser.add_argument("--exp2-image-size", type=int, default=224)
    parser.add_argument("--exp2-passes", type=int, default=64)
    parser.add_argument("--exp2-trials", type=int, default=5)

    parser.add_argument("--exp3-models", type=str, default="clip_b32")
    parser.add_argument("--exp3-dropout-types", type=str, default="A,B,C,D,E")
    parser.add_argument("--exp3-num-images", type=int, default=1000)
    parser.add_argument("--exp3-passes", type=int, default=64)
    parser.add_argument("--exp3-trials", type=int, default=5)

    parser.add_argument(
        "--exp4-models",
        type=str,
        default="clip_b32,siglip2_b16,siglip2_so400m,clip_l14,siglip2_g16",
    )
    parser.add_argument("--exp4-num-images", type=int, default=500)
    parser.add_argument("--exp4-passes", type=int, default=64)
    parser.add_argument("--exp4-trials", type=int, default=10)

    parser.add_argument("--exp5-models", type=str, default="clip_b32,siglip2_b16")
    parser.add_argument("--exp5-num-images", type=int, default=10000)
    parser.add_argument("--exp5-passes", type=int, default=64)
    parser.add_argument("--exp5-trials", type=int, default=1)
    parser.add_argument("--exp5-quantile", type=float, default=0.10)
    parser.add_argument("--exp5-class-map", type=str, default="")
    parser.add_argument("--exp5-templates", type=str, default="a photo of a {}|a {}|an image of a {}")
    parser.add_argument("--exp5-retrieval-json", type=str, default="")
    parser.add_argument("--exp5-num-retrieval", type=int, default=5000)

    parser.add_argument("--exp6-models", type=str, default="siglip2_b16,siglip2_so400m")
    parser.add_argument("--exp6-num-images", type=int, default=500)
    parser.add_argument("--exp6-passes", type=str, default="4,8,16,32,64")
    parser.add_argument("--exp6-trials", type=int, default=3)

    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = [x.strip() for x in args.only.split(",") if x.strip()]
    known = {"exp1", "exp2", "exp3", "exp4", "exp5", "exp6"}
    invalid = [x for x in selected if x not in known]
    if invalid:
        raise ValueError(f"Unknown experiment(s) in --only: {invalid}. Known: {sorted(known)}")

    all_paths = list_images(args.data_dir)

    def make_manifest(count: int, seed: int, filename: str) -> Path:
        path = out_root / filename
        save_manifest(sample_paths(all_paths, count, seed), str(path))
        return path

    manifest_exp1 = make_manifest(args.exp1_num_images, args.seed, "manifest_exp1.json")
    manifest_exp3 = make_manifest(args.exp3_num_images, args.seed, "manifest_exp3.json")
    manifest_exp4 = make_manifest(args.exp4_num_images, args.seed, "manifest_exp4.json")
    manifest_exp5 = make_manifest(args.exp5_num_images, args.seed + 1, "manifest_exp5.json")
    manifest_exp6 = make_manifest(args.exp6_num_images, args.seed, "manifest_exp6.json")

    for exp_name in selected:
        if exp_name == "exp1":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_two.exp1_rank_p",
                    args.data_dir,
                    str(out_root / "exp1_rank_p"),
                    "--manifest",
                    str(manifest_exp1),
                    "--models",
                    args.exp1_models,
                    "--p-values",
                    args.exp1_p_values,
                    "--passes",
                    str(args.exp1_passes),
                    "--trials",
                    str(args.exp1_trials),
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
            )
            continue

        if exp_name == "exp2":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_two.exp2_synthetic_natural",
                    args.data_dir,
                    str(out_root / "exp2_synthetic_natural"),
                    "--models",
                    args.exp2_models,
                    "--num-natural",
                    str(args.exp2_num_natural),
                    "--num-each-synth",
                    str(args.exp2_num_each_synth),
                    "--image-size",
                    str(args.exp2_image_size),
                    "--dropout",
                    str(args.dropout),
                    "--passes",
                    str(args.exp2_passes),
                    "--trials",
                    str(args.exp2_trials),
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
            )
            continue

        if exp_name == "exp3":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_two.exp3_dropout_type",
                    args.data_dir,
                    str(out_root / "exp3_dropout_type"),
                    "--manifest",
                    str(manifest_exp3),
                    "--models",
                    args.exp3_models,
                    "--dropout-types",
                    args.exp3_dropout_types,
                    "--dropout",
                    str(args.dropout),
                    "--passes",
                    str(args.exp3_passes),
                    "--trials",
                    str(args.exp3_trials),
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
            )
            continue

        if exp_name == "exp4":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_two.exp4_full_matrix",
                    args.data_dir,
                    str(out_root / "exp4_full_matrix"),
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
                    "--save-every",
                    str(args.save_every),
                ]
            )
            continue

        if exp_name == "exp5":
            cmd = [
                sys.executable,
                "-m",
                "phase_two.exp5_full_ambiguity",
                args.data_dir,
                str(out_root / "exp5_full_ambiguity"),
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
                "--quantile",
                str(args.exp5_quantile),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--num-retrieval",
                str(args.exp5_num_retrieval),
                "--save-every",
                str(args.save_every),
            ]
            if args.exp5_class_map:
                cmd.extend(["--class-map", args.exp5_class_map])
            if args.exp5_retrieval_json:
                cmd.extend(["--retrieval-json", args.exp5_retrieval_json])
            run_cmd(cmd)
            continue

        if exp_name == "exp6":
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "phase_two.exp6_mean_convergence",
                    args.data_dir,
                    str(out_root / "exp6_mean_convergence"),
                    "--manifest",
                    str(manifest_exp6),
                    "--models",
                    args.exp6_models,
                    "--passes",
                    args.exp6_passes,
                    "--trials",
                    str(args.exp6_trials),
                    "--dropout",
                    str(args.dropout),
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
            )
            continue

    print(f"Phase 2 run completed. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
