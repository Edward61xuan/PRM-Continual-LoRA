#!/usr/bin/env python3
"""Run PRM-fixed with diagnostic alpha sweeps at selected tasks."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _script_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run PRM-fixed and save alpha-sweep JSONs.")
    p.add_argument("--dataset", required=True, choices=("cifar100", "imagenet_r", "sdomainet", "cub200"))
    p.add_argument("--data_root", required=True)
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tasks", type=int, nargs="+", default=[2, 5, 10],
                   help="1-indexed tasks where alpha sweeps are saved.")
    p.add_argument("--out_root", default="outputs/alpha_sweep")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=100)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--eval_workers", type=int, default=2)
    p.add_argument("--model", default="vit_base_patch16_clip_quickgelu_224.openai")
    p.add_argument("--rank", type=int, default=10)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--prm_alpha", type=float, default=0.8)
    p.add_argument("--prm_prox_weight", type=float, default=0.01)
    p.add_argument("--alpha_grid_n", type=int, default=20)
    p.add_argument("--dry_run", action="store_true")
    args, extra = p.parse_known_args()

    root = _script_root()
    train = os.path.join(root, "train_prm.py")

    for seed in args.seeds:
        seed_dir = os.path.join(args.out_root, f"seed{seed}")
        cmd = [
            sys.executable, train,
            "--dataset", args.dataset,
            "--data_root", args.data_root,
            "--num_tasks", str(args.num_tasks),
            "--seed", str(seed),
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--workers", str(args.workers),
            "--eval_workers", str(args.eval_workers),
            "--rank", str(args.rank),
            "--eps", str(args.eps),
            "--prm_method", "fixed",
            "--prm_alpha", str(args.prm_alpha),
            "--prm_prox_weight", str(args.prm_prox_weight),
            "--alpha_grid_n", str(args.alpha_grid_n),
            "--alpha_sweep_tasks", *[str(t) for t in args.tasks],
            "--alpha_sweep_out_dir", os.path.join(seed_dir, "sweep"),
            "--results_file", os.path.join(seed_dir, "final.json"),
        ]
        _run(cmd + extra, args.dry_run)


if __name__ == "__main__":
    main()
