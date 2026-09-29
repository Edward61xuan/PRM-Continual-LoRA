#!/usr/bin/env python3
"""Expand a paper YAML config into reproducible per-seed training commands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


COMMON = {
    "batch_size": 256,
    "eval_batch_size": 100,
    "rank": 10,
    "module": "kv",
    "eps": 0.5,
    "p0": 0.33,
    "lr": 0.001,
    "lr_scale": 10.0,
    "prm_alpha": 0.8,
    "prm_prox_weight": 0.01,
    "workers": 16,
    "eval_workers": 2,
}


def build_commands(config_path: Path, method: str, seeds: list[int], data_root: Path,
                   output_root: Path, pm_control: bool, extra: list[str]) -> list[list[str]]:
    config = yaml.safe_load(config_path.read_text())
    values = {**COMMON, **config}
    name = values.pop("name", config_path.stem)
    values["prm_method"] = method
    if pm_control:
        values["prm_prox_weight"] = 0.0
    train = Path(__file__).resolve().parents[1] / "train_prm.py"
    commands = []
    tag = f"pm-{method}" if pm_control else f"prm-{method}"
    for seed in seeds:
        result = output_root / name / tag / f"seed{seed}.json"
        command = [sys.executable, str(train), "--data_root", str(data_root),
                   "--seed", str(seed), "--results_file", str(result)]
        for key, value in values.items():
            if isinstance(value, bool):
                command.append(f"--{key}" if value else f"--no-{key}")
            elif isinstance(value, list):
                command.extend([f"--{key}", *map(str, value)])
            else:
                command.extend([f"--{key}", str(value)])
        commands.append(command + extra)
    return commands


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--method", choices=("fixed", "fisher"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--pm-control", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    for command in build_commands(args.config, args.method, args.seeds, args.data_root,
                                  args.output_root, args.pm_control, extra):
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            Path(command[command.index("--results_file") + 1]).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
