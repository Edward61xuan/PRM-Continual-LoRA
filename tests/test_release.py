import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from prm.continual import ClassIncrementalManager
from prm.lora import LoRALayer
from prm.metrics import calc_aaa, calc_forgetting
from prm.prm_ops import get_lora_params_snapshot, proximity_loss
from scripts.run_config import build_commands
from tools.dataset_prep.prepare import _bundled_manifest, verify_tree
from train_prm import get_args, write_json


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_public_cli_and_boolean_disables(self):
        args = get_args([
            "--dataset", "cifar100", "--data_root", "/tmp/data",
            "--prm_method", "fisher", "--no-use_amp", "--no-shuffle_classes",
        ])
        self.assertEqual(args.prm_method, "fisher")
        self.assertFalse(args.use_amp)
        self.assertFalse(args.shuffle_classes)

    def test_class_permutation_is_pcg64_and_repeatable(self):
        first = ClassIncrementalManager(range(100), 10, rand_seed=2)
        second = ClassIncrementalManager(range(100), 10, rand_seed=2)
        self.assertEqual(first.task_class_list, second.task_class_list)
        expected = np.arange(100)
        np.random.Generator(np.random.PCG64(2)).shuffle(expected)
        self.assertEqual(first.all_classes, expected.tolist())

    def test_metrics_match_paper_definitions(self):
        acc_mat = np.array([[0.8, 0.0], [0.6, 0.9]])
        acc_list = np.array([0.8, 0.75])
        self.assertAlmostEqual(calc_aaa(acc_list, 1), 0.775)
        self.assertAlmostEqual(calc_forgetting(acc_mat, 1), 0.2)

    def test_config_expands_three_seed_commands(self):
        commands = build_commands(
            ROOT / "configs/imagenet_r_10t.yaml", "fisher", [0, 1, 2],
            Path("data/inr"), Path("outputs"), False, ["--no-use_amp"],
        )
        self.assertEqual(len(commands), 3)
        self.assertIn("fisher", commands[0])
        self.assertIn("--no-use_amp", commands[0])
        self.assertIn("vit_base_patch16_224.augreg2_in21k_ft_in1k", commands[0])

    def test_data_tree_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                for cls in ("0", "1"):
                    path = root / split / cls / "sample.jpg"
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b"not decoded by verifier")
            self.assertEqual(verify_tree(root), {"train": 2, "val": 2, "classes": 2})

    def test_bundled_manifests_exist_and_are_well_formed(self):
        for name in (
            "imagenet_r_train.txt", "imagenet_r_val.txt",
            "sdomain_train.txt", "sdomain_val.txt",
        ):
            self.assertTrue(_bundled_manifest(name).is_file())

    def test_proximity_anchor_is_current_task_initialization(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = LoRALayer(4, 3, rank=2, eps=0.5, num_tasks=2)

        model = TinyModel()
        model.adapter.cal_alpha(1.0)
        model.adapter.update(1)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.adapter.lora_A[1].weight.requires_grad_(True)
        model.adapter.lora_B[1].weight.requires_grad_(True)

        snapshot = get_lora_params_snapshot(model)
        self.assertTrue(snapshot)
        self.assertTrue(all(".1.weight" in name for name in snapshot))
        self.assertEqual(proximity_loss(model, snapshot).item(), 0.0)

        with torch.no_grad():
            model.adapter.lora_B[0].weight.add_(1.0)
        self.assertEqual(proximity_loss(model, snapshot).item(), 0.0)

        with torch.no_grad():
            model.adapter.lora_B[1].weight.add_(1.0)
        self.assertGreater(proximity_loss(model, snapshot).item(), 0.0)

    def test_result_writer_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "nested" / "result.json"
            write_json(str(destination), {"AAA": 0.5})
            self.assertTrue(destination.is_file())

    def test_readme_contains_complete_paper_results_and_citation(self):
        readme = (ROOT / "README.md").read_text()
        self.assertNotIn("reported on arXiv", readme)
        self.assertIn("reported in the paper", readme)
        self.assertIn("Qiu et al., 2025", readme)
        self.assertIn("@article{liu2026mergecoefficient", readme)
        for setting in (
            "ImageNet-R 5T", "ImageNet-R 10T", "ImageNet-R 20T",
            "CIFAR-100 10T", "CIFAR-100 20T", "S-DomainNet 5T",
            "CUB-200 10T",
        ):
            self.assertIn(setting, readme)

    def test_sweep_wrappers_dry_run(self):
        for script in ("run_alpha_sweep.py", "run_lambda_sweep.py"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script),
                 "--dataset", "imagenet_r", "--data_root", "/tmp/missing",
                 "--dry_run", "--seeds", "0"],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("train_prm.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
