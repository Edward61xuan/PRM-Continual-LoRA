"""Exercise real training/Fisher/write-back on a tiny CPU model.

Only dataset access and CUDA transfers are replaced; gradients, optimizer
updates, Fisher accumulation, and coefficient computation run normally.
"""
import contextlib
import io
import json
import math
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.utils.data import TensorDataset

import train_prm as train
from prm.lora import LoRALayer, apply_lora


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(4, 12)
        self.blocks = nn.ModuleList([block])
        self.head = nn.Linear(4, 2)

    def forward(self, images, perturbation=0):
        qkv = self.blocks[0].attn.qkv(images, perturbation=perturbation)
        return self.head(qkv.reshape(-1, 2, 3, 4).mean(dim=(1, 2)))

    def remove_text_features(self):
        pass


class FisherModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyModel()
        self.fisher = train.init_fisher_storage(num_layers=1)
        self.cache, self.labels, self.meta = {}, {}, {}
        self.manager = SimpleNamespace(num_tasks=3)
        self.dataset = TensorDataset(torch.randn(4, 2, 4), torch.tensor([0, 1, 0, 1]))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.object(nn.Module, "cuda", lambda obj, *a, **k: obj))
        self.stack.enter_context(mock.patch.object(torch.Tensor, "cuda", lambda obj, *a, **k: obj))
        self.stack.enter_context(mock.patch.object(train, "build_train_transform", return_value=None))
        self.stack.enter_context(mock.patch.object(train, "get_path_dataset", return_value=None))
        self.stack.enter_context(mock.patch.object(train, "ClassIncrementalDataset", return_value=self.dataset))
        self.accumulate = self.stack.enter_context(mock.patch.object(
            train, "accumulate_fisher", side_effect=partial(train.accumulate_fisher, num_layers=1),
        ))
        self.q_t = self.stack.enter_context(mock.patch.object(
            train, "compute_q_t", side_effect=partial(train.compute_q_t, embed_dim=4),
        ))
        self.alpha = self.stack.enter_context(mock.patch.object(
            train, "compute_prm_fisher_alpha", side_effect=partial(train.compute_prm_fisher_alpha, embed_dim=4),
        ))
        original_epoch = train.train_one_epoch

        def checked_epoch(model, loader, criterion, optimizer, args, taskid, **kwargs):
            # Verify the next task re-enables only its adapters and head.
            for name, param in model.named_parameters():
                expected = (name.startswith("head.") or
                            f"lora_A.{taskid}." in name or f"lora_B.{taskid}." in name)
                self.assertEqual(param.requires_grad, expected, name)
            for module in model.modules():
                if isinstance(module, LoRALayer):
                    self.assertTrue(module.training)
            if not train.needs_fisher(args):
                self.assertEqual(self.fisher, {"0": {}})
            return original_epoch(model, loader, criterion, optimizer, args, taskid, **kwargs)

        self.epoch = self.stack.enter_context(mock.patch.object(
            train, "train_one_epoch", side_effect=checked_epoch,
        ))

    def args(self, *extra):
        return train.get_args([
            "--dataset", "cifar100", "--data_root", "/unused",
            "--epochs", "1", "--batch_size", "2", "--workers", "0",
            "--no-use_amp", "--temperature", "1", *extra,
        ])

    def task(self, args, taskid):
        apply_lora(self.model, taskid, rank=2, num_tasks=3)
        train.train_one_task(
            self.model, args, self.manager, self.labels, {}, self.cache,
            self.fisher, taskid, [taskid * 2, taskid * 2 + 1], self.meta,
        )

    def test_default_fixed_skips_fisher_and_restores_training_next_task(self):
        args = self.args()
        self.assertFalse(args.fisher_diagnostics)
        for taskid in range(2):
            self.task(args, taskid)
            self.assertFalse(self.model.training)
            self.assertFalse(any(p.requires_grad for p in self.model.parameters()))
            self.assertIsNone(self.meta[taskid]["q_t"])
            self.assertIsNone(self.meta[taskid]["delta_norm_F"])
            self.assertEqual(self.meta[taskid]["delta_norm_per_layer"], {})
            self.assertIn('"q_t": null', json.dumps(self.meta[taskid]))
            for module in self.model.modules():
                if isinstance(module, LoRALayer):
                    self.assertFalse(module.training)
                    self.assertGreater(module.lora_B[taskid].weight.abs().sum().item(), 0)
        self.assertEqual(self.fisher, {"0": {}})
        self.assertEqual(self.epoch.call_count, 2)
        self.accumulate.assert_not_called()
        self.q_t.assert_not_called()
        self.alpha.assert_not_called()
        self.assertAlmostEqual(self.meta[1]["alpha_chosen"], args.prm_alpha)

    def assert_fisher_tasks(self, args):
        for taskid in range(2):
            self.task(args, taskid)
            tensor = self.fisher["0"][taskid]
            self.assertEqual(tensor.shape, (12, 4))  # Full Q/K/V, per task.
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertGreater(tensor.sum().item(), 0)
            self.assertTrue(math.isfinite(self.meta[taskid]["q_t"]))
            self.assertGreater(self.meta[taskid]["delta_norm_F"], 0)
            self.assertTrue(math.isfinite(self.meta[taskid]["alpha_chosen"]))
        self.assertEqual(set(self.fisher["0"]), {0, 1})
        self.assertEqual(self.accumulate.call_count, 2)
        self.assertEqual(self.q_t.call_count, 2)

    def test_fisher_method_runs_fisher_without_diagnostic_flag(self):
        self.assert_fisher_tasks(self.args("--prm_method", "fisher"))
        self.alpha.assert_called_once()

    def test_fixed_diagnostics_opt_in_runs_fisher(self):
        self.assert_fisher_tasks(self.args("--fisher_diagnostics"))
        self.alpha.assert_not_called()

    def test_later_sweep_collects_fisher_from_first_task(self):
        self.assert_fisher_tasks(self.args("--alpha_sweep_tasks", "3"))
        self.alpha.assert_not_called()

    def test_explicit_diagnostic_disable(self):
        self.assertFalse(train.needs_fisher(self.args(
            "--fisher_diagnostics", "--no-fisher_diagnostics",
        )))
        self.assertTrue(train.needs_fisher(self.args(
            "--prm_method", "fisher", "--no-fisher_diagnostics",
        )))

    def test_missing_prefix_fisher_fails_before_training(self):
        self.task(self.args(), 0)
        with self.assertRaisesRegex(ValueError, "Regenerate the prefix"):
            self.task(self.args("--fisher_diagnostics"), 1)
        self.assertEqual(self.epoch.call_count, 1)

    def test_stage1_only_serializes_unmeasured_diagnostics_as_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.task(self.args("--stage1_only_save_delta", tmp), 0)
            saved = torch.load(next(Path(tmp).glob("*.pt")), map_location="cpu")
            self.assertIsNone(saved["q_t"])
            self.assertIsNone(saved["delta_norm_F"])
            self.assertIsNone(self.meta[0]["q_t"])
            self.accumulate.assert_not_called()

    def test_resume_keeps_historical_fisher_only_when_needed(self):
        for diagnostic in (False, True):
            with self.subTest(diagnostic=diagnostic), contextlib.ExitStack() as stack:
                extra = ["--fisher_diagnostics"] if diagnostic else []
                args = self.args(
                    "--num_tasks", "2", "--resume_prefix", "/unused/prefix.pt",
                    "--train_task", "2", "--stage1_only_save_delta", "/unused", *extra,
                )
                history = torch.ones(12, 4)
                loaded = {"fisher": {"0": {"0": history}}}
                stack.enter_context(mock.patch.object(train, "get_args", return_value=args))
                stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
                stack.enter_context(mock.patch.object(train, "seed_everything"))
                stack.enter_context(mock.patch.object(train, "get_path_dataset", return_value=SimpleNamespace(class_list=range(4))))
                stack.enter_context(mock.patch.object(train, "load_prefix", return_value=loaded))
                stack.enter_context(mock.patch.object(train, "rebuild_model_from_prefix", return_value=(self.model, {}, 0)))
                stack.enter_context(mock.patch.object(train, "collect_run_metadata", return_value={}))
                stack.enter_context(mock.patch.object(train, "restore_rng"))
                stack.enter_context(mock.patch.object(train, "apply_lora"))
                task = stack.enter_context(mock.patch.object(train, "train_one_task", return_value=self.model))
                train.main()
                restored = task.call_args.args[6]
                if diagnostic:
                    self.assertIs(restored["0"][0], history)
                else:
                    self.assertTrue(all(not tasks for tasks in restored.values()))
                self.assertNotIn("fisher", loaded)


if __name__ == "__main__":
    unittest.main()
