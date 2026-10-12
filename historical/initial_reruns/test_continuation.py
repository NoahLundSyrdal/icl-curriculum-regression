"""Meaningful CPU checks of immutable, paired, restartable continuation.

Run on IDUN's verified environment: python -m unittest -v test_continuation.py
Tiny internal sources exercise the same trainer; the production CLI accepts
only completed canonical full sources with validated runtime/provenance.
"""
import dataclasses
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest import mock

import torch

import run_experiment as base
import continue_experiment as continuation


def assert_nested_equal(test, left, right, path="state"):
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor, path)
        test.assertTrue(torch.equal(left.cpu(), right.cpu()), path)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right), path)
        for key in left:
            assert_nested_equal(test, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_nested_equal(test, a, b, f"{path}[{index}]")
    else:
        test.assertEqual(left, right, path)


class ContinuationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        cls.original = base.preset_config(
            "smoke", seed=3, dims=2, points=5, embedding=16, layers=1, heads=2,
            steps=3, batch_size=4, microbatch=2, eval_episodes=4, eval_microbatch=4,
            checkpoint_every=1, log_every=3, bayes_reference=False)
        cls.sources = {}
        for history in continuation.HISTORIES:
            directory = cls.root / "sources" / history
            base.train(dataclasses.replace(cls.original, mode=history), directory, torch.device("cpu"))
            cls.sources[history] = directory
        cls.source_hashes = {history: continuation.file_sha256(directory / "checkpoint.pt")
                             for history, directory in cls.sources.items()}

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def output(self, name):
        return self.root / self._testMethodName / name

    def config(self, tail="high", eval_steps=(0, 2, 4)):
        return continuation.continuation_config(tail, 4, eval_steps,
                                                checkpoint_every=2, log_every=2)

    def run_tail(self, name, cfg, **kwargs):
        return continuation.train_continuation(
            self.sources["curriculum"], cfg, self.output(name), torch.device("cpu"),
            strict_source=False, **kwargs)

    def load_tail(self, name):
        return torch.load(self.output(name) / "checkpoint.pt", map_location="cpu", weights_only=True)

    def compare_training_states(self, a, b):
        for key in ("model", "optimizer", "episode_rng", "torch_rng", "cuda_rng", "python_rng"):
            assert_nested_equal(self, a[key], b[key], key)

    def test_restore_retains_model_optimizer_and_all_original_rng(self):
        for history, directory in self.sources.items():
            with self.subTest(history=history):
                saved = torch.load(directory / "checkpoint.pt", weights_only=True)
                cfg, provenance, model, optimizer, stream = continuation.initialize_source(
                    directory, torch.device("cpu"), strict=False)
                self.assertEqual(cfg.steps, 3)
                assert_nested_equal(self, saved["model"], model.state_dict())
                assert_nested_equal(self, saved["optimizer"], optimizer.state_dict())
                assert_nested_equal(self, saved["episode_rng"], stream.state_dict())
                assert_nested_equal(self, saved["torch_rng"], torch.get_rng_state())
                self.assertEqual(provenance["checkpoint_sha256"], self.source_hashes[history])
                self.assertTrue(model.training)

    def test_future_episodes_and_mixed_schedule_are_paired_across_histories(self):
        initialized = [continuation.initialize_source(directory, torch.device("cpu"), strict=False)
                       for directory in self.sources.values()]
        left, right = initialized[0][-1], initialized[1][-1]
        for _ in range(4):
            assert_nested_equal(self, left.draw(2, 5, 2), right.draw(2, 5, 2))
        cfg = continuation.continuation_config("mixed", 100, (0, 100))
        first = continuation.tail_schedule(initialized[0][0], cfg)
        second = continuation.tail_schedule(initialized[1][0], cfg)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first.sort().values, torch.linspace(0, 1, 100, dtype=torch.float64)))
        stats = continuation.planned_tail_budget(first)
        self.assertEqual(stats["terminal_windows"]["last_250"]["count"], 100)
        self.assertAlmostEqual(stats["sigma_squared_sum"], float(first.square().sum()))

    def test_evaluation_restores_rng_streams_and_training_mode(self):
        cfg, provenance, model, optimizer, stream = continuation.initialize_source(
            self.sources["curriculum"], torch.device("cpu"), strict=False)
        before = continuation.capture_rng(stream)
        adam_before = optimizer.state_dict()
        original_loss_function = base.clean_target_losses
        def deliberately_consume_rng(*args, **kwargs):
            torch.rand(3)
            random.random()
            stream.draw(1, cfg.points, cfg.dims)
            return original_loss_function(*args, **kwargs)
        with mock.patch.object(base, "clean_target_losses", side_effect=deliberately_consume_rng):
            metrics = continuation.evaluate_tail(
                cfg, self.config(), provenance, model, stream, 0,
                self.output("evaluated"), torch.device("cpu"))
        assert_nested_equal(self, before, continuation.capture_rng(stream))
        assert_nested_equal(self, adam_before, optimizer.state_dict())
        self.assertTrue(model.training)
        self.assertEqual(metrics["config"]["steps"], 3)
        self.assertEqual(metrics["tail_completed_steps"], 0)
        self.assertEqual(metrics["total_completed_steps"], 3)
        self.assertEqual(metrics["experiment"], continuation.KIND)
        self.assertEqual(len(metrics["conditions"]), 12)
        self.assertIn("gaussian_sd0", metrics["conditions"])

    def test_interruption_resume_is_bitwise_equal_for_both_tails(self):
        for tail in continuation.TAILS:
            with self.subTest(tail=tail):
                cfg = self.config(tail)
                _, whole = self.run_tail(tail + "_whole", cfg)
                _, partial = self.run_tail(tail + "_split", cfg, max_steps_this_run=2)
                _, resumed = self.run_tail(tail + "_split", cfg, resume=True)
                self.assertEqual((whole, partial, resumed), (4, 2, 4))
                a, b = self.load_tail(tail + "_whole"), self.load_tail(tail + "_split")
                self.compare_training_states(a, b)
                self.assertEqual(b["tail_completed_steps"], 4)
                self.assertEqual(b["total_completed_steps"], 7)
                self.assertEqual(b["config"]["steps"], 3)
                self.assertTrue(torch.equal(b["original_schedule"], base.sigma_schedule(
                    dataclasses.replace(self.original, mode="curriculum"))))
        for history, source in self.sources.items():
            self.assertEqual(continuation.file_sha256(source / "checkpoint.pt"), self.source_hashes[history])

    def test_intermediate_evaluations_do_not_change_training(self):
        self.run_tail("sparse", self.config("mixed", (0, 4)))
        self.run_tail("frequent", self.config("mixed", (0, 1, 2, 3, 4)))
        self.compare_training_states(self.load_tail("sparse"), self.load_tail("frequent"))

    def test_resume_rejects_changed_tail_config_and_source(self):
        cfg = self.config("high")
        self.run_tail("split", cfg, max_steps_this_run=1)
        with self.assertRaisesRegex(ValueError, "continuation configuration differs"):
            self.run_tail("split", dataclasses.replace(cfg, tail="mixed"), resume=True)
        changed_source = self.output("changed_source")
        shutil.copytree(self.sources["curriculum"], changed_source)
        checkpoint = torch.load(changed_source / "checkpoint.pt", weights_only=True)
        checkpoint["provenance_test_marker"] = "same numerical state, different source bytes"
        base.atomic_torch_save(changed_source / "checkpoint.pt", checkpoint)
        with self.assertRaisesRegex(ValueError, "source provenance differs"):
            continuation.train_continuation(changed_source, cfg, self.output("split"),
                torch.device("cpu"), strict_source=False, resume=True)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            continuation.train_continuation(self.sources["curriculum"], cfg,
                self.sources["curriculum"], torch.device("cpu"), strict_source=False)
        with self.assertRaisesRegex(ValueError, "canonical full"):
            continuation.inspect_source(self.sources["curriculum"], strict=True)

    def test_completed_tail_can_resume_interrupted_terminal_evaluation(self):
        cfg = self.config()
        self.run_tail("finished", cfg)
        state_before = self.load_tail("finished")
        terminal = continuation.evaluation_path(self.output("finished"), 4)
        (terminal / "metrics.json").unlink()
        (terminal / "losses.pt").unlink()
        _, completed = self.run_tail("finished", cfg, resume=True)
        self.assertEqual(completed, 4)
        self.compare_training_states(state_before, self.load_tail("finished"))
        metadata = json.loads((terminal / "metrics.json").read_text())
        self.assertEqual(metadata["tail_completed_steps"], 4)
        self.assertEqual(metadata["total_completed_steps"], 7)
        self.assertEqual(metadata["noise_budget"]["tail_completed"]["count"], 4)
        self.assertEqual(metadata["noise_budget"]["tail_planned"]["sigma_squared_sum"], 4)


if __name__ == "__main__":
    unittest.main()
