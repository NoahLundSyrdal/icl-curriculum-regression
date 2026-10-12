"""Scientific checks: causal ridge, symmetries, calibration and matched training."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import torch
import run_experiment as base
import continue_experiment as tail
import clean_control as control
import checkpoint_diagnostics as diagnostic
import summarize_followup as summary
from test_continuation import assert_nested_equal


class PredictorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def bank(self):
        return base.EpisodeStream(91).draw(12, 7, 3)

    def test_ridge_matches_direct_primal_and_known_sigma_bayes(self):
        xs, clean, epsilon = self.bank()
        ys = clean + epsilon
        grid = [0, 0.25, 1, 4]
        actual, _, _ = diagnostic.ridge_predictions(xs, ys, grid, (2, 3, 6))
        for li, lam in enumerate(grid):
            for ki, k in enumerate((2, 3, 6)):
                x, y = xs[:, :k].double(), ys[:, :k].double()
                gram = x.transpose(1, 2) @ x
                rhs = x.transpose(1, 2) @ y[..., None]
                weights = (torch.linalg.pinv(gram, hermitian=True) @ rhs if lam == 0 else
                           torch.linalg.solve(gram + lam*torch.eye(3), rhs))
                expected = (xs[:, k:k+1].double() @ weights).flatten().float()
                torch.testing.assert_close(actual[li, :, ki], expected, atol=2e-5, rtol=2e-5)
        losses = base.gaussian_bayes_losses(xs, clean, ys, 1, dims=3, microbatch=12)
        torch.testing.assert_close((actual[2] - clean[:, (2, 3, 6)]).square()/3,
                                   losses[:, (2, 3, 6)], atol=2e-5, rtol=2e-5)

    def test_no_query_or_future_target_leakage(self):
        xs, clean, noise = self.bank()
        labels = clean + noise
        first = diagnostic.ridge_predictions(xs, labels, [0, 1, 4], (3,))
        changed = labels.clone()
        changed[:, 3:] += 10000
        second = diagnostic.ridge_predictions(xs, changed, [0, 1, 4], (3,))
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

    def test_orthogonal_and_context_permutation_invariance_of_references(self):
        xs, clean, noise = self.bank()
        ys = clean + noise
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        xs = xs.double()
        original = diagnostic.ridge_predictions(xs, ys, [0, 1, 4], (6,))
        rotated = diagnostic.ridge_predictions(xs @ q, ys, [0, 1, 4], (6,))
        px, py = diagnostic.permutations(xs, ys, 6)
        permuted = diagnostic.ridge_predictions(px, py, [0, 1, 4], (6,))
        for variant in (rotated, permuted):
            torch.testing.assert_close(original[0], variant[0], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(original[1], variant[1], atol=2e-5, rtol=2e-5)
            self.assertTrue(torch.equal(original[2], variant[2]))

    def test_calibration_fits_predictions_and_splits_are_distinct_reproducible(self):
        xs, clean, noise = self.bank()
        grid, _, _ = diagnostic.ridge_predictions(xs, clean + noise, [0, 1, 4], (4, 5, 6))
        self.assertEqual(diagnostic.select_penalty(grid, grid[2]), 2)
        a = base.EpisodeStream(diagnostic.CALIBRATION_SEED).draw(2, 7, 3)
        b = base.EpisodeStream(diagnostic.HELDOUT_SEED).draw(2, 7, 3)
        again = base.EpisodeStream(diagnostic.CALIBRATION_SEED).draw(2, 7, 3)
        self.assertFalse(torch.equal(a[0], b[0]))
        assert_nested_equal(self, a, again)

    def test_empirical_bayes_matches_direct_marginal_likelihood(self):
        xs, clean, noise = self.bank()
        ys = clean + noise
        grid = [0, 0.25, 1, 4]
        _, _, selected = diagnostic.ridge_predictions(xs, ys, grid, (2, 6))
        for ki, k in enumerate((2, 6)):
            x, y = xs[:, :k].double(), ys[:, :k].double()
            covariance = x @ x.transpose(1, 2)
            scores = []
            for lam in grid[1:]:
                c = covariance + lam * torch.eye(k)
                score = torch.linalg.slogdet(c).logabsdet + (y[..., None].transpose(1, 2) @ torch.linalg.solve(c, y[..., None])).flatten()
                scores.append(score)
            expected = torch.tensor(grid[1:])[torch.stack(scores).argmin(0)]
            self.assertTrue(torch.equal(selected[:, ki], expected))


class ControlChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cfg = base.preset_config("smoke", mode="clean", seed=2, dims=2, points=5,
            embedding=16, layers=1, heads=2, steps=6, batch_size=4, microbatch=2,
            eval_episodes=4, eval_microbatch=4, checkpoint_every=1, log_every=3,
            bayes_reference=False)

    def tearDown(self):
        self.temporary.cleanup()

    def run_control(self, name, **kwargs):
        return control.train_control(self.cfg, self.root/name, torch.device("cpu"), (3, 6),
                                     {"test": "immutable"}, **kwargs)

    def load(self, name, sub="training"):
        return torch.load(self.root/name/sub/"checkpoint.pt", map_location="cpu", weights_only=True)

    def compare_states(self, a, b):
        for key in ("model", "optimizer", "episode_rng", "torch_rng", "cuda_rng"):
            assert_nested_equal(self, a[key], b[key], key)

    def test_control_matches_uninterrupted_base_training_and_retains_snapshot(self):
        base.train(self.cfg, self.root/"direct", torch.device("cpu"))
        self.assertEqual(self.run_control("control"), 6)
        self.compare_states(self.load("direct", "."), self.load("control"))
        snapshot = self.load("control", "step_000003")
        self.assertEqual(snapshot["completed_steps"], 3)
        self.assertEqual(snapshot["config"]["steps"], 6)
        metric = json.loads((self.root/"control/step_000003/metrics.json").read_text())
        self.assertEqual(metric["completed_steps"], 3)
        self.assertEqual(metric["config"]["steps"], 3)

    def test_resume_matches_whole_run_and_recovers_milestone_evaluation(self):
        self.run_control("whole")
        self.assertEqual(self.run_control("split", max_steps_this_run=3), 3)
        frozen = self.root/"split/step_000003/checkpoint.pt"
        sha = tail.file_sha256(frozen)
        (frozen.parent/"metrics.json").unlink()
        self.assertEqual(self.run_control("split", resume=True), 6)
        self.compare_states(self.load("whole"), self.load("split"))
        self.assertEqual(sha, tail.file_sha256(frozen))
        self.assertTrue((frozen.parent/"metrics.json").exists())

    def test_midphase_resume_and_episode_pairing_against_shuffled(self):
        self.run_control("whole")
        self.assertEqual(self.run_control("split", max_steps_this_run=2), 2)
        self.assertEqual(self.run_control("split", resume=True), 6)
        self.compare_states(self.load("whole"), self.load("split"))
        base.train(dataclasses.replace(self.cfg, mode="shuffled"), self.root/"noisy", torch.device("cpu"))
        assert_nested_equal(self, self.load("whole")["episode_rng"], self.load("noisy", ".")["episode_rng"])

    def test_resume_rejects_changed_protocol_or_configuration(self):
        self.run_control("partial", max_steps_this_run=2)
        with self.assertRaisesRegex(ValueError, "identical"):
            control.train_control(self.cfg, self.root/"partial", torch.device("cpu"), (3, 6), {"test": "changed"}, resume=True)
        with self.assertRaisesRegex(ValueError, "identical"):
            control.train_control(dataclasses.replace(self.cfg, learning_rate=0.1), self.root/"partial", torch.device("cpu"), (3, 6), {"test": "immutable"}, resume=True)


class SummaryChecks(unittest.TestCase):
    def test_pair_contrast_sign_and_interval_symmetry(self):
        a = summary.paired_summary([-1, -2, -3, -4, -5])
        b = summary.paired_summary([1, 2, 3, 4, 5])
        self.assertEqual(a["mean"], -3)
        self.assertAlmostEqual(a["ci95"][0], -b["ci95"][1])
        self.assertAlmostEqual(a["ci95"][1], -b["ci95"][0])
        self.assertLess(a["ci95"][1], 0)
        with self.assertRaises(ValueError):
            summary.paired_summary([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            summary.paired_summary([1, 2, 3, 4, float("nan")])


if __name__ == "__main__":
    unittest.main()
