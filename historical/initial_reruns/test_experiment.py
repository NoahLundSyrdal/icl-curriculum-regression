"""Fast CPU checks of pairing, causality, clean targets and exact resume.

Run: python -m unittest -v test_experiment.py
These checks verify implementation properties; they cannot establish H2.
"""
import dataclasses
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from run_experiment import (
    Config, EpisodeStream, RegressionTransformer, clean_target_losses,
    evaluation_bank, gaussian_bayes_losses, preset_config, seed_model,
    sigma_schedule, train,
)


class ExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ordering_changes_with_exactly_matched_sigma_budget(self):
        cfg = preset_config("smoke", steps=100, seed=7)
        increasing = sigma_schedule(cfg)
        shuffled = sigma_schedule(dataclasses.replace(cfg, mode="shuffled"))
        reverse = sigma_schedule(dataclasses.replace(cfg, mode="reverse"))
        self.assertTrue(torch.equal(increasing, shuffled.sort().values))
        self.assertTrue(torch.equal(increasing, reverse.sort().values))
        self.assertTrue(torch.equal(increasing.flip(0), reverse))
        self.assertFalse(torch.equal(increasing, shuffled))
        self.assertAlmostEqual(float(increasing.square().sum()),
                               float(shuffled.square().sum()), places=12)
        self.assertTrue(torch.equal(shuffled, sigma_schedule(dataclasses.replace(cfg, mode="shuffled"))))
        self.assertTrue(torch.equal(sigma_schedule(dataclasses.replace(cfg, mode="clean")),
                                    torch.zeros(100, dtype=torch.float64)))

    def test_modes_pair_initialization_and_episode_streams(self):
        cfg = preset_config("smoke", seed=2)
        seed_model(cfg.seed)
        first = RegressionTransformer(cfg)
        sigma_schedule(dataclasses.replace(cfg, mode="shuffled"))
        seed_model(cfg.seed)
        second = RegressionTransformer(dataclasses.replace(cfg, mode="shuffled"))
        for key, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[key]), key)
        left, right = EpisodeStream(cfg.seed), EpisodeStream(cfg.seed)
        for _ in range(3):
            for a, b in zip(left.draw(4, 9, 4), right.draw(4, 9, 4)):
                self.assertTrue(torch.equal(a, b))
        state = left.state_dict()
        expected = left.draw(4, 9, 4)
        restored = EpisodeStream(900)
        restored.load_state_dict(state)
        for a, b in zip(expected, restored.draw(4, 9, 4)):
            self.assertTrue(torch.equal(a, b))

    def test_current_and_future_labels_cannot_leak_to_prediction(self):
        cfg = preset_config("smoke")
        seed_model(0)
        model = RegressionTransformer(cfg).eval()
        xs, labels, _ = EpisodeStream(12).draw(2, cfg.points, cfg.dims)
        with torch.no_grad():
            original = model(xs, labels)
            for k in range(cfg.points):
                changed_labels = labels.clone()
                changed_labels[:, k:] += 100
                changed_xs = xs.clone()
                changed_xs[:, k+1:] -= 100
                changed = model(changed_xs, changed_labels)
                torch.testing.assert_close(original[:, :k+1], changed[:, :k+1],
                                           rtol=0, atol=1e-7)
            # Ensure the test model actually responds to preceding labels.
            preceding = labels.clone()
            preceding[:, 0] += 10
            self.assertGreater(float((model(xs, preceding)[:, 1:] - original[:, 1:]).abs().max()), 1e-6)

    def test_eval_scores_clean_targets_and_shared_bank(self):
        class ZeroPredictor(nn.Module):
            def forward(self, xs, labels):
                return torch.zeros_like(labels)
        xs = torch.zeros(3, 5, 4)
        clean = torch.ones(3, 5)
        noisy = torch.full_like(clean, 100)
        losses = clean_target_losses(ZeroPredictor(), xs, clean, noisy, dims=4,
                                     microbatch=2, device=torch.device("cpu"))
        self.assertTrue(torch.equal(losses, torch.full_like(clean, 0.25)))
        cfg = preset_config("smoke", eval_episodes=16)
        x1, y1, e1 = evaluation_bank(cfg)
        x2, y2, e2 = evaluation_bank(dataclasses.replace(cfg, mode="reverse", seed=800))
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))
        for key in e1:
            self.assertTrue(torch.equal(e1[key], e2[key]))

    def test_gaussian_bayes_reference_uses_only_prefix(self):
        # In d=1, one exact context pair identifies w; the first prediction is zero.
        xs = torch.ones(2, 4, 1)
        clean = torch.full((2, 4), 3.0)
        errors = gaussian_bayes_losses(xs, clean, clean, 0, dims=1, microbatch=2)
        self.assertTrue(torch.equal(errors[:, 0], torch.full((2,), 9.0)))
        torch.testing.assert_close(errors[:, 1:], torch.zeros(2, 3), atol=1e-12, rtol=0)
        contaminated = clean.clone()
        contaminated[:, -1] = 999
        changed = gaussian_bayes_losses(xs, clean, contaminated, 0, dims=1, microbatch=2)
        self.assertTrue(torch.equal(errors, changed))

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        cfg = preset_config("smoke", mode="shuffled", steps=4, batch_size=4,
                            microbatch=2, checkpoint_every=2, log_every=2)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            uninterrupted, done = train(cfg, base / "whole", torch.device("cpu"))
            self.assertEqual(done, 4)
            _, partial = train(cfg, base / "split", torch.device("cpu"), max_steps_this_run=2)
            self.assertEqual(partial, 2)
            resumed, done = train(cfg, base / "split", torch.device("cpu"), resume=True)
            self.assertEqual(done, 4)
            for key, value in uninterrupted.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            a = torch.load(base / "whole" / "checkpoint.pt", weights_only=True)
            b = torch.load(base / "split" / "checkpoint.pt", weights_only=True)
            for key in a["episode_rng"]:
                self.assertTrue(torch.equal(a["episode_rng"][key], b["episode_rng"][key]))
            for param_id, state in a["optimizer"]["state"].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, b["optimizer"]["state"][param_id][key]))
                    else:
                        self.assertEqual(value, b["optimizer"]["state"][param_id][key])
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                train(dataclasses.replace(cfg, learning_rate=1e-3), base / "split",
                      torch.device("cpu"), resume=True)

    def test_full_preset_budget_and_architecture(self):
        cfg = preset_config("full")
        self.assertEqual((cfg.layers, cfg.embedding, cfg.heads), (12, 256, 8))
        self.assertEqual((cfg.dims, cfg.points, cfg.position_capacity), (20, 41, 101))
        self.assertEqual((cfg.steps, cfg.batch_size, cfg.microbatch), (10000, 2048, 512))
        self.assertEqual(cfg.learning_rate, 4e-4)


if __name__ == "__main__":
    unittest.main()
