"""Stdlib fixtures for strict seed-level exploratory continuation inference."""
import copy
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

from summarize_continuation import (
    CANONICAL, HISTORIES, PRIMARY, SEEDS, TAILS, TIMES,
    summarize, write_outputs,
)


def fixture_record(seed, history, tail, elapsed):
    initial = (0.4 if history == "curriculum" else 0.5) + seed * 0.01
    fraction = elapsed / 10000
    if tail == "high":
        improvement = (0.2 + seed * 0.005) if history == "curriculum" else 0.1
    else:
        improvement = 0.1 if history == "curriculum" else 0.05
    primary = initial - improvement * fraction
    conditions = {}
    for name, factor in (("gaussian_sd0", 0.1), ("gaussian_sd1", 0.5), ("gaussian_sd2", 1.0)):
        v = primary * factor
        conditions[name] = {"mean_nmse": v, "late_context_nmse": v, "mean_nmse_by_k": [v] * 41}
    return {
        "schema_version": 1, "config": CANONICAL | {"seed": seed, "mode": history},
        "continuation": {"tail": tail, "steps": 10000, "eval_steps": list(TIMES), "checkpoint_every": 100, "log_every": 25},
        "source": {"checkpoint_sha256": ("a" if history == "curriculum" else "b") * 64,
                   "source_script_sha256": "c" * 64, "config_sha256": "d" * 64,
                   "metrics_sha256": "e" * 64,
                   "original_primary": {"name": PRIMARY, "value": initial},
                   "original_completed_steps": 10000, "original_mode": history, "original_seed": seed},
        "tail_completed_steps": elapsed, "total_completed_steps": 10000 + elapsed,
        "primary": {"name": PRIMARY, "condition": "gaussian_sd2", "k_start": 20, "k_end": 40, "value": primary},
        "conditions": conditions, "runtime": {"script_sha256": "f" * 64, "base_script_sha256": "c" * 64, "python": "test-fixture"},
    }


def record_path(directory, seed, history, tail, elapsed):
    return directory / f"seed_{seed}" / history / tail / "evaluations" / f"step_{elapsed:06d}" / "metrics.json"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class ContinuationSummaryChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "continuation_10000"
        for seed in SEEDS:
            for history in HISTORIES:
                for tail in TAILS:
                    for elapsed in TIMES:
                        save(record_path(self.directory, seed, history, tail, elapsed), fixture_record(seed, history, tail, elapsed))

    def alter(self, key, function):
        path = record_path(self.directory, *key)
        data = json.loads(path.read_text())
        function(data)
        save(path, data)

    def test_fixed_interaction_arithmetic_ci_and_baseline_changes(self):
        result, rows = summarize(self.directory)
        self.assertTrue(result["complete"])
        self.assertEqual(result["observed_records"], 100)
        main = result["main_contrast"]
        self.assertEqual(main["n"], 5)
        self.assertEqual(main["seeds"], list(SEEDS))
        values = [-0.05 - 0.005 * seed for seed in SEEDS]
        expected_mean = statistics.mean(values)
        radius = 2.7764 * statistics.stdev(values) / math.sqrt(5)
        self.assertAlmostEqual(main["mean"], expected_mean, places=12)
        self.assertAlmostEqual(main["ci95"][0], expected_mean - radius, places=12)
        self.assertAlmostEqual(main["ci95"][1], expected_mean + radius, places=12)
        self.assertLess(main["ci95"][1], 0)
        self.assertIn("Exploratory", main["interpretation"])
        final = result["trajectories"][-1]
        high = final["tails"]["high"]["gaussian_sd2"]
        mixed = final["tails"]["mixed"]["gaussian_sd2"]
        self.assertAlmostEqual(mixed["history_gap"]["mean"], -0.15)
        # The seeded high-arm extra reduction is .01 on average: D_high=-.21.
        self.assertAlmostEqual(high["history_gap"]["mean"], -0.21)
        self.assertAlmostEqual(high["history_gap_change_from_baseline"]["mean"], -0.11)
        self.assertAlmostEqual(mixed["history_gap_change_from_baseline"]["mean"], -0.05)
        self.assertAlmostEqual(high["arm_changes_from_baseline"]["curriculum"]["mean"], -0.21)
        self.assertAlmostEqual(high["arm_changes_from_baseline"]["shuffled"]["mean"], -0.1)
        self.assertEqual([x["tail_completed_steps"] for x in result["trajectories"]], list(TIMES))
        baseline = result["trajectories"][0]["interactions"]["gaussian_sd2"]["history_gap_high_minus_mixed"]
        self.assertAlmostEqual(baseline["mean"], 0)
        self.assertAlmostEqual(final["interactions"]["gaussian_sd2"]["interaction_change_from_baseline"]["mean"], main["mean"])
        self.assertEqual(len(result["baseline_consistency"]["checks"]), 20)
        self.assertEqual(result["baseline_consistency"]["high_mixed_baselines_compared"], 10)
        self.assertTrue(all(x["difference"] == 0 for x in result["baseline_consistency"]["checks"]))
        self.assertTrue(result["original_h2"]["analysis_unchanged"])
        self.assertEqual(len(rows), 5 * 2 * 3 * 5)
        text = write_outputs(self.directory, result, rows)
        self.assertIn("different noise budgets", text)
        self.assertTrue((self.directory / "trajectory_pairs.csv").exists())
        self.assertEqual(json.loads((self.directory / "summary.json").read_text())["main_contrast"]["n"], 5)

    def test_missing_final_pair_requires_explicit_partial_report(self):
        record_path(self.directory, 4, "shuffled", "mixed", 10000).unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete.*missing final"):
            summarize(self.directory)
        result, _ = summarize(self.directory, allow_partial=True)
        self.assertFalse(result["complete"])
        self.assertEqual(result["main_contrast"]["n"], 4)
        self.assertEqual(result["main_contrast"]["seeds"], [0, 1, 2, 3])
        self.assertIn("Partial", result["main_contrast"]["interpretation"])
        self.assertEqual(result["missing_records"], [[4, "shuffled", "mixed", 10000]])
        self.assertEqual(result["trajectories"][-1]["tails"]["high"]["gaussian_sd2"]["history_gap"]["n"], 5)

    def test_missing_intermediate_evaluation_is_not_silently_discarded(self):
        record_path(self.directory, 0, "curriculum", "high", 2500).unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize(self.directory)
        result, _ = summarize(self.directory, allow_partial=True)
        self.assertEqual(result["main_contrast"]["n"], 5)
        self.assertIn("Partial", result["main_contrast"]["interpretation"])
        self.assertEqual(result["trajectories"][2]["interactions"]["gaussian_sd2"]["history_gap_high_minus_mixed"]["n"], 4)

    def test_noncanonical_config_and_differing_tail_budgets_rejected(self):
        key = (0, "curriculum", "high", 1000)
        original = fixture_record(*key)
        for field, changed in (("steps", 200), ("microbatch", 64), ("eval_seed", 99)):
            with self.subTest(field=field):
                invalid = copy.deepcopy(original)
                invalid["config"][field] = changed
                save(record_path(self.directory, *key), invalid)
                with self.assertRaisesRegex(ValueError, "Noncanonical"):
                    summarize(self.directory)
        save(record_path(self.directory, *key), original)
        self.alter(key, lambda d: d["continuation"].update(steps=5000))
        with self.assertRaisesRegex(ValueError, "budget differs"):
            summarize(self.directory)
        save(record_path(self.directory, *key), original)
        self.alter(key, lambda d: d["continuation"].update(eval_steps=[0, 10000]))
        with self.assertRaisesRegex(ValueError, "timepoints differ"):
            summarize(self.directory)
        save(record_path(self.directory, *key), original)
        self.alter(key, lambda d: d["continuation"].update(log_every=50))
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            summarize(self.directory)

    def test_duplicate_record_rejected_even_in_partial_mode(self):
        duplicate = self.directory / "zz_duplicate" / "metrics.json"
        save(duplicate, fixture_record(0, "curriculum", "high", 0))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(self.directory, allow_partial=True)

    def test_baseline_disagreement_and_source_provenance_changes_rejected(self):
        key = (1, "curriculum", "mixed", 0)
        def change_baseline(data):
            data["primary"]["value"] += 0.01
            c = data["conditions"]["gaussian_sd2"]
            c.update(mean_nmse=data["primary"]["value"], late_context_nmse=data["primary"]["value"], mean_nmse_by_k=[data["primary"]["value"]] * 41)
        self.alter(key, change_baseline)
        with self.assertRaisesRegex(ValueError, "Baseline inconsistency"):
            summarize(self.directory)
        save(record_path(self.directory, *key), fixture_record(*key))
        self.alter(key, lambda d: d["source"].update(checkpoint_sha256="9" * 64))
        with self.assertRaisesRegex(ValueError, "Source provenance changed"):
            summarize(self.directory)

    def test_original_curve_comparison_and_original_summary_preserved(self):
        original_dir = self.directory.parent / "full"
        for seed in SEEDS:
            for history in HISTORIES:
                data = fixture_record(seed, history, "high", 0)
                save(original_dir / f"seed_{seed}" / history / "metrics.json", data)
        old = {"interpretation": "Original primary inconclusive", "ci95_across_training_seeds": [-0.03, 0.02]}
        save(original_dir / "summary.json", old)
        result, _ = summarize(self.directory, source_directory=original_dir)
        self.assertEqual(result["original_h2"]["original_summary_snapshot"], old)
        self.assertTrue(all(x["original_curve_file_compared"] for x in result["baseline_consistency"]["checks"]))
        path = original_dir / "seed_0" / "curriculum" / "metrics.json"
        data = json.loads(path.read_text())
        data["conditions"]["gaussian_sd1"] = {"mean_nmse": 99.0, "late_context_nmse": 99.0, "mean_nmse_by_k": [99.0] * 41}
        save(path, data)
        with self.assertRaisesRegex(ValueError, "Baseline curve inconsistency"):
            summarize(self.directory, source_directory=original_dir)

    def test_small_baseline_curve_drift_preserves_primary_and_is_recorded(self):
        original_dir = self.directory.parent / "full"
        for seed in SEEDS:
            for history in HISTORIES:
                save(original_dir / f"seed_{seed}" / history / "metrics.json", fixture_record(seed, history, "high", 0))
        key = (0, "curriculum", "high", 0)
        def shift_curve(data):
            curve = data["conditions"]["gaussian_sd2"]["mean_nmse_by_k"]
            curve[21] += 1e-5
            curve[22] -= 1e-5
        self.alter(key, shift_curve)
        result, rows = summarize(self.directory, source_directory=original_dir)
        check = next(x for x in result["baseline_consistency"]["checks"] if (x["seed"], x["history"], x["tail"]) == key[:3])
        self.assertEqual(check["primary_delta_vs_original"], 0)
        self.assertEqual(check["primary_delta_vs_counterpart_tail"], 0)
        self.assertAlmostEqual(check["max_abs_curve_delta"], 1e-5, places=12)
        self.assertAlmostEqual(check["max_abs_curve_delta_vs_original"], 1e-5, places=12)
        self.assertAlmostEqual(check["max_abs_curve_delta_vs_counterpart_tail"], 1e-5, places=12)
        consistency = result["baseline_consistency"]
        self.assertEqual(consistency["primary_tolerance"], {"rtol": 1e-5, "atol": 1e-7})
        self.assertEqual(consistency["curve_tolerance"]["atol"], 2e-5)
        self.assertIn("before inspecting any continuation outcomes", consistency["curve_tolerance"]["basis"])
        text = write_outputs(self.directory, result, rows)
        self.assertIn("baseline-curve-only tolerance", text)
        self.assertIn("max_abs_curve_delta=", text)

    def test_large_baseline_curve_drift_rejected_with_unchanged_primary(self):
        key = (0, "curriculum", "high", 0)
        def shift_curve(data):
            curve = data["conditions"]["gaussian_sd2"]["mean_nmse_by_k"]
            curve[21] += 2.1e-5
            curve[22] -= 2.1e-5
        self.alter(key, shift_curve)
        with self.assertRaisesRegex(ValueError, "Baseline curve inconsistency.*high/mixed"):
            summarize(self.directory)
        # Matching counterpart curves cannot hide drift from the original source.
        original_dir = self.directory.parent / "full"
        for seed in SEEDS:
            for history in HISTORIES:
                save(original_dir / f"seed_{seed}" / history / "metrics.json", fixture_record(seed, history, "high", 0))
        self.alter((0, "curriculum", "mixed", 0), shift_curve)
        with self.assertRaisesRegex(ValueError, "Baseline curve inconsistency.*original"):
            summarize(self.directory, source_directory=original_dir)

    def test_malformed_endpoint_nonfinite_and_counter_records_rejected(self):
        key = (0, "shuffled", "high", 1000)
        original = fixture_record(*key)
        self.alter(key, lambda d: d["primary"].update(name="different_metric"))
        with self.assertRaisesRegex(ValueError, "Primary endpoint differs"):
            summarize(self.directory)
        save(record_path(self.directory, *key), original)
        self.alter(key, lambda d: d["primary"].update(value=float("nan")))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            summarize(self.directory)
        save(record_path(self.directory, *key), original)
        self.alter(key, lambda d: d.update(total_completed_steps=9999))
        with self.assertRaisesRegex(ValueError, "completion counters"):
            summarize(self.directory)

    def test_producer_hash_mismatch_or_missing_hash_is_rejected(self):
        key = (2, "curriculum", "mixed", 5000)
        original = fixture_record(*key)
        self.alter(key, lambda d: d["runtime"].update(script_sha256="1" * 64))
        with self.assertRaisesRegex(ValueError, "script hashes differ"):
            summarize(self.directory)
        save(record_path(self.directory, *key), original)
        self.alter(key, lambda d: d["runtime"].pop("script_sha256"))
        with self.assertRaisesRegex(ValueError, "Malformed provenance digest"):
            summarize(self.directory)
        save(record_path(self.directory, *key), original)
        self.alter(key, lambda d: d["runtime"].update(base_script_sha256="1" * 64))
        with self.assertRaisesRegex(ValueError, "Imported base script differs"):
            summarize(self.directory)

    def test_positive_and_zero_interactions_have_correct_interpretation(self):
        for desired in ("positive", "zero"):
            for seed in SEEDS:
                for history in HISTORIES:
                    for tail in TAILS:
                        for elapsed in TIMES:
                            actual_tail = ("mixed" if tail == "high" else "high") if desired == "positive" else "mixed"
                            data = fixture_record(seed, history, actual_tail, elapsed)
                            data["continuation"]["tail"] = tail
                            save(record_path(self.directory, seed, history, tail, elapsed), data)
            result, _ = summarize(self.directory)
            main = result["main_contrast"]
            if desired == "positive":
                self.assertGreater(main["ci95"][0], 0)
                self.assertIn("more positive", main["interpretation"])
            else:
                self.assertEqual(main["mean"], 0)
                self.assertEqual(main["ci95"], [0, 0])
                self.assertIn("Inconclusive", main["interpretation"])
                self.assertIn("does not establish equivalence", main["interpretation"])


if __name__ == "__main__":
    unittest.main()
