#!/usr/bin/env python3
"""Strict full-cohort report for the separate H1 follow-up."""
import csv
import dataclasses
import json
import math
from pathlib import Path
import statistics

import torch
import run_experiment as base
import continue_experiment as tail
import checkpoint_diagnostics as diagnostic

ROOT = Path("results/followup_20261006")


def paired_summary(values):
    if len(values) != 5 or any(not math.isfinite(x) for x in values):
        raise ValueError("Require five finite paired training-seed differences")
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(5)
    half = 2.7764451051977987 * se
    return {"n": 5, "seeds": list(range(5)), "differences_noisy_minus_clean": values,
            "mean": mean, "ci95": [mean-half, mean+half]}


def read_metric(path, seed, mode, steps):
    data = json.loads(Path(path).read_text())
    cfg = data["config"]
    if cfg["seed"] != seed or cfg["mode"] != mode or cfg["eval_seed"] != 81973 or cfg["eval_episodes"] != 4096 or data["completed_steps"] != steps:
        raise ValueError(f"Invalid metric identity: {path}")
    if data["primary"]["name"] != "gaussian_sd2_nmse_k20_40":
        raise ValueError("Primary endpoint differs")
    for key, condition in data["conditions"].items():
        curve = condition["mean_nmse_by_k"]
        if len(curve) != 41 or any(not math.isfinite(v) for v in curve) or not math.isfinite(condition["late_context_nmse"]):
            raise ValueError("Invalid metric curve")
        if not math.isclose(statistics.mean(curve[20:41]), condition["late_context_nmse"], abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("Endpoint and curve disagree")
    return data


def validate_pairing(seed, steps, clean):
    directory = ROOT / "clean" / f"seed_{seed}" / f"step_{steps:06d}"
    checkpoint = directory / "checkpoint.pt"
    if tail.file_sha256(checkpoint) != clean["control_checkpoint_sha256"]:
        raise ValueError("Frozen clean checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"] != dataclasses.asdict(base.preset_config("full", mode="clean", seed=seed, steps=20000, precision="bf16")):
        raise ValueError("Clean training configuration differs")
    if saved["completed_steps"] != steps or any(int(state["step"]) != steps for state in saved["optimizer"]["state"].values()):
        raise ValueError("Clean optimizer/progress mismatch")
    if not torch.equal(saved["schedule"], torch.zeros(20000, dtype=torch.float64)):
        raise ValueError("Control is not clean-only")
    episode_rng = saved["episode_rng"]
    del saved
    tag = "original_shuffled" if steps == 10000 else "shuffled_mixed"
    noisy_path = diagnostic.checkpoint_source(Path.cwd(), seed, tag)
    noisy = torch.load(noisy_path, map_location="cpu", weights_only=True)
    if any(not torch.equal(episode_rng[key], noisy["episode_rng"][key]) for key in episode_rng):
        raise ValueError("Clean and noisy training episodes are not paired")
    del noisy


def main():
    torch.set_num_threads(1)
    all_metrics = {}
    curves = []
    for steps in (10000, 20000):
        pairs = []
        for seed in range(5):
            clean = read_metric(ROOT/"clean"/f"seed_{seed}"/f"step_{steps:06d}"/"metrics.json", seed, "clean", steps)
            noisy_path = (Path("results/full")/f"seed_{seed}"/"shuffled/metrics.json" if steps == 10000 else
                          Path("results/continuation_10000")/f"seed_{seed}"/"shuffled/mixed/evaluations/step_010000/metrics.json")
            noisy = read_metric(noisy_path, seed, "shuffled", steps)
            validate_pairing(seed, steps, clean)
            pairs.append((clean, noisy))
            for arm, data in (("clean", clean), ("shuffled" if steps == 10000 else "shuffled_mixed", noisy)):
                for condition, metric in data["conditions"].items():
                    curves.extend({"steps": steps, "seed": seed, "arm": arm, "condition": condition,
                                   "k": k, "nmse": value} for k, value in enumerate(metric["mean_nmse_by_k"]))
        conditions = {}
        for condition in pairs[0][0]["conditions"]:
            clean_values = [pair[0]["conditions"][condition]["late_context_nmse"] for pair in pairs]
            noisy_values = [pair[1]["conditions"][condition]["late_context_nmse"] for pair in pairs]
            stats = paired_summary([n-c for c, n in zip(clean_values, noisy_values)])
            stats.update(clean_mean=statistics.mean(clean_values), noisy_mean=statistics.mean(noisy_values))
            conditions[condition] = stats
        all_metrics[str(steps)] = conditions
    tags = ["original_curriculum", "original_shuffled", "curriculum_high", "shuffled_high",
            "curriculum_mixed", "shuffled_mixed", "clean_10000", "clean_20000"]
    diagnostic_rows, diagnostic_reports = [], {}
    identity = diagnostic.cache_identity()
    for seed in range(5):
        for tag in tags:
            path = ROOT/"diagnostics"/f"seed_{seed}"/f"{tag}.json"
            report = json.loads(path.read_text())
            if report["identity"]["cache_identity"] != identity or report["identity"]["seed"] != seed or report["identity"]["tag"] != tag:
                raise ValueError("Diagnostic identity differs")
            if report["identity"]["checkpoint_sha256"] != tail.file_sha256(diagnostic.checkpoint_source(Path.cwd(), seed, tag)):
                raise ValueError("Diagnostic source changed")
            diagnostic_reports[seed, tag] = report
            for condition, values in report["conditions"].items():
                row = {"seed": seed, "tag": tag, "condition": condition,
                       "heldout_task_nmse": values["model_task_nmse"]["mean_k20_40"],
                       "fitted_penalty": values["calibration_selected_penalty"],
                       "fitted_ridge_disagreement": values["references"]["calibration_fitted_ridge"]["prediction_disagreement_nmse"]["mean_k20_40"],
                       "oracle_task_nmse": values["references"]["known_sigma_oracle"]["reference_task_nmse"]["mean_k20_40"],
                       "empirical_bayes_task_nmse": values["references"]["empirical_bayes_context_only"]["reference_task_nmse"]["mean_k20_40"],
                       "rotation_prediction_change": values["invariance"]["orthogonal_rotation"]["prediction_change_nmse"]["mean_k20_40"],
                       "permutation_prediction_change": values["invariance"]["context_permutation"]["prediction_change_nmse"]["mean_k20_40"]}
                if any(not math.isfinite(value) for key, value in row.items() if isinstance(value, (float, int))):
                    raise ValueError("Nonfinite diagnostic summary")
                diagnostic_rows.append(row)
    fresh = {}
    for steps, tag in ((10000, "original_shuffled"), (20000, "shuffled_mixed")):
        differences = [diagnostic_reports[seed, tag]["conditions"]["gaussian_sd2"]["model_task_nmse"]["mean_k20_40"] -
                       diagnostic_reports[seed, f"clean_{steps}"]["conditions"]["gaussian_sd2"]["model_task_nmse"]["mean_k20_40"] for seed in range(5)]
        fresh[str(steps)] = paired_summary(differences)
    primary = all_metrics["20000"]["gaussian_sd2"]
    if primary["ci95"][1] < 0:
        conclusion = "Noise exposure improves the predeclared SD=2 late-context endpoint at 20k in this setting."
    elif primary["ci95"][0] > 0:
        conclusion = "Clean-only training performs better on the predeclared SD=2 late-context endpoint at 20k in this setting."
    else:
        conclusion = "The predeclared H1 comparison is inconclusive; superiority and equivalence are not established."
    result = {"kind": "h1_clean_control_followup_v1", "protocol_sha256": tail.file_sha256("FOLLOWUP_PROTOCOL.txt"),
              "summary_script_sha256": tail.file_sha256(__file__), "primary_20000_gaussian_sd2": primary,
              "secondary_conditions": all_metrics, "secondary_fresh_bank_gaussian_sd2": fresh,
              "controls": 5, "clean_snapshots": 10, "diagnostic_reports": 40,
              "episode_pairing_verified": True, "conclusion": conclusion}
    base.atomic_json(ROOT/"summary.json", result)
    for filename, rows in (("curves.csv", curves), ("diagnostics.csv", diagnostic_rows)):
        with (ROOT/filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["Separate H1 follow-up: complete", conclusion, "",
             f"Primary: 20k Gaussian SD=2, k=20..40, shared 4096-episode bank 81973.",
             f"Noisy mean={primary['noisy_mean']:.9g}; clean-only mean={primary['clean_mean']:.9g}.",
             f"D=noisy-clean: {primary['mean']:+.9g}; 95% t interval {primary['ci95']}.",
             f"All five paired differences: {primary['differences_noisy_minus_clean']}",
             "All five controls, ten frozen snapshots, paired episode RNGs and forty diagnostic reports verified.",
             "Secondary outcomes and fresh-bank comparisons are descriptive and unadjusted.",
             "Diagnostic fits are exploratory prediction agreement, not proof of an internal ridge algorithm.",
             "Known-SD oracle is privileged; context-only EB assumes the known Gaussian task prior.",
             "The completed H2 curriculum study is unchanged; this follow-up does not establish a curriculum benefit."]
    for steps in ("10000", "20000"):
        lines += ["", f"Budget {steps}: descriptive late-context comparisons:"]
        for key in ("gaussian_sd0", "gaussian_sd0.5", "gaussian_sd1", "gaussian_sd2"):
            stat = all_metrics[steps][key]
            lines.append(f"{key}: clean={stat['clean_mean']:.8g}, noisy={stat['noisy_mean']:.8g}, D={stat['mean']:+.8g}, CI={stat['ci95']}")
    (ROOT/"RESULTS.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
