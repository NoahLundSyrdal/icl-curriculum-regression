#!/usr/bin/env python3
"""Paired training-seed inference for the preregistered H2 endpoint."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

# Two-sided Student t 97.5% quantiles for 1..30 degrees of freedom.
T975 = (None, 12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469,
        2.3646, 2.3060, 2.2622, 2.2281, 2.2010, 2.1788, 2.1604,
        2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
        2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518,
        2.0484, 2.0452, 2.0423)


def paired_interval(differences):
    """95% t interval across independent training seeds, not test episodes."""
    mean = statistics.mean(differences)
    if len(differences) < 2:
        return mean, None
    df = len(differences) - 1
    if df >= len(T975):
        raise ValueError("This small-study interval supports 2..31 seeds.")
    radius = T975[df] * statistics.stdev(differences) / math.sqrt(len(differences))
    return mean, [mean - radius, mean + radius]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="One preset's results directory")
    args = parser.parse_args()
    records = {}
    configs = []
    endpoint_names = set()
    for path in sorted(args.directory.rglob("metrics.json")):
        data = json.loads(path.read_text())
        config = data["config"]
        key = (int(config["seed"]), config["mode"])
        if key in records:
            raise ValueError(f"Duplicate seed/mode {key}; summarize one experiment only.")
        value = float(data["primary"]["value"])
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite primary metric in {path}")
        records[key] = value
        configs.append(config)
        endpoint_names.add(data["primary"]["name"])
    if not records:
        raise ValueError("No metrics.json found; jobs must finish evaluation first.")
    if len(endpoint_names) != 1:
        raise ValueError("Cannot combine different primary endpoints.")
    # Only seed and mode should differ between arms of the same experiment.
    ignored = {"mode", "seed"}
    signatures = {json.dumps({k: v for k, v in c.items() if k not in ignored},
                             sort_keys=True) for c in configs}
    if len(signatures) != 1:
        raise ValueError("Configuration mismatch across runs; do not pool these results.")
    preset = configs[0]["preset"]
    all_seeds = sorted({seed for seed, _ in records})
    paired_seeds = [s for s in all_seeds if (s, "curriculum") in records and
                    (s, "shuffled") in records]
    if not paired_seeds:
        raise ValueError("No complete curriculum/shuffled seed pairs found.")
    differences = [records[(s, "curriculum")] - records[(s, "shuffled")]
                   for s in paired_seeds]
    mean, interval = paired_interval(differences)
    missing = [[s, m] for s in all_seeds for m in
               ("clean", "curriculum", "shuffled", "reverse") if (s, m) not in records]
    canonical_full = {
        "dims": 20, "points": 41, "position_capacity": 101,
        "embedding": 256, "layers": 12, "heads": 8,
        "steps": 10000, "batch_size": 2048, "microbatch": 512,
        "learning_rate": 4e-4, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "sigma_max": 1.0, "eval_episodes": 4096, "eval_seed": 81973,
    }
    canonical_budget = all(configs[0].get(k) == v for k, v in canonical_full.items())
    eligible = preset == "full" and canonical_budget and len(paired_seeds) >= 5
    if not eligible:
        interpretation = "Preliminary only: require completed canonical full runs and at least five paired seeds. Modified full settings are exploratory."
    elif interval[1] < 0:
        interpretation = "Evidence for lower curriculum MSE at the primary endpoint; check convergence before a scientific claim."
    elif interval[0] > 0:
        interpretation = "Evidence for higher curriculum MSE at the primary endpoint; check convergence before a scientific claim."
    else:
        interpretation = "Inconclusive: the interval includes zero. This does not establish equivalence."
    summary = {
        "preset": preset,
        "primary_endpoint": next(iter(endpoint_names)),
        "paired_seeds": paired_seeds,
        "missing_secondary_controls": missing,
        "mean_curriculum_minus_shuffled_nmse": mean,
        "ci95_across_training_seeds": interval,
        "method": "Paired Student t interval across training seeds; fixed shared test set.",
        "interpretation": interpretation,
        "limitations": "Five seeds provide limited precision; the t interval assumes roughly normal seed differences. Test-episode uncertainty and protocol sensitivity are separate.",
    }
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.directory / "primary_pairs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "clean_nmse", "curriculum_nmse", "shuffled_nmse",
                         "reverse_nmse", "curriculum_minus_shuffled"])
        for seed in paired_seeds:
            writer.writerow([seed] + [records.get((seed, mode), "") for mode in
                            ("clean", "curriculum", "shuffled", "reverse")] +
                            [records[(seed, "curriculum")] - records[(seed, "shuffled")]])
    ci_text = "unavailable (one seed)" if interval is None else f"[{interval[0]:.6f}, {interval[1]:.6f}]"
    text = (f"Preset: {preset}\nEndpoint: {summary['primary_endpoint']}\n"
            f"Paired seeds: {paired_seeds}\n"
            f"Curriculum minus shuffled NMSE: {mean:.6f}\n95% interval: {ci_text}\n"
            f"Negative differences favor curriculum.\n{interpretation}\n"
            f"Missing conditions (secondary controls): {missing}\n{summary['limitations']}\n")
    (args.directory / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
