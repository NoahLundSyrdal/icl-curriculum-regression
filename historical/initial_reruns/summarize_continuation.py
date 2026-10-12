#!/usr/bin/env python3
"""Summarize the predeclared exploratory continuation diagnostic using only stdlib.

Strict mode requires seeds 0..4, both original histories, both continuation tails,
and every fixed evaluation time. The independent unit is a training seed.
The original H2 analysis and its inconclusive conclusion remain unchanged.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from summarize import paired_interval

SEEDS = tuple(range(5))
HISTORIES = ("curriculum", "shuffled")
TAILS = ("high", "mixed")
TIMES = (0, 1000, 2500, 5000, 10000)
PRIMARY = "gaussian_sd2_nmse_k20_40"
REQUIRED_CONDITIONS = ("gaussian_sd0", "gaussian_sd1", "gaussian_sd2")
BASELINE_RTOL, BASELINE_ATOL = 1e-5, 1e-7
# Set from step-0-only cross-A100 form-factor diagnostics before inspecting
# any continuation outcomes. This tolerance applies only to baseline curves;
# primary endpoints, finiteness and curve aggregation checks remain unchanged.
BASELINE_CURVE_RTOL, BASELINE_CURVE_ATOL = BASELINE_RTOL, 2e-5
CANONICAL = {
    "preset": "full", "dims": 20, "points": 41, "position_capacity": 101,
    "embedding": 256, "layers": 12, "heads": 8,
    "steps": 10000, "batch_size": 2048, "microbatch": 512,
    "learning_rate": 4e-4, "weight_decay": 0.01, "max_grad_norm": 1.0,
    "sigma_max": 1.0, "eval_seed": 81973, "eval_episodes": 4096,
    "eval_microbatch": 64, "checkpoint_every": 100, "log_every": 25,
    "precision": "bf16", "bayes_reference": True,
}


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Malformed numeric {label}")
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {label}")
    return float(value)


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Malformed integer {label}")
    return value


def signature(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Malformed provenance digest {label}")
    return value


def require_close(left, right, label):
    if not math.isclose(left, right, rel_tol=BASELINE_RTOL, abs_tol=BASELINE_ATOL):
        raise ValueError(f"Baseline inconsistency for {label}: {left} versus {right}")


def require_baseline_curve_close(left, right, label):
    if not math.isclose(left, right, rel_tol=BASELINE_CURVE_RTOL, abs_tol=BASELINE_CURVE_ATOL):
        raise ValueError(f"Baseline curve inconsistency for {label}: {left} versus {right}")


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Malformed original config")
    if set(config) != set(CANONICAL) | {"seed", "mode"}:
        raise ValueError("Original configuration keys differ from canonical full config")
    for key, expected in CANONICAL.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"Noncanonical original configuration: {key}")
    seed = integer(config["seed"], "seed")
    history = config["mode"]
    if seed not in SEEDS or history not in HISTORIES:
        raise ValueError("Only original seeds 0..4 and curriculum/shuffled histories are allowed")
    return seed, history


def validate_primary(primary):
    if not isinstance(primary, dict) or primary.get("name") != PRIMARY:
        raise ValueError("Primary endpoint differs from Gaussian SD2 NMSE k20..40")
    value = finite_number(primary.get("value"), "primary value")
    if value < 0:
        raise ValueError("Negative primary NMSE")
    for key, expected in (("condition", "gaussian_sd2"), ("k_start", 20), ("k_end", 40)):
        if key in primary and primary[key] != expected:
            raise ValueError(f"Primary endpoint {key} differs")
    return value


def validate_conditions(conditions):
    if not isinstance(conditions, dict) or not set(REQUIRED_CONDITIONS) <= set(conditions):
        raise ValueError("Missing Gaussian clean/SD1/SD2 condition metrics")
    for key, condition in conditions.items():
        if not isinstance(condition, dict):
            raise ValueError(f"Malformed condition {key}")
        curve = condition.get("mean_nmse_by_k")
        if not isinstance(curve, list) or len(curve) != CANONICAL["points"]:
            raise ValueError(f"Malformed context curve for {key}")
        values = [finite_number(v, key) for v in curve]
        late = finite_number(condition.get("late_context_nmse"), key)
        mean = finite_number(condition.get("mean_nmse"), key)
        if min(values + [late, mean]) < 0:
            raise ValueError(f"Negative NMSE for {key}")
        require_close(late, sum(values[20:]) / 21, f"{key} late-context aggregation")
        require_close(mean, sum(values) / 41, f"{key} full-context aggregation")


def validate_source(source, seed, history):
    if not isinstance(source, dict):
        raise ValueError("Missing source provenance")
    for key in ("checkpoint_sha256", "source_script_sha256", "config_sha256", "metrics_sha256"):
        digest(source.get(key), key)
    if integer(source.get("original_seed"), "source original_seed") != seed or source.get("original_mode") != history:
        raise ValueError("Source provenance seed/history differs")
    if integer(source.get("original_completed_steps"), "source original_completed_steps") != CANONICAL["steps"]:
        raise ValueError("Source model did not complete canonical original training")
    validate_primary(source.get("original_primary"))


def load_records(directory, allow_partial=False):
    directory = Path(directory)
    records, sources = {}, {}
    continuation_signatures, config_signatures, condition_signatures = set(), set(), set()
    script_hashes = set()
    for path in sorted(directory.rglob("metrics.json")):
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("Malformed metrics object")
            seed, history = validate_config(data.get("config"))
            continuation = data.get("continuation")
            if not isinstance(continuation, dict):
                raise ValueError("Missing continuation configuration")
            tail = continuation.get("tail")
            if tail not in TAILS or integer(continuation.get("steps"), "continuation steps") != TIMES[-1]:
                raise ValueError("Continuation tail or step budget differs")
            eval_steps = continuation.get("eval_steps")
            if not isinstance(eval_steps, list) or any(type(n) is not int for n in eval_steps) or eval_steps != list(TIMES):
                raise ValueError("Continuation evaluation timepoints differ")
            elapsed = integer(data.get("tail_completed_steps"), "tail_completed_steps")
            total = integer(data.get("total_completed_steps"), "total_completed_steps")
            if elapsed not in TIMES or total != CANONICAL["steps"] + elapsed:
                raise ValueError("Evaluation completion counters differ")
            key = (seed, history, tail, elapsed)
            if key in records:
                raise ValueError(f"Duplicate evaluation {key}")
            expected = Path(f"seed_{seed}") / history / tail / "evaluations" / f"step_{elapsed:06d}" / "metrics.json"
            if path.relative_to(directory) != expected:
                raise ValueError(f"Evaluation path differs from record identity: expected {expected}")
            primary = validate_primary(data.get("primary"))
            validate_conditions(data.get("conditions"))
            require_close(primary, data["conditions"]["gaussian_sd2"]["late_context_nmse"], "primary versus SD2")
            validate_source(data.get("source"), seed, history)
            source_key = (seed, history)
            if source_key in sources and signature(sources[source_key]) != signature(data["source"]):
                raise ValueError(f"Source provenance changed across tails/timepoints for {source_key}")
            sources[source_key] = data["source"]
            runtime = data.get("runtime")
            if not isinstance(runtime, dict) or not runtime:
                raise ValueError("Missing evaluation runtime provenance")
            script_hashes.add(digest(runtime.get("script_sha256"), "continuation script_sha256"))
            base_digest = digest(runtime.get("base_script_sha256"), "base_script_sha256")
            if base_digest != data["source"]["source_script_sha256"]:
                raise ValueError("Imported base script differs from source original script")
            records[key] = data
            continuation_signatures.add(signature({k: v for k, v in continuation.items() if k != "tail"}))
            config_signatures.add(signature({k: v for k, v in data["config"].items() if k not in ("seed", "mode")}))
            condition_signatures.add(tuple(sorted(data["conditions"])))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Invalid record {path}: {error}") from error
    if not records:
        raise ValueError("No continuation metrics.json records found")
    if len(continuation_signatures) != 1 or len(config_signatures) != 1:
        raise ValueError("Configuration or continuation budget mismatch across runs")
    if len(condition_signatures) != 1:
        raise ValueError("Evaluation condition sets differ across runs")
    if len(script_hashes) > 1:
        raise ValueError("Continuation script hashes differ across runs")
    expected_keys = {(s, h, t, n) for s in SEEDS for h in HISTORIES for t in TAILS for n in TIMES}
    missing = sorted(expected_keys - records.keys())
    if missing and not allow_partial:
        final_missing = sum(key[-1] == TIMES[-1] for key in missing)
        raise ValueError(f"Incomplete continuation study: {len(missing)} missing evaluations, including {final_missing} missing final records; use --allow-partial explicitly")
    return records, sources, missing


def compare_baselines(records, sources, source_directory=None):
    comparisons = []
    for seed in SEEDS:
        for history in HISTORIES:
            source_key = (seed, history)
            if source_key not in sources:
                continue
            baseline = {tail: records.get((seed, history, tail, 0)) for tail in TAILS}
            counterpart_max_curve_delta = None
            if all(baseline.values()):
                high, mixed = baseline["high"], baseline["mixed"]
                require_close(high["primary"]["value"], mixed["primary"]["value"], f"seed {seed}/{history} high/mixed primary")
                counterpart_max_curve_delta = 0.0
                for condition in high["conditions"]:
                    for k, (left, right) in enumerate(zip(high["conditions"][condition]["mean_nmse_by_k"], mixed["conditions"][condition]["mean_nmse_by_k"])):
                        require_baseline_curve_close(left, right, f"seed {seed}/{history} high/mixed {condition}/k{k}")
                        counterpart_max_curve_delta = max(counterpart_max_curve_delta, abs(left - right))
            original = sources[source_key]["original_primary"]["value"]
            source_path = (Path(source_directory) / f"seed_{seed}" / history / "metrics.json") if source_directory else None
            source_data = None
            if source_path and source_path.exists():
                source_data = json.loads(source_path.read_text())
                if validate_config(source_data.get("config")) != source_key:
                    raise ValueError(f"Original source identity mismatch: {source_path}")
                source_value = validate_primary(source_data.get("primary"))
                require_close(original, source_value, f"seed {seed}/{history} embedded versus source original")
                validate_conditions(source_data.get("conditions"))
            for tail, data in baseline.items():
                if data is None:
                    continue
                value = data["primary"]["value"]
                require_close(value, original, f"seed {seed}/{history}/{tail} reevaluation versus original")
                original_max_curve_delta = None
                if source_data:
                    original_max_curve_delta = 0.0
                    for condition in data["conditions"]:
                        if condition not in source_data["conditions"]:
                            raise ValueError(f"Original source lacks condition {condition}")
                        for k, (left, right) in enumerate(zip(data["conditions"][condition]["mean_nmse_by_k"], source_data["conditions"][condition]["mean_nmse_by_k"])):
                            require_baseline_curve_close(left, right, f"seed {seed}/{history}/{tail} original {condition}/k{k}")
                            original_max_curve_delta = max(original_max_curve_delta, abs(left - right))
                counterpart = baseline["mixed" if tail == "high" else "high"]
                available_curve_deltas = [v for v in (original_max_curve_delta, counterpart_max_curve_delta) if v is not None]
                comparisons.append({"seed": seed, "history": history, "tail": tail,
                                    "reevaluated_primary": value, "original_primary": original,
                                    "difference": value - original,
                                    "primary_delta_vs_original": value - original,
                                    "primary_delta_vs_counterpart_tail": value - counterpart["primary"]["value"] if counterpart else None,
                                    "max_abs_curve_delta": max(available_curve_deltas) if available_curve_deltas else None,
                                    "max_abs_curve_delta_vs_original": original_max_curve_delta,
                                    "max_abs_curve_delta_vs_counterpart_tail": counterpart_max_curve_delta,
                                    "original_curve_file_compared": str(source_path) if source_data else None})
    return comparisons


def estimate(values):
    seeds = sorted(values)
    if not seeds:
        return {"n": 0, "seeds": [], "mean": None, "ci95": None}
    mean, interval = paired_interval([values[s] for s in seeds])
    return {"n": len(seeds), "seeds": seeds, "mean": mean, "ci95": interval}


def value(data, condition):
    return data["conditions"][condition]["late_context_nmse"]


def history_values(records, tail, elapsed, condition):
    return {s: value(records[(s, "curriculum", tail, elapsed)], condition) - value(records[(s, "shuffled", tail, elapsed)], condition)
            for s in SEEDS if (s, "curriculum", tail, elapsed) in records and (s, "shuffled", tail, elapsed) in records}


def summarize(directory, allow_partial=False, source_directory=None):
    records, sources, missing = load_records(directory, allow_partial)
    baseline_checks = compare_baselines(records, sources, source_directory)
    conditions = sorted(next(iter(records.values()))["conditions"])
    trajectories, csv_rows = [], []
    for elapsed in TIMES:
        trajectory = {"tail_completed_steps": elapsed, "tails": {}, "interactions": {}}
        for tail in TAILS:
            tail_summary = {}
            for condition in conditions:
                gaps = history_values(records, tail, elapsed, condition)
                initial = history_values(records, tail, 0, condition)
                gap_changes = {s: gaps[s] - initial[s] for s in gaps.keys() & initial.keys()}
                arm_means, arm_changes = {}, {}
                for history in HISTORIES:
                    current = {s: value(records[(s, history, tail, elapsed)], condition) for s in SEEDS if (s, history, tail, elapsed) in records}
                    baseline = {s: value(records[(s, history, tail, 0)], condition) for s in SEEDS if (s, history, tail, 0) in records}
                    arm_means[history] = estimate(current)
                    arm_changes[history] = estimate({s: current[s] - baseline[s] for s in current.keys() & baseline.keys()})
                tail_summary[condition] = {"history_gap": estimate(gaps), "history_gap_change_from_baseline": estimate(gap_changes),
                                           "arm_means": arm_means, "arm_changes_from_baseline": arm_changes}
                for seed in SEEDS:
                    row = {"tail_completed_steps": elapsed, "tail": tail, "condition": condition, "seed": seed,
                           "curriculum_nmse": "", "shuffled_nmse": "", "curriculum_minus_shuffled": gaps.get(seed, ""),
                           "gap_change_from_baseline": gap_changes.get(seed, ""), "high_minus_mixed_history_gap": ""}
                    for history in HISTORIES:
                        data = records.get((seed, history, tail, elapsed))
                        if data:
                            row[f"{history}_nmse"] = value(data, condition)
                    csv_rows.append(row)
            trajectory["tails"][tail] = tail_summary
        for condition in conditions:
            high = history_values(records, "high", elapsed, condition)
            mixed = history_values(records, "mixed", elapsed, condition)
            interactions = {s: high[s] - mixed[s] for s in high.keys() & mixed.keys()}
            initial_high = history_values(records, "high", 0, condition)
            initial_mixed = history_values(records, "mixed", 0, condition)
            changes = {s: interactions[s] - (initial_high[s] - initial_mixed[s])
                       for s in interactions.keys() & initial_high.keys() & initial_mixed.keys()}
            trajectory["interactions"][condition] = {"history_gap_high_minus_mixed": estimate(interactions),
                                                       "interaction_change_from_baseline": estimate(changes)}
            for row in csv_rows:
                if row["tail_completed_steps"] == elapsed and row["condition"] == condition:
                    row["high_minus_mixed_history_gap"] = interactions.get(row["seed"], "")
        trajectories.append(trajectory)
    final = trajectories[-1]["interactions"]["gaussian_sd2"]["history_gap_high_minus_mixed"]
    if missing or final["n"] != len(SEEDS):
        interpretation = "Partial exploratory diagnostic: study is incomplete; no complete-study inference."
    elif final["ci95"][1] < 0:
        interpretation = "Exploratory evidence of a more negative original-history gap under the high tail than the mixed tail at the fixed final endpoint."
    elif final["ci95"][0] > 0:
        interpretation = "Exploratory evidence of a more positive original-history gap under the high tail than the mixed tail at the fixed final endpoint."
    else:
        interpretation = "Inconclusive exploratory interaction: the 95% interval includes zero; this does not establish equivalence."
    original_values = {s: sources[(s, "curriculum")]["original_primary"]["value"] - sources[(s, "shuffled")]["original_primary"]["value"]
                       for s in SEEDS if (s, "curriculum") in sources and (s, "shuffled") in sources}
    source_summary = None
    source_summary_path = Path(source_directory) / "summary.json" if source_directory else None
    if source_summary_path and source_summary_path.exists():
        source_summary = json.loads(source_summary_path.read_text())
    summary = {
        "schema_version": 1, "study": "exploratory_continuation_diagnostic", "complete": not missing,
        "allow_partial": allow_partial, "expected_seeds": list(SEEDS), "expected_records": 100,
        "observed_records": len(records), "missing_records": [list(k) for k in missing],
        "original_config": dict(CANONICAL), "continuation_steps": TIMES[-1], "fixed_eval_steps": list(TIMES),
        "main_contrast": {"name": "final_gaussian_sd2_history_interaction_k20_40", "formula": "(C_high - S_high) - (C_mixed - S_mixed)",
                          "tail_completed_steps": TIMES[-1], **final, "interpretation": interpretation},
        "trajectories": trajectories,
        "original_h2": {"analysis_unchanged": True, "original_study_conclusion": "inconclusive",
                        "source_original_primary_gap": estimate(original_values), "original_summary_snapshot": source_summary},
        "baseline_consistency": {"primary_tolerance": {"rtol": BASELINE_RTOL, "atol": BASELINE_ATOL},
                                 "curve_tolerance": {"rtol": BASELINE_CURVE_RTOL, "atol": BASELINE_CURVE_ATOL,
                                                     "scope": "Step-0 original-versus-new and counterpart-tail curve comparisons only; aggregation and primary tolerances are unchanged.",
                                                     "basis": "Threshold set using only step-0 cross-A100 form-factor reevaluations before inspecting any continuation outcomes."},
                                 "checks": baseline_checks,
                                 "high_mixed_baselines_compared": sum(all((s, h, t, 0) in records for t in TAILS) for s in SEEDS for h in HISTORIES)},
        "sign_direction": "Negative C-S favors curriculum history. Negative interaction means C-S is lower under high than mixed; this alone does not establish curriculum superiority within either tail. Negative change means the gap moves toward curriculum after continuation.",
        "inference": "Paired Student t 95% intervals across independent original training seeds; tails and histories are paired within each seed. Trajectory and secondary intervals are descriptive and unadjusted for multiplicity.",
        "secondary_endpoints": ["gaussian_sd0_nmse_k20_40", "gaussian_sd1_nmse_k20_40"],
        "limitations": "Both tails run 10000 additional optimizer steps. High SD1 and mixed evenly sampled SD0..1 intentionally have different noise budgets. This diagnostic cannot claim equal-budget tail superiority, automatically confirm original H2, establish equivalence, or establish publication readiness. Five seeds give limited precision; t intervals assume roughly normal seed differences. No favorable tail, timepoint, or endpoint is selected.",
        "sources": [{"seed": s, "history": h, "provenance": source} for (s, h), source in sorted(sources.items())],
        "summary_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return summary, csv_rows


def interval_text(estimate_):
    if estimate_["n"] == 0:
        return "n=0; unavailable"
    interval = estimate_["ci95"]
    ci = "unavailable (one seed)" if interval is None else f"[{interval[0]:.7f}, {interval[1]:.7f}]"
    return f"n={estimate_['n']}; mean={estimate_['mean']:.7f}; 95% CI={ci}"


def write_outputs(directory, summary, rows):
    directory = Path(directory)
    main = summary["main_contrast"]
    lines = ["Exploratory continuation diagnostic; original H2 remains inconclusive.",
             f"Complete study: {summary['complete']}; records {summary['observed_records']}/100",
             f"Fixed main contrast at 10000 continuation steps: {main['formula']}", interval_text(main), main["interpretation"],
             summary["sign_direction"], "", "All predeclared times; no timepoint or tail selected:"]
    for trajectory in summary["trajectories"]:
        lines.append(f"Continuation step {trajectory['tail_completed_steps']}:")
        for condition in REQUIRED_CONDITIONS:
            lines.append(f"  {condition} NMSE k20..40 (clean/SD1 are descriptive secondary):")
            for tail in TAILS:
                arm = trajectory["tails"][tail][condition]
                lines.append(f"    {tail} C-S: {interval_text(arm['history_gap'])}; change from baseline: {interval_text(arm['history_gap_change_from_baseline'])}")
            lines.append("    Interaction: " + interval_text(trajectory["interactions"][condition]["history_gap_high_minus_mixed"]))
    lines += ["", "Original source primary C-S: " + interval_text(summary["original_h2"]["source_original_primary_gap"]),
              f"Baseline comparisons: {len(summary['baseline_consistency']['checks'])}; high/mixed history pairs: {summary['baseline_consistency']['high_mixed_baselines_compared']}",
              f"Baseline primary tolerance: rtol={BASELINE_RTOL:g}, atol={BASELINE_ATOL:g}; baseline-curve-only tolerance: rtol={BASELINE_CURVE_RTOL:g}, atol={BASELINE_CURVE_ATOL:g}.",
              summary["baseline_consistency"]["curve_tolerance"]["basis"],
              "Per-record baseline primary deltas and maximum absolute curve deltas:"]
    for check in summary["baseline_consistency"]["checks"]:
        lines.append(f"  seed={check['seed']} history={check['history']} tail={check['tail']}: primary_delta_original={check['primary_delta_vs_original']:.9g}; primary_delta_counterpart={check['primary_delta_vs_counterpart_tail']}; max_abs_curve_delta={check['max_abs_curve_delta']} (original={check['max_abs_curve_delta_vs_original']}, counterpart={check['max_abs_curve_delta_vs_counterpart_tail']})")
    lines += [
              f"Missing records: {summary['missing_records']}", summary["inference"], summary["limitations"]]
    text = "\n".join(lines) + "\n"
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (directory / "summary.txt").write_text(text)
    with (directory / "trajectory_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Explicit incomplete-study report; missing observations remain visible")
    parser.add_argument("--source-directory", type=Path, help="Optional original results/full directory for baseline curve comparison")
    args = parser.parse_args()
    source_directory = args.source_directory or args.directory.parent / "full"
    summary, rows = summarize(args.directory, args.allow_partial, source_directory)
    print(write_outputs(args.directory, summary, rows))


if __name__ == "__main__":
    main()
