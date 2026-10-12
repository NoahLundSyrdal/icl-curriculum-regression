#!/usr/bin/env python3
"""Descriptive report of all existing checkpoints; new clean controls pending."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path("results/followup_20261006")
TAGS = ("original_curriculum", "original_shuffled", "curriculum_high",
        "shuffled_high", "curriculum_mixed", "shuffled_mixed")


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    reports = {}
    identity = None
    for seed in range(5):
        for tag in TAGS:
            report = json.loads((ROOT/"diagnostics"/f"seed_{seed}"/f"{tag}.json").read_text())
            if not finite_tree(report) or report["identity"]["seed"] != seed or report["identity"]["tag"] != tag:
                raise ValueError("Invalid diagnostic record")
            if identity is None:
                identity = report["identity"]["cache_identity"]
            if report["identity"]["cache_identity"] != identity:
                raise ValueError("Diagnostic banks/protocol/code differ")
            if set(report["conditions"]) != {"gaussian_sd0", "gaussian_sd0.5", "gaussian_sd1", "gaussian_sd2"}:
                raise ValueError("Missing diagnostic condition")
            reports[seed, tag] = report
    aggregates = {}
    for tag in TAGS:
        aggregates[tag] = {}
        for key in reports[0, tag]["conditions"]:
            values = [reports[seed, tag]["conditions"][key] for seed in range(5)]
            aggregate = {
                "task_nmse_mean": statistics.mean(v["model_task_nmse"]["mean_k20_40"] for v in values),
                "calibration_selected_penalties": [v["calibration_selected_penalty"] for v in values],
                "references": {name: {
                    "prediction_disagreement_mean": statistics.mean(v["references"][name]["prediction_disagreement_nmse"]["mean_k20_40"] for v in values),
                    "reference_task_nmse_mean": statistics.mean(v["references"][name]["reference_task_nmse"]["mean_k20_40"] for v in values)}
                    for name in values[0]["references"]},
                "invariance": {name: {
                    "prediction_change_mean": statistics.mean(v["invariance"][name]["prediction_change_nmse"]["mean_k20_40"] for v in values),
                    "variant_task_nmse_mean": statistics.mean(v["invariance"][name]["variant_task_nmse"]["mean_k20_40"] for v in values)}
                    for name in values[0]["invariance"]}}
            aggregates[tag][key] = aggregate
    result = {"kind": "existing_checkpoint_diagnostics_descriptive", "records": 30,
              "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "cache_identity": identity, "aggregates": aggregates,
              "reporting_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "clean_only_control_status": "running; primary H1 comparison unavailable"}
    (ROOT/"existing_diagnostics_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["Existing checkpoint diagnostics: all 30 models complete", "",
             "Exploratory means across five training seeds on fresh held-out bank 921733.",
             "Fit penalties selected to match predictions on separate calibration bank 921731.",
             "NMSE and prediction discrepancies divide squared error by dimension 20.",
             "Late context means use k=20..40; symmetry checks use 512 shared episodes.",
             "", "Arm | SD | model NMSE | fitted lambdas by seed | fitted-ridge prediction discrepancy | rotation change | permutation change"]
    for tag in TAGS:
        for sigma in (0, 0.5, 1, 2):
            a = aggregates[tag][f"gaussian_sd{sigma:g}"]
            penalty = ",".join(f"{x:.5g}" for x in a["calibration_selected_penalties"])
            discrepancy = a["references"]["calibration_fitted_ridge"]["prediction_disagreement_mean"]
            rotation = a["invariance"]["orthogonal_rotation"]["prediction_change_mean"]
            permutation = a["invariance"]["context_permutation"]["prediction_change_mean"]
            lines.append(f"{tag} | {sigma:g} | {a['task_nmse_mean']:.7g} | {penalty} | {discrepancy:.7g} | {rotation:.7g} | {permutation:.7g}")
    lines += ["", "Reference scores on the common held-out bank:"]
    for sigma in (0, 0.5, 1, 2):
        a = aggregates[TAGS[0]][f"gaussian_sd{sigma:g}"]["references"]
        lines.append(f"SD={sigma:g}: known-SD oracle={a['known_sigma_oracle']['reference_task_nmse_mean']:.8g}, context-only empirical Bayes={a['empirical_bayes_context_only']['reference_task_nmse_mean']:.8g}")
    lines += ["", "Interpretation limits:",
              "Prediction agreement with fitted ridge is behavioral evidence, not proof of an internal algorithm.",
              "Symmetry changes quantify departures from literal ridge invariance; fitted penalties can also vary with test condition.",
              "Known-SD oracle uses privileged noise information; empirical Bayes assumes the true Gaussian task prior.",
              "A gap to a privileged oracle is not an achievable curriculum gain estimate.",
              "The new clean-only models are still training. No H1 conclusion is available yet.",
              "Completed H2 conclusions remain unchanged; these exploratory diagnostics do not establish a curriculum benefit."]
    (ROOT/"EXISTING_DIAGNOSTICS.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
