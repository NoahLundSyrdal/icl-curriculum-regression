#!/usr/bin/env python3
"""Check archived run provenance, recompute raw outcomes and print the main results."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'data/confirmation_20261008.zip'
    verification_path = output / 'repository-verification.json'
    analysis_path = output / 'review_recomputed.json'
    for command in [
        [sys.executable, str(ROOT / 'tools/verify_archive.py'), '--output', str(verification_path)],
        [sys.executable, str(ROOT / 'analysis/recompute_review.py'), str(archive), str(analysis_path)],
    ]:
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode:
            sys.stderr.write(process.stdout + process.stderr)
            raise SystemExit(process.returncode)
    verification = json.loads(verification_path.read_text())
    review = json.loads(analysis_path.read_text())
    with zipfile.ZipFile(archive) as source:
        summary = json.loads(source.read('results/confirmation/summary.json'))
    print('PASS: frozen source, archived identities and raw-metric recomputation')
    print(f"Models: {verification['models']['development']} development + {verification['models']['confirmation']} confirmation; evaluations: 1,140")
    print('\nFinal clean-target NMSE at 125k updates, contexts k=50..100; 20 paired seeds')
    print(f"{'Condition':<16} {'Curriculum mean':>17} {'Shuffled mean':>17} {'C better':>10} {'C worse':>10}")
    for condition, label in [('gaussian_sd0', 'Clean'), ('gaussian_sd1', 'Noise SD1'), ('gaussian_sd2', 'Noise SD2')]:
        counts = review['noise'][condition]['final']
        c = summary['arm_means']['curriculum'][condition]['final']
        s = summary['arm_means']['shuffled'][condition]['final']
        print(f"{label:<16} {c:>17.8f} {s:>17.8f} {counts['curriculum_better']:>7}/20 {counts['curriculum_worse']:>7}/20")
    ratio = review['ratios']['final_excluded_None']['geometric_mean_paired_ratio']
    print(f'Final clean paired geometric error ratio, curriculum/shuffled: {ratio:.8f} (exploratory)')
    spec = importlib.util.spec_from_file_location('frozen_plan', ROOT / 'experiments/confirmation_20261008/confirmation_plan.py')
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    records = {(r['seed'], r['arm']): r for r in summary['individual_runs']}
    gains = [records[s, 'shuffled']['outcomes']['gaussian_sd0']['auc'] - records[s, 'curriculum']['outcomes']['gaussian_sd0']['auc'] for s in summary['seeds']]
    test = plan.exact_sign_test([-gain for gain in gains])
    assert test == summary['ordering_vs_matched_shuffled']['sign_test']
    print('\nPrespecified clean AUC, averaged over updates 25k..125k (different endpoint)')
    print(f"Curriculum wins: {test['wins']}/20; exact sign-test p={test['two_sided_p']:.8f}")
    print(f'Mean paired AUC reduction: {statistics.mean(gains):.8f}; median: {statistics.median(gains):.8f}')
    without = [g for seed, g in zip(summary['seeds'], gains) if seed != 1017]
    print(f'Mean reduction excluding seed 1017: {statistics.mean(without):.8f} (post hoc; below frozen 0.02 minimum)')
    print('\nFull-difficulty final outcomes: limited optimizer grid')
    for key, label in [('learned', 'Learned'), ('did_not_learn', 'Did not learn')]:
        group = review['hard_groups'][key]
        print(f"{label}: {group['n']}/20; mean={group['final_mean']:.8f}; median={group['final_median']:.8f}")
    print('\nNoise comparisons and new ratio/signed-rank analyses are descriptive or exploratory.')
    print('Checkpoint binaries remain on IDUN; this does not repeat the original binary audit.')
    print(f'Fresh reports: {verification_path}\n               {analysis_path}')


if __name__ == '__main__':
    main()
