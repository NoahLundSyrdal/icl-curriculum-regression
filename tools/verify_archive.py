#!/usr/bin/env python3
"""Verify archived source and run identities without importing Torch or training."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/confirmation_20261008'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    provenance = json.loads((ROOT / 'provenance/REPOSITORY_PROVENANCE.json').read_text())
    archive_path = ROOT / provenance['data_archive']['path']
    require(digest(archive_path.read_bytes()) == provenance['data_archive']['sha256'], 'Run archive checksum changed')
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'Invalid ZIP member checksum')
        freeze_bytes = archive.read('submissions/freeze.json')
        freeze = json.loads(freeze_bytes)
        require((EXPERIMENT / 'submissions/freeze.json').read_bytes() == freeze_bytes, 'Local source freeze differs')
        for name, expected in freeze['source_hashes'].items():
            require(digest(archive.read(name)) == expected, f'Archived frozen source changed: {name}')
            require(digest((EXPERIMENT / name).read_bytes()) == expected, f'Local frozen source changed: {name}')
        captures = json.loads((ROOT / 'provenance/CONFIRMATION_SOURCE_MANIFEST.json').read_text())
        for name, record in captures.items():
            require(digest((EXPERIMENT / name).read_bytes()) == record['sha256'], f'Captured auxiliary source changed: {name}')
        for name in ['selection.json', 'confirmation_resolved_tasks.json']:
            require((EXPERIMENT / 'submissions' / name).read_bytes() == archive.read('submissions/' + name), f'Recorded selection changed: {name}')
        spec = importlib.util.spec_from_file_location('frozen_confirmation_plan', EXPERIMENT / 'confirmation_plan.py')
        plan = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plan)
        previous = Path.cwd()
        try:
            os.chdir(EXPERIMENT)
            plan.verify_sources()
            selection, resolved = plan.load_selection()
        finally:
            os.chdir(previous)
        require(len(resolved) == 60, 'Incomplete resolved task list')
        source_hashes = freeze['source_hashes']
        wanted_hashes = {
            'adapter_sha256': source_hashes['confirmation_experiment.py'],
            'trainer_sha256': source_hashes['robustness_experiment.py'],
            'base_sha256': source_hashes['run_experiment.py'],
            'protocol_sha256': source_hashes['CONFIRMATION_PROTOCOL.txt'],
            'tasks_sha256': source_hashes['CONFIRMATION_TASKS.json'],
            'plan_source_sha256': source_hashes['confirmation_plan.py'],
            'freeze_sha256': digest(freeze_bytes),
        }
        tasks = {'development': plan.development_tasks(), 'confirmation': resolved}
        records = {}; initial = {}; counts = {'development': 0, 'confirmation': 0}; cohorts = {}
        for cohort, entries in tasks.items():
            cohort_models = 0
            for task in entries:
                base = f"results/{cohort}/seed_{task['seed']}/{task['variant']}"
                manifest = json.loads(archive.read(base + '/manifest.json'))
                require(manifest['kind'] == plan.KIND, 'Wrong study identity')
                require(all(manifest.get(k) == v for k, v in wanted_hashes.items()), 'Source hash missing or changed in model identity')
                cfg = manifest['config']; study = manifest['study']
                require(cfg['seed'] == task['seed'] and cfg['steps'] == task['steps'], 'Wrong seed or update budget')
                require(cfg['learning_rate'] == task['learning_rate'], 'Optimizer differs from resolved task')
                require((cfg['dims'], cfg['points'], cfg['batch_size'], cfg['microbatch']) == (50, 101, 256, 128), 'Wrong production shape')
                require((cfg['embedding'], cfg['layers'], cfg['heads'], cfg['precision']) == (256, 12, 8, 'bf16'), 'Wrong architecture or precision')
                require(study['arm'] == task['arm'] and study['objective'] == task['objective'] and study['target_variance'] == 'constant', 'Wrong treatment')
                require(study['cohort'] == cohort, 'Wrong training cohort')
                if cohort == 'confirmation':
                    require(manifest['selection_sha256'] == digest(archive.read('submissions/selection.json')), 'Wrong selection identity')
                    require(manifest['resolved_tasks_sha256'] == digest(archive.read('submissions/confirmation_resolved_tasks.json')), 'Wrong resolved tasks identity')
                initial.setdefault((cohort, task['seed']), set()).add(manifest['initial_model_sha256'])
                cohort_models += 1
                times = plan.TIMES if cohort == 'development' else plan.LONG_TIMES
                bank = plan.DEV_BANK if cohort == 'development' else plan.CONFIRM_BANK
                holdout_bank = plan.DEV_HOLDOUT if cohort == 'development' else plan.CONFIRM_HOLDOUT
                paths = [(f'{base}/evaluations/step_{t:06d}/metrics.json', t, 'primary', bank) for t in times]
                paths += [(f"{base}/final_holdout/step_{task['steps']:06d}/metrics.json", task['steps'], 'final_holdout', holdout_bank)]
                for name, step, split, expected_bank in paths:
                    m = json.loads(archive.read(name))
                    require(m['identity'] == manifest, 'Metric identity differs from model manifest')
                    require((m['completed_steps'], m['evaluation_split'], m['eval_seed'], m['eval_episodes']) == (step, split, expected_bank, 4096), 'Wrong evaluation progress/bank')
                    require((m['evaluation_dimension'], m['evaluation_positions'], m['k_start'], m['k_end']) == (50, 101, 50, 100), 'Wrong evaluation shape')
                    require(set(m['conditions']) == {'gaussian_sd0', 'gaussian_sd1', 'gaussian_sd2'}, 'Missing noise conditions')
                    for v in m['conditions'].values():
                        curve = v['mean_nmse_by_k']; value = v['late_context_nmse']
                        require(len(curve) == 101 and all(math.isfinite(x) and x >= 0 for x in curve), 'Invalid error curve')
                        require(math.isfinite(value) and math.isclose(statistics.mean(curve[50:]), value, abs_tol=2e-6, rel_tol=2e-6), 'Curve/scalar mismatch')
                    records[cohort, task['index'], step, split] = m
                    counts[cohort] += 1
            cohorts[cohort] = cohort_models
        require(counts == {'development': 360, 'confirmation': 780}, 'Wrong number of evaluation records')
        require(all(len(v) == 1 for v in initial.values()), 'Paired initializations differ')
        actual_metric_count = sum(n.endswith('/metrics.json') for n in archive.namelist())
        require(actual_metric_count == 1140, 'Unexpected additional or missing evaluation records')
        scores = {}
        for task in tasks['development']:
            times = [t for t in plan.TIMES if t >= 25000]
            values = [records['development', task['index'], t, 'primary']['conditions']['gaussian_sd0']['late_context_nmse'] for t in times]
            scores[task['seed'], task['setting'], task['arm']] = plan.trapezoidal_mean(times, values)
        require(plan.rank_development(scores) == selection['rankings'], 'Raw development results do not reproduce selection')
        summary = json.loads(archive.read('results/confirmation/summary.json'))
        require(summary['verification']['all_checks_passed'], 'Archived checkpoint audit did not pass')
        require(summary['verification']['source_hashes'] == source_hashes, 'Archived audit uses different source')
        # Hashes above check recorded bytes, not the absent checkpoint binaries.
        source_count = len(source_hashes)
    historical = json.loads((ROOT / 'provenance/HISTORICAL_SOURCE_MANIFEST.json').read_text())
    for name, entry in historical.items():
        require(digest((ROOT / 'historical/initial_reruns' / name).read_bytes()) == entry['sha256_at_packaging'], f'Historical snapshot changed: {name}')
    return {
        'status': 'passed', 'archive_sha256': provenance['data_archive']['sha256'],
        'frozen_source_files': source_count, 'captured_source_files': len(captures),
        'historical_source_files': len(historical), 'models': cohorts, 'evaluation_records': counts,
        'paired_initializations_verified': True, 'raw_development_selection_reproduced': True,
        'binary_checkpoint_audit_repeated': False,
        'note': 'The original checkpoint audit is archived; checkpoints/schedules are not shipped or re-audited here.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/repository-verification.json')
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
