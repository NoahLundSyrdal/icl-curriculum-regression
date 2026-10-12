#!/usr/bin/env python3
"""Submit the fixed diagnostic once, with correctness checks as a dependency."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


SOURCE_HASH = '4df516cd368733d00795d9a3cf16f4f114db06e156cad5a4cabe3a22ce0776b7'


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def submit(arguments):
    result = subprocess.run(['sbatch', '--parsable'] + arguments,
                            capture_output=True, text=True, check=True)
    token = result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'[0-9]+', token):
        raise RuntimeError(f'Unexpected sbatch response: {result.stdout!r}')
    return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, choices=[10000], default=10000,
                        help='This fixed overnight protocol uses10000 additional steps.')
    parser.add_argument('--parallel', type=int, default=20)
    parser.add_argument('--account', default='share-ie-idi')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.parallel <= 20:
        parser.error('--parallel must be1..20')
    directory = Path(__file__).resolve().parent
    if Path.cwd().resolve() != directory:
        parser.error('Run from the experiment directory on IDUN.')
    files = ['run_experiment.py', 'continue_experiment.py', 'summarize.py',
             'summarize_continuation.py', 'test_continuation.py',
             'test_summary_continuation.py', 'preflight_continuation.py',
             'continuation.slurm', 'continuation_preflight.slurm',
             'continuation_summary.slurm', 'OVERNIGHT_PROTOCOL.txt']
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}
    if hashes['run_experiment.py'] != SOURCE_HASH:
        raise ValueError('Original training script changed; do not submit.')
    for seed in range(5):
        for history in ('curriculum', 'shuffled'):
            source = directory / 'results/full' / f'seed_{seed}' / history
            for filename in ('checkpoint.pt', 'metrics.json', 'config.json'):
                if not (source / filename).is_file():
                    raise FileNotFoundError(source / filename)
    output = f'results/continuation_{args.steps}'
    if (directory / output).exists() and any((directory / output).iterdir()):
        raise FileExistsError(f'{output} already has files. Inspect existing runs; do not duplicate them.')
    (directory / 'logs').mkdir(exist_ok=True)
    (directory / 'submissions').mkdir(exist_ok=True)
    manifest = directory / 'submissions' / f'continuation_{args.steps}-submission.json'
    if manifest.exists():
        raise FileExistsError(f'Already submitted or partially submitted: {manifest}. Inspect saved job IDs.')
    exports = f'ALL,ICL_TAIL_STEPS={args.steps},ICL_TAIL_OUTPUT={output},ICL_TAIL_RESUME=0'
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'account': args.account, 'steps': args.steps, 'parallel_ceiling': args.parallel,
        'output': output, 'array': '0-19', 'exports': exports,
        'file_sha256': hashes, 'status': 'preparing', 'jobs': {},
        'original_primary_result': 'inconclusive; preserved',
    }
    if args.dry_run:
        print(json.dumps(record, indent=2))
        print('No jobs submitted.')
        return
    atomic_json(manifest, record)
    try:
        checks = submit(['--account=' + args.account, '--export=' + exports,
                         'continuation_preflight.slurm'])
        record['jobs']['preflight'] = checks
        record['status'] = 'preflight_submitted'
        atomic_json(manifest, record)
        array = submit(['--account=' + args.account, '--export=' + exports,
                        '--dependency=afterok:' + checks,
                        '--array=0-19%' + str(args.parallel), 'continuation.slurm'])
        record['jobs']['continuations'] = array
        record['status'] = 'array_submitted'
        atomic_json(manifest, record)
        summary = submit(['--account=' + args.account, '--export=' + exports,
                          '--dependency=afterok:' + array, 'continuation_summary.slurm'])
        record['jobs']['summary'] = summary
        record['status'] = 'submitted'
        atomic_json(manifest, record)
    except Exception:
        record['status'] = 'partial_submission; inspect saved IDs before recovery'
        atomic_json(manifest, record)
        raise
    print(json.dumps(record, indent=2))
    print(f'CPU checks:{checks}; continuation array:{array}; final summary:{summary}.')
    print('GPU jobs wait for all correctness/source checks to pass.')


if __name__ == '__main__':
    main()
