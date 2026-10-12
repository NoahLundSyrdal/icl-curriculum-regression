#!/usr/bin/env python3
"""Stage unchanged source into a clean directory without launching any jobs."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/confirmation_20261008'


def stage(destination, mode):
    destination = Path(destination).resolve()
    if destination == ROOT or destination.is_relative_to(ROOT / 'experiments') or destination.is_relative_to(ROOT / 'historical'):
        raise ValueError('Stage outside the archived source directories')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Destination is nonempty; preserve the existing run')
    destination.mkdir(parents=True, exist_ok=True)
    for path in SOURCE.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    if mode == 'confirmation':
        (destination / 'submissions').mkdir(exist_ok=True)
        for name in ['freeze.json', 'selection.json', 'confirmation_resolved_tasks.json']:
            shutil.copy2(SOURCE / 'submissions' / name, destination / 'submissions' / name)
    return {'destination': str(destination), 'mode': mode, 'jobs_submitted': False,
            'selection': 'recorded original selection' if mode == 'confirmation' else 'fresh development and selection required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['confirmation', 'full'], required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.destination, args.mode), indent=2))


if __name__ == '__main__':
    main()
