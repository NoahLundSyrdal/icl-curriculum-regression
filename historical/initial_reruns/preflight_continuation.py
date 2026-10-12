#!/usr/bin/env python3
"""Verify all ten trusted original checkpoints before overnight continuations."""
import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path

import torch
import run_experiment as original

SOURCE_SHA256 = '4df516cd368733d00795d9a3cf16f4f114db06e156cad5a4cabe3a22ce0776b7'


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_sources(source):
    actual_source_hash = file_sha256(Path(original.__file__))
    if actual_source_hash != SOURCE_SHA256:
        raise ValueError('Original training source changed; do not submit.')
    records = []
    for seed in range(5):
        paired_states = None
        for mode in ('curriculum', 'shuffled'):
            directory = source / f'seed_{seed}' / mode
            metrics = json.loads((directory / 'metrics.json').read_text())
            cfg = original.preset_config('full', seed=seed, mode=mode, precision='bf16')
            expected = dataclasses.asdict(cfg)
            if metrics['config'] != expected:
                raise ValueError(f'Noncanonical original config: {directory}')
            if json.loads((directory / 'config.json').read_text()) != expected:
                raise ValueError(f'Metrics/config disagreement: {directory}')
            if metrics['completed_steps'] != cfg.steps:
                raise ValueError(f'Original evaluation incomplete: {directory}')
            if metrics['primary']['name'] != 'gaussian_sd2_nmse_k20_40':
                raise ValueError(f'Wrong original endpoint: {directory}')
            if metrics['runtime']['script_sha256'] != SOURCE_SHA256:
                raise ValueError(f'Original provenance hash mismatch: {directory}')
            path = directory / 'checkpoint.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved.get('schema_version') != original.SCHEMA_VERSION:
                raise ValueError(f'Wrong checkpoint schema: {path}')
            if saved['config'] != expected or saved['completed_steps'] != cfg.steps:
                raise ValueError(f'Incomplete or mismatched checkpoint: {path}')
            if not torch.equal(saved['schedule'], original.sigma_schedule(cfg)):
                raise ValueError(f'Original schedule mismatch: {path}')
            if not saved['model'] or not saved['optimizer']['state']:
                raise ValueError(f'Missing trained model/AdamW state: {path}')
            for state in saved['optimizer']['state'].values():
                step = state.get('step')
                if step is None or int(step) != cfg.steps:
                    raise ValueError(f'AdamW progress mismatch: {path}')
            for group in saved['optimizer']['param_groups']:
                if group['lr'] != cfg.learning_rate or group['weight_decay'] != cfg.weight_decay:
                    raise ValueError(f'Optimizer hyperparameter mismatch: {path}')
            states = {key: value.clone() for key, value in saved['episode_rng'].items()}
            if set(states) != {'tasks', 'inputs', 'corruption'}:
                raise ValueError(f'Episode stream names differ: {path}')
            if paired_states is not None:
                if any(not torch.equal(states[key], paired_states[key]) for key in states):
                    raise ValueError(f'Future episode RNG streams are unpaired at seed {seed}')
            else:
                paired_states = states
            records.append({
                'seed': seed, 'history': mode, 'checkpoint': str(path.resolve()),
                'checkpoint_sha256': file_sha256(path), 'completed_steps': cfg.steps,
                'original_primary': metrics['primary'], 'runtime': metrics['runtime'],
                'episode_rng_sha256': {key: hashlib.sha256(bytes(value.tolist())).hexdigest()
                                      for key, value in states.items()},
            })
            del saved
            gc.collect()
            print(f'Checked seed={seed} history={mode}: completed source, optimizer and paired streams.', flush=True)
    return {'source_script_sha256': actual_source_hash, 'original_models': records,
            'source_checkpoint_count': len(records), 'paired_seeds': list(range(5)),
            'checks_passed': True, 'original_result': 'inconclusive; unchanged'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/full'))
    parser.add_argument('--output', type=Path, default=Path('submissions/continuation-preflight.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = check_sources(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original.atomic_json(args.output, result)
    print(f'PASS: all ten source checkpoints verified. {args.output}', flush=True)


if __name__ == '__main__':
    main()
