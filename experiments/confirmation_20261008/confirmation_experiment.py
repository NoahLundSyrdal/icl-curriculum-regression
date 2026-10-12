#!/usr/bin/env python3
"""Provenance adapter around the unchanged, verified curriculum trainer."""
import argparse
import dataclasses
import json
from pathlib import Path
import signal
import torch
import confirmation_plan as plan
import robustness_experiment as training

ROOT=Path('results')


def identity(cfg,study):
    ident=dict(kind=plan.KIND,config=dataclasses.asdict(cfg),study=training.study_dict(study),
        adapter_sha256=plan.digest(__file__),trainer_sha256=plan.digest(training.__file__),
        base_sha256=plan.digest(training.base.__file__),protocol_sha256=plan.digest(plan.PROTOCOL),
        tasks_sha256=plan.digest(plan.PLAN_FILE),plan_source_sha256=plan.digest(plan.__file__),
        freeze_sha256=plan.digest(plan.FREEZE))
    if study.cohort=='confirmation':
        plan.load_selection()
        ident.update(selection_sha256=plan.digest(plan.SELECTION),resolved_tasks_sha256=plan.digest(plan.RESOLVED))
    return ident


# Only the provenance callback changes. Batch generation, causal forward pass,
# objective, update loop, checkpoint/RNG handling and evaluation remain intact.
training.identity=identity


def task_config(task, *, smoke=False):
    cohort=task['cohort']
    cfg=training.base.preset_config('full',mode='clean',seed=task['seed'],dims=50,points=101,
        steps=task['steps'],batch_size=256,microbatch=128,learning_rate=task['learning_rate'],
        eval_seed=plan.DEV_BANK if cohort=='development' else plan.CONFIRM_BANK,
        eval_episodes=64 if smoke else 4096,eval_microbatch=64,precision='bf16',bayes_reference=False)
    study=training.Study(arm=task['arm'],cohort='gpu_smoke' if smoke else cohort,
        variant=task['variant'],component=task['component'],target_variance='constant',
        objective=task['objective'],evaluation_steps=plan.TIMES if cfg.steps==50000 else plan.LONG_TIMES)
    training.validate(cfg,study)
    return cfg,study


def task_directory(task, root=ROOT):
    return Path(root)/task['cohort']/f"seed_{task['seed']}"/task['variant']


def tasks(cohort):
    return plan.development_tasks() if cohort=='development' else plan.load_selection()[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',choices=['development','confirmation'],required=True)
    p.add_argument('--task-index',type=int,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();plan.verify_sources();torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('Verified CUDA BF16 environment required')
    entries=tasks(args.cohort)
    if not 0<=args.task_index<len(entries): raise ValueError('Task index outside fixed cohort')
    task=entries[args.task_index];cfg,study=task_config(task);directory=task_directory(task)
    stopping=[False]
    def stop(signum,frame):
        stopping[0]=True;print(f'Signal {signum}: saving after current update',flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1): signal.signal(sig,stop)
    completed=training.train(cfg,study,directory,torch.device('cuda'),resume=args.resume,stop_requested=lambda:stopping[0])
    if completed!=cfg.steps: return 3
    saved=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=training.base.RegressionTransformer(cfg).to('cuda');model.load_state_dict(saved['model'])
    holdout=plan.DEV_HOLDOUT if args.cohort=='development' else plan.CONFIRM_HOLDOUT
    training.evaluate(dataclasses.replace(cfg,eval_seed=holdout),study,saved['identity'],model,
        completed,saved['training_seconds'],directory,torch.device('cuda'),final_holdout=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
