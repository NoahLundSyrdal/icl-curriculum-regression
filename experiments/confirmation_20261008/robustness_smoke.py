#!/usr/bin/env python3
import dataclasses
import json
from pathlib import Path
import unittest
import torch
import robustness_experiment as exp
from test_robustness import equal


def main():
    exp.verify_plan();torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Verified CUDA environment required')
    tasks=exp.planned_tasks();indices=(0,1,2,3,6,9,12,32,34);root=Path('results/robustness_smoke_20261007');records=[];starts={}
    for index in indices:
        task=tasks[index];cfg,study=exp.task_config(task)
        cfg=dataclasses.replace(cfg,eval_episodes=64);directory=root/f'task_{index}_split';whole=root/f'task_{index}_whole'
        if exp.train(cfg,study,directory,torch.device('cuda'),max_steps_this_run=2)!=2:raise ValueError('Smoke step mismatch')
        if exp.train(cfg,study,whole,torch.device('cuda'),max_steps_this_run=4)!=4:raise ValueError('Whole step mismatch')
        if exp.train(cfg,study,directory,torch.device('cuda'),resume=True,max_steps_this_run=2)!=4:raise ValueError('Resume step mismatch')
        a=torch.load(whole/'checkpoint.pt',map_location='cpu',weights_only=True);b=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
        for key in ('model','optimizer','torch_rng','cuda_rng','python_rng'):equal(unittest.TestCase(),a[key],b[key])
        starts.setdefault(cfg.precision,[]).append(json.loads((directory/'evaluations/step_000000/metrics.json').read_text()))
        records.append(dict(task_index=index,variant=task['variant'],precision=cfg.precision,objective=study.objective,initial_model_sha256=a['identity']['initial_model_sha256'],gpu_resume_bitwise_passed=True,finite_updates=True));del a,b
    if len({r['initial_model_sha256'] for r in records})!=1:raise ValueError('Initial parameter mismatch')
    for group in starts.values():
        for m in group[1:]:
            for c in group[0]['conditions']:
                if any(abs(a-b)>2e-5 for a,b in zip(m['conditions'][c]['mean_nmse_by_k'],group[0]['conditions'][c]['mean_nmse_by_k'])):raise ValueError('Initial evaluation mismatch')
    exp.base.atomic_json(Path('submissions/robustness-gpu-smoke.json'),dict(production_training_shapes_passed=True,bf16_and_fp32_passed=True,bitwise_restart_passed=True,paired_initialization_passed=True,records=records,gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),runner_sha256=exp.sha(exp.__file__),protocol_sha256=exp.sha(exp.PROTOCOL)))
    print('Canonical GPU BF16/FP32, treatment/objective and bitwise restart checks passed',flush=True)

if __name__=='__main__':main()
