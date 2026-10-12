#!/usr/bin/env python3
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import torch
import robustness_experiment as exp


def main():
    torch.set_num_threads(2);exp.verify_plan()
    subprocess.run([sys.executable,'-m','unittest','-v','test_robustness'],check=True)
    probes=[]
    for task in exp.planned_tasks():
        cfg,study=exp.task_config(task);probes.append(dict(index=task['index'],cohort=study.cohort,variant=study.variant,shape=[256,101,50],steps=cfg.steps))
        for bid in (0,999,1000,12345,24999,25000,cfg.steps-1):
            x,y,active,eligible=exp.batch(cfg,study,bid)
            if x.shape!=(256,101,50) or y.shape!=(256,101) or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Batch invalid')
            if x[:,:,active:].count_nonzero()!=0:raise ValueError('Dimension mask invalid')
            if bid>=25000 and (active,eligible)!=(50,101):raise ValueError('Common terminal phase invalid')
        if study.arm=='curriculum':
            shuffled=dataclasses.replace(study,arm='shuffled')
            for bid in (0,12345,25000):
                a,b=exp.batch(cfg,study,bid),exp.batch(cfg,shuffled,bid)
                if any(not torch.equal(u,v) if isinstance(u,torch.Tensor) else u!=v for u,v in zip(a,b)):raise ValueError('Ordered/shuffled batch mismatch')
    exp.base.atomic_json(Path('submissions/robustness-preflight.json'),dict(all_scientific_tests_passed=True,all_44_production_batch_probes_passed=True,tasks=probes,runner_sha256=exp.sha(exp.__file__),protocol_sha256=exp.sha(exp.PROTOCOL),tasks_sha256=exp.sha('ROBUSTNESS_TASKS.json')))
    print('All scientific tests and production task probes passed',flush=True)

if __name__=='__main__':main()
