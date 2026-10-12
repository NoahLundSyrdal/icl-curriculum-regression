#!/usr/bin/env python3
"""CPU and canonical GPU preflight for the frozen 96-model study."""
import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import unittest
import torch
import confirmation_plan as plan
import confirmation_experiment as exp
from test_robustness import equal


def fresh_seed_audit():
    previous=Path.cwd().parent/'results';seen_seeds=set();seen_banks=set();count=0
    for pattern in ('manifest.json','config.json','metrics.json'):
        for path in previous.rglob(pattern):
            value=json.loads(path.read_text())
            if not isinstance(value,dict):continue
            config=value.get('identity',value).get('config',value)
            if isinstance(config,dict):
                if isinstance(config.get('seed'),int):seen_seeds.add(config['seed'])
                if isinstance(config.get('eval_seed'),int):seen_banks.add(config['eval_seed'])
            if isinstance(value.get('eval_seed'),int):seen_banks.add(value['eval_seed'])
            count+=1
    if set(plan.DEV_SEEDS+plan.CONFIRM_SEEDS)&seen_seeds: raise ValueError('Training seed already used')
    if {plan.DEV_BANK,plan.DEV_HOLDOUT,plan.CONFIRM_BANK,plan.CONFIRM_HOLDOUT}&seen_banks: raise ValueError('Evaluation bank already used')
    return dict(prior_records=count,prior_training_seeds=sorted(seen_seeds),prior_evaluation_banks=sorted(seen_banks),fresh_training_seeds=True,fresh_evaluation_banks=True)


def cpu_preflight():
    plan.verify_sources();torch.set_num_threads(2)
    subprocess.run([sys.executable,'-m','unittest','-v','test_robustness','test_confirmation'],check=True)
    fresh=fresh_seed_audit();probes=[]
    tasks=plan.development_tasks()+[
        t|dict(setting='lr_low',learning_rate=.0001,objective='raw',variant='lr_low_'+t['arm'])
        for t in plan.confirmation_templates()]
    for task in tasks:
        cfg,study=exp.task_config(task)
        ids=exp.training.batch_order(cfg,study)
        if not torch.equal(ids[:25000].sort().values,torch.arange(25000)) or not torch.equal(ids[25000:],torch.arange(25000,cfg.steps)): raise ValueError('Canonical batch multiset/ending differs')
        for bid in (0,999,1000,12345,24999,25000,cfg.steps-1):
            x,y,active,eligible=exp.training.batch(cfg,study,bid)
            if x.shape!=(256,101,50) or y.shape!=(256,101) or not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError('Production batch invalid')
            if x[:,:,active:].count_nonzero()!=0: raise ValueError('Mask invalid')
            if abs(active*exp.training.treatment(cfg,study,bid)[2]**2-50)>1e-10:raise ValueError('Target variance scale invalid')
            if bid>=25000 and (active,eligible)!=(50,101):raise ValueError('Terminal distribution differs')
        probes.append(dict(cohort=task['cohort'],index=task['index'],seed=task['seed'],arm=task['arm'],steps=cfg.steps))
    report=dict(kind=plan.KIND,scientific_tests_passed=True,all_96_task_batch_probes_passed=True,
        fresh_seed_audit=fresh,production_probes=probes,freeze_sha256=plan.digest(plan.FREEZE),
        runtime={'torch':str(torch.__version__),'transformers':exp.training.base.transformers.__version__,'python':sys.version})
    plan.atomic_json('submissions/cpu-preflight.json',report)
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=open('submissions/environment.txt','w'),check=True)
    print('All scientific checks and 96 production task probes passed.',flush=True)


def independent_query_prediction(model,x,y,k):
    labels=torch.zeros_like(x[:,:k]);labels[:,:,0]=y[:,:k]
    tokens=torch.stack((x[:,:k],labels),dim=2).flatten(1,2)
    tokens=torch.cat((tokens,x[:,k:k+1]),dim=1)
    hidden=model.backbone(inputs_embeds=model.input_projection(tokens)).last_hidden_state
    return model.output_projection(hidden[:,-1]).squeeze(-1)


def gpu_smoke():
    plan.verify_sources();torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 required')
    root=Path('results/gpu_smoke');records=[];initial=set();baselines=[]
    for task in plan.development_tasks()[:12]:
        cfg,study=exp.task_config(task,smoke=True);whole=root/f"task_{task['index']}_whole";split=root/f"task_{task['index']}_split"
        if exp.training.train(cfg,study,split,torch.device('cuda'),max_steps_this_run=2)!=2:raise ValueError('Smoke progress differs')
        if exp.training.train(cfg,study,whole,torch.device('cuda'),max_steps_this_run=4)!=4:raise ValueError('Whole progress differs')
        if exp.training.train(cfg,study,split,torch.device('cuda'),resume=True,max_steps_this_run=2)!=4:raise ValueError('Resume progress differs')
        a=torch.load(whole/'checkpoint.pt',map_location='cpu',weights_only=True);b=torch.load(split/'checkpoint.pt',map_location='cpu',weights_only=True)
        for key in ('model','optimizer','torch_rng','cuda_rng','python_rng'):equal(unittest.TestCase(),a[key],b[key])
        initial.add(a['identity']['initial_model_sha256']);baselines.append(json.loads((whole/'evaluations/step_000000/metrics.json').read_text())['conditions'])
        model=exp.training.base.RegressionTransformer(cfg).to('cuda').eval();model.load_state_dict(a['model'])
        x,y,_=exp.training.base.EpisodeStream(plan.DEV_BANK).draw(8,101,50);x=x.to('cuda');y=y.to('cuda')
        with torch.no_grad(),exp.training.base.autocast_context(torch.device('cuda'),'bf16'):full=model(x,y)
        for k in (0,50,100):
            xm=x.clone();ym=y.clone();xm[:,k+1:]+=1000;ym[:,k:]+=10000
            with torch.no_grad(),exp.training.base.autocast_context(torch.device('cuda'),'bf16'):
                changed=model(xm,ym);query=independent_query_prediction(model,x,y,k)
            if not torch.equal(full[:,:k+1],changed[:,:k+1]): raise ValueError('Own/future targets affect prediction')
            # Cropped BF16 matmuls can round differently; report, never call bitwise.
            delta=float((full[:,k].float()-query.float()).abs().max())
            if not torch.allclose(full[:,k].float(),query.float(),atol=.01,rtol=.02): raise ValueError('Target-absent independent predictions disagree')
            records.append(dict(task_index=task['index'],arm=task['arm'],setting=task['setting'],k=k,
                query_target_absent_max_difference=delta,causal_mutation_bitwise=True,bitwise_training_resume=True))
        del a,b,model;torch.cuda.empty_cache()
    if len(initial)!=1:raise ValueError('Paired starting parameters differ')
    for other in baselines[1:]:
        for condition in baselines[0]:
            if any(abs(a-b)>2e-5 for a,b in zip(baselines[0][condition]['mean_nmse_by_k'],other[condition]['mean_nmse_by_k'])):raise ValueError('Initial predictions differ')
    report=dict(kind=plan.KIND,production_shapes_passed=True,all_4_optimizers_3_arms_passed=True,
        bitwise_restart_passed=True,paired_initialization_passed=True,causal_and_query_target_absent_passed=True,
        freeze_sha256=plan.digest(plan.FREEZE),records=records,gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__))
    plan.atomic_json('submissions/gpu-smoke.json',report)
    print('Production GPU updates, pairing, bitwise restart and causal/target-absent checks passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');args=p.parse_args()
    gpu_smoke() if args.gpu else cpu_preflight()
