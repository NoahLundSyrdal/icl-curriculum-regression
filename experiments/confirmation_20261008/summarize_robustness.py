#!/usr/bin/env python3
"""Strict diagnostic cohort summary; no confirmatory p-value claims."""
import csv
import json
import math
from pathlib import Path
import statistics
import torch
import robustness_experiment as exp


def trapezoidal_mean(times,values):
    if len(times)!=len(values) or len(times)<2 or any(b<=a for a,b in zip(times,times[1:])) or any(not math.isfinite(x) for x in values):
        raise ValueError('Finite values and increasing times required')
    return sum((b-a)*(x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))/(times[-1]-times[0])


def paired(left,right):
    if len(left)!=2 or len(right)!=2:raise ValueError('Two complete paired diagnostic seeds required')
    ds=[a-b for a,b in zip(left,right)]
    if any(not math.isfinite(x) for x in ds):raise ValueError('Nonfinite paired differences')
    return dict(seeds=list(exp.SEEDS),seed_differences=ds,mean=statistics.mean(ds),wins=sum(v<0 for v in ds),interpretation='Descriptive two-seed comparison; no significance claim')


def check_metric(m,manifest,cfg,step,split):
    wanted=cfg.eval_seed if split=='primary' else exp.FINAL_HOLDOUT_SEED
    if m['identity']!=manifest or m['completed_steps']!=step or m['evaluation_dimension']!=50 or m['evaluation_positions']!=101 or m['eval_seed']!=wanted or m['eval_episodes']!=4096 or m['evaluation_split']!=split or m['k_start']!=50 or m['k_end']!=100:
        raise ValueError('Wrong identity, progress, shapes or bank')
    if set(m['conditions'])!={'gaussian_sd0','gaussian_sd1','gaussian_sd2'}:raise ValueError('Missing condition')
    if not math.isfinite(m['zero_predictor_late_nmse']):raise ValueError('Invalid baseline')
    for v in m['conditions'].values():
        if len(v['mean_nmse_by_k'])!=101 or any(not math.isfinite(x) or x<0 for x in v['mean_nmse_by_k']):raise ValueError('Invalid curve')
        if not math.isclose(statistics.mean(v['mean_nmse_by_k'][50:]),v['late_context_nmse'],abs_tol=2e-6,rel_tol=2e-6):raise ValueError('Curve endpoint mismatch')


def aggregate(data,heldout):
    conditions=('gaussian_sd0','gaussian_sd1','gaussian_sd2');scores={};endpoints={};holdouts={};crossings={}
    for task in exp.planned_tasks():
        key=task['cohort']+'/'+task['variant'];seed=task['seed'];cfg,study=exp.task_config(task)
        times=[t for t in study.evaluation_steps if t>=study.prefix_steps]
        for c in conditions:
            scores.setdefault(key,{}).setdefault(c,{})[seed]=trapezoidal_mean(times,[data[task['index'],t]['conditions'][c]['late_context_nmse'] for t in times])
            endpoints.setdefault(key,{}).setdefault(c,{})[seed]=data[task['index'],cfg.steps]['conditions'][c]['late_context_nmse']
            holdouts.setdefault(key,{}).setdefault(c,{})[seed]=heldout[task['index']]['conditions'][c]['late_context_nmse']
        crossings.setdefault(key,{})[seed]=next((t for t in study.evaluation_steps if data[task['index'],t]['conditions']['gaussian_sd0']['late_context_nmse']<=.25),None)
    def vector(key,c='gaussian_sd0'):return [scores[key][c][s] for s in exp.SEEDS]
    ranking={}
    for arm in exp.ARMS:
        entries=[dict(setting=name,mean_clean_auc=statistics.mean(vector('optimizer_screen/'+name+'_'+arm)),per_seed_clean_auc=vector('optimizer_screen/'+name+'_'+arm),fixed_order=i) for i,(name,*_) in enumerate(exp.OPTIMIZERS)]
        entries.sort(key=lambda r:(r['mean_clean_auc'],r['fixed_order']))
        ranking[arm]=dict(candidates=entries,recommended_setting_for_fresh_long_confirmation=entries[0]['setting'],validated_best_setting=False)
    contrasts={component:{c:paired(vector('components/'+component+'_ordered',c),vector('components/'+component+'_shuffled',c)) for c in conditions} for component in ('both','dimension','context')}
    b=vector('components/both_ordered');d=vector('components/dimension_ordered');c=vector('components/context_ordered');h=vector('components/hard')
    effects={'dimension_main_effect':[(bi-ci+di-hi)/2 for bi,ci,di,hi in zip(b,c,d,h)],'context_main_effect':[(bi-di+ci-hi)/2 for bi,di,ci,hi in zip(b,d,c,h)],'interaction':[bi-di-ci+hi for bi,di,ci,hi in zip(b,d,c,h)]}
    scale={}
    for arm,variant in [('curriculum','both_ordered'),('shuffled','both_shuffled'),('hard','hard')]:
        raw=[];constant=[]
        for seed in exp.SEEDS:
            left=next(t for t in exp.planned_tasks() if t['cohort']=='optimizer_screen' and t['seed']==seed and t['variant']=='baseline_'+arm)
            right=next(t for t in exp.planned_tasks() if t['cohort']=='components' and t['seed']==seed and t['variant']==variant)
            times=[25000,30000,40000,50000]
            raw.append(trapezoidal_mean(times,[data[left['index'],t]['conditions']['gaussian_sd0']['late_context_nmse'] for t in times]))
            constant.append(trapezoidal_mean(times,[data[right['index'],t]['conditions']['gaussian_sd0']['late_context_nmse'] for t in times]))
        scale[arm]=dict(common_budget_steps=50000,constant_minus_raw=paired(constant,raw),raw_per_seed=raw,constant_per_seed=constant)
    return dict(auc=scores,final=endpoints,final_holdout=holdouts,first_scheduled_clean_nmse_at_most_0_25=crossings,optimizer_candidate_ranking=ranking,component_ordering_contrasts=contrasts,constant_scale_factorial_clean_auc_effects=effects,constant_vs_raw_matched_50000_update_comparison=scale)


def main():
    torch.set_num_threads(1);exp.verify_plan();root=exp.ROOT
    if (root/'summary.json').exists():raise FileExistsError('Preserve prior summary')
    manifest_submission=json.loads(Path('submissions/robustness-20261007.json').read_text())
    if any(exp.sha(name)!=digest for name,digest in manifest_submission['source_hashes'].items()):raise ValueError('Pre-submission source hash mismatch')
    data={};heldout={};rows=[];initial={};baselines={};records=0;checkpoints=0
    for task in exp.planned_tasks():
        cfg,study=exp.task_config(task);directory=exp.task_directory(task);manifest=json.loads((directory/'manifest.json').read_text())
        if any(manifest.get(k)!=v for k,v in exp.identity(cfg,study).items()):raise ValueError('Treatment/source/config mismatch')
        initial.setdefault(cfg.seed,set()).add(manifest['initial_model_sha256']);ids=exp.batch_order(cfg,study)
        schedule=torch.load(directory/'schedule.pt',map_location='cpu',weights_only=True)
        expected=[exp.treatment(cfg,study,int(i)) for i in ids]
        if not torch.equal(schedule['batch_ids'],ids) or not torch.equal(schedule['active_dims'],torch.tensor([e[0] for e in expected])) or not torch.equal(schedule['eligible_positions'],torch.tensor([e[1] for e in expected])) or not torch.equal(schedule['label_multipliers'],torch.tensor([e[2] for e in expected],dtype=torch.float64)):raise ValueError('Saved treatment schedule differs')
        for t in study.evaluation_steps:
            m=json.loads((directory/'evaluations'/f'step_{t:06d}'/'metrics.json').read_text());check_metric(m,manifest,cfg,t,'primary');data[task['index'],t]=m;records+=1
            if t==0:baselines.setdefault((cfg.seed,cfg.precision),[]).append(m['conditions'])
            for condition,value in m['conditions'].items():rows.append(dict(task_index=task['index'],cohort=study.cohort,variant=study.variant,seed=cfg.seed,step=t,condition=condition,late_context_nmse=value['late_context_nmse'],training_seconds=m['training_seconds']))
            path=directory/f'checkpoint_step_{t:06d}.pt'
            if exp.sha(path)!=m['checkpoint_sha256']:raise ValueError('Frozen weights hash differs')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['identity']!=manifest or saved['completed_steps']!=t or not torch.equal(saved['batch_ids'],ids):raise ValueError('Checkpoint identity/progress/schedule differs')
            states=saved['optimizer']['state']
            if (t>0 and not states) or any(int(v['step'])!=t for v in states.values()):raise ValueError('Adam progress differs')
            del saved;checkpoints+=1
        m=json.loads((directory/'final_holdout'/f'step_{cfg.steps:06d}'/'metrics.json').read_text());check_metric(m,manifest,cfg,cfg.steps,'final_holdout')
        if m['checkpoint_sha256']!=data[task['index'],cfg.steps]['checkpoint_sha256']:raise ValueError('Holdout weights differ')
        heldout[task['index']]=m;records+=1
        with (directory/'train.csv').open(newline='') as f:logs=list(csv.DictReader(f))
        if not logs or int(logs[-1]['step'])!=cfg.steps or any(not math.isfinite(float(r[k])) for r in logs for k in ['train_mse','objective_loss','grad_norm']):raise ValueError('Incomplete/nonfinite training log')
    if any(len(v)!=1 for v in initial.values()):raise ValueError('Paired initialization mismatch')
    for values in baselines.values():
        for v in values[1:]:
            for c in values[0]:
                if any(abs(a-b)>2e-5 for a,b in zip(values[0][c]['mean_nmse_by_k'],v[c]['mean_nmse_by_k'])):raise ValueError('Paired initial predictions differ')
    if records!=482 or checkpoints!=438:raise ValueError('Incomplete cohort')
    result=dict(kind=exp.KIND,status='complete_diagnostic_cohort',seeds=list(exp.SEEDS),training_runs=44,evaluation_records=records,frozen_checkpoints_verified=checkpoints,source_hashes=manifest_submission['source_hashes'],interpretation='Two-seed diagnostic study. No confirmatory significance claim. Optimizer candidates require fresh longer confirmation; do not compare unequal budgets.')
    result.update(aggregate(data,heldout));exp.base.atomic_json(root/'summary.json',result)
    with (root/'learning_curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['All 44 planned diagnostic models completed and verified. Two paired training seeds, not a definitive publication test.']
    for arm,r in result['optimizer_candidate_ranking'].items():lines.append(f"Optimizer screen {arm}: {r}")
    for component,r in result['component_ordering_contrasts'].items():lines.append(f"Constant-scale {component} ordered-minus-shuffled clean AUC: {r['gaussian_sd0']}")
    lines += [f"Constant-scale ordered factorial effects: {result['constant_scale_factorial_clean_auc_effects']}",result['interpretation']]
    (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
