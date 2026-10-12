#!/usr/bin/env python3
"""Fixed cohort summary with paired training-seed uncertainty, no endpoint selection."""
import csv
import json
import math
from pathlib import Path
import statistics
import torch
import run_experiment as base
import dimension_long_experiment as exp

ROOT=Path('results/dimension_long_20261006')
T95_DF4=2.7764451051977987

def trapezoidal_mean(times,values):
    if len(times)!=len(values) or len(times)<2 or any(b<=a for a,b in zip(times,times[1:])) or any(not math.isfinite(x) for x in values):
        raise ValueError('AUC requires finite values at increasing times')
    return sum((b-a)*(x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))/(times[-1]-times[0])

def paired_contrast(left,right):
    if len(left)!=5 or len(right)!=5:raise ValueError('Exactly five complete paired training seeds required')
    ds=[a-b for a,b in zip(left,right)]
    if any(not math.isfinite(x) for x in ds):raise ValueError('Nonfinite paired difference')
    mean=statistics.mean(ds);se=statistics.stdev(ds)/math.sqrt(5);half=T95_DF4*se
    return {'seed_differences':ds,'mean':mean,'standard_error_across_training_seeds':se,'ci95_t_df4':[mean-half,mean+half],
            'curriculum_wins':sum(x<0 for x in ds),'n_training_seeds':5}

def check_metric(m,manifest,cfg,step,split='primary'):
    wanted=cfg.eval_seed if split=='primary' else exp.FINAL_HOLDOUT_SEED
    if m['identity']!=manifest or m['completed_steps']!=step or m['evaluation_dimension']!=50 or m['evaluation_positions']!=101 or m['eval_seed']!=wanted or m['eval_episodes']!=4096 or m['evaluation_split']!=split:
        raise ValueError('Wrong model, cohort, progress or evaluation bank')
    if set(m['conditions'])!={'gaussian_sd0','gaussian_sd1','gaussian_sd2'}:raise ValueError('Missing condition')
    if not math.isfinite(m['zero_predictor_late_nmse']):raise ValueError('Invalid baseline')
    for v in m['conditions'].values():
        if len(v['mean_nmse_by_k'])!=101 or any(not math.isfinite(x) for x in v['mean_nmse_by_k']):raise ValueError('Invalid curve')
        if not math.isclose(statistics.mean(v['mean_nmse_by_k'][50:]),v['late_context_nmse'],abs_tol=2e-6,rel_tol=2e-6):raise ValueError('Curve/endpoint mismatch')

def main():
    torch.set_num_threads(1);data={};heldout={};rows=[];records=0;checkpoints=0
    for seed in exp.SEEDS:
        hashes=[]
        for arm in exp.ARMS:
            cfg=exp.long_config(seed);study=exp.Study(arm=arm);d=ROOT/f'seed_{seed}'/arm
            manifest=json.loads((d/'manifest.json').read_text());expected=exp.identity(cfg,study)
            if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Source or configuration changed')
            hashes.append(manifest['initial_model_sha256']);ids=exp.batch_order(cfg,study)
            for t in study.evaluation_steps:
                m=json.loads((d/'evaluations'/f'step_{t:06d}'/'metrics.json').read_text());check_metric(m,manifest,cfg,t)
                data[seed,arm,t]=m;records+=1
                for condition,v in m['conditions'].items():rows.append({'seed':seed,'arm':arm,'step':t,'condition':condition,'late_context_nmse':v['late_context_nmse'],'training_seconds':m['training_seconds'],'zero_predictor_late_nmse':m['zero_predictor_late_nmse']})
                path=d/f'checkpoint_step_{t:06d}.pt'
                if exp.sha(path)!=m['checkpoint_sha256']:raise ValueError('Frozen checkpoint differs from evaluated weights')
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if saved['identity']!=manifest or saved['completed_steps']!=t or not torch.equal(saved['batch_ids'],ids):raise ValueError('Checkpoint identity/schedule mismatch')
                states=saved['optimizer']['state']
                if (t>0 and not states) or any(int(v['step'])!=t for v in states.values()):raise ValueError('Optimizer progress mismatch')
                del saved;checkpoints+=1
            m=json.loads((d/'final_holdout'/f'step_{cfg.steps:06d}'/'metrics.json').read_text());check_metric(m,manifest,cfg,cfg.steps,'final_holdout')
            if m['checkpoint_sha256']!=data[seed,arm,cfg.steps]['checkpoint_sha256']:raise ValueError('Holdout evaluated different final weights')
            heldout[seed,arm]=m;records+=1
        if len(set(hashes))!=1:raise ValueError('Paired initialization mismatch')
        ref=data[seed,'curriculum',0]
        for arm in ('shuffled','hard'):
            for c in ref['conditions']:
                if any(abs(a-b)>2e-5 for a,b in zip(ref['conditions'][c]['mean_nmse_by_k'],data[seed,arm,0]['conditions'][c]['mean_nmse_by_k'])):raise ValueError('Paired initial evaluations differ')
    if records!=195 or checkpoints!=180:raise ValueError('Incomplete fixed cohort')
    conditions=('gaussian_sd0','gaussian_sd1','gaussian_sd2');scores={};contrasts={};final={};fresh={};crossings={}
    for c in conditions:
        scores[c]={a:[trapezoidal_mean(exp.PRIMARY_TIMES,[data[s,a,t]['conditions'][c]['late_context_nmse'] for t in exp.PRIMARY_TIMES]) for s in exp.SEEDS] for a in exp.ARMS}
        contrasts[c]={a:paired_contrast(scores[c]['curriculum'],scores[c][a]) for a in ('shuffled','hard')}
        final[c]={a:[data[s,a,125000]['conditions'][c]['late_context_nmse'] for s in exp.SEEDS] for a in exp.ARMS}
        fresh[c]={a:[heldout[s,a]['conditions'][c]['late_context_nmse'] for s in exp.SEEDS] for a in exp.ARMS}
    for a in exp.ARMS:
        crossings[a]=[next((t for t in exp.Study().evaluation_steps if data[s,a,t]['conditions']['gaussian_sd0']['late_context_nmse']<=.25),None) for s in exp.SEEDS]
    primary=contrasts['gaussian_sd0']['shuffled'];final_mean=statistics.mean(final['gaussian_sd0']['curriculum'])
    criteria={'primary_ci_upper_below_zero':primary['ci95_t_df4'][1]<0,'mean_primary_reduction_at_least_0_02':primary['mean']<=-.02,'curriculum_final_mean_nmse_at_most_0_25':final_mean<=.25}
    result={'kind':exp.KIND,'status':'complete_five_seed_study','seeds':list(exp.SEEDS),'evaluation_records':records,'frozen_checkpoints_verified':checkpoints,
      'primary_metric':'trapezoidal_mean_clean_target_nmse_k50_100_steps25000_125000','primary_contrast':primary,'auc_by_arm':scores,'secondary_contrasts':contrasts,
      'final_by_arm':final,'final_holdout_by_arm':fresh,'first_scheduled_clean_nmse_at_most_0_25':crossings,'prespecified_support_criteria':criteria,'all_support_criteria_met':all(criteria.values()),
      'zero_predictor_primary_bank':data[200,'curriculum',0]['zero_predictor_late_nmse'],'zero_predictor_final_holdout_bank':heldout[200,'curriculum']['zero_predictor_late_nmse'],
      'protocol_sha256':exp.sha(exp.PROTOCOL),'runner_sha256':exp.sha(exp.__file__),'summary_sha256':exp.sha(__file__),
      'uncertainty_note':'95% paired t interval across five training seeds; approximate normality, small n, conditional on this task/model/schedule. Secondary contrasts are descriptive, not multiple confirmatory tests.'}
    base.atomic_json(ROOT/'summary.json',result)
    with (ROOT/'learning_curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['Fixed 125,000-update dimension/context study: all 15 fresh-seed runs complete.',
           'Primary: curriculum minus matched shuffled, clean full-50D NMSE k50..100, trapezoidal mean updates25,000..125,000. Negative favors curriculum.']
    for a,v in scores['gaussian_sd0'].items():lines.append(f'{a}: primary per-seed={v}, mean={statistics.mean(v):.9g}; final mean={statistics.mean(final["gaussian_sd0"][a]):.9g}')
    lines += [f'Primary paired contrast: {primary}',f'Prespecified support criteria: {criteria}',f'All criteria met: {all(criteria.values())}',result['uncertainty_note'],
              'Pilot seeds are excluded. Incomplete cohorts produce no conclusion. No new runs or parameter searches are submitted automatically.']
    (ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
