#!/usr/bin/env python3
"""Strict verification, selection, and fixed-sequence confirmation analysis."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch
import confirmation_plan as plan
import confirmation_experiment as exp

CONDITIONS=('gaussian_sd0','gaussian_sd1','gaussian_sd2')


def check_metric(m,manifest,cfg,step,split,bank):
    fields={'identity':manifest,'completed_steps':step,'evaluation_dimension':50,
            'evaluation_positions':101,'eval_seed':bank,'eval_episodes':4096,
            'evaluation_split':split,'k_start':50,'k_end':100}
    if any(m.get(k)!=v for k,v in fields.items()): raise ValueError('Wrong identity, shapes, progress or bank')
    if set(m['conditions'])!=set(CONDITIONS): raise ValueError('Missing conditions')
    if not math.isfinite(m['zero_predictor_late_nmse']) or m['zero_predictor_late_nmse']<0: raise ValueError('Invalid zero reference')
    for condition in m['conditions'].values():
        curve=condition['mean_nmse_by_k'];value=condition['late_context_nmse']
        if len(curve)!=101 or any(not math.isfinite(x) or x<0 for x in curve): raise ValueError('Invalid curve')
        if not math.isfinite(value) or not math.isclose(statistics.mean(curve[50:]),value,abs_tol=2e-6,rel_tol=2e-6): raise ValueError('Curve/late-context mismatch')
    if not math.isfinite(m['training_seconds']) or m['training_seconds']<0: raise ValueError('Invalid runtime')


def state_fingerprint(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_cohort(cohort):
    """Fail closed on missing/changed sources, checkpoints, schedules or outcomes."""
    freeze=plan.verify_sources();tasks=exp.tasks(cohort)
    metrics={};holdouts={};rows=[];initial={};baselines={};records=0;checkpoints=0
    for task in tasks:
        cfg,study=exp.task_config(task);directory=exp.task_directory(task)
        manifest=json.loads((directory/'manifest.json').read_text())
        expected_identity=exp.identity(cfg,study)
        if set(manifest)!=set(expected_identity)|{'initial_model_sha256'} or any(manifest.get(k)!=v for k,v in expected_identity.items()): raise ValueError('Source/treatment identity mismatch')
        initial.setdefault(cfg.seed,set()).add(manifest['initial_model_sha256'])
        ids=exp.training.batch_order(cfg,study)
        schedule=torch.load(directory/'schedule.pt',map_location='cpu',weights_only=True)
        expected=[exp.training.treatment(cfg,study,int(i)) for i in ids]
        wanted={'batch_ids':ids,'active_dims':torch.tensor([x[0] for x in expected]),
                'eligible_positions':torch.tensor([x[1] for x in expected]),
                'label_multipliers':torch.tensor([x[2] for x in expected],dtype=torch.float64)}
        if set(schedule)!=set(wanted) or any(not torch.equal(schedule[k],v) for k,v in wanted.items()): raise ValueError('Recorded schedule differs')
        for step in study.evaluation_steps:
            metric_path=directory/'evaluations'/f'step_{step:06d}'/'metrics.json'
            m=json.loads(metric_path.read_text());check_metric(m,manifest,cfg,step,'primary',cfg.eval_seed)
            metrics[task['index'],step]=m;records+=1
            if step==0: baselines.setdefault(cfg.seed,[]).append(m['conditions'])
            for condition,v in m['conditions'].items():
                rows.append(dict(task_index=task['index'],cohort=cohort,setting=task['setting'],arm=task['arm'],seed=cfg.seed,step=step,condition=condition,late_context_nmse=v['late_context_nmse'],training_seconds=m['training_seconds']))
            checkpoint=directory/f'checkpoint_step_{step:06d}.pt'
            if plan.digest(checkpoint)!=m['checkpoint_sha256']: raise ValueError('Frozen checkpoint hash differs')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if saved['identity']!=manifest or saved['completed_steps']!=step or not torch.equal(saved['batch_ids'],ids): raise ValueError('Checkpoint progress/identity mismatch')
            states=saved['optimizer']['state']
            if (step>0 and not states) or any(int(v['step'])!=step for v in states.values()): raise ValueError('Wrong AdamW update count')
            groups=saved['optimizer']['param_groups']
            if not groups or any(g['lr']!=cfg.learning_rate or g['weight_decay']!=cfg.weight_decay for g in groups): raise ValueError('Wrong checkpoint optimizer')
            if step==0 and state_fingerprint(saved['model'])!=manifest['initial_model_sha256']: raise ValueError('Initial weights hash differs')
            del saved;checkpoints+=1
        holdout_bank=plan.DEV_HOLDOUT if cohort=='development' else plan.CONFIRM_HOLDOUT
        m=json.loads((directory/'final_holdout'/f'step_{cfg.steps:06d}'/'metrics.json').read_text())
        check_metric(m,manifest,cfg,cfg.steps,'final_holdout',holdout_bank)
        if m['checkpoint_sha256']!=metrics[task['index'],cfg.steps]['checkpoint_sha256']: raise ValueError('Holdout uses different weights')
        if plan.digest(directory/'checkpoint.pt')!=m['checkpoint_sha256']: raise ValueError('Rolling final weights differ')
        holdouts[task['index']]=m;records+=1
        with (directory/'train.csv').open(newline='') as f: logs=list(csv.DictReader(f))
        expected_steps=[1]+list(range(100,cfg.steps+1,100))
        if [int(r['step']) for r in logs]!=expected_steps: raise ValueError('Incomplete/duplicated training log')
        for r in logs:
            step=int(r['step']);bid=int(ids[step-1]);active,eligible,_=exp.training.treatment(cfg,study,bid)
            if (int(r['batch_id']),int(r['active_dims']),int(r['eligible_positions']))!=(bid,active,eligible): raise ValueError('Logged batch treatment differs')
            if any(not math.isfinite(float(r[k])) for k in ('train_mse','objective_loss','grad_norm','seconds_per_step','training_seconds')): raise ValueError('Nonfinite training log')
            divisor=active if study.objective=='nmse' else 1
            if not math.isclose(float(r['objective_loss']),float(r['train_mse'])/divisor,rel_tol=2e-6,abs_tol=2e-6): raise ValueError('Logged objective differs')
        print(f'Verified {cohort} task {task["index"]+1}/{len(tasks)} seed={cfg.seed} arm={study.arm}',flush=True)
    if any(len(v)!=1 for v in initial.values()): raise ValueError('Paired initialization differs')
    for group in baselines.values():
        for other in group[1:]:
            for condition in CONDITIONS:
                if any(abs(a-b)>2e-5 for a,b in zip(group[0][condition]['mean_nmse_by_k'],other[condition]['mean_nmse_by_k'])): raise ValueError('Paired initial predictions differ')
    expected_records,expected_checkpoints=(360,324) if cohort=='development' else (780,720)
    if (records,checkpoints)!=(expected_records,expected_checkpoints): raise ValueError('Incomplete cohort')
    audit=dict(kind=plan.KIND,cohort=cohort,training_runs=len(tasks),evaluation_records=records,
               frozen_checkpoints_verified=checkpoints,all_checks_passed=True,source_hashes=freeze['source_hashes'])
    return tasks,metrics,holdouts,rows,audit


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def development_selection():
    if plan.SELECTION.exists() or plan.RESOLVED.exists(): raise FileExistsError('Selection already/partially exists; preserve and inspect')
    tasks,metrics,holdouts,rows,audit=validate_cohort('development')
    auc={};clean_scores=[]
    for task in tasks:
        times=[t for t in plan.TIMES if t>=25000]
        score=plan.trapezoidal_mean(times,[metrics[task['index'],t]['conditions']['gaussian_sd0']['late_context_nmse'] for t in times])
        auc[task['seed'],task['setting'],task['arm']]=score
        clean_scores.append(dict(seed=task['seed'],setting=task['setting'],arm=task['arm'],auc=score))
    ranking=plan.rank_development(auc);resolved=plan.resolve_templates(ranking)
    plan.atomic_json(plan.RESOLVED,resolved)
    decision=dict(kind=plan.KIND,status='complete_locked_development_selection',
        freeze_sha256=plan.digest(plan.FREEZE),protocol_sha256=plan.digest(plan.PROTOCOL),
        development_clean_auc=clean_scores,rankings=ranking,
        selected_common_setting=ranking['shuffled'][0]['setting'],
        selected_hard_setting=ranking['hard'][0]['setting'],
        resolved_tasks_sha256=plan.digest(plan.RESOLVED),verification=audit,
        interpretation='Development selects optimizers only. Its outcomes are not confirmation; holdouts and all confirmation outcomes are excluded from selection.')
    plan.atomic_json(plan.SELECTION,decision);plan.load_selection()
    write_csv(Path('results/development/learning_curves.csv'),rows)
    plan.atomic_json(Path('results/development/selection.json'),decision)
    print(json.dumps({'selected_common_setting':decision['selected_common_setting'],'selected_hard_setting':decision['selected_hard_setting'],'status':decision['status']},indent=2),flush=True)


def confirmation_summary():
    target=Path('results/confirmation/summary.json')
    if target.exists(): raise FileExistsError('Preserve prior summary')
    tasks,metrics,holdouts,rows,audit=validate_cohort('confirmation')
    auc={};finals={};details=[]
    for task in tasks:
        seed,arm=task['seed'],task['arm'];times=[t for t in plan.LONG_TIMES if t>=25000]
        outcomes={}
        for condition in CONDITIONS:
            ys=[metrics[task['index'],t]['conditions'][condition]['late_context_nmse'] for t in times]
            outcomes[condition]=dict(auc=plan.trapezoidal_mean(times,ys),final=ys[-1],holdout=holdouts[task['index']]['conditions'][condition]['late_context_nmse'])
        auc[seed,arm]=outcomes['gaussian_sd0']['auc'];finals[seed,arm]=outcomes['gaussian_sd0']['final']
        crossing=next((i for i,t in enumerate(plan.LONG_TIMES) if metrics[task['index'],t]['conditions']['gaussian_sd0']['late_context_nmse']<=.25),None)
        threshold=dict(first_scheduled_step=plan.LONG_TIMES[crossing] if crossing is not None else None,
            previous_scheduled_step=plan.LONG_TIMES[crossing-1] if crossing not in (None,0) else None,
            right_censored=crossing is None,budget_end=125000)
        details.append(dict(task_index=task['index'],seed=seed,arm=arm,setting=task['setting'],
            outcomes=outcomes,threshold_nmse_0_25=threshold,
            training_seconds=metrics[task['index'],125000]['training_seconds']))
    result=dict(kind=plan.KIND,status='complete_fresh_twenty_seed_confirmation',
        verification=audit,selection_sha256=plan.digest(plan.SELECTION),
        resolved_tasks_sha256=plan.digest(plan.RESOLVED),seeds=list(plan.CONFIRM_SEEDS),
        individual_runs=details)
    result.update(plan.analyze_scores(auc,finals))
    result['arm_means']={arm:{condition:{metric:statistics.mean(r['outcomes'][condition][metric] for r in details if r['arm']==arm) for metric in ('auc','final','holdout')} for condition in CONDITIONS} for arm in plan.ARMS}
    result['learned_at_final']={arm:sum(finals[s,arm]<=.25 for s in plan.CONFIRM_SEEDS) for arm in plan.ARMS}
    result['claim']=('Curriculum helps acquisition versus tuned hard-only in this fixed setup; chronology also helps beyond matched exposure.' if result['ordering_supported'] else 'Curriculum helps acquisition versus tuned hard-only in this fixed setup; ordering beyond matched exposure is not confirmed.' if result['primary_supported'] else 'The prespecified primary criteria did not confirm a useful curriculum benefit versus tuned hard-only in this fixed setup.')
    plan.atomic_json(target,result);write_csv(Path('results/confirmation/learning_curves.csv'),rows)
    lines=[result['claim'],'All 60 runs complete and verified; 20 fresh paired training seeds.',
        'Primary clean AUC means: '+json.dumps({a:result['arm_means'][a]['gaussian_sd0']['auc'] for a in plan.ARMS}),
        'Final learned counts at NMSE <= .25: '+json.dumps(result['learned_at_final']),
        'Primary support criteria: '+json.dumps(result['primary_criteria']),
        'Primary sign test: '+json.dumps(result['primary_curriculum_vs_tuned_hard']['sign_test']),
        'Ordering test eligible for confirmation: '+str(result['ordering_test_eligible']),
        'Ordering sign test (descriptive if not eligible): '+json.dumps(result['ordering_vs_matched_shuffled']['sign_test']),
        result['testing_rule'],
        'No formal proof, universal ICL claim, mechanism claim, or independent noisy-context significance claim follows.']
    Path('results/confirmation/RESULTS.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');args=p.parse_args()
    torch.set_num_threads(1)
    development_selection() if args.select else confirmation_summary()


if __name__=='__main__': main()
