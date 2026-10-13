"""Fail-closed selection, checkpoint audit and all-seed results from raw metrics."""
import argparse, csv, json, math, random, statistics
from pathlib import Path
import torch
import followup_plan as plan
import followup_experiment as exp

def read_task(t):
    cfg,study=exp.task_config(t);root=exp.task_directory(t)
    manifest=json.loads((root/'manifest.json').read_text());expected=exp.identity(cfg,study)
    if {k:v for k,v in manifest.items() if k!='initial_model_sha256'}!=expected:raise ValueError('Manifest differs')
    metrics={}
    for time in plan.TIMES:
        m=json.loads((root/f'evaluations/step_{time:06d}/metrics.json').read_text())
        check_metric(m,manifest,cfg,time,'primary');metrics[time]=m
    held=json.loads((root/f'final_holdout/step_{cfg.steps:06d}/metrics.json').read_text())
    check_metric(held,manifest,cfg,cfg.steps,'final_holdout')
    for time in (0,study.prefix_steps,cfg.steps):
        cp=root/f'checkpoint_step_{time:06d}.pt'
        if plan.digest(cp)!=metrics[time]['checkpoint_sha256']:raise ValueError('Snapshot hash differs')
    if held['checkpoint_sha256']!=metrics[cfg.steps]['checkpoint_sha256']:raise ValueError('Holdout model differs')
    cp=torch.load(root/f'checkpoint_step_{cfg.steps:06d}.pt',map_location='cpu',weights_only=True)
    ids=exp.training.batch_order(cfg,study)
    if cp['identity']!=manifest or cp['completed_steps']!=cfg.steps or not torch.equal(ids,cp['batch_ids'].cpu()):
        raise ValueError('Final binary identity or order differs')
    if not cp['optimizer']['state'] or any(int(s['step'])!=cfg.steps for s in cp['optimizer']['state'].values()):
        raise ValueError('Optimizer completion differs')
    schedule=torch.load(root/'schedule.pt',map_location='cpu',weights_only=True)
    if not torch.equal(schedule['batch_ids'],ids):raise ValueError('Schedule differs')
    expected_treatments=[exp.training.treatment(cfg,study,int(i)) for i in ids]
    for key,column in (('active_dims',0),('eligible_positions',1),('label_multipliers',2)):
        expected_values=torch.tensor([x[column] for x in expected_treatments],dtype=schedule[key].dtype)
        if not torch.equal(schedule[key],expected_values):raise ValueError('Treatment schedule differs')
    del cp,schedule
    return manifest,metrics,held

def check_metric(m,manifest,cfg,step,split):
    expected=dict(identity=manifest,completed_steps=step,evaluation_dimension=cfg.dims,
        evaluation_positions=cfg.points,eval_seed=cfg.eval_seed if split=='primary' else plan.BANKS[manifest['study']['cohort']][1],
        eval_episodes=4096,evaluation_split=split,k_start=cfg.dims,k_end=2*cfg.dims)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong metric identity/split/shape')
    if set(m['conditions'])!={'gaussian_sd0','gaussian_sd1','gaussian_sd2'}:raise ValueError('Conditions differ')
    for row in m['conditions'].values():
        values=row['mean_nmse_by_k']
        if len(values)!=cfg.points or any(not math.isfinite(v) or v<0 for v in values):raise ValueError('Invalid curve')
        if not math.isclose(statistics.mean(values[cfg.dims:]),row['late_context_nmse'],rel_tol=2e-5,abs_tol=1e-7):
            raise ValueError('Scalar differs from curve')
    if not .8<m['zero_predictor_late_nmse']<1.2:raise ValueError('Evaluation scale invalid')

def select():
    plan.verify_sources()
    if plan.SELECT.exists() or plan.RESOLVED.exists():raise FileExistsError('Selection already exists, do not overwrite')
    scores={};rows=[];initial={}
    for t in plan.development_tasks():
        manifest,metrics,held=read_task(t)
        key=(t['seed'],t['dims']);fp=manifest['initial_model_sha256']
        if key in initial and initial[key]!=fp:raise ValueError('Paired initialization differs')
        initial[key]=fp
        score=metrics[plan.STEPS]['conditions']['gaussian_sd0']['late_context_nmse']
        scores[t['seed'],t['strategy'],t['learning_rate']]=score
        rows.append({k:t[k] for k in ('seed','strategy','learning_rate','index')}|dict(final_clean_nmse=score))
    ranking=plan.rank_development(scores);resolved=plan.resolve(ranking)
    plan.atomic_json(plan.RESOLVED,resolved)
    boundary=[key for key,r in ranking['tuned'].items() if r[0]['learning_rate'] in (plan.LRS[0],plan.LRS[-1])]
    plan.atomic_json(plan.SELECT,dict(kind=plan.KIND,locked_utc=plan.now(),freeze_sha256=plan.digest(plan.FREEZE),
        ranking=ranking,scores=rows,resolved_sha256=plan.digest(plan.RESOLVED),boundary_winners=boundary,
        unique_confirmation_models=sum(t['alias_of'] is None for t in resolved),
        rule='Full equal development budgets; all scores are primary-bank final clean NMSE. Holdout/noise cannot select.',
        caveat='Boundary winners remain reported; no post-confirmation retuning or claim of optimal baseline.'))
    print('Selection locked. Unique confirmation models:',sum(t['alias_of'] is None for t in resolved),flush=True)

def auc(metrics,condition):
    times=plan.TIMES[5:];values=[metrics[t]['conditions'][condition]['late_context_nmse'] for t in times]
    return sum((b-a)*(x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))/(times[-1]-times[0])

def paired(left,right):
    if len(left)!=20 or len(right)!=20 or any(v<=0 or not math.isfinite(v) for v in left+right):raise ValueError('Twenty positive pairs required')
    differences=[a-b for a,b in zip(left,right)];logs=[math.log(a/b) for a,b in zip(left,right)]
    rng=random.Random(921733);draws=sorted(statistics.mean(rng.choices(logs,k=20)) for _ in range(10000))
    return dict(left=left,right=right,differences=differences,mean_difference=statistics.mean(differences),
        median_difference=statistics.median(differences),sign_test=plan.sign_test(differences),
        geometric_mean_ratio=math.exp(statistics.mean(logs)),
        descriptive_geometric_ratio_ci95=[math.exp(draws[249]),math.exp(draws[9749])],
        leave_one_out_geometric_ratios=[math.exp(statistics.mean(logs[:i]+logs[i+1:])) for i in range(20)])


def classify(contrasts):
    if len(contrasts)!=8 or 'joint_d20_ordering' not in contrasts:raise ValueError('Eight predefined contrasts required')
    def practical(c):return c['geometric_mean_ratio']<=.90 and statistics.mean(c['left'])<=.25
    primary=contrasts['joint_d20_ordering']
    supported=primary['sign_test']['p_two_sided']<.05 and practical(primary)
    adjusted=plan.holm({k:v['sign_test']['p_two_sided'] for k,v in contrasts.items() if k!='joint_d20_ordering'})
    for key,c in contrasts.items():
        c['practical_threshold_passed']=practical(c)
        c['confirmatory_supported']=(supported if key=='joint_d20_ordering' else supported and adjusted[key]<.05 and practical(c))
        if key in adjusted:c['holm_adjusted_p']=adjusted[key]
    return supported

def summary():
    tasks=plan.load_selection();raw={};initial={};rows=[]
    for t in tasks:
        if t['alias_of'] is not None:continue
        manifest,metrics,held=read_task(t);raw[t['index']]=(manifest,metrics,held)
        key=(t['seed'],t['dims']);fp=manifest['initial_model_sha256']
        if key in initial and initial[key]!=fp:raise ValueError('Paired initialization differs')
        initial[key]=fp
    lookup={}
    for t in tasks:
        manifest,metrics,held=raw[t['index'] if t['alias_of'] is None else t['alias_of']]
        lookup[t['seed'],t['group'],t['panel'],t['arm']]=(metrics,held)
        for condition in ('gaussian_sd0','gaussian_sd1','gaussian_sd2'):
            curve=metrics[plan.STEPS]['conditions'][condition]['mean_nmse_by_k'];d=t['dims']
            bump=statistics.mean(curve[d-2:d+3])/statistics.mean(curve[d-12:d-7])
            rows.append({k:t[k] for k in ('index','seed','dims','group','panel','arm','learning_rate','alias_of')}|
                dict(condition=condition,final_nmse=metrics[plan.STEPS]['conditions'][condition]['late_context_nmse'],
                    clean_or_noisy_auc=auc(metrics,condition),holdout_nmse=held['conditions'][condition]['late_context_nmse'],
                    peak_ratio=bump,canonical_index=t['index'] if t['alias_of'] is None else t['alias_of']))
    contrasts={};noise={};tuned_order={}
    for group,d,_ in plan.GROUPS:
        for suffix,panel,right_group,right_panel,right_arm in (
            ('ordering','shared',group,'shared','shuffled'),('policy','tuned',f'hard_d{d}','tuned','hard')):
            left=[lookup[s,group,panel,'curriculum'][0][plan.STEPS]['conditions']['gaussian_sd0']['late_context_nmse'] for s in plan.CONFIRM_SEEDS]
            right=[lookup[s,right_group,right_panel,right_arm][0][plan.STEPS]['conditions']['gaussian_sd0']['late_context_nmse'] for s in plan.CONFIRM_SEEDS]
            contrasts[group+'_'+suffix]=paired(left,right)
        for condition in ('gaussian_sd1','gaussian_sd2'):
            left=[lookup[s,group,'shared','curriculum'][0][plan.STEPS]['conditions'][condition]['late_context_nmse'] for s in plan.CONFIRM_SEEDS]
            right=[lookup[s,group,'shared','shuffled'][0][plan.STEPS]['conditions'][condition]['late_context_nmse'] for s in plan.CONFIRM_SEEDS]
            noise[group+'_'+condition]=paired(left,right)
        tuned_order[group]=paired(
            [lookup[s,group,'tuned','curriculum'][0][plan.STEPS]['conditions']['gaussian_sd0']['late_context_nmse'] for s in plan.CONFIRM_SEEDS],
            [lookup[s,group,'tuned','shuffled'][0][plan.STEPS]['conditions']['gaussian_sd0']['late_context_nmse'] for s in plan.CONFIRM_SEEDS])
    supported=classify(contrasts)
    groups={}
    for group,panel,arm in sorted({(t['group'],t['panel'],t['arm']) for t in tasks}):
        values=[lookup[seed,group,panel,arm][0][plan.STEPS]['conditions']['gaussian_sd0']['late_context_nmse'] for seed in plan.CONFIRM_SEEDS]
        learned=[v for v in values if v<=.25];failed=[v for v in values if v>.25]
        groups[group+'_'+panel+'_'+arm]=dict(mean=statistics.mean(values),median=statistics.median(values),
            learned_count=len(learned),failed_count=len(failed),learned_mean=statistics.mean(learned) if learned else None,
            failed_mean=statistics.mean(failed) if failed else None,per_seed_values=values)
    out=dict(kind=plan.KIND,completed_utc=plan.now(),freeze_sha256=plan.digest(plan.FREEZE),
        selection_sha256=plan.digest(plan.SELECT),unique_models=len(raw),confirmation_seeds=list(plan.CONFIRM_SEEDS),
        contrasts=contrasts,strategy_outcomes=groups,exploratory_noise=noise,exploratory_tuned_curriculum_vs_shuffled=tuned_order,
        primary_supported=supported,secondary_family_eligible=supported,
        note='Only final clean ordering/policy contrasts follow the frozen sequence. Noise, AUC, peak ratios, tuned ordering, bootstrap and leave-one-out are descriptive; retain every seed.')
    plan.atomic_json('results/summary.json',out)
    path=Path('results/all_seed_results.csv')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(primary_supported=supported,unique_models=len(raw)),indent=2),flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');a=p.parse_args()
    select() if a.select else summary()
