"""Prospective follow-up design; standard-library only."""
import datetime, hashlib, json, math, statistics
from pathlib import Path
KIND='icl_curriculum_generalization_components_20261013_v1'
DEV_SEEDS=(1600,1601,1602)
CONFIRM_SEEDS=tuple(range(2000,2020))
LRS=(1e-5,3e-5,1e-4,3e-4,1e-3)
GROUPS=(('joint_d20',20,'both'),('joint_d50',50,'both'),('dimension_d50',50,'dimension'),('context_d50',50,'context'))
TIMES=(0,1000,5000,10000,20000,25000,30000,40000,50000,75000,100000,125000)
STEPS=125000
BANKS={'development':(842131,842132),'confirmation':(842133,842134)}
MAX_GPUS=15
PLAN=Path('FOLLOWUP_TASKS.json');FREEZE=Path('submissions/freeze.json')
SELECT=Path('submissions/selection.json');RESOLVED=Path('submissions/resolved_tasks.json')
SUBMISSION=Path('submissions/pipeline.json')
BASE_SHA='4df516cd368733d00795d9a3cf16f4f114db06e156cad5a4cabe3a22ce0776b7'
TRAINER_SHA='83bcb5847c8194fca7d463666f588d010564e00186ac36e1d461cb566a3e1db3'
FILES=('FOLLOWUP_PROTOCOL.md','FOLLOWUP_TASKS.json','followup_plan.py','followup_experiment.py',
       'followup_analysis.py','archive_followup.py','followup_checks.py','test_followup.py','submit_followup.py',
       'followup_preflight.slurm','followup_smoke.slurm','followup_dev.slurm','followup_select.slurm',
       'followup_confirm.slurm','followup_summary.slurm','run_experiment.py','robustness_experiment.py')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def strategies():
    out=[]
    for group,d,component in GROUPS:
        for arm in ('curriculum','shuffled'):
            out.append(dict(strategy=group+'_'+arm,group=group,dims=d,component=component,arm=arm))
    for d in (20,50):out.append(dict(strategy=f'hard_d{d}',group=f'hard_d{d}',dims=d,component='none',arm='hard'))
    return out

def development_tasks():
    out=[]
    for seed in DEV_SEEDS:
        for lr in LRS:
            for strategy in strategies():
                out.append(strategy|dict(index=len(out),cohort='development',seed=seed,learning_rate=lr,
                    steps=STEPS,variant=f"{strategy['strategy']}_lr{lr:g}"))
    return out

def templates():
    out=[]
    for seed in CONFIRM_SEEDS:
        for group,d,component in GROUPS:
            for panel in ('shared','tuned'):
                for arm in ('curriculum','shuffled'):
                    out.append(dict(index=len(out),cohort='confirmation',seed=seed,dims=d,group=group,
                        component=component,arm=arm,strategy=group+'_'+arm,panel=panel,steps=STEPS,
                        variant=group+'_'+panel+'_'+arm))
        for d in (20,50):out.append(dict(index=len(out),cohort='confirmation',seed=seed,dims=d,
            group=f'hard_d{d}',component='none',arm='hard',strategy=f'hard_d{d}',panel='tuned',
            steps=STEPS,variant=f'hard_d{d}_tuned'))
    return out

def frozen_plan():
    return dict(kind=KIND,development=development_tasks(),confirmation_templates=templates(),
        learning_rates=list(LRS),development_seeds=list(DEV_SEEDS),confirmation_seeds=list(CONFIRM_SEEDS),
        banks={k:list(v) for k,v in BANKS.items()},max_concurrent_gpus=MAX_GPUS,
        selection_endpoint='arithmetic mean final clean NMSE across all three development seeds at 125000',
        shared_selection='minimum pooled curriculum/shuffled development mean; same LR for both',
        tuned_selection='minimum own-strategy development mean; identical candidate grid and training budget',
        primary='joint_d20_shared_curriculum_vs_shuffled_final_clean',secondary_family_size=7,
        tie_tolerance=1e-8,practical_geometric_ratio_max=.90,maximum_curriculum_mean_nmse=.25)

def verify_sources():
    if json.loads(PLAN.read_text())!=frozen_plan():raise ValueError('Prospective plan changed')
    if digest('run_experiment.py')!=BASE_SHA or digest('robustness_experiment.py')!=TRAINER_SHA:
        raise ValueError('Verified upstream training source changed')
    f=json.loads(FREEZE.read_text())
    if f['kind']!=KIND or set(f['source_hashes'])!=set(FILES):raise ValueError('Freeze incomplete')
    for name,h in f['source_hashes'].items():
        if digest(name)!=h:raise ValueError('Frozen source changed: '+name)
    return f

def rank_development(scores):
    expected={(s,x['strategy'],lr) for s in DEV_SEEDS for x in strategies() for lr in LRS}
    if set(scores)!=expected or any(not math.isfinite(v) or v<=0 for v in scores.values()):
        raise ValueError('Complete finite positive development scores required')
    ranks={}
    for x in strategies():
        rows=[dict(learning_rate=lr,mean_nmse=statistics.mean(scores[s,x['strategy'],lr] for s in DEV_SEEDS),
            per_seed_nmse=[scores[s,x['strategy'],lr] for s in DEV_SEEDS],grid_order=i)
            for i,lr in enumerate(LRS)]
        ranks[x['strategy']]=sorted(rows,key=lambda r:(r['mean_nmse'],r['grid_order']))
    shared={}
    for group,_,_ in GROUPS:
        rows=[dict(learning_rate=lr,pooled_mean=statistics.mean(scores[s,group+'_'+a,lr]
              for s in DEV_SEEDS for a in ('curriculum','shuffled')),grid_order=i) for i,lr in enumerate(LRS)]
        shared[group]=sorted(rows,key=lambda r:(r['pooled_mean'],r['grid_order']))
    return dict(tuned=ranks,shared=shared)

def resolve(ranking):
    out=[];seen={}
    for t in templates():
        lr=(ranking['shared'][t['group']][0] if t['panel']=='shared' else ranking['tuned'][t['strategy']][0])['learning_rate']
        key=(t['seed'],t['strategy'],lr)
        alias=seen.get(key);seen.setdefault(key,t['index'])
        out.append(t|dict(learning_rate=lr,alias_of=alias))
    return out

def load_selection():
    verify_sources();value=json.loads(SELECT.read_text())
    if value['kind']!=KIND or value['freeze_sha256']!=digest(FREEZE):raise ValueError('Wrong selection freeze')
    scores={(r['seed'],r['strategy'],r['learning_rate']):r['final_clean_nmse'] for r in value['scores']}
    if len(scores)!=len(value['scores']):raise ValueError('Duplicate development scores')
    ranking=rank_development(scores)
    if ranking!=value['ranking']:raise ValueError('Selection ranking changed')
    tasks=resolve(ranking)
    if json.loads(RESOLVED.read_text())!=tasks or value['resolved_sha256']!=digest(RESOLVED):
        raise ValueError('Resolved confirmation changed')
    return tasks

def sign_test(ds):
    if not ds or any(not math.isfinite(x) for x in ds):raise ValueError('Finite pairs required')
    wins=sum(x < -1e-8 for x in ds);losses=sum(x > 1e-8 for x in ds);n=wins+losses
    p=min(1.,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n) if n else 1.
    return dict(wins=wins,losses=losses,ties=len(ds)-n,p_two_sided=p)

def holm(ps):
    if any(not math.isfinite(p) or not 0<=p<=1 for p in ps.values()):raise ValueError('Invalid p-values')
    out={};running=0.
    for i,(key,p) in enumerate(sorted(ps.items(),key=lambda x:(x[1],x[0]))):
        running=max(running,(len(ps)-i)*p);out[key]=min(1.,running)
    return out
