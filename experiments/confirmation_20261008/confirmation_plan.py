"""Frozen design and statistical primitives. No Torch import or GPU use."""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

KIND = 'icl_curriculum_strong_baseline_confirmation_v1'
DEV_SEEDS = (600, 601, 602)
CONFIRM_SEEDS = tuple(range(1000, 1020))
ARMS = ('curriculum', 'shuffled', 'hard')
OPTIMIZERS = (
    ('lr_low', 0.0001, 'raw'),
    ('baseline', 0.0004, 'raw'),
    ('lr_high', 0.001, 'raw'),
    ('loss_nmse', 0.0004, 'nmse'),
)
DEV_BANK = 731839
DEV_HOLDOUT = 731840
CONFIRM_BANK = 731841
CONFIRM_HOLDOUT = 731842
TIMES = (0, 1000, 5000, 10000, 20000, 25000, 30000, 40000, 50000)
LONG_TIMES = TIMES + (75000, 100000, 125000)
PROTOCOL = Path('CONFIRMATION_PROTOCOL.txt')
PLAN_FILE = Path('CONFIRMATION_TASKS.json')
FREEZE = Path('submissions/freeze.json')
SELECTION = Path('submissions/selection.json')
RESOLVED = Path('submissions/confirmation_resolved_tasks.json')
SUBMISSION = Path('submissions/confirmation-20261008.json')
BASE_SHA = '4df516cd368733d00795d9a3cf16f4f114db06e156cad5a4cabe3a22ce0776b7'
TRAINER_SHA = '83bcb5847c8194fca7d463666f588d010564e00186ac36e1d461cb566a3e1db3'
FILES = (
    'CONFIRMATION_PROTOCOL.txt', 'CONFIRMATION_TASKS.json', 'confirmation_plan.py',
    'confirmation_experiment.py', 'confirmation_analysis.py', 'confirmation_checks.py',
    'test_confirmation.py', 'select_confirmation.py', 'submit_confirmation.py',
    'confirmation_preflight.slurm', 'confirmation_smoke.slurm', 'confirmation_dev.slurm',
    'confirmation_select.slurm', 'confirmation_train.slurm', 'confirmation_summary.slurm',
    'run_experiment.py', 'robustness_experiment.py', 'ROBUSTNESS_PROTOCOL.txt',
    'ROBUSTNESS_TASKS.json', 'test_robustness.py', 'summarize_robustness.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def development_tasks():
    tasks = []
    for seed in DEV_SEEDS:
        for name, lr, objective in OPTIMIZERS:
            for arm in ARMS:
                tasks.append(dict(index=len(tasks), cohort='development', seed=seed,
                    variant=name+'_'+arm, setting=name, arm=arm,
                    component='none' if arm=='hard' else 'both',
                    target_variance='constant', objective=objective,
                    learning_rate=lr, precision='bf16', steps=50000))
    return tasks


def confirmation_templates():
    return [dict(index=i, cohort='confirmation', seed=seed, arm=arm,
                 component='none' if arm=='hard' else 'both',
                 target_variance='constant', precision='bf16', steps=125000,
                 selection='hard_best' if arm=='hard' else 'shuffled_best_common')
            for i, (seed, arm) in enumerate((s,a) for s in CONFIRM_SEEDS for a in ARMS)]


def frozen_plan():
    return dict(kind=KIND, development=development_tasks(), confirmation=confirmation_templates(),
                optimizer_order=[name for name, *_ in OPTIMIZERS],
                development_bank=DEV_BANK, development_holdout=DEV_HOLDOUT,
                confirmation_bank=CONFIRM_BANK, confirmation_holdout=CONFIRM_HOLDOUT,
                primary='curriculum_minus_tuned_hard_clean_auc',
                gated_secondary='curriculum_minus_matched_shuffled_clean_auc',
                tie_tolerance=1e-8, minimum_mean_auc_reduction=0.02,
                maximum_curriculum_final_mean_nmse=0.25,
                bootstrap_repetitions=10000, bootstrap_seed=845731,
                max_concurrent_gpus=15)


def verify_sources():
    if json.loads(PLAN_FILE.read_text()) != frozen_plan():
        raise ValueError('Frozen plan differs from code')
    if digest('run_experiment.py') != BASE_SHA or digest('robustness_experiment.py') != TRAINER_SHA:
        raise ValueError('Verified training sources changed')
    freeze = json.loads(FREEZE.read_text())
    if set(freeze['source_hashes']) != set(FILES): raise ValueError('Incomplete source freeze')
    for name, sha in freeze['source_hashes'].items():
        if digest(name) != sha: raise ValueError('Frozen source changed: '+name)
    return freeze


def trapezoidal_mean(times, values):
    if len(times)!=len(values) or len(times)<2 or any(b<=a for a,b in zip(times,times[1:])) or any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('Finite nonnegative values and increasing times required')
    return sum((b-a)*(x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))/(times[-1]-times[0])


def rank_development(auc):
    """Only development primary-bank clean AUCs enter selection; no holdouts."""
    result = {}
    for arm in ARMS:
        candidates=[]
        for order,(name,lr,objective) in enumerate(OPTIMIZERS):
            values=[auc[(seed,name,arm)] for seed in DEV_SEEDS]
            if any(not math.isfinite(v) or v<0 for v in values): raise ValueError('Invalid development score')
            candidates.append(dict(setting=name,learning_rate=lr,objective=objective,
                fixed_order=order,per_seed_clean_auc=values,mean_clean_auc=statistics.mean(values)))
        result[arm]=sorted(candidates,key=lambda c:(c['mean_clean_auc'],c['fixed_order']))
    return result


def resolve_templates(ranking):
    tasks=[]
    for template in confirmation_templates():
        pick=ranking['hard' if template['arm']=='hard' else 'shuffled'][0]
        tasks.append(template | dict(setting=pick['setting'],learning_rate=pick['learning_rate'],
            objective=pick['objective'],variant=pick['setting']+'_'+template['arm']))
    return tasks


def load_selection():
    verify_sources()
    decision=json.loads(SELECTION.read_text())
    if decision['kind']!=KIND or decision['status']!='complete_locked_development_selection':
        raise ValueError('Missing locked selection')
    if decision['freeze_sha256']!=digest(FREEZE) or decision['protocol_sha256']!=digest(PROTOCOL):
        raise ValueError('Selection protocol/freeze differs')
    scores={}
    for row in decision['development_clean_auc']:
        key=(row['seed'],row['setting'],row['arm'])
        if key in scores: raise ValueError('Duplicate development score')
        scores[key]=row['auc']
    if len(scores)!=36 or rank_development(scores)!=decision['rankings']:
        raise ValueError('Selection is not the frozen development ranking')
    expected=resolve_templates(decision['rankings'])
    if json.loads(RESOLVED.read_text())!=expected or decision['resolved_tasks_sha256']!=digest(RESOLVED):
        raise ValueError('Resolved confirmation task list differs')
    return decision, expected


def exact_sign_test(differences, tolerance=1e-8):
    if not differences or any(not math.isfinite(d) for d in differences): raise ValueError('Finite pairs required')
    wins=sum(d < -tolerance for d in differences)
    losses=sum(d > tolerance for d in differences)
    n=wins+losses;small=min(wins,losses)
    p=min(1.,2*sum(math.comb(n,k) for k in range(small+1))/(2**n)) if n else 1.
    return dict(wins=wins,losses=losses,ties=len(differences)-n,effective_n=n,
                two_sided_p=p,null_win_probability=0.5,tie_tolerance=tolerance)


def paired_statistics(left, right):
    if len(left)!=len(right) or len(left)!=len(CONFIRM_SEEDS): raise ValueError('Twenty complete paired seeds required')
    ds=[a-b for a,b in zip(left,right)]
    test=exact_sign_test(ds)
    rng=random.Random(845731)
    draws=sorted(statistics.mean(rng.choices(ds,k=len(ds))) for _ in range(10000))
    def quantile(q):
        at=q*(len(draws)-1); lo=math.floor(at); hi=math.ceil(at)
        return draws[lo]+(draws[hi]-draws[lo])*(at-lo)
    return dict(seeds=list(CONFIRM_SEEDS),seed_differences=ds,mean=statistics.mean(ds),
                median=statistics.median(ds),sign_test=test,
                descriptive_bootstrap_mean_ci95=[quantile(.025),quantile(.975)],
                bootstrap_repetitions=10000,bootstrap_seed=845731,
                inference_note='Sign test concerns seed win probability; bootstrap mean interval is descriptive, not the primary test.')


def support_gate(contrast, curriculum_final_mean):
    return dict(two_sided_significant=contrast['sign_test']['two_sided_p']<.05,
                practical_mean_reduction=contrast['mean']<=-.02,
                curriculum_learned=curriculum_final_mean<=.25)


def analyze_scores(auc, finals):
    """Fixed-sequence testing controls two claims; primary policy benefit first."""
    c=[auc[(s,'curriculum')] for s in CONFIRM_SEEDS]
    primary=paired_statistics(c,[auc[(s,'hard')] for s in CONFIRM_SEEDS])
    ordering=paired_statistics(c,[auc[(s,'shuffled')] for s in CONFIRM_SEEDS])
    final_mean=statistics.mean(finals[(s,'curriculum')] for s in CONFIRM_SEEDS)
    gate=support_gate(primary,final_mean);supported=all(gate.values())
    order_gate=support_gate(ordering,final_mean)
    return dict(primary_curriculum_vs_tuned_hard=primary,primary_criteria=gate,
                primary_supported=supported,
                ordering_vs_matched_shuffled=ordering,ordering_criteria=order_gate,
                ordering_test_eligible=supported,ordering_supported=supported and all(order_gate.values()),
                testing_rule='Fixed sequence: test broad curriculum-vs-tuned-hard claim first; only if its full support gate passes may the ordering contrast support a confirmatory claim. Otherwise ordering p-value is descriptive.')
