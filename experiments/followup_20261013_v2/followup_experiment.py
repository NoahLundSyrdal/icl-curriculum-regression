"""New provenance adapter; upstream causal trainer and batch generation unchanged."""
import argparse, dataclasses, json, signal
from pathlib import Path
import torch
import followup_plan as plan
import robustness_experiment as training
ROOT=Path('results')

def identity(cfg,study):
    plan.verify_sources()
    value=dict(kind=plan.KIND,config=dataclasses.asdict(cfg),study=training.study_dict(study),
        source_hashes=json.loads(plan.FREEZE.read_text())['source_hashes'],freeze_sha256=plan.digest(plan.FREEZE))
    if study.cohort=='confirmation':
        plan.load_selection();value.update(selection_sha256=plan.digest(plan.SELECT),resolved_sha256=plan.digest(plan.RESOLVED))
    return value
training.identity=identity
# Keep baseline, curriculum-boundary and final snapshots, plus the rolling checkpoint.
# Every other evaluated checkpoint hash remains in its immutable metric JSON.
def retain(directory,completed,study,cfg):
    if completed in (0,study.prefix_steps,cfg.steps):
        snapshot=Path(directory)/f'checkpoint_step_{completed:06d}.pt'
        if not snapshot.exists():
            import shutil
            tmp=snapshot.with_suffix('.pt.tmp');shutil.copyfile(Path(directory)/'checkpoint.pt',tmp);tmp.replace(snapshot)
training.retain=retain

def task_config(t,smoke=False):
    cfg=training.base.preset_config('full',mode='clean',seed=t['seed'],dims=t['dims'],points=2*t['dims']+1,
        steps=t['steps'],batch_size=256,microbatch=128,learning_rate=t['learning_rate'],
        eval_seed=plan.BANKS[t['cohort']][0],eval_episodes=64 if smoke else 4096,
        eval_microbatch=64,precision='bf16',bayes_reference=False)
    study=training.Study(arm=t['arm'],cohort='gpu_smoke' if smoke else t['cohort'],variant=t['variant'],
        component=t['component'],target_variance='constant',objective='raw',evaluation_steps=plan.TIMES)
    training.validate(cfg,study);return cfg,study

def task_directory(t):return ROOT/t['cohort']/f"seed_{t['seed']}"/t['variant']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['development','confirmation'],required=True)
    parser.add_argument('--task-index',type=int,required=True);parser.add_argument('--resume',action='store_true')
    a=parser.parse_args();plan.verify_sources();torch.set_num_threads(2)
    entries=plan.development_tasks() if a.cohort=='development' else plan.load_selection()
    if not 0<=a.task_index<len(entries):raise ValueError('Invalid task index')
    t=entries[a.task_index]
    if t.get('alias_of') is not None:
        print(json.dumps(dict(index=t['index'],alias_of=t['alias_of'],reason='Identical training setting, reuse without treating as independent replication')),flush=True);return 0
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    cfg,study=task_config(t);directory=task_directory(t);stop=[False]
    def request_stop(signum,frame):stop[0]=True;print(f'Signal {signum}: checkpointing',flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,request_stop)
    done=training.train(cfg,study,directory,torch.device('cuda'),resume=a.resume,stop_requested=lambda:stop[0])
    if done!=cfg.steps:return 3
    saved=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=training.base.RegressionTransformer(cfg).to('cuda');model.load_state_dict(saved['model'])
    training.evaluate(dataclasses.replace(cfg,eval_seed=plan.BANKS[a.cohort][1]),study,saved['identity'],
        model,done,saved['training_seconds'],directory,torch.device('cuda'),final_holdout=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
