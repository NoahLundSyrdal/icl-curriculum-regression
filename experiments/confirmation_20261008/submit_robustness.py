#!/usr/bin/env python3
"""Submit only after checks; preserve IDs immediately to prevent duplicate work."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

FILES=('ROBUSTNESS_PROTOCOL.txt','ROBUSTNESS_TASKS.json','run_experiment.py','robustness_experiment.py','test_robustness.py','summarize_robustness.py','preflight_robustness.py','robustness_smoke.py','robustness_preflight.slurm','robustness_smoke.slurm','robustness_train.slurm','robustness_summary.slurm','submit_robustness.py')
PATH=Path('submissions/robustness-20261007.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}
    if hashes['run_experiment.py']!='4df516cd368733d00795d9a3cf16f4f114db06e156cad5a4cabe3a22ce0776b7':raise ValueError('Original trainer changed')
    if args.preflight_only:
        if PATH.exists() or Path('results/curriculum_robustness_20261007').exists() or Path('results/robustness_smoke_20261007').exists():raise FileExistsError('Prior submission/output exists; inspect before retry')
        data=dict(kind='curriculum_robustness_components_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='submitting_preflight',source_hashes=hashes,seeds=[300,301],training_runs=44,max_concurrent_gpus=15,screen_steps=50000,component_steps=125000,primary_bank_seed=315831,holdout_bank_seed=315833,remote_root=str(Path.cwd()),account='share-ie-idi',qos='high',interpretation='Exploratory two-seed diagnostic study; all outcomes retained; no automatic confirmatory expansion')
    else:
        data=json.loads(PATH.read_text())
        if data['status']!='preflight_submitted' or any(k in data for k in ['gpu_smoke_job','training_array_job','summary_job']):raise ValueError('Already/partially submitted; inspect manifest')
        if data['source_hashes']!=hashes:raise ValueError('Frozen sources changed after preflight submission')
        state=subprocess.check_output(['sacct','-j',data['preflight_job'],'-X','-n','-P','--format=State,ExitCode'],text=True).strip()
        if state!='COMPLETED|0:0':raise ValueError('CPU preflight is not successfully completed: '+state)
        report=json.loads(Path('submissions/robustness-preflight.json').read_text())
        if not report['all_scientific_tests_passed'] or not report['all_44_production_batch_probes_passed']:raise ValueError('Checks missing')
        for field,file in [('runner_sha256','robustness_experiment.py'),('protocol_sha256','ROBUSTNESS_PROTOCOL.txt'),('tasks_sha256','ROBUSTNESS_TASKS.json')]:
            if report[field]!=hashes[file]:raise ValueError('Preflight report sources differ')
        data['status']='submitting_remaining'
    def save():
        PATH.parent.mkdir(exist_ok=True);tmp=PATH.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(PATH)
    def submit(key,script,deps=()):
        command=['sbatch','--parsable','--account=share-ie-idi','--qos=high']
        if deps:command+=['--dependency=afterok:'+':'.join(deps),'--kill-on-invalid-dep=yes']
        command.append(script);job=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not job.isdigit():raise ValueError('Unknown sbatch output; inspect before retry')
        data[key]=job;save();print(key+'='+job,flush=True);return job
    Path('logs').mkdir(exist_ok=True);save()
    if args.preflight_only:
        submit('preflight_job','robustness_preflight.slurm');data['status']='preflight_submitted';save();return
    smoke=submit('gpu_smoke_job','robustness_smoke.slurm',[data['preflight_job']])
    array=submit('training_array_job','robustness_train.slurm',[smoke])
    submit('summary_job','robustness_summary.slurm',[array]);data['status']='submitted';save()

if __name__=='__main__':main()
