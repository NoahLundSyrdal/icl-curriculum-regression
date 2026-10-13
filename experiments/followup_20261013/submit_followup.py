"""Idempotent, source-frozen Slurm pipeline. Every stage fails closed."""
import argparse, json, subprocess
from pathlib import Path
import followup_plan as plan

def successful(job):
    lines=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=State,ExitCode'],text=True).strip().splitlines()
    return bool(lines) and all(line=='COMPLETED|0:0' for line in lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    Path('submissions').mkdir(exist_ok=True);Path('logs').mkdir(exist_ok=True)
    if a.preflight_only:
        if plan.FREEZE.exists() or plan.SUBMISSION.exists() or Path('results').exists():raise FileExistsError('Prior run/freeze exists; inspect before retry')
        if json.loads(plan.PLAN.read_text())!=plan.frozen_plan():raise ValueError('Plan changed')
        hashes={name:plan.digest(name) for name in plan.FILES}
        if hashes['run_experiment.py']!=plan.BASE_SHA or hashes['robustness_experiment.py']!=plan.TRAINER_SHA:raise ValueError('Upstream changed')
        plan.atomic_json(plan.FREEZE,dict(kind=plan.KIND,created_utc=plan.now(),source_hashes=hashes))
        state=dict(kind=plan.KIND,status='submitting_preflight',created_utc=plan.now(),remote_root=str(Path.cwd()),
            development_models=150,confirmation_slots=360,confirmation_unique_min=200,confirmation_unique_max=360,
            account='share-ie-idi',qos='high',max_concurrent_gpus=15,git_commit=subprocess.check_output(['cat','submissions/repository_commit.txt'],text=True).strip(),
            note='Five LRs x three fresh development seeds x ten strategies, all 125k. Aliases skip identical confirmation runs. No external messages.')
    else:
        plan.verify_sources();state=json.loads(plan.SUBMISSION.read_text())
        if state['status']!='preflight_submitted' or any(k in state for k in ('gpu_smoke_job','development_job','selection_job','confirmation_job','summary_job')):raise ValueError('Already/partially submitted; inspect saved IDs')
        if not successful(state['preflight_job']):raise ValueError('CPU preflight not complete successfully')
        report=json.loads(Path('submissions/cpu-preflight.json').read_text())
        if not report['tests_passed'] or not report['all_150_production_batch_probes_passed'] or report['freeze_sha256']!=plan.digest(plan.FREEZE):raise ValueError('Missing CPU evidence')
        state['status']='submitting_pipeline'
    plan.atomic_json(plan.SUBMISSION,state)
    def submit(key,script,dependency=None):
        command=['sbatch','--parsable','--account=share-ie-idi','--qos=high']
        if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        job=subprocess.check_output(command+[script],text=True).strip().split(';')[0]
        if not job.isdigit():raise ValueError('Unknown submission outcome; inspect Slurm')
        state[key]=job;plan.atomic_json(plan.SUBMISSION,state);print(key+'='+job,flush=True);return job
    if a.preflight_only:
        submit('preflight_job','followup_preflight.slurm');state['status']='preflight_submitted'
    else:
        smoke=submit('gpu_smoke_job','followup_smoke.slurm',state['preflight_job'])
        dev=submit('development_job','followup_dev.slurm',smoke)
        select=submit('selection_job','followup_select.slurm',dev)
        confirm=submit('confirmation_job','followup_confirm.slurm',select)
        submit('summary_job','followup_summary.slurm',confirm);state['status']='pipeline_submitted'
    plan.atomic_json(plan.SUBMISSION,state)
if __name__=='__main__':main()
