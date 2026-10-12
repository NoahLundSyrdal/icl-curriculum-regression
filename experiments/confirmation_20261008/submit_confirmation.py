#!/usr/bin/env python3
"""Idempotence guard, source freeze, and fail-closed Slurm dependency pipeline."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import confirmation_plan as plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    if args.preflight_only:
        if plan.SUBMISSION.exists() or plan.FREEZE.exists() or Path('results').exists():raise FileExistsError('Existing submission/results; inspect before retry')
        if json.loads(plan.PLAN_FILE.read_text())!=plan.frozen_plan():raise ValueError('Plan differs')
        hashes={name:plan.digest(name) for name in plan.FILES}
        if hashes['run_experiment.py']!=plan.BASE_SHA or hashes['robustness_experiment.py']!=plan.TRAINER_SHA:raise ValueError('Original training sources changed')
        now=datetime.datetime.now(datetime.timezone.utc).isoformat()
        plan.atomic_json(plan.FREEZE,dict(kind=plan.KIND,created_utc=now,source_hashes=hashes))
        state=dict(kind=plan.KIND,created_utc=now,status='submitting_preflight',source_hashes=hashes,
            remote_root=str(Path.cwd()),development_runs=36,confirmation_runs=60,development_seeds=list(plan.DEV_SEEDS),
            confirmation_seeds=list(plan.CONFIRM_SEEDS),max_concurrent_gpus=15,account='share-ie-idi',qos='high',
            primary='curriculum_vs_tuned_hard',gated_secondary='curriculum_vs_matched_shuffled',
            estimated_training_gpu_hours=[664,767],contact_srikar=False)
    else:
        plan.verify_sources();state=json.loads(plan.SUBMISSION.read_text())
        if state['status']!='preflight_submitted' or any(k in state for k in ('gpu_smoke_job','development_array_job','selection_job','confirmation_array_job','summary_job')):raise ValueError('Already/partially submitted; inspect saved IDs')
        observed=subprocess.check_output(['sacct','-j',state['preflight_job'],'-X','-n','-P','--format=State,ExitCode'],text=True).strip()
        if observed!='COMPLETED|0:0':raise ValueError('CPU preflight not successfully complete: '+observed)
        report=json.loads(Path('submissions/cpu-preflight.json').read_text())
        if not report['scientific_tests_passed'] or not report['all_96_task_batch_probes_passed'] or report['freeze_sha256']!=plan.digest(plan.FREEZE):raise ValueError('CPU checks missing/incompatible')
        state['status']='submitting_remaining'
    Path('logs').mkdir(exist_ok=True);plan.atomic_json(plan.SUBMISSION,state)
    def submit(field,script,dependency=None):
        command=['sbatch','--parsable','--account=share-ie-idi','--qos=high']
        if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        command.append(script)
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        if not job.isdigit():raise ValueError('Unknown submission outcome; inspect Slurm before retry')
        state[field]=job;plan.atomic_json(plan.SUBMISSION,state);print(field+'='+job,flush=True)
        return job
    if args.preflight_only:
        submit('preflight_job','confirmation_preflight.slurm');state['status']='preflight_submitted';plan.atomic_json(plan.SUBMISSION,state);return
    smoke=submit('gpu_smoke_job','confirmation_smoke.slurm',state['preflight_job'])
    dev=submit('development_array_job','confirmation_dev.slurm',smoke)
    selection=submit('selection_job','confirmation_select.slurm',dev)
    confirm=submit('confirmation_array_job','confirmation_train.slurm',selection)
    submit('summary_job','confirmation_summary.slurm',confirm)
    state['status']='pipeline_submitted';plan.atomic_json(plan.SUBMISSION,state)


if __name__=='__main__':main()
