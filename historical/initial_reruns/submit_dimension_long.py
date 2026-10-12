#!/usr/bin/env python3
"""Submit once: CPU checks -> GPU smoke -> six-arm pilot -> complete summary."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

FILES=("DIMENSION_LONG_PROTOCOL.txt","run_experiment.py","dimension_long_experiment.py",
       "test_dimension_long.py","preflight_dimension_long.py","dimension_long_smoke.py",
       "summarize_dimension_long.py","dimension_long_preflight.slurm","dimension_long_smoke.slurm",
       "dimension_long.slurm","dimension_long_summary.slurm","submit_dimension_long.py")


def main():
    path=Path("submissions/dimension-long-20261006.json")
    if path.exists():raise FileExistsError("Long study already submitted or partially submitted; inspect manifest before retry")
    if Path("results/dimension_long_20261006").exists() or Path("results/dimension_long_smoke_20261006").exists():
        raise FileExistsError("Long/smoke result directory already exists")
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}
    if hashes["run_experiment.py"]!="4df516cd368733d00795d9a3cf16f4f114db06e156cad5a4cabe3a22ce0776b7":raise ValueError("Original trainer changed")
    data={"kind":"dimension_context_curriculum_long_v1","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "source_hashes":hashes,"seeds":[200,201,202,203,204],"arms":["curriculum","shuffled","hard"],
          "steps":125000,"prefix_steps":25000,"initial_easy_steps":1000,"target_dimension":50,
          "primary":"curriculum-minus-shuffled clean target error trapezoidal mean k50..100, steps25000..125000",
          "eval_seed":273819,"eval_episodes":4096,"final_holdout_seed":273821,"fresh_training_seeds_only":True,"status":"submitting","remote_root":str(Path.cwd()),"account":"share-ie-idi","qos":"high"}
    def save():
        tmp=path.with_suffix(".json.tmp");tmp.write_text(json.dumps(data,indent=2)+"\n");tmp.replace(path)
    Path("logs").mkdir(exist_ok=True);path.parent.mkdir(exist_ok=True);save()
    def submit(key,script,deps=()):
        cmd=["sbatch","--parsable","--account=share-ie-idi","--qos=high"]
        if deps:cmd += ["--dependency=afterok:"+":".join(deps),"--kill-on-invalid-dep=yes"]
        cmd.append(script)
        job=subprocess.check_output(cmd,text=True).strip().split(";")[0]
        if not job.isdigit():raise ValueError("Unknown Slurm response; inspect before retry")
        data[key]=job;save();print(f"{key}={job}",flush=True);return job
    checks=submit("preflight_job","dimension_long_preflight.slurm")
    smoke=submit("gpu_smoke_job","dimension_long_smoke.slurm",[checks])
    pilot=submit("pilot_array_job","dimension_long.slurm",[smoke])
    submit("summary_job","dimension_long_summary.slurm",[pilot])
    data["status"]="submitted";save()


if __name__=="__main__":main()
