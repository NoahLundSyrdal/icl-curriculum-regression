#!/usr/bin/env python3
"""Submit once: CPU checks -> GPU smoke -> six-arm pilot -> complete summary."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

FILES=("DIMENSION_PILOT_PROTOCOL.txt","run_experiment.py","dimension_experiment.py",
       "test_dimension.py","preflight_dimension.py","dimension_smoke.py",
       "summarize_dimension.py","dimension_preflight.slurm","dimension_smoke.slurm",
       "dimension_pilot.slurm","dimension_summary.slurm","submit_dimension.py")


def main():
    path=Path("submissions/dimension-pilot-20261006.json")
    if path.exists():raise FileExistsError("Pilot already submitted or partially submitted; inspect manifest before retry")
    if Path("results/dimension_pilot_20261006").exists() or Path("results/dimension_smoke_20261006").exists():
        raise FileExistsError("Pilot/smoke result directory already exists")
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}
    if hashes["run_experiment.py"]!="4df516cd368733d00795d9a3cf16f4f114db06e156cad5a4cabe3a22ce0776b7":raise ValueError("Original trainer changed")
    data={"kind":"dimension_context_curriculum_pilot_v1","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "source_hashes":hashes,"seeds":[100,101],"arms":["curriculum","shuffled","hard"],
          "steps":6000,"prefix_steps":3000,"initial_easy_steps":1000,"target_dimension":50,
          "primary":"curriculum-minus-shuffled clean target error trapezoidal mean k50..100, steps3000..6000",
          "status":"submitting","remote_root":str(Path.cwd()),"account":"share-ie-idi","qos":"high"}
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
    checks=submit("preflight_job","dimension_preflight.slurm")
    smoke=submit("gpu_smoke_job","dimension_smoke.slurm",[checks])
    pilot=submit("pilot_array_job","dimension_pilot.slurm",[smoke])
    submit("summary_job","dimension_summary.slurm",[pilot])
    data["status"]="submitted";save()


if __name__=="__main__":main()
