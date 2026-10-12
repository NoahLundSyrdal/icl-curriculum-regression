#!/usr/bin/env python3
"""Submit the fixed follow-up once, recording every successful submission."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

FILES = ("FOLLOWUP_PROTOCOL.txt", "run_experiment.py", "continue_experiment.py",
         "clean_control.py", "checkpoint_diagnostics.py", "summarize_followup.py",
         "test_followup.py", "followup_preflight.slurm", "followup_clean.slurm",
         "followup_diagnostics.slurm", "followup_summary.slurm", "submit_followup.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    path = Path("submissions/followup_20261006.json")
    hashes = {name: sha(name) for name in FILES}
    if path.exists():
        data = json.loads(path.read_text())
        if data["preflight_job"] != args.preflight or data["source_hashes"] != hashes:
            raise ValueError("Existing manifest has different jobs or source hashes")
        if any(name in data for name in ("clean_array_job", "diagnostic_array_job", "summary_job")):
            raise ValueError("GPU jobs already submitted; inspect manifest rather than resubmit")
    else:
        data = {"kind": "h1_followup_20261006", "preflight_job": args.preflight,
                "source_hashes": hashes, "account": "share-ie-idi", "qos": "high",
                "primary": "20k Gaussian SD2 NMSE k20..40: shuffled+mixed minus clean",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "remote_root": str(Path.cwd()), "state": "preflight_pending"}
    def save():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2)+"\n")
        temporary.replace(path)
    save()
    if args.record_only:
        print(json.dumps(data, indent=2))
        return
    status = subprocess.check_output(["sacct", "-j", args.preflight, "-n", "-P", "--format=JobID,State,ExitCode"], text=True)
    main_row = next(line.split("|") for line in status.splitlines() if line.split("|")[0] == args.preflight)
    if main_row[1:3] != ["COMPLETED", "0:0"]:
        raise ValueError("Preflight must have completed successfully before GPU submission")
    if not Path("results/followup_20261006/references.pt").exists():
        raise ValueError("Missing prepared reference bank")
    def submit(key, script, dependencies):
        job = subprocess.check_output(["sbatch", "--parsable", "--account=share-ie-idi", "--qos=high",
            "--dependency=afterok:"+":".join(dependencies), "--kill-on-invalid-dep=yes", script], text=True).strip().split(";")[0]
        if not job.isdigit():
            raise ValueError("Unexpected Slurm submission response; inspect before retry")
        data[key] = job
        save()
        print(f"{key}={job}", flush=True)
        return job
    clean = submit("clean_array_job", "followup_clean.slurm", [args.preflight])
    diagnostics = submit("diagnostic_array_job", "followup_diagnostics.slurm", [args.preflight])
    submit("summary_job", "followup_summary.slurm", [clean, diagnostics])
    data["state"] = "submitted"
    data["submitted_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
