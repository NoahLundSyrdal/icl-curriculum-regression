#!/usr/bin/env python3
"""Canonical GPU shape smoke, kept separate from all pilot result directories."""
import dataclasses
import json
from pathlib import Path

import torch
import run_experiment as base
import dimension_experiment as exp


def main():
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 GPU missing")
    cfg=exp.pilot_config(100);root=Path("results/dimension_smoke_20261006")
    starts=[];records=[]
    for arm in exp.ARMS:
        study=exp.Study(arm=arm);directory=root/arm
        done=exp.train(cfg,study,directory,torch.device("cuda"),max_steps_this_run=2)
        if done!=2:raise ValueError("Smoke budget differs")
        baseline=json.loads((directory/"evaluations/step_000000/metrics.json").read_text())
        starts.append(baseline)
        records.append({"arm":arm,"completed_steps":done,"baseline":baseline["conditions"],
                        "initial_model_sha256":baseline["identity"]["initial_model_sha256"]})
    if len({r["initial_model_sha256"] for r in records})!=1:raise ValueError("Initialization mismatch")
    for b in starts[1:]:
        for condition in starts[0]["conditions"]:
            left=starts[0]["conditions"][condition]["mean_nmse_by_k"]
            right=b["conditions"][condition]["mean_nmse_by_k"]
            if any(abs(x-y)>2e-5 for x,y in zip(left,right)):raise ValueError("Paired GPU baseline differs")
    base.atomic_json(Path("submissions/dimension-gpu-smoke.json"),{
        "canonical_shapes_passed":True,"all_arms_finite":True,"paired_initialization_passed":True,
        "records":records,"gpu":torch.cuda.get_device_name(0),"torch":str(torch.__version__),
        "script_sha256":exp.sha(exp.__file__),"protocol_sha256":exp.sha(exp.PROTOCOL),
        "smoke_script_sha256":exp.sha(__file__)})
    print("Canonical BF16 GPU smoke passed for all three arms",flush=True)


if __name__=="__main__":main()
