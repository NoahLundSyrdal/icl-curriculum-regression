#!/usr/bin/env python3
"""All-arm, all-time exploratory summary; no significance claims from two seeds."""
import csv
import dataclasses
import json
import math
from pathlib import Path
import statistics

import torch
import run_experiment as base
import dimension_experiment as exp

ROOT=Path("results/dimension_pilot_20261006")


def trapezoidal_mean(times, values):
    if len(times)!=len(values) or len(times)<2 or any(b<=a for a,b in zip(times,times[1:])) or any(not math.isfinite(v) for v in values):
        raise ValueError("AUC requires finite values at increasing times")
    return sum((b-a)*(x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))/(times[-1]-times[0])


def main():
    torch.set_num_threads(1);all_data={};rows=[];counts=0
    for seed in (100,101):
        hashes=[]
        for arm in exp.ARMS:
            cfg=exp.pilot_config(seed);study=exp.Study(arm=arm);directory=ROOT/f"seed_{seed}"/arm
            manifest=json.loads((directory/"manifest.json").read_text())
            expected=exp.identity(cfg,study)
            if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError("Run sources/settings differ")
            hashes.append(manifest["initial_model_sha256"])
            for t in study.evaluation_steps:
                m=json.loads((directory/"evaluations"/f"step_{t:06d}"/"metrics.json").read_text())
                if m["identity"]!=manifest or m["completed_steps"]!=t or m["evaluation_dimension"]!=50 or m["evaluation_positions"]!=101 or m["eval_seed"]!=173819 or m["eval_episodes"]!=1024:
                    raise ValueError("Evaluation identity, progress or target shape differs")
                if set(m["conditions"])!={"gaussian_sd0","gaussian_sd1","gaussian_sd2"}:raise ValueError("Missing condition")
                for condition,values in m["conditions"].items():
                    curve=values["mean_nmse_by_k"]
                    if len(curve)!=101 or any(not math.isfinite(x) for x in curve):raise ValueError("Invalid curve")
                    if not math.isclose(statistics.mean(curve[50:]),values["late_context_nmse"],abs_tol=2e-6,rel_tol=2e-6):raise ValueError("Endpoint/curve mismatch")
                    rows.append({"seed":seed,"arm":arm,"step":t,"condition":condition,"late_context_nmse":values["late_context_nmse"],"training_seconds":m["training_seconds"]})
                all_data[seed,arm,t]=m;counts+=1
            for t in (3000,6000):
                checkpoint=directory/f"checkpoint_step_{t:06d}.pt"
                if exp.sha(checkpoint)!=all_data[seed,arm,t]["checkpoint_sha256"]:raise ValueError("Frozen weights differ from evaluated weights")
                saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
                if saved["identity"]!=manifest or saved["completed_steps"]!=t or not torch.equal(saved["batch_ids"],exp.batch_order(cfg,study)):
                    raise ValueError("Checkpoint identity/schedule/progress differs")
                if not saved["optimizer"]["state"] or any(int(v["step"])!=t for v in saved["optimizer"]["state"].values()):raise ValueError("AdamW progress mismatch")
                del saved
        if len(set(hashes))!=1:raise ValueError("Arms have different initial parameters")
        for t in exp.Study().evaluation_steps:
            if t==0:
                reference=all_data[seed,"curriculum",0]
                for arm in ("shuffled","hard"):
                    for condition in reference["conditions"]:
                        left=reference["conditions"][condition]["mean_nmse_by_k"]
                        right=all_data[seed,arm,0]["conditions"][condition]["mean_nmse_by_k"]
                        if any(abs(a-b)>2e-5 for a,b in zip(left,right)):raise ValueError("GPU paired baseline differs")
    if counts!=48:raise ValueError("Incomplete cohort")
    times=[3000,4000,5000,6000];contrasts={};arm_scores={}
    for condition in ("gaussian_sd0","gaussian_sd1","gaussian_sd2"):
        scores={arm:[trapezoidal_mean(times,[all_data[seed,arm,t]["conditions"][condition]["late_context_nmse"] for t in times]) for seed in (100,101)] for arm in exp.ARMS}
        arm_scores[condition]=scores
        contrasts[condition]={other:{"differences_curriculum_minus_comparator":[a-b for a,b in zip(scores["curriculum"],scores[other])],
            "mean":statistics.mean(a-b for a,b in zip(scores["curriculum"],scores[other]))} for other in ("shuffled","hard")}
    result={"kind":exp.KIND,"status":"complete_exploratory_pilot","seeds":[100,101],"evaluation_records":48,
            "primary_condition":"gaussian_sd0","primary_metric":"trapezoidal_mean_target_nmse_k50_100_steps3000_6000",
            "primary_contrast":contrasts["gaussian_sd0"]["shuffled"],"auc_by_arm":arm_scores,"secondary_contrasts":contrasts,
            "final_by_arm":{condition:{arm:[all_data[seed,arm,6000]["conditions"][condition]["late_context_nmse"] for seed in (100,101)] for arm in exp.ARMS} for condition in arm_scores},
            "protocol_sha256":exp.sha(exp.PROTOCOL),"runner_sha256":exp.sha(exp.__file__),"summary_sha256":exp.sha(__file__),
            "interpretation":"Exploratory two-seed pilot; no confirmatory significance or publication claim."}
    base.atomic_json(ROOT/"summary.json",result)
    with (ROOT/"learning_curves.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=["Dimension/context curriculum pilot: all six arms complete.","Two fresh training seeds; exploratory, not confirmatory.",
           "Primary: full 50D clean target NMSE k50..100, trapezoidal mean during updates3000..6000.",
           "Negative curriculum-minus-comparator differences favor curriculum."]
    for arm,values in arm_scores["gaussian_sd0"].items():lines.append(f"{arm}: per-seed primary values={values}; mean={statistics.mean(values):.9g}")
    for other,values in contrasts["gaussian_sd0"].items():lines.append(f"curriculum-{other}: {values}")
    lines+= ["An ordering benefit requires curriculum to improve over the matched-batch shuffled arm.",
             "A benefit versus hard-only alone supports staged exposure, while ordering remains unresolved.",
             "All evaluations use the same full-difficulty task; easier training losses are not the endpoint.",
             "No new full confirmation or parameter grid is launched automatically."]
    (ROOT/"RESULTS.txt").write_text("\n".join(lines)+"\n");print("\n".join(lines),flush=True)


if __name__=="__main__":main()
