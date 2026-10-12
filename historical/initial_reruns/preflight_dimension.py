#!/usr/bin/env python3
"""Audit canonical production schedules, data identity and shared evaluation."""
import dataclasses
import hashlib
from pathlib import Path

import torch
import run_experiment as base
import dimension_experiment as exp


def fingerprint(xs,y):
    h=hashlib.sha256();h.update(xs.numpy().tobytes());h.update(y.numpy().tobytes());return h.hexdigest()


def main():
    torch.set_num_threads(2)
    if exp.sha(base.__file__)!=exp.BASE_SHA:raise ValueError("Base trainer differs")
    records=[]
    for seed in (100,101):
        cfg=exp.pilot_config(seed);c=exp.Study();s=dataclasses.replace(c,arm="shuffled");hard=dataclasses.replace(c,arm="hard")
        exp.validate(cfg,c);left,right=exp.batch_order(cfg,c),exp.batch_order(cfg,s)
        if not torch.equal(left[:3000].sort().values,right[:3000].sort().values) or not torch.equal(left[3000:],right[3000:]):
            raise ValueError("Training-batch schedule is not matched")
        dims=torch.tensor([exp.canonical_dims(cfg,c,i) for i in left.tolist()])
        shuffled_dims=torch.tensor([exp.canonical_dims(cfg,s,i) for i in right.tolist()])
        if not torch.equal(dims.sort().values,shuffled_dims.sort().values):raise ValueError("Difficulty histograms differ")
        fingerprints={}
        for i in (0,999,1000,1500,2999,3000,5999):
            a,b=exp.batch(cfg,c,i),exp.batch(cfg,s,i)
            if not torch.equal(a[0],b[0]) or not torch.equal(a[1],b[1]) or a[2:]!=b[2:]:
                raise ValueError("Batch contents depend on order/arm")
            if i>=3000:
                h=exp.batch(cfg,hard,i)
                if not torch.equal(a[0],h[0]) or not torch.equal(a[1],h[1]):raise ValueError("Common final batches differ")
            fingerprints[str(i)]={"sha256":fingerprint(a[0],a[1]),"active_dims":a[2],"eligible_positions":a[3]}
        records.append({"seed":seed,"matched_prefix_batch_ids":True,"common_terminal_batches":True,
            "active_dim_sum":int(dims.sum()),"eligible_position_sum":int((2*dims+1).sum()),
            "physically_processed_positions":cfg.steps*cfg.batch_size*cfg.points,
            "canonical_batch_checks":fingerprints})
        print(f"seed={seed}: matched production schedules and representative batches passed",flush=True)
    base.atomic_json(Path("submissions/dimension-preflight.json"),{
        "records":records,"source_sha256":exp.sha(exp.__file__),"base_sha256":exp.sha(base.__file__),
        "protocol_sha256":exp.sha(exp.PROTOCOL),"torch":str(torch.__version__),
        "transformers":base.transformers.__version__,"validation_script_sha256":exp.sha(__file__)})


if __name__=="__main__":main()
