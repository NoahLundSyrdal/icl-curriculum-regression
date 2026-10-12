#!/usr/bin/env python3
"""Matched-batch difficulty curricula, equal shapes and shared terminal phase."""
import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import sys
import time

import torch
import run_experiment as base

KIND = "dimension_context_curriculum_long_v1"
SEEDS = (200, 201, 202, 203, 204)
PRIMARY_TIMES = (25000, 30000, 40000, 50000, 75000, 100000, 125000)
FINAL_HOLDOUT_SEED = 273821
ARMS = ("curriculum", "shuffled", "hard")
PROTOCOL = Path("DIMENSION_LONG_PROTOCOL.txt")
BASE_SHA = "4df516cd368733d00795d9a3cf16f4f114db06e156cad5a4cabe3a22ce0776b7"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class Study:
    arm: str = "curriculum"
    initial_dims: int = 5
    easy_steps: int = 1000
    prefix_steps: int = 25000
    evaluation_steps: tuple = (0, 1000, 5000, 10000, 20000, 25000, 30000, 40000, 50000, 75000, 100000, 125000)
    checkpoint_every: int = 1000
    log_every: int = 100


def study_dict(study):
    d = dataclasses.asdict(study)
    d["evaluation_steps"] = list(study.evaluation_steps)
    return d


def validate(cfg, study):
    if study.arm not in ARMS or cfg.mode != "clean" or cfg.points != 2*cfg.dims+1:
        raise ValueError("Invalid arm or physical model shape")
    if not 1 <= study.initial_dims <= cfg.dims or not 0 <= study.easy_steps < study.prefix_steps-1 < cfg.steps-1:
        raise ValueError("Invalid curriculum phases")
    e = tuple(study.evaluation_steps)
    if e != tuple(sorted(set(e))) or e[0] != 0 or e[-1] != cfg.steps or study.prefix_steps not in e:
        raise ValueError("Evaluations must include baseline, boundary and final update")
    if any(k < 0 or k > cfg.steps for k in e) or study.checkpoint_every <= 0 or study.log_every <= 0:
        raise ValueError("Invalid evaluation/checkpoint intervals")


def canonical_dims(cfg, study, batch_id):
    if not 0 <= batch_id < cfg.steps:
        raise ValueError("Batch ID outside fixed budget")
    if batch_id < study.easy_steps:
        return study.initial_dims
    if batch_id >= study.prefix_steps:
        return cfg.dims
    fraction = (batch_id-study.easy_steps)/(study.prefix_steps-study.easy_steps-1)
    return study.initial_dims + int((cfg.dims-study.initial_dims)*fraction)


def batch_order(cfg, study):
    ids = torch.arange(cfg.steps)
    if study.arm == "shuffled":
        g = torch.Generator().manual_seed(cfg.seed+9000003)
        ids[:study.prefix_steps] = torch.randperm(study.prefix_steps, generator=g)
    return ids


def generator(seed, batch_id, stream):
    payload = f"dimension_batch_v1|{seed}|{batch_id}|{stream}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63)-1)
    return torch.Generator().manual_seed(value)


def batch(cfg, study, batch_id):
    active = cfg.dims if study.arm == "hard" else canonical_dims(cfg, study, batch_id)
    weights = torch.randn(cfg.batch_size, cfg.dims, 1, generator=generator(cfg.seed,batch_id,"tasks"))
    xs = torch.randn(cfg.batch_size,cfg.points,cfg.dims,generator=generator(cfg.seed,batch_id,"inputs"))
    xs[:, :, active:] = 0
    clean = (xs @ weights).squeeze(-1)
    return xs, clean, active, 2*active+1


def identity(cfg, study):
    return {"kind": KIND, "config": dataclasses.asdict(cfg), "study": study_dict(study),
            "script_sha256": sha(__file__), "base_sha256": sha(base.__file__),
            "protocol_sha256": sha(PROTOCOL)}


def model_fingerprint(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def retain(directory, completed, study, cfg):
    if completed in study.evaluation_steps:
        frozen=Path(directory)/f"checkpoint_step_{completed:06d}.pt"
        if not frozen.exists():
            tmp=frozen.with_suffix(".pt.tmp")
            shutil.copyfile(Path(directory)/"checkpoint.pt",tmp);tmp.replace(frozen)


def capture_rng():
    return {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(states, device):
    torch.set_rng_state(states["torch_rng"].cpu())
    random.setstate(states["python_rng"])
    if device.type == "cuda" and states["cuda_rng"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in states["cuda_rng"]])


def save(path, ident, ids, completed, model, optimizer, seconds):
    base.atomic_torch_save(path, {"identity": ident, "batch_ids": ids,
        "completed_steps": completed, "training_seconds": seconds,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), **capture_rng()})


def load(path, ident, ids, cfg, model, optimizer, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["identity"] != ident or not torch.equal(saved["batch_ids"].cpu(),ids):
        raise ValueError("Resume identity/code/protocol/schedule differs")
    completed = int(saved["completed_steps"])
    if not 0 <= completed <= cfg.steps:
        raise ValueError("Invalid checkpoint progress")
    if any(int(s["step"]) != completed for s in saved["optimizer"]["state"].values()):
        raise ValueError("AdamW progress mismatch")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved, device)
    return completed, float(saved["training_seconds"])


def evaluation_bank(cfg):
    return base.EpisodeStream(cfg.eval_seed).draw(cfg.eval_episodes,cfg.points,cfg.dims)


def evaluate(cfg, study, ident, model, completed, seconds, directory, device, *, final_holdout=False):
    checkpoint_hash=sha(Path(directory)/"checkpoint.pt")
    directory = Path(directory)/("final_holdout" if final_holdout else "evaluations")/f"step_{completed:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    metric = directory/"metrics.json"
    if metric.exists():
        existing = json.loads(metric.read_text())
        if existing["identity"] != ident or existing["completed_steps"] != completed or existing["checkpoint_sha256"]!=checkpoint_hash or existing["eval_seed"]!=cfg.eval_seed or existing["eval_episodes"]!=cfg.eval_episodes:
            raise ValueError("Existing evaluation has incompatible identity")
        return existing
    rng, was_training = capture_rng(), model.training
    try:
        xs, clean, noise = evaluation_bank(cfg)
        conditions = {}
        for sigma in (0.0,1.0,2.0):
            losses = base.clean_target_losses(model,xs,clean,clean+sigma*noise,
                dims=cfg.dims,microbatch=cfg.eval_microbatch,device=device,precision=cfg.precision)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite target evaluation")
            conditions[f"gaussian_sd{sigma:g}"] = {
                "mean_nmse_by_k": losses.mean(0).tolist(),
                "late_context_nmse": float(losses[:,cfg.dims:].mean())}
        data = {"identity": ident,"completed_steps":completed,"training_seconds":seconds,
                "checkpoint_sha256":checkpoint_hash,
                "evaluation_dimension":cfg.dims,"evaluation_positions":cfg.points,
                "eval_seed":cfg.eval_seed,"eval_episodes":cfg.eval_episodes,
                "normalization":"clean-target squared error / full target dimension",
                "k_start":cfg.dims,"k_end":cfg.points-1,"conditions":conditions,
                "zero_predictor_late_nmse":float(clean[:,cfg.dims:].square().mean()/cfg.dims),
                "evaluation_split":"final_holdout" if final_holdout else "primary",
                "runtime":{"torch":str(torch.__version__),"transformers":base.transformers.__version__,
                    "python":sys.version,"device":str(device),
                    "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None}}
        base.atomic_json(metric,data)
        print(f"eval seed={cfg.seed} arm={study.arm} step={completed} full_d={cfg.dims} "
              f"clean_nmse={conditions['gaussian_sd0']['late_context_nmse']:.6g}",flush=True)
        return data
    finally:
        restore_rng(rng, device)
        model.train(was_training)


def train(cfg, study, directory, device, *, resume=False,
          max_steps_this_run=None, stop_requested=lambda:False):
    validate(cfg,study)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    ident=identity(cfg,study); checkpoint=directory/"checkpoint.pt"
    base.seed_model(cfg.seed)
    model=base.RegressionTransformer(cfg).to(device)
    ident["initial_model_sha256"]=model_fingerprint(model)
    manifest=directory/"manifest.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text())!=ident:
            raise ValueError("Existing run requires unchanged resume identity")
    elif resume or any(directory.iterdir()):
        raise ValueError("Nonempty output or missing resume identity")
    else:
        base.atomic_json(manifest,ident)
    if resume and not checkpoint.exists():
        raise FileNotFoundError("Resume requested but rolling checkpoint is missing")
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    ids=batch_order(cfg,study)
    completed,seconds=(load(checkpoint,ident,ids,cfg,model,optimizer,device) if checkpoint.exists() else (0,0.0))
    base.atomic_torch_save(directory/"schedule.pt",{
        "batch_ids":ids,"active_dims":torch.tensor([cfg.dims if study.arm=="hard" else canonical_dims(cfg,study,int(i)) for i in ids])})
    if not checkpoint.exists(): save(checkpoint,ident,ids,completed,model,optimizer,seconds)
    retain(directory,completed,study,cfg)
    if completed in study.evaluation_steps:
        evaluate(cfg,study,ident,model,completed,seconds,directory,device)
    if stop_requested() and completed<cfg.steps: return completed
    fields=("step","batch_id","active_dims","eligible_positions","train_mse","grad_norm","seconds_per_step","training_seconds")
    log=directory/"train.csv"
    if resume and log.exists():
        with log.open(newline="") as f: rows=[r for r in csv.DictReader(f) if int(r["step"])<=completed]
        with log.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    start_step=completed;started=time.monotonic();model.train()
    with log.open("a",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if f.tell()==0:writer.writeheader()
        for step in range(completed,cfg.steps):
            step_start=time.monotonic();batch_id=int(ids[step])
            xs,clean,active,eligible=batch(cfg,study,batch_id)
            optimizer.zero_grad(set_to_none=True);loss_sum=0.0
            accumulation=cfg.batch_size//cfg.microbatch
            for start in range(0,cfg.batch_size,cfg.microbatch):
                sl=slice(start,start+cfg.microbatch)
                xb,yb=xs[sl].to(device),clean[sl].to(device)
                with base.autocast_context(device,cfg.precision):
                    prediction=model(xb,yb)
                    loss=(prediction.float()[:,:eligible]-yb[:,:eligible]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
                (loss/accumulation).backward();loss_sum+=float(loss.detach())/accumulation
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm,error_if_nonfinite=True))
            optimizer.step();completed=step+1;duration=time.monotonic()-step_start;seconds+=duration
            if completed==1 or completed%study.log_every==0 or completed==cfg.steps:
                writer.writerow(dict(zip(fields,(completed,batch_id,active,eligible,loss_sum,norm,duration,seconds))));f.flush()
                rate=(time.monotonic()-started)/(completed-start_step)
                print(f"arm={study.arm} seed={cfg.seed} step={completed}/{cfg.steps} "
                      f"active_d={active} eligible={eligible} mse={loss_sum:.5g} "
                      f"s/step={rate:.3f} eta_hours={(cfg.steps-completed)*rate/3600:.2f}",flush=True)
            interrupted=stop_requested() or (max_steps_this_run is not None and completed-start_step>=max_steps_this_run)
            evaluation_due=completed in study.evaluation_steps
            if completed%study.checkpoint_every==0 or evaluation_due or interrupted:
                save(checkpoint,ident,ids,completed,model,optimizer,seconds)
            retain(directory,completed,study,cfg)
            if evaluation_due:evaluate(cfg,study,ident,model,completed,seconds,directory,device)
            if (interrupted or stop_requested()) and completed<cfg.steps:return completed
    return completed


def long_config(seed):
    return base.preset_config("full",mode="clean",seed=seed,dims=50,points=101,
        steps=125000,batch_size=256,microbatch=128,eval_seed=273819,eval_episodes=4096,
        eval_microbatch=64,precision="bf16",bayes_reference=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--output",type=Path,default=Path("results/dimension_long_20261006"))
    p.add_argument("--resume",action="store_true")
    args=p.parse_args();torch.set_num_threads(2)
    if sha(base.__file__)!=BASE_SHA:raise ValueError("Original trainer source changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("Verified BF16 CUDA environment required")
    cfg=long_config(args.seed);study=Study(arm=args.arm)
    stop=[False]
    def request_stop(signum,frame):
        stop[0]=True;print(f"Signal {signum}: saving after this update",flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,request_stop)
    done=train(cfg,study,args.output/f"seed_{cfg.seed}"/args.arm,torch.device("cuda"),resume=args.resume,stop_requested=lambda:stop[0])
    if done==cfg.steps:
        directory=args.output/f"seed_{cfg.seed}"/args.arm
        saved=torch.load(directory/"checkpoint.pt",map_location="cpu",weights_only=True)
        model=base.RegressionTransformer(cfg).to("cuda")
        model.load_state_dict(saved["model"])
        evaluate(dataclasses.replace(cfg,eval_seed=FINAL_HOLDOUT_SEED),study,saved["identity"],
                 model,done,saved["training_seconds"],directory,torch.device("cuda"),final_holdout=True)
        del saved,model
    return 0 if done==cfg.steps else 3


if __name__=="__main__":raise SystemExit(main())
