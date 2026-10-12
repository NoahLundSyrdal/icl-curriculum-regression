#!/usr/bin/env python3
"""Train a clean-only control with the unchanged original trainer.

The 10k snapshot is retained before resuming to 20k. Evaluation is isolated
from training RNG, and resume must retain the same protocol and source hashes.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import shutil
import signal
import sys

import torch
import run_experiment as base
import continue_experiment as tail

KIND = "clean_only_followup_v1"


def train_control(cfg, run_dir, device, milestones, provenance, *, resume=False,
                  stop_requested=lambda: False, max_steps_this_run=None):
    run_dir = Path(run_dir)
    milestones = tuple(milestones)
    if cfg.mode != "clean" or not milestones or milestones != tuple(sorted(set(milestones))) or milestones[-1] != cfg.steps or milestones[0] <= 0:
        raise ValueError("Clean configuration requires increasing fixed milestones ending at its budget")
    identity = {"kind": KIND, "config": dataclasses.asdict(cfg),
                "milestones": list(milestones), "provenance": provenance,
                "base_script_sha256": tail.file_sha256(base.__file__),
                "control_script_sha256": tail.file_sha256(__file__)}
    progress = run_dir / "training"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text()) != identity:
            raise ValueError("Existing control requires resume with identical source/protocol/code")
    elif resume or any(run_dir.iterdir()):
        raise ValueError("Missing manifest or nonempty control directory")
    else:
        base.atomic_json(manifest, identity)
    checkpoint = progress / "checkpoint.pt"
    completed = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != dataclasses.asdict(cfg):
            raise ValueError("Control checkpoint configuration differs")
        completed = int(saved["completed_steps"])
    elif resume:
        # A process may have exited after the manifest write, before step 1.
        if progress.exists() and any(progress.iterdir()):
            raise ValueError("Incomplete training initialization; inspect before retry")
    if not 0 <= completed <= cfg.steps:
        raise ValueError("Invalid checkpoint progress")
    started_at = completed
    for milestone in milestones:
        snapshot = run_dir / f"step_{milestone:06d}"
        if completed > milestone:
            if not (snapshot / "checkpoint.pt").exists() or not (snapshot / "metrics.json").exists():
                raise ValueError("Cannot reconstruct missing earlier fixed snapshot")
            continue
        limit = milestone - completed
        if max_steps_this_run is not None:
            limit = min(limit, max_steps_this_run - (completed - started_at))
        if limit < 0:
            raise ValueError("Invalid interruption budget")
        # Loading through base.train recreates the model and restores AdamW and RNG.
        # A completed milestone with interrupted evaluation needs no new update.
        if limit == 0 and completed < milestone:
            return completed
        if completed == milestone:
            base.seed_model(cfg.seed)
            model = base.RegressionTransformer(cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                         weight_decay=cfg.weight_decay)
            stream = base.EpisodeStream(cfg.seed)
            base.load_checkpoint(checkpoint, cfg, model, optimizer, stream,
                                 device, base.sigma_schedule(cfg))
        else:
            model, completed = base.train(cfg, progress, device, resume=checkpoint.exists(),
                max_steps_this_run=limit, stop_requested=stop_requested)
        if completed == milestone:
            snapshot.mkdir(parents=True, exist_ok=True)
            frozen = snapshot / "checkpoint.pt"
            if not frozen.exists():
                tmp = frozen.with_suffix(".pt.tmp")
                shutil.copyfile(checkpoint, tmp)
                tmp.replace(frozen)
            if not (snapshot / "metrics.json").exists():
                # Save/restore global RNG defensively; separate evaluation streams
                # cannot alter the episode RNG stored in the frozen checkpoint.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                try:
                    metrics = base.evaluate(dataclasses.replace(cfg, steps=milestone), model, snapshot, device)
                    metrics.update(control_kind=KIND, control_identity=identity,
                                   control_checkpoint_sha256=tail.file_sha256(frozen))
                    base.atomic_json(snapshot / "metrics.json", metrics)
                finally:
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng:
                        torch.cuda.set_rng_state_all(cuda_rng)
        if completed < milestone or (stop_requested() and completed < cfg.steps) or (max_steps_this_run is not None and completed - started_at >= max_steps_this_run):
            return completed
    return completed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=range(5), required=True)
    p.add_argument("--root", type=Path, default=Path("results/followup_20261006"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full control requires the verified BF16 CUDA environment")
    source = Path("results/full") / f"seed_{args.seed}" / "shuffled"
    original, provenance = tail.inspect_source(source)
    cfg = dataclasses.replace(original, mode="clean", steps=20000)
    provenance["protocol_sha256"] = tail.file_sha256("FOLLOWUP_PROTOCOL.txt")
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
        print(f"Signal {signum}: saving after this optimizer step", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    completed = train_control(cfg, args.root / "clean" / f"seed_{args.seed}",
        torch.device("cuda"), (10000, 20000), provenance, resume=args.resume,
        stop_requested=lambda: stop[0])
    print(f"Control seed {args.seed}: saved {completed}/20000 steps", flush=True)
    return 0 if completed == 20000 else 3


if __name__ == "__main__":
    raise SystemExit(main())
