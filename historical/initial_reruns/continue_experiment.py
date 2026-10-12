#!/usr/bin/env python3
"""Continue completed paired models under two fixed, preregistered noise tails.

Original checkpoints/configuration/schedules remain immutable. Each original
history branches to constant SD=1 (high) or a seeded permutation of SDs from
0 through 1 (mixed). Comparisons between histories within a tail share future
episodes and noise draws; the two tail regimes have different noise budgets.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import signal
import sys
import time

import run_experiment as base
import torch

SCHEMA_VERSION = 1
KIND = "paired_fixed_budget_continuation"
HISTORIES = ("curriculum", "shuffled")
TAILS = ("high", "mixed")


@dataclasses.dataclass(frozen=True)
class Continuation:
    tail: str = "high"
    steps: int = 10000
    eval_steps: tuple[int, ...] = (0, 1000, 2500, 5000, 10000)
    checkpoint_every: int = 100
    log_every: int = 25


def continuation_config(tail="high", steps=10000, eval_steps=None,
                        checkpoint_every=100, log_every=25):
    if eval_steps is None:
        eval_steps = sorted({0, steps} | {k for k in (1000, 2500, 5000, 10000) if k <= steps})
    cfg = Continuation(tail, steps, tuple(eval_steps), checkpoint_every, log_every)
    if tail not in TAILS or steps <= 0 or checkpoint_every <= 0 or log_every <= 0:
        raise ValueError("Invalid continuation tail or nonpositive budget/interval")
    if (not cfg.eval_steps or cfg.eval_steps != tuple(sorted(set(cfg.eval_steps))) or cfg.eval_steps[0] != 0
            or cfg.eval_steps[-1] != steps or any(k < 0 or k > steps for k in cfg.eval_steps)):
        raise ValueError("Evaluation steps must be unique, increasing, and include 0 and the final step")
    return cfg


def continuation_dict(cfg):
    result = dataclasses.asdict(cfg)
    result["eval_steps"] = list(cfg.eval_steps)
    return result


def tail_schedule(original, continuation):
    if continuation.tail == "high":
        return torch.ones(continuation.steps, dtype=torch.float64)
    evenly_spaced = torch.linspace(0, 1, continuation.steps, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(original.seed + 7000003)
    return evenly_spaced[torch.randperm(continuation.steps, generator=generator)]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def budget_stats(sigmas):
    result = {"count": len(sigmas), "sigma_sum": float(sigmas.sum()),
              "sigma_squared_sum": float(sigmas.square().sum())}
    result["sigma_squared_mean"] = float(sigmas.square().mean()) if len(sigmas) else 0.0
    return result


def planned_tail_budget(sigmas):
    result = budget_stats(sigmas)
    result["terminal_windows"] = {
        f"last_{count}": budget_stats(sigmas[-count:]) for count in (250, 1000)
    }
    return result


def inspect_source(source_dir, *, strict=True):
    """Validate a trusted completed source; tiny sources are allowed only internally."""
    source_dir = Path(source_dir)
    checkpoint = source_dir / "checkpoint.pt"
    checkpoint_sha = file_sha256(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    original = base.preset_config(saved["config"]["preset"], **saved["config"])
    if saved.get("schema_version") != base.SCHEMA_VERSION:
        raise ValueError("Original checkpoint schema differs")
    if saved.get("completed_steps") != original.steps:
        raise ValueError("Source checkpoint must have completed its entire original budget")
    if not torch.equal(saved["schedule"].cpu(), base.sigma_schedule(original)):
        raise ValueError("Source checkpoint's original schedule differs")
    if original.mode not in HISTORIES:
        raise ValueError("Source history must be curriculum or shuffled")
    if not saved["optimizer"]["state"]:
        raise ValueError("Source has no trained AdamW state")
    for group in saved["optimizer"]["param_groups"]:
        if group["lr"] != original.learning_rate or group["weight_decay"] != original.weight_decay:
            raise ValueError("Source optimizer differs from original configuration")
    if any(int(state["step"]) != original.steps for state in saved["optimizer"]["state"].values()):
        raise ValueError("Source AdamW progress differs from the completed original budget")
    config_path, metrics_path = source_dir / "config.json", source_dir / "metrics.json"
    if config_path.exists() and json.loads(config_path.read_text()) != saved["config"]:
        raise ValueError("Source config.json and checkpoint configuration differ")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    source_script_sha = file_sha256(Path(base.__file__))
    if metrics is not None:
        if metrics["config"] != saved["config"] or metrics["completed_steps"] != original.steps:
            raise ValueError("Source metrics and checkpoint configuration/progress differ")
        if metrics["runtime"]["script_sha256"] != source_script_sha:
            raise ValueError("Original source implementation differs from its recorded script hash")
    if strict:
        canonical = base.preset_config("full", seed=original.seed, mode=original.mode, precision="bf16")
        if original != canonical or original.seed not in range(5):
            raise ValueError("CLI requires an unmodified canonical full source from seeds 0..4")
        if not config_path.exists() or metrics is None:
            raise ValueError("CLI source requires completed config.json and metrics.json")
        runtime = metrics["runtime"]
        if (str(torch.__version__) != runtime["torch"] or
                base.transformers.__version__ != runtime["transformers"] or
                sys.version != runtime["python"]):
            raise ValueError("Continuation runtime differs from the original recorded runtime")
        if not saved["cuda_rng"]:
            raise ValueError("Canonical GPU source is missing CUDA RNG state")
        if metrics["primary"]["name"] != "gaussian_sd2_nmse_k20_40":
            raise ValueError("Source primary endpoint differs")
    if file_sha256(checkpoint) != checkpoint_sha:
        raise ValueError("Source checkpoint changed during validation")
    provenance = {
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": file_sha256(config_path) if config_path.exists() else None,
        "metrics_sha256": file_sha256(metrics_path) if metrics_path.exists() else None,
        "source_script_sha256": source_script_sha,
        "original_completed_steps": original.steps,
        "original_mode": original.mode, "original_seed": original.seed,
        "original_primary": metrics["primary"] if metrics else None,
        "source_runtime": metrics["runtime"] if metrics else None,
    }
    return original, provenance


def capture_rng(stream):
    return {"episode_rng": stream.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate()}


def restore_rng(states, stream, device):
    stream.load_state_dict(states["episode_rng"])
    torch.set_rng_state(states["torch_rng"].cpu())
    random.setstate(states["python_rng"])
    if device.type == "cuda" and states["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in states["cuda_rng"]])


def initialize_source(source_dir, device, *, strict=True):
    original, provenance = inspect_source(source_dir, strict=strict)
    base.seed_model(original.seed)
    model = base.RegressionTransformer(original).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=original.learning_rate,
                                 weight_decay=original.weight_decay)
    stream = base.EpisodeStream(original.seed)
    completed = base.load_checkpoint(Path(source_dir) / "checkpoint.pt", original,
                                     model, optimizer, stream, device, base.sigma_schedule(original))
    if completed != original.steps:
        raise ValueError("Original source changed during initialization")
    model.train()
    return original, provenance, model, optimizer, stream


def save_tail_checkpoint(path, original, continuation, provenance, model, optimizer,
                         stream, completed, schedule):
    base.atomic_torch_save(path, {
        "schema_version": SCHEMA_VERSION, "kind": KIND,
        "config": dataclasses.asdict(original),
        "continuation": continuation_dict(continuation), "source": provenance,
        "continuation_script_sha256": file_sha256(__file__),
        "original_schedule": base.sigma_schedule(original), "tail_schedule": schedule,
        "tail_completed_steps": completed, "total_completed_steps": original.steps + completed,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        **capture_rng(stream),
    })


def load_tail_checkpoint(path, original, continuation, provenance, model, optimizer,
                         stream, device, schedule):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("schema_version") != SCHEMA_VERSION or saved.get("kind") != KIND:
        raise ValueError("Continuation checkpoint schema/kind differs")
    if saved["config"] != dataclasses.asdict(original):
        raise ValueError("Resume original configuration differs")
    if saved["continuation"] != continuation_dict(continuation):
        raise ValueError("Resume continuation configuration differs")
    if saved["source"] != provenance:
        raise ValueError("Resume source provenance differs")
    if saved["continuation_script_sha256"] != file_sha256(__file__):
        raise ValueError("Resume continuation implementation differs")
    if (not torch.equal(saved["original_schedule"].cpu(), base.sigma_schedule(original)) or
            not torch.equal(saved["tail_schedule"].cpu(), schedule)):
        raise ValueError("Resume original or continuation schedule differs")
    completed = int(saved["tail_completed_steps"])
    if not 0 <= completed <= continuation.steps or saved["total_completed_steps"] != original.steps + completed:
        raise ValueError("Resume progress counters are invalid")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved, stream, device)
    return completed


def evaluation_path(run_dir, completed):
    return Path(run_dir) / "evaluations" / f"step_{completed:06d}"


def evaluate_tail(original, continuation, provenance, model, stream, completed,
                  run_dir, device):
    """Evaluation is restartable and cannot advance training RNG or leave eval mode."""
    directory = evaluation_path(run_dir, completed)
    directory.mkdir(parents=True, exist_ok=True)
    expected = {
        "config": dataclasses.asdict(original), "continuation": continuation_dict(continuation),
        "source": provenance, "tail_completed_steps": completed,
        "total_completed_steps": original.steps + completed,
    }
    if (directory / "metrics.json").exists():
        existing = json.loads((directory / "metrics.json").read_text())
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing evaluation metadata differs; refusing to reuse it")
        if (directory / "losses.pt").exists():
            return existing
    rng, was_training = capture_rng(stream), model.training
    started = time.monotonic()
    try:
        xs, clean_y, noises = base.evaluation_bank(original)
        losses, reference = {}, {}
        late_start = min(20, original.points - 1)
        for distribution, epsilon in noises.items():
            for sigma in base.NOISE_SDS:
                key = f"{distribution}_sd{sigma:g}"
                labels = clean_y + sigma * epsilon
                losses[key] = base.clean_target_losses(
                    model, xs, clean_y, labels, dims=original.dims,
                    microbatch=original.eval_microbatch, device=device,
                    precision=original.precision)
                if distribution == "gaussian" and original.bayes_reference:
                    reference[key] = base.gaussian_bayes_losses(
                        xs, clean_y, labels, sigma, dims=original.dims,
                        microbatch=original.eval_microbatch)
        conditions = {key: base.condition_summary(value, late_start)
                      for key, value in losses.items()}
        original_budget = budget_stats(base.sigma_schedule(original))
        schedule = tail_schedule(original, continuation)
        used_budget = budget_stats(schedule[:completed])
        metadata = {
            "schema_version": SCHEMA_VERSION, "experiment": KIND, **expected,
            "completed_steps": original.steps + completed,
            "normalization": "squared error / dims; w,x ~ N(0,I), so Var(y)=dims",
            "k_definition": "number of preceding context labels available at x_k",
            "primary": {"name": f"gaussian_sd2_nmse_k{late_start}_{original.points - 1}",
                        "condition": "gaussian_sd2", "k_start": late_start,
                        "k_end": original.points - 1,
                        "value": conditions["gaussian_sd2"]["late_context_nmse"]},
            "conditions": conditions,
            "bayes_reference": {key: base.condition_summary(value, late_start)
                                for key, value in reference.items()},
            "noise_budget": {
                "source": original_budget, "tail_planned": planned_tail_budget(schedule),
                "tail_completed": used_budget,
                "combined_completed": {
                    "count": original_budget["count"] + used_budget["count"],
                    "sigma_sum": original_budget["sigma_sum"] + used_budget["sigma_sum"],
                    "sigma_squared_sum": original_budget["sigma_squared_sum"] + used_budget["sigma_squared_sum"],
                },
            },
            "runtime": {"torch": str(torch.__version__),
                        "transformers": base.transformers.__version__, "python": sys.version,
                        "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "script_sha256": file_sha256(__file__),
                        "base_script_sha256": file_sha256(base.__file__),
                        "eval_seconds": time.monotonic() - started},
        }
        base.atomic_torch_save(directory / "losses.pt", {
            "schema_version": SCHEMA_VERSION, **expected, "losses": losses,
            "bayes_losses": reference, "k": torch.arange(original.points),
            "eval_seed": original.eval_seed,
        })
        base.atomic_json(directory / "metrics.json", metadata)
        print(f"eval history={original.mode} tail={continuation.tail} seed={original.seed} "
              f"tail_step={completed} total_step={original.steps + completed} "
              f"primary_nmse={metadata['primary']['value']:.7g}", flush=True)
        return metadata
    finally:
        restore_rng(rng, stream, device)
        model.train(was_training)


def train_continuation(source_dir, continuation, run_dir, device, *, strict_source=True,
                       resume=False, max_steps_this_run=None, stop_requested=lambda: False,
                       expected_identity=None):
    source_dir, run_dir = Path(source_dir), Path(run_dir)
    source_resolved, output_resolved = source_dir.resolve(), run_dir.resolve()
    if (source_resolved == output_resolved or source_resolved in output_resolved.parents or
            output_resolved in source_resolved.parents):
        raise ValueError("Continuation output must not overlap its original source directory")
    continuation = continuation_config(**continuation_dict(continuation))
    if max_steps_this_run is not None and max_steps_this_run <= 0:
        raise ValueError("max_steps_this_run must be positive")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot resume: {checkpoint_path} missing")
    if not resume and any(run_dir.iterdir()):
        raise FileExistsError(f"Continuation directory contains files: {run_dir}; use --resume")
    original, provenance, model, optimizer, stream = initialize_source(
        source_dir, device, strict=strict_source)
    if expected_identity is not None and (original.seed, original.mode) != expected_identity:
        raise ValueError("Source checkpoint identity differs from the requested seed/history")
    schedule = tail_schedule(original, continuation)
    completed = (load_tail_checkpoint(checkpoint_path, original, continuation, provenance,
                                     model, optimizer, stream, device, schedule) if resume else 0)
    base.atomic_json(run_dir / "config.json", {
        "config": dataclasses.asdict(original), "continuation": continuation_dict(continuation),
        "source": provenance, "tail_noise_budget": planned_tail_budget(schedule),
    })
    base.atomic_torch_save(run_dir / "schedule.pt", schedule)
    if not resume:
        save_tail_checkpoint(checkpoint_path, original, continuation, provenance,
                             model, optimizer, stream, completed, schedule)
    if completed in continuation.eval_steps:
        evaluate_tail(original, continuation, provenance, model, stream, completed, run_dir, device)
    if stop_requested() and completed < continuation.steps:
        return model, completed
    log_path = run_dir / "train.csv"
    fields = ("tail_step", "total_step", "sigma", "train_mse", "train_nmse",
              "grad_norm", "seconds_per_step")
    if resume and log_path.exists():
        with log_path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["tail_step"]) <= completed]
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    start_step, started = completed, time.monotonic()
    model.train()
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        for step in range(completed, continuation.steps):
            step_started = time.monotonic()
            sigma = float(schedule[step])
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            accumulation = original.batch_size // original.microbatch
            for _ in range(accumulation):
                xs, clean_y, epsilon = stream.draw(original.microbatch, original.points, original.dims)
                context_y = clean_y + sigma * epsilon
                xs, clean_y, context_y = (tensor.to(device) for tensor in (xs, clean_y, context_y))
                with base.autocast_context(device, original.precision):
                    prediction = model(xs, context_y)
                    loss = (prediction.float() - clean_y).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite continuation loss at step {step + 1}")
                (loss / accumulation).backward()
                loss_sum += float(loss.detach()) / accumulation
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), original.max_grad_norm, error_if_nonfinite=True))
            optimizer.step()
            completed = step + 1
            duration = time.monotonic() - step_started
            if completed == 1 or completed % continuation.log_every == 0 or completed == continuation.steps:
                writer.writerow(dict(zip(fields, (completed, original.steps + completed, sigma,
                    loss_sum, loss_sum / original.dims, grad_norm, duration))))
                handle.flush()
                rate = (time.monotonic() - started) / (completed - start_step)
                print(f"history={original.mode} tail={continuation.tail} seed={original.seed} "
                      f"tail_step={completed}/{continuation.steps} total_step={original.steps + completed} "
                      f"sigma={sigma:.5g} nmse={loss_sum / original.dims:.6g} "
                      f"s/step={rate:.3f} eta_hours={(continuation.steps-completed)*rate/3600:.2f}", flush=True)
            interrupted = stop_requested() or (
                max_steps_this_run is not None and completed - start_step >= max_steps_this_run)
            evaluation_due = completed in continuation.eval_steps
            if (completed % continuation.checkpoint_every == 0 or evaluation_due or
                    completed == continuation.steps or interrupted):
                save_tail_checkpoint(checkpoint_path, original, continuation, provenance,
                                     model, optimizer, stream, completed, schedule)
            if evaluation_due:
                evaluate_tail(original, continuation, provenance, model, stream, completed, run_dir, device)
            if (interrupted or stop_requested()) and completed < continuation.steps:
                return model, completed
    return model, completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("results/full"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history", choices=HISTORIES, required=True)
    parser.add_argument("--tail", choices=TAILS, required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-steps", type=int, nargs="+")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps-this-run", type=int)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Canonical full continuation requires CUDA with bfloat16 support")
    continuation = continuation_config(args.tail, args.steps, args.eval_steps,
                                       args.checkpoint_every, args.log_every)
    source = args.source_root / f"seed_{args.seed}" / args.history
    root = args.output if args.output is not None else Path("results") / f"continuation_{continuation.steps}"
    run_dir = root / f"seed_{args.seed}" / args.history / args.tail
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
        print(f"Signal {signum}: checkpointing after the current completed step", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    print(f"Continuing {source} -> {run_dir}; tail={continuation.tail} "
          f"added_steps={continuation.steps} eval_steps={continuation.eval_steps}", flush=True)
    _, completed = train_continuation(source, continuation, run_dir, device, resume=args.resume,
                                     max_steps_this_run=args.max_steps_this_run,
                                     stop_requested=lambda: stop[0],
                                     expected_identity=(args.seed, args.history))
    if completed < continuation.steps:
        print(f"Saved {completed}/{continuation.steps} tail steps; rerun with --resume.", flush=True)
        return 3
    print(f"Complete: {evaluation_path(run_dir, completed) / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
