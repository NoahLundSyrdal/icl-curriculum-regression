#!/usr/bin/env python3
"""A paired, fixed-budget rerun of synthetic noisy-label linear regression.

This is a corrected experiment, not an exact historical replication. The GPT-2
architecture follows Garg et al. and the public robust-icl-curriculum repository:
https://github.com/neilpattanaik/robust-icl-curriculum
This implementation is independent; it does not import that repository.

Episodes have w,x ~ N(0,I), clean y=xw and context labels y+sigma*epsilon.
Training minimizes clean-target MSE at every x token, including k=0. The paper's
query-only wording and the repository's noisy-target loss differ from this choice.
The three noisy arms use an identical multiset of noise standard deviations.
They share initialization, base episodes and Gaussian epsilon at each train step.
The realized epsilon-weighted corruption energy need not be exactly identical.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
import transformers
from transformers import GPT2Config, GPT2Model

MODES = ("clean", "curriculum", "shuffled", "reverse")
NOISE_SDS = (0.0, 0.5, 1.0, 2.0)
SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class Config:
    preset: str = "smoke"
    mode: str = "curriculum"
    seed: int = 0
    dims: int = 4
    points: int = 9
    position_capacity: int = 101
    embedding: int = 32
    layers: int = 2
    heads: int = 4
    steps: int = 6
    batch_size: int = 8
    microbatch: int = 4
    learning_rate: float = 4e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    sigma_max: float = 1.0
    eval_seed: int = 81973
    eval_episodes: int = 32
    eval_microbatch: int = 32
    checkpoint_every: int = 100
    log_every: int = 25
    precision: str = "fp32"
    bayes_reference: bool = True


def preset_config(name: str, **overrides) -> Config:
    settings = {"smoke": {}, "pilot": {
        "dims": 20, "points": 41, "embedding": 256, "layers": 12,
        "heads": 8, "steps": 500, "batch_size": 128, "microbatch": 32,
        "eval_episodes": 1024, "eval_microbatch": 64,
    }, "full": {
        "dims": 20, "points": 41, "embedding": 256, "layers": 12,
        "heads": 8, "steps": 10000, "batch_size": 2048, "microbatch": 512,
        "eval_episodes": 4096, "eval_microbatch": 64,
        "checkpoint_every": 100, "log_every": 25,
    }}
    if name not in settings:
        raise ValueError(f"Unknown preset: {name}")
    cfg = Config(**(settings[name] | overrides | {"preset": name}))
    if cfg.mode not in MODES or cfg.seed < 0 or cfg.eval_seed < 0:
        raise ValueError("Invalid mode or negative seed")
    for field in ("dims", "points", "position_capacity", "embedding", "layers",
                  "heads", "steps", "batch_size", "microbatch", "eval_episodes",
                  "eval_microbatch", "checkpoint_every", "log_every"):
        if getattr(cfg, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if cfg.points > cfg.position_capacity or cfg.embedding % cfg.heads:
        raise ValueError("Invalid positional capacity or embedding/head ratio")
    if cfg.batch_size % cfg.microbatch:
        raise ValueError("Effective batch must be divisible by microbatch")
    if cfg.learning_rate <= 0 or cfg.sigma_max < 0 or cfg.max_grad_norm <= 0:
        raise ValueError("Invalid learning rate, noise SD, or gradient clipping")
    if cfg.precision not in ("fp32", "bf16"):
        raise ValueError("Precision must resolve to fp32 or bf16")
    return cfg


def sigma_schedule(cfg: Config) -> torch.Tensor:
    """Exactly match the SD multiset; randomize ordering with a separate RNG."""
    base = torch.linspace(0, cfg.sigma_max, cfg.steps, dtype=torch.float64)
    if cfg.mode == "clean":
        return torch.zeros_like(base)
    if cfg.mode == "curriculum":
        return base
    if cfg.mode == "reverse":
        return base.flip(0)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 3000003)
    return base[torch.randperm(cfg.steps, generator=gen)]


class EpisodeStream:
    """Three independent CPU RNG streams, unaffected by device or noise mode."""
    def __init__(self, seed: int):
        self.generators = {
            "tasks": torch.Generator().manual_seed(seed + 1000003),
            "inputs": torch.Generator().manual_seed(seed + 2000003),
            "corruption": torch.Generator().manual_seed(seed + 4000003),
        }

    def draw(self, batch: int, points: int, dims: int):
        weights = torch.randn(batch, dims, 1, generator=self.generators["tasks"])
        xs = torch.randn(batch, points, dims, generator=self.generators["inputs"])
        clean = (xs @ weights).squeeze(-1)
        epsilon = torch.randn(batch, points, generator=self.generators["corruption"])
        return xs, clean, epsilon

    def state_dict(self):
        return {key: gen.get_state() for key, gen in self.generators.items()}

    def load_state_dict(self, states):
        if set(states) != set(self.generators):
            raise ValueError("Checkpoint RNG stream names differ")
        for key, gen in self.generators.items():
            gen.set_state(states[key].cpu())


class RegressionTransformer(nn.Module):
    """Predict y_k at x_k; its causal prefix contains only labels y_0..y_{k-1}."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_projection = nn.Linear(cfg.dims, cfg.embedding)
        backbone_config = GPT2Config(
            n_positions=2 * cfg.position_capacity, n_embd=cfg.embedding,
            n_layer=cfg.layers, n_head=cfg.heads, resid_pdrop=0.0,
            embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
        )
        # Eager causal attention is deterministic and supported on CPU and CUDA.
        backbone_config._attn_implementation = "eager"
        self.backbone = GPT2Model(backbone_config)
        self.output_projection = nn.Linear(cfg.embedding, 1)

    def forward(self, xs, context_ys):
        label_tokens = torch.zeros_like(xs)
        label_tokens[:, :, 0] = context_ys
        tokens = torch.stack((xs, label_tokens), dim=2).flatten(1, 2)
        hidden = self.backbone(inputs_embeds=self.input_projection(tokens)).last_hidden_state
        return self.output_projection(hidden[:, 0::2]).squeeze(-1)


def seed_model(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def autocast_context(device: torch.device, precision: str):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and precision == "bf16"
            else contextlib.nullcontext())


def atomic_json(path: Path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch_save(path: Path, data):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def save_checkpoint(path, cfg, model, optimizer, stream, completed_steps, schedule):
    atomic_torch_save(path, {
        "schema_version": SCHEMA_VERSION, "config": dataclasses.asdict(cfg),
        "completed_steps": completed_steps, "schedule": schedule,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "episode_rng": stream.state_dict(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    })


def load_checkpoint(path, cfg, model, optimizer, stream, device, schedule):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Checkpoint schema differs")
    expected = dataclasses.asdict(cfg)
    if saved["config"] != expected:
        changed = [key for key in expected if saved["config"].get(key) != expected[key]]
        raise ValueError("Resume configuration differs: " + ", ".join(changed))
    if not torch.equal(saved["schedule"].cpu(), schedule):
        raise ValueError("Resume schedule differs")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    stream.load_state_dict(saved["episode_rng"])
    torch.set_rng_state(saved["torch_rng"].cpu())
    if device.type == "cuda" and saved["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    return int(saved["completed_steps"])


def train(cfg: Config, run_dir: Path, device: torch.device, resume=False,
          max_steps_this_run=None, stop_requested=lambda: False):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot resume: {checkpoint_path} missing")
    if not resume and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {run_dir}; use --resume")
    seed_model(cfg.seed)
    model = RegressionTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    stream = EpisodeStream(cfg.seed)
    schedule = sigma_schedule(cfg)
    completed = (load_checkpoint(checkpoint_path, cfg, model, optimizer, stream,
                                 device, schedule) if resume else 0)
    atomic_json(run_dir / "config.json", dataclasses.asdict(cfg))
    atomic_torch_save(run_dir / "schedule.pt", schedule)
    log_path = run_dir / "train.csv"
    # A crash can leave rows newer than the last checkpoint; trim on resumption.
    if resume and log_path.exists():
        with log_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row["step"]) <= completed]
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    fields = ("step", "sigma", "train_mse", "train_nmse", "grad_norm", "seconds_per_step")
    start_step = completed
    start_time = time.monotonic()
    model.train()
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        for step in range(completed, cfg.steps):
            step_start = time.monotonic()
            sigma = float(schedule[step])
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            accumulation = cfg.batch_size // cfg.microbatch
            for _ in range(accumulation):
                xs, clean_y, epsilon = stream.draw(cfg.microbatch, cfg.points, cfg.dims)
                context_y = clean_y + sigma * epsilon
                xs, clean_y, context_y = (tensor.to(device) for tensor in (xs, clean_y, context_y))
                with autocast_context(device, cfg.precision):
                    pred = model(xs, context_y)
                    loss = (pred.float() - clean_y).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
                (loss / accumulation).backward()
                loss_sum += float(loss.detach()) / accumulation
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm,
                                                           error_if_nonfinite=True))
            optimizer.step()
            completed = step + 1
            duration = time.monotonic() - step_start
            if completed == 1 or completed % cfg.log_every == 0 or completed == cfg.steps:
                writer.writerow(dict(zip(fields, (completed, sigma, loss_sum,
                                                  loss_sum / cfg.dims, grad_norm, duration))))
                handle.flush()
                elapsed = time.monotonic() - start_time
                rate = elapsed / (completed - start_step)
                eta = (cfg.steps - completed) * rate
                print(f"{cfg.mode} seed={cfg.seed} step={completed}/{cfg.steps} "
                      f"sigma={sigma:.4g} nmse={loss_sum / cfg.dims:.5g} "
                      f"s/step={rate:.3f} eta_hours={eta / 3600:.2f}", flush=True)
            interrupted = stop_requested() or (
                max_steps_this_run is not None and completed - start_step >= max_steps_this_run)
            if completed % cfg.checkpoint_every == 0 or completed == cfg.steps or interrupted:
                save_checkpoint(checkpoint_path, cfg, model, optimizer, stream, completed, schedule)
            if interrupted and completed < cfg.steps:
                return model, completed
    # Also handles resuming a fully trained model whose evaluation was interrupted.
    if completed == cfg.steps and not checkpoint_path.exists():
        save_checkpoint(checkpoint_path, cfg, model, optimizer, stream, completed, schedule)
    return model, completed


@torch.no_grad()
def clean_target_losses(model, xs, clean_y, context_y, *, dims, microbatch, device,
                        precision="fp32"):
    """Return per-episode, per-prefix squared errors divided by Var(y)=dims."""
    model.eval()
    loss_batches = []
    for start in range(0, len(xs), microbatch):
        sl = slice(start, start + microbatch)
        with autocast_context(device, precision):
            predictions = model(xs[sl].to(device), context_y[sl].to(device))
        losses = (predictions.float().cpu() - clean_y[sl]).square() / dims
        loss_batches.append(losses)
    return torch.cat(loss_batches)


def evaluation_bank(cfg):
    xs, clean, gaussian = EpisodeStream(cfg.eval_seed).draw(cfg.eval_episodes,
                                                          cfg.points, cfg.dims)
    gen = torch.Generator().manual_seed(cfg.eval_seed + 5000003)
    uniform_u = torch.rand(cfg.eval_episodes, cfg.points, generator=gen)
    laplace_u = torch.rand(cfg.eval_episodes, cfg.points, generator=gen).clamp_(1e-7, 1-1e-7) - 0.5
    # These have zero mean and population SD=1; no sample renormalization.
    uniform = (uniform_u * 2 - 1) * math.sqrt(3)
    laplace = -laplace_u.sign() * torch.log1p(-2 * laplace_u.abs()) / math.sqrt(2)
    return xs, clean, {"gaussian": gaussian, "laplace": laplace, "uniform": uniform}


@torch.no_grad()
def gaussian_bayes_losses(xs, clean_y, context_y, sigma, *, dims, microbatch):
    """Bayes posterior-mean ridge reference for the known Gaussian task prior.

    At sigma=0 use the minimum-norm noiseless solution. Calculations use CPU
    float64 to avoid singular float32 Gram matrices at short context lengths.
    """
    output = []
    for start in range(0, len(xs), microbatch):
        xb = xs[start:start + microbatch].double()
        yb = context_y[start:start + microbatch].double()
        batch, points, _ = xb.shape
        gram = torch.zeros(batch, dims, dims, dtype=torch.float64)
        rhs = torch.zeros(batch, dims, 1, dtype=torch.float64)
        identity = torch.eye(dims, dtype=torch.float64).unsqueeze(0)
        predictions = []
        for k in range(points):
            if k == 0:
                predicted = torch.zeros(batch, dtype=torch.float64)
            else:
                weights = (torch.linalg.solve(gram + sigma**2 * identity, rhs)
                           if sigma > 0 else torch.linalg.pinv(gram, hermitian=True) @ rhs)
                predicted = (xb[:, k:k+1] @ weights).flatten()
            predictions.append(predicted)
            xk = xb[:, k, :, None]
            gram += xk @ xk.transpose(1, 2)
            rhs += xk * yb[:, k, None, None]
        pred = torch.stack(predictions, dim=1).float()
        output.append((pred - clean_y[start:start + microbatch]).square() / dims)
    return torch.cat(output)


def condition_summary(losses, late_start):
    return {
        "mean_nmse": float(losses.mean()),
        "mean_nmse_by_k": losses.mean(0).tolist(),
        "late_context_nmse": float(losses[:, late_start:].mean()),
    }


def evaluate(cfg, model, run_dir, device):
    xs, clean_y, noises = evaluation_bank(cfg)
    losses, reference = {}, {}
    late_start = min(20, cfg.points - 1)
    eval_start = time.monotonic()
    for distribution, epsilon in noises.items():
        for sigma in NOISE_SDS:
            key = f"{distribution}_sd{sigma:g}"
            labels = clean_y + sigma * epsilon
            losses[key] = clean_target_losses(
                model, xs, clean_y, labels, dims=cfg.dims, microbatch=cfg.eval_microbatch,
                device=device, precision=cfg.precision)
            if distribution == "gaussian" and cfg.bayes_reference:
                reference[key] = gaussian_bayes_losses(
                    xs, clean_y, labels, sigma, dims=cfg.dims, microbatch=cfg.eval_microbatch)
            print(f"eval {key} late_nmse={losses[key][:, late_start:].mean():.6g}", flush=True)
    config = dataclasses.asdict(cfg)
    conditions = {key: condition_summary(value, late_start) for key, value in losses.items()}
    primary_key = "gaussian_sd2"
    metric_name = f"gaussian_sd2_nmse_k{late_start}_{cfg.points - 1}"
    schedule = sigma_schedule(cfg)
    metrics = {
        "schema_version": SCHEMA_VERSION, "config": config,
        "completed_steps": cfg.steps,
        "experiment": "corrected_fixed_budget_clean_target_all_prefixes",
        "normalization": "squared error / dims; w,x ~ N(0,I), so Var(y)=dims",
        "k_definition": "number of preceding context labels available at x_k",
        "primary": {"name": metric_name, "condition": primary_key,
                    "k_start": late_start, "k_end": cfg.points - 1,
                    "value": conditions[primary_key]["late_context_nmse"]},
        "conditions": conditions,
        "bayes_reference": {key: condition_summary(value, late_start)
                            for key, value in reference.items()},
        "noise_budget": {"sigma_sum": float(schedule.sum()),
                         "sigma_squared_sum": float(schedule.square().sum())},
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "python": sys.version, "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "eval_seconds": time.monotonic() - eval_start},
    }
    atomic_torch_save(run_dir / "losses.pt", {
        "schema_version": SCHEMA_VERSION, "config": config, "losses": losses,
        "bayes_losses": reference, "k": torch.arange(cfg.points),
        "eval_seed": cfg.eval_seed,
    })
    atomic_json(run_dir / "metrics.json", metrics)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--mode", choices=MODES, default="curriculum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-bayes-reference", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-steps-this-run", type=int, help="Checkpoint and stop for later resume; total budget stays fixed")
    for flag in ("steps", "batch-size", "microbatch", "eval-seed", "eval-episodes",
                 "eval-microbatch", "checkpoint-every", "log-every"):
        parser.add_argument("--" + flag, type=int)
    for flag in ("learning-rate", "max-grad-norm", "sigma-max"):
        parser.add_argument("--" + flag, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the allocated GPU and PyTorch build")
    if args.preset == "full" and device.type != "cuda":
        raise RuntimeError("The full preset requires CUDA. Use smoke for CPU verification.")
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 requires a CUDA GPU with bfloat16 support")
    overrides = {field.name: getattr(args, field.name) for field in dataclasses.fields(Config)
                 if hasattr(args, field.name) and getattr(args, field.name) is not None}
    overrides.update(precision=precision, bayes_reference=not args.no_bayes_reference)
    cfg = preset_config(args.preset, **overrides)
    if args.max_steps_this_run is not None and args.max_steps_this_run <= 0:
        raise ValueError("--max-steps-this-run must be positive")
    run_dir = args.output / f"seed_{cfg.seed}" / cfg.mode
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
        print(f"Signal {signum}: checkpointing after the current optimizer step", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    print(f"preset={cfg.preset} mode={cfg.mode} seed={cfg.seed} device={device} "
          f"precision={cfg.precision} effective_batch={cfg.batch_size} "
          f"accumulation={cfg.batch_size // cfg.microbatch} output={run_dir}", flush=True)
    model, completed = train(cfg, run_dir, device, args.resume, args.max_steps_this_run,
                             stop_requested=lambda: stop[0])
    if completed < cfg.steps:
        print(f"Saved {completed}/{cfg.steps} steps; rerun the same command with --resume.", flush=True)
        return 3
    evaluate(cfg, model, run_dir, device)
    print(f"Complete: {run_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
