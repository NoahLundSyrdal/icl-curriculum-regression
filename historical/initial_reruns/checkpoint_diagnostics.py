#!/usr/bin/env python3
"""Exploratory ridge agreement on independent calibration and held-out banks.

Ridge penalties are selected to match model predictions on calibration episodes,
never held-out targets. Known-SD ridge is privileged; empirical Bayes chooses SD
by the context-label marginal likelihood under the known N(0,I) task prior.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import torch
import run_experiment as base
import continue_experiment as tail

CALIBRATION_SEED = 921731
HELDOUT_SEED = 921733
PERMUTATION_SEED = 921739
ROTATION_SEED = 921743
SIGMAS = (0.0, 0.5, 1.0, 2.0)
GRID = sorted(set([0.0, 0.25, 1/3, 1.0, 4.0] +
                  [10 ** (-4 + 6*i/40) for i in range(41)]))
KIND = "ridge_prediction_diagnostics_v1"


def ridge_predictions(xs, ys, penalties, ks):
    """Return grid-by-episode-by-k predictions and context-only EB predictions.

    Batched float64 eigendecomposition; y_k and all later labels are excluded.
    EB maximizes p(y_<k | X_<k, sigma^2=lambda), integrating the N(0,I) w prior.
    The positive grid has a numerical lower bound 1e-4 and upper bound 100.
    """
    xs, ys = xs.double(), ys.double()
    penalties = torch.as_tensor(penalties, dtype=torch.float64)
    if (penalties < 0).any() or not (penalties > 0).any():
        raise ValueError("Ridge grid needs nonnegative penalties and positive EB candidates")
    outputs, adaptive, fitted = [], [], []
    for k in ks:
        if not 1 <= k < xs.shape[1]:
            raise ValueError("Prediction index must have a nonempty causal context")
        x, y = xs[:, :k], ys[:, :k]
        gram = x.transpose(1, 2) @ x
        s, u = torch.linalg.eigh(gram)
        valid = s > (s[:, -1:] * 1e-12).clamp_min(1e-14)
        s = s.clamp_min(1e-14)
        a = (u.transpose(1, 2) @ (x.transpose(1, 2) @ y[..., None])).squeeze(-1)
        a = a * valid
        q = (xs[:, k:k+1] @ u).squeeze(1)
        denominator = s[None] + penalties[:, None, None]
        prediction = ((q*a)[None] / denominator).sum(-1)
        outputs.append(prediction.float())
        # Decompose the marginal covariance into the column-space eigenvalues
        # and an orthogonal complement of multiplicity k-rank(X).
        residual = (y.square().sum(-1) - (a.square()/s).sum(-1)).clamp_min(0)
        residual = torch.where(valid.sum(-1) == k, torch.zeros_like(residual), residual)
        positive = penalties > 0
        lam = penalties[positive, None]
        denom = s[None] + lam[..., None]
        logdet = (torch.log(denom)*valid[None]).sum(-1) + (k-valid.sum(-1))[None]*torch.log(lam)
        quadratic = (a.square()[None]/s[None]/denom).sum(-1) + residual[None]/lam
        index = (logdet + quadratic).argmin(0)
        eb_preds = prediction[positive]
        adaptive.append(eb_preds.gather(0, index[None]).squeeze(0))
        fitted.append(penalties[positive][index].float())
    return torch.stack(outputs, -1), torch.stack(adaptive, -1), torch.stack(fitted, -1)


def cache_identity():
    return {"kind": KIND, "diagnostic_script_sha256": tail.file_sha256(__file__),
            "base_script_sha256": tail.file_sha256(base.__file__),
            "protocol_sha256": tail.file_sha256("FOLLOWUP_PROTOCOL.txt"),
            "calibration_seed": CALIBRATION_SEED, "heldout_seed": HELDOUT_SEED,
            "calibration_episodes": 512, "heldout_episodes": 4096,
            "grid": GRID, "sigmas": list(SIGMAS), "ks": list(range(20, 41))}


def prepare_cache(path):
    path = Path(path)
    identity = cache_identity()
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["identity"] != identity:
            raise ValueError("Existing reference cache code/protocol/split differs")
        return
    data = {"identity": identity, "banks": {}}
    for name, seed, episodes in (("calibration", CALIBRATION_SEED, 512),
                                 ("heldout", HELDOUT_SEED, 4096)):
        cfg = base.preset_config("full", eval_seed=seed, eval_episodes=episodes)
        xs, clean, noise = base.EpisodeStream(seed).draw(episodes, cfg.points, cfg.dims)
        bank = {"xs": xs, "clean": clean, "epsilon": noise, "references": {}}
        for sigma in SIGMAS:
            grid, eb, fitted = ridge_predictions(xs, clean + sigma*noise, GRID, range(20, 41))
            if not torch.isfinite(grid).all() or not torch.isfinite(eb).all():
                raise FloatingPointError("Nonfinite classical reference")
            bank["references"][f"{sigma:g}"] = {"grid": grid, "empirical_bayes": eb,
                                                     "eb_penalty": fitted}
            print(f"Prepared {name}, SD={sigma:g}", flush=True)
        data["banks"][name] = bank
    generator = torch.Generator().manual_seed(ROTATION_SEED)
    q, _ = torch.linalg.qr(torch.randn(20, 20, generator=generator, dtype=torch.float64))
    data["rotation"] = q.float()
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_torch_save(path, data)


@torch.no_grad()
def model_predictions(model, xs, labels, cfg, device):
    model.eval()
    outputs = []
    for start in range(0, len(xs), cfg.eval_microbatch):
        sl = slice(start, start + cfg.eval_microbatch)
        with base.autocast_context(device, cfg.precision):
            pred = model(xs[sl].to(device), labels[sl].to(device))
        outputs.append(pred.float().cpu())
    prediction = torch.cat(outputs)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite transformer prediction")
    return prediction


def prediction_metrics(prediction, target, dims=20):
    squared = (prediction.double() - target.double()).square() / dims
    if not torch.isfinite(squared).all():
        raise FloatingPointError("Nonfinite metric")
    return {"mean_k20_40": float(squared.mean()),
            "by_k": squared.mean(0).tolist(), "k20": float(squared[:, 0].mean()),
            "k40": float(squared[:, -1].mean())}


def select_penalty(calibration_grid, calibration_prediction):
    discrepancy = (calibration_grid.double() - calibration_prediction.double()[None]).square().mean((1, 2))
    return int(discrepancy.argmin())


def permutations(xs, labels, k, seed=PERMUTATION_SEED):
    generator = torch.Generator().manual_seed(seed + k)
    index = torch.rand(len(xs), k, generator=generator).argsort(-1)
    permuted_x = xs[:, :k].gather(1, index[..., None].expand(-1, -1, xs.shape[-1]))
    permuted_y = labels[:, :k].gather(1, index)
    # The query label is arbitrary and causally inaccessible; zero is explicit.
    return torch.cat((permuted_x, xs[:, k:k+1]), 1), torch.cat((permuted_y, torch.zeros(len(xs), 1)), 1)


def checkpoint_source(root, seed, tag):
    root = Path(root)
    if tag.startswith("original_"):
        return root / "results/full" / f"seed_{seed}" / tag.removeprefix("original_") / "checkpoint.pt"
    if tag.startswith("clean_"):
        return root / "results/followup_20261006/clean" / f"seed_{seed}" / f"step_{int(tag.removeprefix('clean_')):06d}" / "checkpoint.pt"
    history, regime = tag.split("_")
    return root / "results/continuation_10000" / f"seed_{seed}" / history / regime / "checkpoint.pt"


def diagnose(checkpoint, destination, cache, tag, seed, device):
    checkpoint, destination = Path(checkpoint), Path(destination)
    identity = {"kind": KIND, "tag": tag, "seed": seed, "cache_identity": cache["identity"],
                "checkpoint_sha256": tail.file_sha256(checkpoint)}
    if destination.exists():
        result = json.loads(destination.read_text())
        if result["identity"] != identity:
            raise ValueError("Existing diagnostic identity differs")
        return
    started = time.monotonic()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = base.preset_config(saved["config"]["preset"], **saved["config"])
    total_steps = int(saved.get("total_completed_steps", saved.get("completed_steps", -1)))
    expected_steps = int(tag.removeprefix("clean_")) if tag.startswith("clean_") else 10000 if tag.startswith("original_") else 20000
    canonical = base.preset_config("full", seed=seed,
        mode="clean" if tag.startswith("clean_") else tag.removeprefix("original_") if tag.startswith("original_") else tag.split("_")[0],
        steps=20000 if tag.startswith("clean_") else 10000, precision="bf16")
    if cfg != canonical or total_steps != expected_steps:
        raise ValueError("Diagnostic checkpoint identity/settings/progress differs")
    if tag.startswith("original_"):
        _, source = tail.inspect_source(checkpoint.parent)
        if source["checkpoint_sha256"] != identity["checkpoint_sha256"]:
            raise ValueError("Original source changed")
    elif tag.startswith("clean_"):
        metric = json.loads((checkpoint.parent / "metrics.json").read_text())
        if metric["control_checkpoint_sha256"] != identity["checkpoint_sha256"]:
            raise ValueError("Clean snapshot differs from recorded evaluation")
    else:
        history, regime = tag.split("_")
        original_path = checkpoint_source(Path.cwd(), seed, "original_" + history)
        if saved["source"]["checkpoint_sha256"] != tail.file_sha256(original_path):
            raise ValueError("Continuation source provenance differs")
        if saved["continuation_script_sha256"] != tail.file_sha256(tail.__file__) or saved["continuation"]["tail"] != regime:
            raise ValueError("Continuation code/tail differs")
    base.seed_model(seed)
    model = base.RegressionTransformer(cfg).to(device)
    model.load_state_dict(saved["model"])
    del saved
    report = {"identity": identity, "total_steps": total_steps, "conditions": {}}
    for sigma in SIGMAS:
        predictions = {}
        for split, bank in cache["banks"].items():
            predictions[split] = model_predictions(model, bank["xs"], bank["clean"] + sigma*bank["epsilon"], cfg, device)[:, 20:41]
        calib = cache["banks"]["calibration"]["references"][f"{sigma:g}"]
        hold = cache["banks"]["heldout"]
        references = hold["references"][f"{sigma:g}"]
        best = select_penalty(calib["grid"], predictions["calibration"])
        condition = {"model_task_nmse": prediction_metrics(predictions["heldout"], hold["clean"][:, 20:41]),
                     "calibration_selected_penalty": GRID[best], "references": {}}
        named = {"ols": references["grid"][GRID.index(0.0)],
                 "ridge_lambda_1over3": references["grid"][GRID.index(1/3)],
                 "ridge_lambda_1": references["grid"][GRID.index(1.0)],
                 "ridge_lambda_4": references["grid"][GRID.index(4.0)],
                 "known_sigma_oracle": references["grid"][GRID.index(sigma*sigma)],
                 "calibration_fitted_ridge": references["grid"][best],
                 "empirical_bayes_context_only": references["empirical_bayes"]}
        for name, reference in named.items():
            condition["references"][name] = {
                "prediction_disagreement_nmse": prediction_metrics(predictions["heldout"], reference),
                "reference_task_nmse": prediction_metrics(reference, hold["clean"][:, 20:41])}
        # One fixed orthogonal rotation and independent permutations for each k.
        # These are distribution-preserving diagnostics, not mechanistic proof.
        count = 512
        xs, labels = hold["xs"][:count], (hold["clean"] + sigma*hold["epsilon"])[:count]
        rotated = model_predictions(model, xs @ cache["rotation"], labels, cfg, device)[:, 20:41]
        permuted = []
        for k in range(20, 41):
            px, py = permutations(xs, labels, k)
            permuted.append(model_predictions(model, px, py, cfg, device)[:, -1])
        permuted = torch.stack(permuted, -1)
        condition["invariance_episodes"] = count
        condition["invariance"] = {
            name: {"prediction_change_nmse": prediction_metrics(variant, predictions["heldout"][:count]),
                   "variant_task_nmse": prediction_metrics(variant, hold["clean"][:count, 20:41])}
            for name, variant in (("orthogonal_rotation", rotated), ("context_permutation", permuted))}
        condition["eb_selected_penalty_mean_by_k"] = references["eb_penalty"].mean(0).tolist()
        report["conditions"][f"gaussian_sd{sigma:g}"] = condition
        print(f"seed={seed} tag={tag} SD={sigma:g}: heldout={condition['model_task_nmse']['mean_k20_40']:.6g}, fitted_lambda={GRID[best]:.5g}", flush=True)
    if tail.file_sha256(checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    report["runtime"] = {"seconds": time.monotonic()-started, "torch": str(torch.__version__),
                         "transformers": base.transformers.__version__, "gpu": torch.cuda.get_device_name(device)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(destination, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--cache", type=Path, default=Path("results/followup_20261006/references.pt"))
    p.add_argument("--seed", type=int, choices=range(5))
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.prepare:
        prepare_cache(args.cache)
        return
    if args.seed is None or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Diagnostics require seed and BF16 CUDA allocation")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache["identity"] != cache_identity():
        raise ValueError("Reference bank identity differs")
    tags = ["clean_10000", "clean_20000"] if args.clean else [
        "original_curriculum", "original_shuffled", "curriculum_high", "shuffled_high", "curriculum_mixed", "shuffled_mixed"]
    for tag in tags:
        destination = args.cache.parent / "diagnostics" / f"seed_{args.seed}" / f"{tag}.json"
        diagnose(checkpoint_source(Path.cwd(), args.seed, tag), destination, cache, tag, args.seed, torch.device("cuda"))


if __name__ == "__main__":
    main()
