#!/usr/bin/env python3
"""Plot one paired training seed; requires matplotlib==3.10.7.

These figures describe fixed trained models, without across-seed uncertainty.
Usage: python plot_results.py results/full --seed 0
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    modes = ("curriculum", "shuffled")
    colors = {"curriculum": "#007C91", "shuffled": "#C65524"}
    labels = {"curriculum": "Curriculum", "shuffled": "Shuffled noise"}
    runs = {mode: args.directory / f"seed_{args.seed}" / mode for mode in modes}
    metrics = {mode: json.loads((runs[mode] / "metrics.json").read_text())
               for mode in modes}
    configs = [data["config"] for data in metrics.values()]
    if any(configs[0][key] != configs[1][key]
           for key in configs[0] if key != "mode"):
        raise ValueError("The paired configurations differ.")
    cfg = configs[0]
    k = list(range(cfg["points"]))
    destination = args.directory / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160,
                         "savefig.facecolor": "white"})

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    ax.axvspan(20, cfg["points"] - 1, color="#E9EEF2", zorder=0)
    for mode in modes:
        ax.plot(k, metrics[mode]["conditions"]["gaussian_sd2"]["mean_nmse_by_k"],
                color=colors[mode], label=labels[mode], linewidth=2)
    ax.plot(k, metrics["curriculum"]["bayes_reference"]["gaussian_sd2"]["mean_nmse_by_k"],
            color="#555555", linestyle="--", linewidth=1.5,
            label="Gaussian Bayes reference")
    ax.set(title="High-noise test: Gaussian SD = 2",
           xlabel="Preceding demonstrations, k", ylabel="Normalized MSE (lower is better)",
           xlim=(0, cfg["points"] - 1), ylim=(0, 1.08))
    ax.text(30, 1.04, "Primary window: k20–40", ha="center", fontsize=9)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1]
    sds = (0, 0.5, 1, 2)
    for mode in modes:
        values = [metrics[mode]["conditions"][f"gaussian_sd{sd}"]["late_context_nmse"]
                  for sd in sds]
        ax.plot(sds, values, marker="o", linewidth=2,
                color=colors[mode], label=labels[mode])
    ax.set(title="Tradeoff across Gaussian noise levels",
           xlabel="Evaluation noise standard deviation",
           ylabel="Mean normalized MSE over k20–40",
           xticks=sds, ylim=(0, 0.39))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle(f"Controlled synthetic regression rerun · paired training seed {args.seed}",
                 fontsize=13, y=0.98)
    fig.text(0.5, 0.02,
             f"{cfg['eval_episodes']:,} shared test episodes · equal offered noise budget · "
             "one training seed; no across-seed confidence interval",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.065, 1, 0.93))
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"seed_{args.seed}_results.{suffix}")
    plt.close(fig)

    logs = {}
    for mode in modes:
        with (runs[mode] / "train.csv").open(newline="") as handle:
            logs[mode] = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for mode in modes:
        rows = logs[mode]
        steps = [int(row["step"]) for row in rows]
        losses = [float(row["train_nmse"]) for row in rows]
        smooth = [sum(losses[max(0, i - 8):i + 1]) / min(i + 1, 9)
                  for i in range(len(losses))]
        axes[0].plot(steps, losses, color=colors[mode], linewidth=0.6, alpha=0.25)
        axes[0].plot(steps, smooth, color=colors[mode], linewidth=1.8,
                     label=labels[mode] + " (9 logged points, trailing mean)")
        axes[1].plot(steps, [float(row["sigma"]) for row in rows],
                     color=colors[mode], linewidth=0.7, alpha=0.7,
                     label=labels[mode])
    axes[0].set(title="Training loss on the current noise level", ylabel="Training normalized MSE")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(title="Noise levels at logged steps (every 25 steps after step 1)",
                xlabel="Optimizer step", ylabel="Context noise SD", ylim=(0, 1.05))
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle(f"Training diagnostics · paired seed {args.seed}", fontsize=13, y=0.99)
    fig.text(0.5, 0.02,
             "Noise changes during training; this is not a fixed-condition validation curve "
             "and does not establish convergence.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"seed_{args.seed}_training.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    main()
