#!/usr/bin/env python3
"""Export a scientific figure of every predeclared continuation evaluation.

Reads the completed, validated summary; writes only the requested PNG and PDF.
No selection of a favorable tail, timepoint, or metric is performed.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TIMES = [0, 1000, 2500, 5000, 10000]
HISTORIES = ("curriculum", "shuffled")
TAILS = ("high", "mixed")
COLORS = {"curriculum": "#0072B2", "shuffled": "#D55E00"}
TAIL_COLORS = {"high": "#555555", "mixed": "#009E73"}
STYLES = {"high": "-", "mixed": (0, (5, 3))}


def verify_estimate(data):
    if data["n"] != 5 or data["seeds"] != list(range(5)):
        raise ValueError("Every plotted estimate must contain all five paired training seeds")
    if not math.isfinite(data["mean"]) or len(data["ci95"]) != 2:
        raise ValueError("Invalid estimate or uncertainty interval")
    if not all(math.isfinite(x) for x in data["ci95"]):
        raise ValueError("Nonfinite interval")
    if not data["ci95"][0] <= data["mean"] <= data["ci95"][1]:
        raise ValueError("Interval does not contain the estimate")


def collect(trajectories, getter):
    estimates = [getter(t) for t in trajectories]
    for estimate in estimates:
        verify_estimate(estimate)
    return ([e["mean"] for e in estimates],
            [e["ci95"][0] for e in estimates],
            [e["ci95"][1] for e in estimates])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/continuation_10000"))
    args = parser.parse_args()
    source = args.directory / "summary.json"
    summary = json.loads(source.read_text())
    if not summary["complete"] or summary["observed_records"] != 100 or summary["missing_records"]:
        raise ValueError("A complete 100-record continuation study is required")
    trajectories = summary["trajectories"]
    if [t["tail_completed_steps"] for t in trajectories] != TIMES:
        raise ValueError("The fixed evaluation times differ")
    final = summary["main_contrast"]
    verify_estimate(final)
    if final["tail_completed_steps"] != 10000:
        raise ValueError("Final main interaction must be evaluated at +10000 steps")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.7))
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.225,
                        top=0.805, hspace=0.47, wspace=0.32)
    x = [n / 1000 for n in TIMES]
    fig.text(0.095, 0.966, "Noise continuation in synthetic linear in-context learning",
             fontsize=17, weight="semibold", va="top")
    fig.text(0.095, 0.929,
             "Original noise histories: ascending (C) or shuffled (S); both receive each shared continuation tail",
             fontsize=10, color="#444444", va="top")
    fig.text(0.095, 0.907,
             "Endpoints average predictions with k = 20-40 preceding demonstrations; normalized MSE = squared error / 20",
             fontsize=9.5, color="#555555", va="top")
    legends = []
    for history in HISTORIES:
        for tail in TAILS:
            legends.append(Line2D([0], [0], color=COLORS[history], linestyle=STYLES[tail],
                                  marker="o" if history == "curriculum" else "s", markersize=4,
                                  label=f"{'Ascending' if history == 'curriculum' else 'Shuffled'} history / {tail} tail"))
    fig.legend(handles=legends, loc="upper left", bbox_to_anchor=(0.087, 0.881),
               ncol=2, frameon=False, fontsize=9, handlelength=3.5, columnspacing=2.7)

    for ax, condition, title in ((axes[0, 0], "gaussian_sd2", "(a) Noisy evaluation: Gaussian SD 2"),
                                 (axes[0, 1], "gaussian_sd0", "(b) Clean evaluation: SD 0")):
        for history in HISTORIES:
            for tail in TAILS:
                means, _, _ = collect(trajectories, lambda t, h=history, a=tail: t["tails"][a][condition]["arm_means"][h])
                ax.plot(x, means, color=COLORS[history], linestyle=STYLES[tail],
                        marker="o" if history == "curriculum" else "s", markersize=4.5,
                        linewidth=1.7, zorder=3)
        ax.set_title(title, loc="left", pad=12)
        ax.set_ylabel("Mean normalized MSE")
        ax.text(0.02, 0.035, "Arm means across five training seeds",
                transform=ax.transAxes, fontsize=8, color="#666666")
        ax.margins(y=0.16)
    axes[0, 1].set_ylim(0, 0.047)

    gap_ax = axes[1, 0]
    for tail in TAILS:
        means, lower, upper = collect(trajectories, lambda t, a=tail: t["tails"][a]["gaussian_sd2"]["history_gap"])
        gap_ax.fill_between(x, lower, upper, color=TAIL_COLORS[tail], alpha=0.12, linewidth=0)
        gap_ax.plot(x, means, color=TAIL_COLORS[tail], linestyle=STYLES[tail],
                    marker="o", linewidth=1.7, markersize=4.5, label=f"{tail.capitalize()} tail")
    gap_ax.axhline(0, color="#777777", linestyle=(0, (2, 3)), linewidth=0.9)
    gap_ax.set_title("(c) Paired history gaps: Gaussian SD 2", loc="left", pad=12)
    gap_ax.set_ylabel("D = C - S normalized MSE")
    gap_ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    gap_ax.text(0.02, 0.04, "Negative gap favors ascending history",
                transform=gap_ax.transAxes, fontsize=8, color="#555555")
    gap_ax.set_ylim(-0.046, 0.061)

    interaction_ax = axes[1, 1]
    means, lower, upper = collect(trajectories, lambda t: t["interactions"]["gaussian_sd2"]["history_gap_high_minus_mixed"])
    if not math.isclose(means[-1], final["mean"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Trajectory and predeclared final interaction disagree")
    purple = "#6F428D"
    interaction_ax.axvspan(9.55, 10.35, color="#E6C55D", alpha=0.19, linewidth=0, zorder=0)
    interaction_ax.fill_between(x, lower, upper, color=purple, alpha=0.11, linewidth=0)
    interaction_ax.plot(x, means, color=purple, marker="o", markersize=4.5, linewidth=1.7)
    interaction_ax.errorbar(x[-1], means[-1], yerr=[[means[-1] - lower[-1]], [upper[-1] - means[-1]]],
                           color=purple, marker="o", markersize=7, capsize=4, linewidth=1.8, zorder=5)
    interaction_ax.axhline(0, color="#777777", linestyle=(0, (2, 3)), linewidth=0.9)
    interaction_ax.set_title("(d) History-by-tail interaction: Gaussian SD 2", loc="left", pad=12)
    interaction_ax.set_ylabel("I = D_high - D_mixed normalized MSE")
    interaction_ax.set_ylim(-0.065, 0.071)
    final_text = ("Fixed final endpoint (+10k steps)\n"
                  f"I = {final['mean']:+.5f}\n"
                  f"95% CI [{final['ci95'][0]:+.5f}, {final['ci95'][1]:+.5f}]\n"
                  "Inconclusive; interval includes zero")
    interaction_ax.text(0.03, 0.965, final_text, transform=interaction_ax.transAxes,
                        ha="left", va="top", fontsize=8.4, linespacing=1.4,
                        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "linewidth": 0.7, "pad": 5})

    for row in axes:
        for ax in row:
            ax.set_xticks(x, ["0", "1", "2.5", "5", "10"])
            ax.set_xlim(-0.3, 10.4)
            ax.set_xlabel("Additional optimizer steps (thousands)")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65, zorder=0)
            ax.set_axisbelow(True)

    caption = [
        "Five independent original training seeds; 4,096 shared held-out episodes. All five evaluation times are displayed.",
        "Bands in (c,d) are paired Student t 95% intervals across seeds, conditional on this fixed test bank.",
        "The +10k-step interaction is the predeclared exploratory contrast (20k total training steps); intermediate intervals are secondary",
        "and unadjusted for multiplicity. A zero-containing interval is inconclusive, rather than evidence of equivalence.",
        "High tail: SD 1 throughout. Mixed tail: a shuffled finite list spanning SD 0-1. Tail noise budgets deliberately differ.",
        "The original H2 result remains inconclusive. Mean clean/noisy performance differences between tails remain visible in (a,b).",
    ]
    for i, line in enumerate(caption):
        fig.text(0.095, 0.167 - i * 0.023, line, fontsize=8.3, color="#444444", ha="left", va="top")
    output = args.directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    png, pdf = output / "continuation_results.png", output / "continuation_results.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf, metadata={"Title": "Exploratory synthetic ICL continuation results",
                              "Subject": "All fixed trajectories and the predeclared final interaction; original H2 unchanged"})
    plt.close(fig)
    print(f"Created {png}\nCreated {pdf}")


if __name__ == "__main__":
    main()
