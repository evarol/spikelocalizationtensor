"""Plot completed peak-channel temporal-codebook Q sweep diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


COLORS = {
    "ink": "#183047",
    "blue": "#4c78a8",
    "orange": "#f58518",
    "green": "#54a24b",
    "red": "#e45756",
    "purple": "#b279a2",
}


def load_comparison(path):
    comparison = json.loads(path.read_text())
    runs = []
    for key, item in sorted(
        comparison["runs"].items(), key=lambda pair: int(pair[0])
    ):
        run_path = Path(item["path"])
        captured = np.load(run_path / "captured_fraction.npy", mmap_mode="r")
        residual_pass = np.load(run_path / "residual_pass.npy", mmap_mode="r")
        detection_score = np.load(run_path / "detection_score.npy", mmap_mode="r")
        runs.append(
            {
                "q": int(key),
                "path": run_path,
                "summary": item,
                "captured": np.asarray(captured, dtype=np.float64),
                "residual_pass": np.asarray(residual_pass, dtype=np.int64),
                "detection_score": np.asarray(detection_score, dtype=np.float64),
            }
        )
    return runs


def round_count_matrix(runs):
    max_round = max(int(run["residual_pass"].max()) for run in runs) + 1
    matrix = np.zeros((len(runs), max_round), dtype=np.float64)
    for row, run in enumerate(runs):
        counts = np.bincount(run["residual_pass"], minlength=max_round)
        matrix[row] = counts
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--captured-floor", type=float, default=0.05)
    args = parser.parse_args()

    runs = load_comparison(args.comparison)
    q_values = np.array([run["q"] for run in runs])
    labels = [f"Q={q}" for q in q_values]
    event_rates = np.array(
        [run["summary"]["events_per_second"] for run in runs], dtype=np.float64
    )
    remaining = np.array(
        [
            run["summary"]["remaining_core_energy_fraction_mean"]
            for run in runs
        ],
        dtype=np.float64,
    )
    rounds = np.array(
        [run["summary"]["rounds_completed_mean"] for run in runs], dtype=np.float64
    )
    captured_median = np.array(
        [
            run["summary"]["captured_fraction_median"]
            for run in runs
        ],
        dtype=np.float64,
    )
    captured_mean = np.array(
        [run["summary"]["captured_fraction_mean"] for run in runs], dtype=np.float64
    )
    effective_rows = np.array(
        [
            run["summary"]["row_usage_effective_count"]
            for run in runs
        ],
        dtype=np.float64,
    )
    codebook_cosine = np.array(
        [
            run["summary"]["codebook"]["maximum_pairwise_absolute_cosine"]
            for run in runs
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)

    bars = axes[0, 0].bar(labels, event_rates, color=COLORS["blue"])
    axes[0, 0].bar_label(bars, labels=[f"{value:,.0f}" for value in event_rates], padding=3)
    axes[0, 0].set(title="accepted event rate", ylabel="events / second")

    axes[0, 1].plot(q_values, remaining, "-o", color=COLORS["red"], label="remaining")
    axes[0, 1].plot(
        q_values,
        1 - remaining,
        "-o",
        color=COLORS["green"],
        label="removed",
    )
    axes[0, 1].set(
        title="mean core energy after pursuit",
        xlabel="temporal rows Q",
        ylabel="fraction of starting core energy",
        xticks=q_values,
        ylim=(0, 1),
    )
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(
        q_values,
        captured_mean,
        "-o",
        color=COLORS["orange"],
        label="mean",
    )
    axes[0, 2].plot(
        q_values,
        captured_median,
        "-o",
        color=COLORS["purple"],
        label="median",
    )
    axes[0, 2].axhline(
        args.captured_floor,
        color="0.25",
        linewidth=1.0,
        linestyle="--",
        label=f"{100 * args.captured_floor:.0f}% floor",
    )
    axes[0, 2].set(
        title="raw local-energy captured fraction",
        xlabel="temporal rows Q",
        ylabel="captured fraction",
        xticks=q_values,
        ylim=(0, max(captured_mean.max(), captured_median.max()) * 1.35),
    )
    axes[0, 2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 2].legend(frameon=False)

    samples = [run["captured"] for run in runs]
    violin = axes[1, 0].violinplot(
        samples, positions=np.arange(len(runs)), showmedians=True, showextrema=False
    )
    for body, color in zip(
        violin["bodies"], plt.colormaps["viridis"](np.linspace(0.15, 0.85, len(runs)))
    ):
        body.set_facecolor(color)
        body.set_alpha(0.78)
    violin["cmedians"].set_color("black")
    axes[1, 0].axhline(args.captured_floor, color="0.25", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="accepted-fit captured-fraction distributions",
        xlabel="temporal rows Q",
        ylabel="captured fraction",
        xticks=np.arange(len(runs)),
        xticklabels=labels,
        ylim=(0, min(1.0, max(np.quantile(sample, 0.995) for sample in samples) * 1.08)),
    )
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))

    count_matrix = round_count_matrix(runs)
    image = axes[1, 1].imshow(
        count_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axes[1, 1].set(
        title="accepted events by pursuit round",
        xlabel="pursuit round",
        ylabel="Q",
        yticks=np.arange(len(runs)),
        yticklabels=labels,
    )
    tick_rounds = np.unique(
        np.rint(np.linspace(1, count_matrix.shape[1], 7)).astype(int)
    )
    axes[1, 1].set_xticks(tick_rounds - 1, tick_rounds)
    figure.colorbar(image, ax=axes[1, 1], label="accepted events")

    axes[1, 2].plot(
        q_values,
        rounds,
        "-o",
        color=COLORS["ink"],
        label="mean rounds",
    )
    axes[1, 2].plot(
        q_values,
        effective_rows,
        "-o",
        color=COLORS["green"],
        label="effective row usage",
    )
    axes_right = axes[1, 2].twinx()
    axes_right.plot(
        q_values,
        codebook_cosine,
        "--s",
        color=COLORS["red"],
        label="max row cosine",
    )
    axes[1, 2].set(
        title="round pressure and row redundancy",
        xlabel="temporal rows Q",
        ylabel="rounds / effective rows",
        xticks=q_values,
    )
    axes_right.set(ylabel="max pairwise absolute cosine", ylim=(0, 1))
    handles, handle_labels = axes[1, 2].get_legend_handles_labels()
    right_handles, right_labels = axes_right.get_legend_handles_labels()
    axes[1, 2].legend(
        handles + right_handles,
        handle_labels + right_labels,
        frameon=False,
        loc="lower right",
        fontsize=8,
    )

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Completed peak-channel fixed projection-threshold Q sweep "
        "(job array 16358281)"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)
    for run in runs:
        print(
            f"Q={run['q']:2d}: events={run['summary']['n_events']:,}, "
            f"median_capture={100 * run['summary']['captured_fraction_median']:.2f}%, "
            f"mean_rounds={run['summary']['rounds_completed_mean']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
