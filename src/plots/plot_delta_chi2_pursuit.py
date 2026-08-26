"""Compare the baseline Q8 pursuit with the delta-chi2-gated run."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


COLORS = {"baseline": "#4c78a8", "delta-chi2": "#e45756"}


def load_run(path):
    return {
        "pass": np.load(path / "residual_pass.npy", mmap_mode="r"),
        "captured": np.load(path / "captured_fraction.npy", mmap_mode="r"),
        "drop": np.load(path / "pass_energy_drop_fraction.npy", mmap_mode="r"),
        "delta_chi2": (
            np.load(path / "delta_chi2.npy", mmap_mode="r")
            if (path / "delta_chi2.npy").exists()
            else None
        ),
    }


def round_values(run):
    passes = np.asarray(run["pass"], dtype=np.int64)
    n_rounds = int(passes.max()) + 1
    rounds = np.arange(1, n_rounds + 1)
    counts = np.bincount(passes, minlength=n_rounds)
    median_capture = np.array(
        [np.median(run["captured"][passes == value]) for value in range(n_rounds)]
    )
    drops = np.array(
        [np.median(run["drop"][passes == value]) for value in range(n_rounds)]
    )
    return rounds, counts, median_capture, np.cumprod(1 - drops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gated", type=Path, required=True)
    parser.add_argument("--min-delta-chi2", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        "baseline": load_run(args.baseline),
        "delta-chi2": load_run(args.gated),
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    for label, run in runs.items():
        rounds, counts, median_capture, remaining = round_values(run)
        color = COLORS[label]
        axes[0, 0].plot(rounds, counts, color=color, linewidth=1.6, label=label)
        axes[0, 1].plot(rounds, remaining, color=color, linewidth=1.6, label=label)
        axes[1, 0].plot(
            rounds, median_capture, color=color, linewidth=1.6, label=label
        )

    axes[0, 0].set(
        title="accepted events per pursuit round",
        xlabel="round",
        ylabel="accepted events",
    )
    axes[0, 1].set(
        title="remaining core energy",
        xlabel="round",
        ylabel="fraction remaining",
        ylim=(0, 1),
    )
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].set(
        title="median captured fraction",
        xlabel="round",
        ylabel="local-energy fraction",
    )
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))

    delta_chi2 = np.asarray(runs["delta-chi2"]["delta_chi2"], dtype=np.float64)
    finite_delta = delta_chi2[np.isfinite(delta_chi2)]
    bins = np.geomspace(args.min_delta_chi2, max(finite_delta.max(), args.min_delta_chi2) * 1.05, 80)
    axes[1, 1].hist(finite_delta, bins=bins, color=COLORS["delta-chi2"], alpha=0.8)
    axes[1, 1].axvline(
        args.min_delta_chi2,
        color="0.2",
        linestyle="--",
        linewidth=1,
        label=rf"gate $\Delta\chi^2={args.min_delta_chi2:g}$",
    )
    axes[1, 1].set(
        title="accepted noise-weighted fit improvement",
        xlabel=r"$\Delta\chi^2$",
        ylabel="accepted events",
        xscale="log",
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Q8 frozen pursuit: 5% raw-capture floor vs noise-weighted gate")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
