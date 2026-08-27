"""Plot round-level diagnostics for peak-channel Q-sweep pursuit runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def load_runs(comparison_path):
    comparison = json.loads(comparison_path.read_text())
    runs = []
    for key, value in sorted(
        comparison["runs"].items(), key=lambda item: int(item[0])
    ):
        path = Path(value["path"])
        runs.append(
            {
                "q": int(key),
                "path": path,
                "summary": value,
                "pass": np.load(path / "residual_pass.npy", mmap_mode="r"),
                "captured": np.load(path / "captured_fraction.npy", mmap_mode="r"),
                "score": np.load(path / "detection_score.npy", mmap_mode="r"),
                "drop": np.load(path / "pass_energy_drop_fraction.npy", mmap_mode="r"),
                "input_energy": np.load(path / "input_energy.npy", mmap_mode="r"),
            }
        )
    return runs


def by_round(values, passes, n_rounds, fn):
    out = np.full(n_rounds, np.nan, dtype=np.float64)
    values = np.asarray(values)
    passes = np.asarray(passes)
    for round_index in range(n_rounds):
        keep = passes == round_index
        if np.any(keep):
            out[round_index] = fn(values[keep])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--captured-floor", type=float, default=0.05)
    parser.add_argument("--energy-stop", type=float, default=0.002)
    args = parser.parse_args()

    runs = load_runs(args.comparison)
    n_rounds = max(int(np.max(run["pass"])) for run in runs) + 1
    rounds = np.arange(1, n_rounds + 1)
    colors = plt.colormaps["viridis"](np.linspace(0.12, 0.88, len(runs)))

    figure, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)

    for run, color in zip(runs, colors):
        passes = np.asarray(run["pass"])
        captured = np.asarray(run["captured"], dtype=np.float64)
        score = np.asarray(run["score"], dtype=np.float64)
        drop = np.asarray(run["drop"], dtype=np.float64)
        input_energy = np.asarray(run["input_energy"], dtype=np.float64)
        label = f"Q={run['q']}"

        counts = np.bincount(passes, minlength=n_rounds).astype(np.float64)
        axes[0, 0].plot(rounds, counts, color=color, linewidth=1.5, label=label)

        median_drop = by_round(drop, passes, n_rounds, np.median)
        axes[0, 1].plot(rounds, median_drop, color=color, linewidth=1.5, label=label)

        median_capture = by_round(captured, passes, n_rounds, np.median)
        q25_capture = by_round(captured, passes, n_rounds, lambda x: np.quantile(x, 0.25))
        q75_capture = by_round(captured, passes, n_rounds, lambda x: np.quantile(x, 0.75))
        axes[0, 2].plot(
            rounds,
            median_capture,
            color=color,
            linewidth=1.5,
            label=label,
        )
        axes[0, 2].fill_between(
            rounds,
            q25_capture,
            q75_capture,
            color=color,
            alpha=0.10,
            linewidth=0,
        )

        near_floor = by_round(
            captured <= 2 * args.captured_floor,
            passes,
            n_rounds,
            np.mean,
        )
        below_twenty = by_round(captured <= 0.20, passes, n_rounds, np.mean)
        axes[1, 0].plot(
            rounds,
            near_floor,
            color=color,
            linewidth=1.6,
            label=label,
        )
        axes[1, 1].plot(
            rounds,
            below_twenty,
            color=color,
            linewidth=1.6,
            label=label,
        )

        median_score = by_round(score, passes, n_rounds, np.median)
        median_energy = by_round(input_energy, passes, n_rounds, np.median)
        axes[1, 2].plot(
            rounds,
            median_score,
            color=color,
            linewidth=1.6,
            label=f"{label} score",
        )
        axes[1, 2].plot(
            rounds,
            np.sqrt(np.maximum(median_energy, 0)),
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
        )

    axes[0, 0].set(
        title="accepted events per pursuit round",
        xlabel="round",
        ylabel="accepted events",
        xlim=(1, n_rounds),
    )
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    axes[0, 1].axhline(
        args.energy_stop,
        color="0.25",
        linewidth=1,
        linestyle="--",
        label=f"{100 * args.energy_stop:.1f}% stop",
    )
    axes[0, 1].set(
        title="core-energy drop per round",
        xlabel="round",
        ylabel="fraction removed",
        xlim=(1, n_rounds),
        yscale="log",
    )
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[0, 2].axhline(args.captured_floor, color="0.25", linestyle="--", linewidth=1)
    axes[0, 2].set(
        title="median captured fraction with IQR",
        xlabel="round",
        ylabel="local-energy fraction",
        xlim=(1, n_rounds),
    )
    axes[0, 2].yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[1, 0].axhline(0.5, color="0.7", linewidth=0.8)
    axes[1, 0].set(
        title=f"accepted fits at or below {100 * 2 * args.captured_floor:.0f}% capture",
        xlabel="round",
        ylabel="fraction of accepted fits",
        xlim=(1, n_rounds),
        ylim=(0, 1),
    )
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[1, 1].axhline(0.5, color="0.7", linewidth=0.8)
    axes[1, 1].set(
        title="accepted fits at or below 20% capture",
        xlabel="round",
        ylabel="fraction of accepted fits",
        xlim=(1, n_rounds),
        ylim=(0, 1),
    )
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[1, 2].set(
        title="median detection score and sqrt local energy",
        xlabel="round",
        ylabel="score / sqrt energy",
        xlim=(1, n_rounds),
    )
    axes[1, 2].text(
        0.02,
        0.05,
        "solid: detection score\n dashed: sqrt input energy",
        transform=axes[1, 2].transAxes,
        fontsize=8,
        color="0.25",
    )

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Round-level diagnostics for completed peak-channel Q sweep "
        "(fixed projection threshold 6, captured-fraction floor 5%)"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
