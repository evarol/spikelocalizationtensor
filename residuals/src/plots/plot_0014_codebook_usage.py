"""Plot 0014-schema temporal codebook values and per-pass usage fractions."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-cols", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()
    if args.n_cols < 1:
        raise ValueError("--n-cols must be a positive integer")

    omega = np.load(args.run / "omega.npy")
    temporal_idx = np.load(args.run / "temporal_idx.npy", mmap_mode="r")
    residual_pass = np.load(args.run / "residual_pass.npy", mmap_mode="r")
    if omega.ndim != 2:
        raise ValueError(f"expected omega to have shape (Q, T), got {omega.shape}")
    if temporal_idx.shape != residual_pass.shape:
        raise ValueError(
            f"temporal_idx {temporal_idx.shape} and residual_pass "
            f"{residual_pass.shape} must match"
        )
    if temporal_idx.min() < 0 or temporal_idx.max() >= len(omega):
        raise ValueError("temporal_idx contains a row outside the temporal codebook")

    n_passes = int(residual_pass.max()) + 1
    n_total = len(temporal_idx)
    if args.max_events is not None and args.max_events < n_total:
        step = int(np.ceil(n_total / args.max_events))
        temporal_idx = temporal_idx[::step]
        residual_pass = residual_pass[::step]
        n_total = len(temporal_idx)

    counts_by_pass = np.zeros((n_passes, len(omega)), dtype=np.int64)
    for pass_index in range(n_passes):
        counts_by_pass[pass_index] = np.bincount(
            temporal_idx[residual_pass == pass_index], minlength=len(omega)
        )
    counts_total = counts_by_pass.sum(axis=0)
    fractions_total = counts_total / counts_total.sum()
    fractions_by_pass = counts_by_pass / counts_by_pass.sum(axis=1, keepdims=True)

    colors = plt.colormaps["tab20"](np.arange(len(omega)) % 20)
    pass_colors = plt.colormaps["viridis"](np.linspace(0.15, 0.9, n_passes))
    amplitude_limit = max(float(np.abs(omega).max()), np.finfo(np.float32).eps)
    usage_limit = max(0.01, float(fractions_total.max()) * 1.32)
    samples = np.arange(omega.shape[1])

    nrows = int(np.ceil(len(omega) / args.n_cols))
    figure, axes = plt.subplots(
        nrows, 3 * args.n_cols,
        figsize=(6.0 * args.n_cols, 1.5 * nrows + 1.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (3.4, 1.6, 1.6) * args.n_cols},
    )
    axes = np.atleast_2d(axes)
    grid = axes.reshape(nrows, args.n_cols, 3)
    for row_index, color in enumerate(colors):
        row, col = divmod(row_index, args.n_cols)
        waveform_axis = grid[row, col, 0]
        overall_axis = grid[row, col, 1]
        pass_axis = grid[row, col, 2]

        waveform_axis.plot(samples, omega[row_index], color=color, linewidth=1.3)
        waveform_axis.axhline(0, color="0.8", linewidth=0.6)
        waveform_axis.set_ylim(-1.05 * amplitude_limit, 1.05 * amplitude_limit)
        waveform_axis.set_ylabel(rf"$\Omega_{{{row_index}}}$")
        waveform_axis.grid(alpha=0.18)

        overall_axis.barh(0, fractions_total[row_index], color=color, height=0.58)
        overall_axis.text(
            fractions_total[row_index] + usage_limit * 0.018, 0,
            f"{100 * fractions_total[row_index]:.1f}%  "
            f"({counts_total[row_index]:,})",
            va="center", fontsize=8,
        )
        overall_axis.set_xlim(0, usage_limit)
        overall_axis.set_ylim(-0.75, 0.75)
        overall_axis.set_yticks([])
        overall_axis.grid(axis="x", alpha=0.2)
        overall_axis.spines[["left", "right", "top"]].set_visible(False)

        pass_axis.set_xlim(-0.5, n_passes - 0.5)
        pass_axis.set_ylim(0, 1.0)
        pass_axis.bar(
            np.arange(n_passes),
            fractions_by_pass[:, row_index],
            color=pass_colors,
            width=0.7,
        )
        pass_axis.set_xticks(np.arange(n_passes))
        pass_axis.set_xticklabels([f"P{i + 1}" for i in range(n_passes)], fontsize=7)
        pass_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        pass_axis.grid(axis="y", alpha=0.2)

    for row_index in range(len(omega), nrows * args.n_cols):
        row, col = divmod(row_index, args.n_cols)
        for panel in range(3):
            grid[row, col, panel].set_visible(False)

    for col in range(args.n_cols):
        axes[-1, 3 * col].set_xlabel("sample")
        axes[-1, 3 * col + 1].set_xlabel("fraction of fitted spikes")
        axes[-1, 3 * col + 2].set_xlabel("fraction within pass")
        axes[-1, 3 * col + 1].xaxis.set_major_formatter(PercentFormatter(1.0))

    figure.suptitle(
        f"Q={len(omega)} temporal codebook · values and usage across "
        f"{n_total:,} spikes · {n_passes} residual passes"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800)
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)
    for row_index, (count, fraction) in enumerate(zip(counts_total, fractions_total)):
        print(
            f"row {row_index:2d}: {count:9,d} ({100 * fraction:6.2f}%)  "
            + "  ".join(f"P{p + 1} {100 * v:.1f}%" for p, v in enumerate(fractions_by_pass[:, row_index])),
            flush=True,
        )


if __name__ == "__main__":
    main()