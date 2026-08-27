"""Plot recording-wide usage of every learned temporal waveform."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-cols", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_cols < 1:
        raise ValueError("--n-cols must be a positive integer")

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")

    with np.load(args.fit, allow_pickle=True) as fit:
        omega = np.asarray(fit["omega"], dtype=np.float32)
        temporal_idx = np.asarray(fit["temporal_idx"], dtype=np.int64)
    if omega.ndim != 2:
        raise ValueError(f"expected omega to have shape (Q, T), got {omega.shape}")
    if temporal_idx.ndim != 1:
        raise ValueError(
            f"expected temporal_idx to have shape (spikes,), got {temporal_idx.shape}"
        )
    if len(temporal_idx) == 0:
        raise ValueError("cannot plot usage for an empty recording")
    if temporal_idx.min() < 0 or temporal_idx.max() >= len(omega):
        raise ValueError("temporal_idx contains a row outside the temporal codebook")

    counts = np.bincount(temporal_idx, minlength=len(omega))
    fractions = counts / counts.sum()
    colors = plt.colormaps["tab20"](np.arange(len(omega)) % 20)
    amplitude_limit = max(float(np.abs(omega).max()), np.finfo(np.float32).eps)
    usage_limit = max(0.01, float(fractions.max()) * 1.32)
    samples = np.arange(omega.shape[1])

    nrows = int(np.ceil(len(omega) / args.n_cols))
    figure, axes = plt.subplots(
        nrows, 2 * args.n_cols,
        figsize=(4.2 * args.n_cols, 1.18 * nrows + 1.5),
        sharex="col", constrained_layout=True,
        gridspec_kw={"width_ratios": (3.4, 1.6) * args.n_cols},
    )
    axes = np.atleast_2d(axes)
    grid = axes.reshape(nrows, args.n_cols, 2)
    for pair_index, color in enumerate(colors):
        row, col = divmod(pair_index, args.n_cols)
        waveform_axis = grid[row, col, 0]
        usage_axis = grid[row, col, 1]
        waveform_axis.plot(samples, omega[pair_index], color=color, linewidth=1.3)
        waveform_axis.axhline(0, color="0.8", linewidth=0.6)
        waveform_axis.set_ylim(-1.05 * amplitude_limit, 1.05 * amplitude_limit)
        waveform_axis.set_ylabel(rf"$\Omega_{{{pair_index}}}$")
        waveform_axis.grid(alpha=0.18)

        usage_axis.barh(0, fractions[pair_index], color=color, height=0.58)
        usage_axis.text(
            fractions[pair_index] + usage_limit * 0.018, 0,
            f"{100 * fractions[pair_index]:.1f}%  ({counts[pair_index]:,})",
            va="center", fontsize=8,
        )
        usage_axis.set_xlim(0, usage_limit)
        usage_axis.set_ylim(-0.75, 0.75)
        usage_axis.set_yticks([])
        usage_axis.grid(axis="x", alpha=0.2)
        usage_axis.spines[["left", "right", "top"]].set_visible(False)

    for pair_index in range(len(omega), nrows * args.n_cols):
        row, col = divmod(pair_index, args.n_cols)
        grid[row, col, 0].set_visible(False)
        grid[row, col, 1].set_visible(False)

    for col in range(args.n_cols):
        axes[-1, 2 * col].set_xlabel("sample")
        axes[-1, 2 * col + 1].set_xlabel("fraction of fitted spikes")
        axes[-1, 2 * col + 1].xaxis.set_major_formatter(PercentFormatter(1.0))
    figure.suptitle(
        f"Q={len(omega)} temporal-waveform usage across {len(temporal_idx):,} spikes"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800)
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)
    for row, (count, fraction) in enumerate(zip(counts, fractions)):
        print(f"row {row:2d}: {count:9,d} ({100 * fraction:6.2f}%)", flush=True)


if __name__ == "__main__":
    main()
