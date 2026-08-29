"""Plot the fitted spatial-spread distribution across residual passes."""

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
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    sigma = np.load(args.run / "sigma.npy", mmap_mode="r")
    residual_pass = np.load(args.run / "residual_pass.npy", mmap_mode="r")
    if sigma.shape != residual_pass.shape:
        raise ValueError("sigma and residual_pass must be event-aligned")
    valid = np.isfinite(sigma) & (sigma > 0) & (residual_pass >= 0)
    if not np.any(valid):
        raise ValueError("no finite positive spatial spreads")
    sigma_values = np.unique(np.asarray(sigma[valid]))
    n_passes = int(residual_pass[valid].max()) + 1
    counts = np.zeros((n_passes, len(sigma_values)), dtype=np.int64)
    for pass_index in range(n_passes):
        values = np.asarray(sigma[valid & (residual_pass == pass_index)])
        indices = np.searchsorted(sigma_values, values)
        counts[pass_index] = np.bincount(indices, minlength=len(sigma_values))
    fractions = counts / counts.sum(axis=1, keepdims=True)

    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    image = axis.imshow(
        fractions, origin="upper", aspect="auto", cmap="inferno", vmin=0,
        vmax=max(0.01, float(fractions.max())), interpolation="nearest",
    )
    for row in range(n_passes):
        for column in range(len(sigma_values)):
            value = fractions[row, column]
            axis.text(
                column, row, f"{100 * value:.1f}%\n{counts[row, column]:,}",
                ha="center", va="center", fontsize=8,
                color="white" if value < 0.58 * fractions.max() else "black",
            )
    axis.set(
        xlabel="fitted spatial spread σ (µm)", ylabel="residual pass",
        xticks=np.arange(len(sigma_values)),
        xticklabels=[f"{value:g}" for value in sigma_values],
        yticks=np.arange(n_passes),
        yticklabels=[f"pass {index + 1} · {counts[index].sum():,} fits" for index in range(n_passes)],
        title="accepted-localization spatial spread by residual pass",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("fraction of accepted localizations within pass")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
