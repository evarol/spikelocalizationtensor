"""Plot the learned temporal codebook from a saved fit."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-cols", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.n_cols < 1:
        raise ValueError("--n-cols must be a positive integer")

    fit = np.load(args.fit, allow_pickle=True)
    omega = np.asarray(fit["omega"])
    temporal_idx = np.asarray(fit["temporal_idx"])
    if omega.ndim != 2:
        raise ValueError(f"expected omega to have shape (Q, T), got {omega.shape}")
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")

    counts = np.bincount(temporal_idx, minlength=len(omega))
    limit = max(float(np.abs(omega).max()), np.finfo(np.float32).eps)
    colors = plt.colormaps["tab10"](np.arange(len(omega)) % 10)
    nrows = int(np.ceil(len(omega) / args.n_cols))
    fig, axes = plt.subplots(
        nrows, args.n_cols, figsize=(3.2 * args.n_cols, 1.5 * nrows + 1.0),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    samples = np.arange(omega.shape[1])
    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(omega):
            ax.set_visible(False)
            continue
        q = idx
        ax.plot(samples, omega[q], color=colors[q], linewidth=1.2)
        ax.axhline(0, color="0.75", linewidth=0.6)
        ax.set_ylim(-1.05 * limit, 1.05 * limit)
        ax.set_ylabel(rf"$\Omega_{{{q}}}$")
        ax.text(0.005, 0.82, f"n={counts[q]:,}", transform=ax.transAxes,
                fontsize=8, color="0.35")
        ax.grid(alpha=0.2)
    for col in range(args.n_cols):
        axes[-1, col].set_xlabel("sample")
    fig.suptitle(f"temporal codebook · {len(omega)} rows × {omega.shape[1]} samples")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=800)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
