"""Compare two learned temporal codebooks and their fit diagnostics."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_fit(path):
    with np.load(path, allow_pickle=True) as archive:
        omega = np.asarray(archive["omega"], dtype=np.float32)
        temporal_idx = np.asarray(archive["temporal_idx"], dtype=np.int64)
        history = archive["history"].tolist()
        nmse = float(archive["nmse"])
    order = np.lexsort((np.arange(len(omega)), np.argmin(omega, axis=1)))
    counts = np.bincount(temporal_idx, minlength=len(omega)).astype(np.float64)
    return {
        "omega": omega[order],
        "fraction": counts[order] / counts.sum(),
        "history": history,
        "nmse": nmse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--q12", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    fits = (("Q=8", load_fit(args.q8)), ("Q=12", load_fit(args.q12)))
    limit = max(float(np.abs(fit["omega"]).max()) for _, fit in fits)
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.4, 1, 1)},
    )
    image = None
    for row, (label, fit) in enumerate(fits):
        omega = fit["omega"]
        image = axes[row, 0].imshow(
            omega, aspect="auto", interpolation="nearest", cmap="RdBu_r",
            vmin=-limit, vmax=limit,
        )
        axes[row, 0].set(
            xlabel="sample", ylabel="row sorted by trough time",
            title=f"{label} temporal dictionary · {omega.shape[1]} samples",
            yticks=np.arange(len(omega)),
        )
        axes[row, 1].bar(
            np.arange(len(omega)), fit["fraction"], color="#756bb1"
        )
        axes[row, 1].set(
            xlabel="row sorted by trough time", ylabel="spike fraction",
            title=f"{label} row usage", xticks=np.arange(len(omega)),
        )
        steps = [entry["step"] for entry in fit["history"]]
        losses = [entry["nmse"] for entry in fit["history"]]
        axes[row, 2].plot(steps, losses, "-o", color="#3182bd", markersize=4)
        axes[row, 2].axhline(
            fit["nmse"], color="#cb181d", linestyle="--",
            label=f"refined {fit['nmse']:.4f}",
        )
        axes[row, 2].set(
            xlabel="alternating-fit iteration", ylabel="nMSE",
            title=f"{label} convergence",
        )
        axes[row, 2].grid(alpha=0.25)
        axes[row, 2].legend(fontsize=8)
    fig.colorbar(image, ax=axes[:, 0], label="normalized amplitude", shrink=0.85)
    fig.suptitle("Global temporal-codebook comparison")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=800)
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
