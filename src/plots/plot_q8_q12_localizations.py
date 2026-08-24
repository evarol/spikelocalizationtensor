"""Compare paired localizations from the Q8 and Q12 temporal codebooks."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--q12", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    centroids = np.load(args.session / "centroids.npy", mmap_mode="r")
    with np.load(args.q8, allow_pickle=False) as archive:
        q8 = np.asarray(archive["sources"], dtype=np.float32)
    with np.load(args.q12, allow_pickle=False) as archive:
        q12 = np.asarray(archive["sources"], dtype=np.float32)
    if q8.shape != q12.shape or len(q8) != len(centroids):
        raise ValueError(
            f"paired shapes disagree: q8={q8.shape}, q12={q12.shape}, "
            f"centroids={centroids.shape}"
        )

    rng = np.random.default_rng(args.seed)
    keep = np.sort(rng.choice(len(q8), min(args.max_points, len(q8)), replace=False))
    global_q8 = np.asarray(centroids[keep, :2]) + q8[keep, :2]
    global_q12 = np.asarray(centroids[keep, :2]) + q12[keep, :2]
    delta = q12 - q8
    displacement = np.linalg.norm(delta, axis=1)
    finite = np.isfinite(displacement)
    display_hi = float(np.quantile(displacement[finite], 0.995))

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for axis, coordinates, label in (
        (axes[0, 0], global_q8, "Q=8"),
        (axes[0, 1], global_q12, "Q=12"),
    ):
        axis.scatter(
            coordinates[:, 0], coordinates[:, 1], s=0.15, color="#252525",
            alpha=0.35, rasterized=True,
        )
        axis.set(
            xlabel="global x (µm)", ylabel="global y (µm)",
            title=f"{label} discrete 1 µm localizations",
        )

    image = axes[1, 0].hist2d(
        delta[finite, 0], delta[finite, 1], bins=121,
        range=((-60, 60), (-60, 60)), cmap="magma", norm=LogNorm(),
    )
    figure.colorbar(image[3], ax=axes[1, 0], label="paired spikes")
    axes[1, 0].axhline(0, color="white", linewidth=0.6, alpha=0.7)
    axes[1, 0].axvline(0, color="white", linewidth=0.6, alpha=0.7)
    axes[1, 0].set(
        xlabel=r"$x_{Q12}-x_{Q8}$ (µm)",
        ylabel=r"$y_{Q12}-y_{Q8}$ (µm)",
        title="paired lateral localization change",
    )

    axes[1, 1].hist(
        displacement[finite & (displacement <= display_hi)], bins=100,
        color="#3182bd",
    )
    median = float(np.median(displacement[finite]))
    p90 = float(np.quantile(displacement[finite], 0.9))
    unchanged = float(np.mean(displacement[finite] == 0))
    axes[1, 1].axvline(median, color="#cb181d", linestyle="--",
                       label=f"median {median:.1f} µm")
    axes[1, 1].axvline(p90, color="#756bb1", linestyle=":",
                       label=f"p90 {p90:.1f} µm")
    axes[1, 1].set(
        xlabel="paired 3D displacement (µm)", ylabel="spikes",
        title=f"Q8 → Q12 change · unchanged {100 * unchanged:.1f}%",
    )
    axes[1, 1].legend()
    figure.suptitle(
        f"Temporal-codebook effect on localization · {len(keep):,} displayed / "
        f"{len(q8):,} paired spikes"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800)
    plt.close(figure)
    print(
        f"paired displacement: median={median:.6f} um, p90={p90:.6f} um, "
        f"unchanged={100 * unchanged:.4f}%",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
