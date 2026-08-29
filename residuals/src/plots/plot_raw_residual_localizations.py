"""Plot probe-global localizations from a raw residual run."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def localization_density(horizontal, depth, x_edges, depth_edges):
    counts, _, _ = np.histogram2d(depth, horizontal, bins=(depth_edges, x_edges))
    return counts


def draw_density(axis, counts, x_edges, depth_edges, contacts, title, norm):
    image = axis.imshow(
        counts,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], depth_edges[0], depth_edges[-1]),
        aspect="auto",
        cmap="inferno",
        norm=norm,
        interpolation="nearest",
    )
    axis.scatter(
        contacts[:, 0],
        contacts[:, 1],
        s=3,
        marker="s",
        facecolors="none",
        edgecolors="#55d6d6",
        linewidths=0.25,
    )
    axis.set(title=title, xlabel="global lateral position (µm)", ylabel="probe depth (µm)")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sources = np.load(args.run / "global_sources.npy", mmap_mode="r")
    residual_pass = np.load(args.run / "residual_pass.npy", mmap_mode="r")
    captured_fraction = np.load(args.run / "captured_fraction.npy", mmap_mode="r")
    contacts = np.load(args.run / "channel_positions.npy")

    horizontal = np.asarray(sources[:, 0])
    depth = np.asarray(sources[:, 1])
    axial = np.asarray(sources[:, 2])
    x_edges = np.linspace(contacts[:, 0].min() - 170, contacts[:, 0].max() + 170, 321)
    depth_edges = np.linspace(
        contacts[:, 1].min() - 170, contacts[:, 1].max() + 170, 769
    )
    selections = [np.ones(len(sources), dtype=bool)]
    selections.extend(residual_pass == index for index in range(4))
    counts = [
        localization_density(horizontal[keep], depth[keep], x_edges, depth_edges)
        for keep in selections
    ]
    nonzero = np.concatenate([item[item > 0] for item in counts])
    vmax = max(1.0, float(np.percentile(nonzero, 99.7)))
    norm = LogNorm(vmin=1, vmax=vmax)

    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    labels = ["all residual passes"] + [f"residual pass {index + 1}" for index in range(4)]
    images = []
    for axis, keep, count, label in zip(axes.flat[:5], selections, counts, labels):
        median = float(np.median(np.asarray(captured_fraction)[keep]))
        title = f"{label} · {int(keep.sum()):,} fits · median capture {median:.3f}"
        images.append(draw_density(axis, count, x_edges, depth_edges, contacts, title, norm))

    axis = axes.flat[5]
    bins = np.linspace(0, float(np.percentile(axial, 99.9)), 121)
    for index in range(4):
        keep = residual_pass == index
        axis.hist(
            axial[keep],
            bins=bins,
            histtype="step",
            linewidth=1.2,
            density=True,
            label=f"pass {index + 1}",
        )
    axis.set(
        title="distance from probe plane",
        xlabel="axial distance (µm)",
        ylabel="density",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.colorbar(images[0], ax=axes[:, :2], label="fits per spatial bin", pad=0.01)
    figure.suptitle(f"raw residual localizations · {len(sources):,} accepted fits")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
