"""Show padded spike neighborhoods across the probe and in example patches."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle
import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from maths import load_channel_map


COLORS = ["#238b45", "#9ecae1", "#fec44f", "#fc8d59", "#b30000"]


def channel_neighborhood_sizes(anchors, counts, n_channels, max_neighbors):
    low = np.full(n_channels, max_neighbors + 1, dtype=np.int16)
    high = np.full(n_channels, -1, dtype=np.int16)
    np.minimum.at(low, anchors, counts)
    np.maximum.at(high, anchors, counts)
    used = high >= 0
    if np.any(low[used] != high[used]):
        raise ValueError("neighborhood size is not fixed for an anchor channel")
    sizes = np.full(n_channels, -1, dtype=np.int16)
    sizes[used] = low[used]
    return sizes, used


def plot_summary(out, positions, anchors, counts, sizes, used, max_neighbors):
    missing = max_neighbors - sizes
    cmap = ListedColormap(COLORS[:max_neighbors - counts.min() + 1])
    norm = BoundaryNorm(
        np.arange(-0.5, max_neighbors - counts.min() + 1.5), cmap.N)

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(0.85, 1.1, 1.25))
    ax_probe = fig.add_subplot(gs[:, 0])
    ax_depth = fig.add_subplot(gs[:, 1], sharey=ax_probe)
    ax_counts = fig.add_subplot(gs[0, 2])
    ax_columns = fig.add_subplot(gs[1, 2])

    image = ax_probe.scatter(
        positions[used, 0], positions[used, 1], c=missing[used], s=16,
        marker="s", cmap=cmap, norm=norm, linewidths=0,
    )
    ax_probe.set(
        xlabel="contact x (\u00b5m; expanded)", ylabel="probe depth (\u00b5m)",
        title="padding by anchor contact",
    )
    ax_probe.set_xlim(positions[:, 0].min() - 12, positions[:, 0].max() + 12)
    ticks = np.arange(cmap.N)
    cbar = fig.colorbar(image, ax=ax_probe, ticks=ticks, pad=0.03)
    cbar.set_label("padded slots out of 8")

    anchor_depth = positions[anchors, 1]
    depth_edges = np.arange(
        positions[:, 1].min(), positions[:, 1].max() + 101, 100)
    total, _ = np.histogram(anchor_depth, bins=depth_edges)
    padded, _ = np.histogram(anchor_depth[counts < max_neighbors], bins=depth_edges)
    fraction = np.divide(
        padded, total, out=np.zeros_like(padded, dtype=float), where=total > 0)
    centers = (depth_edges[:-1] + depth_edges[1:]) / 2
    overall = float(np.mean(counts < max_neighbors))
    ax_depth.fill_betweenx(centers, 0, fraction, color="#3182bd", alpha=0.35)
    ax_depth.plot(fraction, centers, color="#08519c", linewidth=1.2)
    ax_depth.axvline(overall, color="#cb181d", linestyle="--", linewidth=1,
                    label=f"all depths: {overall:.1%}")
    ax_depth.set(
        xlim=(0, 1), xlabel="spikes with padded slots", title="padding vs. depth",
    )
    ax_depth.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1))
    ax_depth.tick_params(labelleft=False)
    ax_depth.grid(alpha=0.2)
    ax_depth.legend(loc="upper right", fontsize=8)

    possible = np.arange(int(counts.min()), max_neighbors + 1)
    number = np.array([(counts == value).sum() for value in possible])
    percent = 100 * number / len(counts)
    ax_counts.bar(
        possible, percent,
        color=[COLORS[max_neighbors - value] for value in possible],
    )
    for value, height, n_spikes in zip(possible, percent, number):
        ax_counts.text(value, height + 1, f"{height:.2f}%\n{n_spikes:,}",
                       ha="center", va="bottom", fontsize=7)
    ax_counts.set(
        xlabel="real channels in 8-slot patch", ylabel="spikes (%)",
        title="neighborhood-size distribution", xticks=possible,
    )
    ax_counts.set_ylim(0, max(percent) * 1.18)
    ax_counts.grid(alpha=0.2, axis="y")

    anchor_x = positions[anchors, 0]
    x_values = np.unique(positions[:, 0])
    bottom = np.zeros(len(x_values))
    for n_missing in range(max_neighbors - int(counts.min()) + 1):
        value = max_neighbors - n_missing
        share = np.zeros(len(x_values))
        for j, x in enumerate(x_values):
            at_x = anchor_x == x
            share[j] = np.mean(counts[at_x] == value) if np.any(at_x) else 0
        ax_columns.bar(
            np.arange(len(x_values)), share, bottom=bottom,
            color=COLORS[n_missing], label=f"{value} real / {n_missing} padded",
        )
        bottom += share
    ax_columns.set(
        xlabel="anchor-contact x (\u00b5m)", ylabel="fraction of spikes",
        title="interior padding follows probe column",
        xticks=np.arange(len(x_values)), xticklabels=[f"{x:g}" for x in x_values],
        ylim=(0, 1),
    )
    ax_columns.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1))
    ax_columns.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)

    fig.suptitle(
        f"Saved 48 \u00b5m channel neighborhoods \u00b7 {len(counts):,} spikes \u00b7 "
        f"{overall:.1%} have padding"
    )
    fig.savefig(out / "missing_channels_on_probe.png", dpi=800,
                bbox_inches="tight")
    plt.close(fig)


def closest_channel(sizes, positions, count, target_depth):
    channels = np.flatnonzero(sizes == count)
    return int(channels[np.argmin(np.abs(positions[channels, 1] - target_depth))])


def plot_examples(out, positions, anchors, ids, sizes, max_neighbors):
    _, first = np.unique(anchors, return_index=True)
    first_row = np.full(len(positions), -1, dtype=np.int64)
    first_row[anchors[first]] = first
    lo, hi = positions[:, 1].min(), positions[:, 1].max()
    middle = (lo + hi) / 2
    examples = [
        ("bottom edge", closest_channel(sizes, positions, 4, lo)),
        ("bottom edge", closest_channel(sizes, positions, 5, lo)),
        ("interior outer column", closest_channel(sizes, positions, 6, middle)),
        ("interior inner column", closest_channel(sizes, positions, 8, middle)),
        ("top edge", closest_channel(sizes, positions, 7, hi)),
        ("top edge", closest_channel(sizes, positions, 4, hi)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), constrained_layout=True)
    for ax, (where, channel) in zip(axes.flat, examples):
        anchor = positions[channel]
        selected = ids[first_row[channel]]
        selected = selected[selected >= 0]
        local = (
            (np.abs(positions[:, 0] - anchor[0]) <= 70)
            & (np.abs(positions[:, 1] - anchor[1]) <= 70)
        )
        ax.scatter(
            positions[local, 0], positions[local, 1], s=38, marker="s",
            facecolors="none", edgecolors="#969696", linewidths=0.8,
            label="nearby contacts",
        )
        ax.scatter(
            positions[selected, 0], positions[selected, 1], s=48, marker="s",
            color="#2b8cbe", edgecolors="white", linewidths=0.5,
            label="saved real channels",
        )
        ax.scatter(
            anchor[0], anchor[1], s=110, marker="*", color="#de2d26",
            edgecolors="black", linewidths=0.5, label="anchor",
        )
        ax.add_patch(Circle(anchor, 48, fill=False, color="#756bb1",
                            linestyle="--", linewidth=1))
        for selected_channel in selected:
            xy = positions[selected_channel]
            ax.text(xy[0] + 2, xy[1] + 2, str(selected_channel), fontsize=5)
        ax.set(
            xlim=(anchor[0] - 70, anchor[0] + 70),
            ylim=(anchor[1] - 70, anchor[1] + 70),
            xlabel="contact x (\u00b5m)", ylabel="probe depth (\u00b5m)",
            title=(f"{where}: channel {channel}\n{sizes[channel]} real, "
                   f"{max_neighbors - sizes[channel]} padded"),
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("Representative saved neighborhoods (dashed circle = 48 \u00b5m)")
    fig.savefig(out / "missing_channel_examples.png", dpi=800,
                bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("residuals/out/plots/missing_channels"))
    args = parser.parse_args()

    counts = np.asarray(np.load(args.session / "neighbor_counts.npy", mmap_mode="r"))
    ids = np.load(args.session / "neighbor_ids.npy", mmap_mode="r")
    anchors = np.asarray(np.load(args.session / "spike_channels.npy", mmap_mode="r"))
    positions = np.asarray(load_channel_map(args.recording).contact_positions)
    max_neighbors = ids.shape[1]
    if not (len(counts) == len(ids) == len(anchors)):
        raise ValueError("spike neighborhood arrays have different lengths")
    sizes, used = channel_neighborhood_sizes(
        anchors, counts, len(positions), max_neighbors)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_summary(args.out, positions, anchors, counts, sizes, used, max_neighbors)
    plot_examples(args.out, positions, anchors, ids, sizes, max_neighbors)
    values, number = np.unique(counts, return_counts=True)
    print("neighborhood sizes:", dict(zip(values.tolist(), number.tolist())))
    print(f"padded spikes: {(counts < max_neighbors).sum():,}/{len(counts):,} "
          f"({np.mean(counts < max_neighbors):.1%})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
