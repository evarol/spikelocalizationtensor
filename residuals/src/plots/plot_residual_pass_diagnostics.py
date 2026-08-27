"""Summarize how detections and fitted energy evolve across residual passes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def nearest_prior_event_ms(times, channels, passes, positions, fs, radius_um):
    distance = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=2
    )
    nearby_channels = [np.flatnonzero(row <= radius_um) for row in distance]
    result = {}
    for residual_pass in range(1, int(passes.max()) + 1):
        prior = passes < residual_pass
        lookup = {
            channel: np.sort(times[prior & (channels == channel)])
            for channel in range(len(positions))
        }
        rows = np.flatnonzero(passes == residual_pass)
        separation = np.full(len(rows), np.inf, dtype=np.float64)
        for output_index, row in enumerate(rows):
            event_time = times[row]
            for channel in nearby_channels[channels[row]]:
                candidates = lookup[channel]
                insertion = np.searchsorted(candidates, event_time)
                if insertion < len(candidates):
                    separation[output_index] = min(
                        separation[output_index], candidates[insertion] - event_time
                    )
                if insertion:
                    separation[output_index] = min(
                        separation[output_index], event_time - candidates[insertion - 1]
                    )
        result[residual_pass] = 1000 * separation / fs
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    args = parser.parse_args()

    metadata = json.loads((args.run / "config.json").read_text())
    config = metadata["config"]
    fs = float(metadata["fs"])
    chunk_samples = max(1, int(round(float(config["chunk_seconds"]) * fs)))
    chunk_start = int(metadata["first_sample"]) + args.chunk_index * chunk_samples
    chunk_path = args.run / "chunks" / f"chunk_{args.chunk_index:06d}.npz"
    with np.load(chunk_path, allow_pickle=False) as archive:
        times = np.asarray(archive["spike_times"], dtype=np.int64)
        channels = np.asarray(archive["spike_channels"], dtype=np.int64)
        passes = np.asarray(archive["residual_pass"], dtype=np.int64)
        temporal_idx = np.asarray(archive["temporal_idx"], dtype=np.int64)
        captured_fraction = np.asarray(archive["captured_fraction"], dtype=np.float64)
        energy_drop = np.asarray(
            archive["pass_energy_drop_fraction"], dtype=np.float64
        )
    positions = np.load(args.run / "channel_positions.npy")
    omega = np.load(args.run / "omega.npy")

    n_passes = int(passes.max()) + 1
    pass_numbers = np.arange(1, n_passes + 1)
    display_ticks = pass_numbers
    if n_passes > 12:
        display_ticks = np.unique(
            np.rint(np.linspace(1, n_passes, 10)).astype(np.int64)
        )
    counts = np.bincount(passes, minlength=n_passes)
    drops = np.array(
        [np.median(energy_drop[passes == index]) for index in range(n_passes)]
    )
    remaining_rms = np.sqrt(np.cumprod(1 - drops))
    recurrence = nearest_prior_event_ms(
        times,
        channels,
        passes,
        positions,
        fs,
        float(config["radius_um"]),
    )
    colors = plt.colormaps["viridis"](np.linspace(0.12, 0.88, n_passes))

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    relative_ms = 1000 * (times - chunk_start) / fs
    for index in range(n_passes):
        keep = passes == index
        axes[0, 0].scatter(
            relative_ms[keep],
            np.full(keep.sum(), index + 1),
            s=0.8,
            alpha=0.35,
            color=colors[index],
            rasterized=True,
        )
    axes[0, 0].set(
        title="accepted-event raster",
        xlabel="time in chunk (ms)",
        ylabel="residual pass",
        yticks=display_ticks,
    )

    bars = axes[0, 1].bar(pass_numbers, counts, color=colors)
    if n_passes <= 12:
        axes[0, 1].bar_label(
            bars, labels=[f"{count:,}" for count in counts], padding=3
        )
    axes[0, 1].set(
        title="accepted fits remain nearly flat",
        xlabel="residual pass",
        ylabel="accepted fits",
        xticks=display_ticks,
    )

    samples = [captured_fraction[passes == index] for index in range(n_passes)]
    violin = axes[0, 2].violinplot(
        samples, positions=pass_numbers, showmedians=True, showextrema=False
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    violin["cmedians"].set_color("black")
    axes[0, 2].set(
        title="sequential captured fraction",
        xlabel="residual pass",
        ylabel="fraction of local waveform energy",
        xticks=display_ticks,
        ylim=(0, min(1, np.percentile(captured_fraction, 99.8) * 1.08)),
    )

    width = 0.36
    axes[1, 0].bar(
        pass_numbers - width / 2,
        drops,
        width=width,
        color=colors,
        label="energy removed this pass",
    )
    axes[1, 0].bar(
        pass_numbers + width / 2,
        remaining_rms,
        width=width,
        color="0.45",
        label="cumulative RMS remaining",
    )
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].set(
        title="residual energy trajectory",
        xlabel="residual pass",
        ylabel="fraction of input",
        xticks=display_ticks,
        ylim=(0, 1),
    )
    axes[1, 0].legend(fontsize=8)

    limits = np.linspace(0, 2.0, 401)
    recurrence_text = []
    legend_passes = set(
        np.unique(
            np.rint(
                np.linspace(1, n_passes - 1, min(6, n_passes - 1))
            ).astype(np.int64)
        )
    )
    for residual_pass, separation in recurrence.items():
        finite = separation[np.isfinite(separation)]
        cdf = np.searchsorted(np.sort(finite), limits, side="right") / len(separation)
        axes[1, 1].plot(
            limits, cdf, color=colors[residual_pass],
            label=(
                f"pass {residual_pass + 1}"
                if residual_pass in legend_passes
                else None
            ),
        )
        recurrence_text.append(
            f"P{residual_pass + 1}: {100 * np.mean(separation <= 0.5):.1f}%"
        )
    axes[1, 1].axvline(0.5, color="0.3", linestyle="--", linewidth=0.8)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].set(
        title="proximity to any earlier-pass event",
        xlabel="nearest event within 48 µm (ms)",
        ylabel="later-pass events within distance",
        xlim=(0, 2),
        ylim=(0, 1),
    )
    axes[1, 1].legend(fontsize=8)

    usage = np.zeros((n_passes, len(omega)), dtype=np.float64)
    for index in range(n_passes):
        usage[index] = np.bincount(
            temporal_idx[passes == index], minlength=len(omega)
        ) / counts[index]
    image = axes[1, 2].imshow(
        usage,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=max(0.01, float(usage.max())),
        interpolation="nearest",
    )
    if n_passes * len(omega) <= 256:
        for row in range(n_passes):
            for column in range(len(omega)):
                axes[1, 2].text(
                    column, row, f"{100 * usage[row, column]:.1f}",
                    ha="center", va="center", fontsize=7,
                    color=(
                        "white"
                        if usage[row, column] > 0.55 * usage.max()
                        else "black"
                    ),
                )
    axes[1, 2].set(
        title="temporal-waveform usage (%)",
        xlabel="temporal row",
        ylabel="residual pass",
        xticks=np.arange(len(omega)),
        yticks=display_ticks - 1,
        yticklabels=display_ticks,
    )
    figure.colorbar(image, ax=axes[1, 2], label="fraction of pass fits")

    figure.suptitle(
        f"Residual peeling diagnostics · chunk {args.chunk_index} · "
        f"{len(times):,} accepted fits"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)
    print("within 0.5 ms of an earlier nearby event: " + ", ".join(recurrence_text))


if __name__ == "__main__":
    main()
