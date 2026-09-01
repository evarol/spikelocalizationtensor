"""Render the full 0019 recording through its saved residual passes."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from preprocessing.raw_residual import preprocess_voltage


PASS_COLORS = {
    0: np.asarray((0.12, 0.31, 0.47), dtype=np.float32),
    1: np.asarray((0.88, 0.48, 0.00), dtype=np.float32),
    2: np.asarray((0.75, 0.07, 0.18), dtype=np.float32),
}


def pass_colour_raster(x, y, pass_id, xlim, ylim, nx, ny):
    ix = np.floor((x - xlim[0]) * nx / (xlim[1] - xlim[0])).astype(np.int64)
    iy = np.floor((y - ylim[0]) * ny / (ylim[1] - ylim[0])).astype(np.int64)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[keep] * nx + ix[keep]
    count = np.bincount(flat, minlength=nx * ny).reshape(ny, nx).astype(np.float32)
    rgb = np.zeros((ny, nx, 3), dtype=np.float32)
    for pass_index, color in PASS_COLORS.items():
        selected = keep & (pass_id == pass_index)
        if not selected.any():
            continue
        event_flat = iy[selected] * nx + ix[selected]
        counts = np.bincount(event_flat, minlength=nx * ny).reshape(ny, nx)
        rgb += counts[..., None] * color
    mass = gaussian_filter(count, 0.5)
    for channel in range(3):
        rgb[..., channel] = gaussian_filter(
            rgb[..., channel], 0.5
        ) / np.maximum(mass, 1e-6)
    intensity = np.clip(1.0 * (1 - np.exp(-mass / 0.35)), 0, 1)
    white = np.ones_like(rgb)
    return white * (1 - intensity[..., None]) + rgb * intensity[..., None]


def signed_block_extrema(values, samples_per_bin):
    n_bins = (len(values) + samples_per_bin - 1) // samples_per_bin
    padded = np.full(
        (n_bins * samples_per_bin, values.shape[1]), np.nan, dtype=values.dtype
    )
    padded[: len(values)] = values
    blocks = padded.reshape(n_bins, samples_per_bin, values.shape[1])
    peaks = np.nan_to_num(np.abs(blocks)).argmax(axis=1)
    return np.take_along_axis(blocks, peaks[:, None, :], axis=1)[:, 0]


def load_pass_events(run, recording_passes):
    passes = []
    for recording_pass in range(recording_passes):
        directory = run / f"pass_{recording_pass:02d}"
        times, ids, preds = [], [], []
        for path in sorted(directory.glob("chunk_*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                times.append(archive["spike_times"])
                ids.append(archive["neighbor_ids"])
                preds.append(archive["predictions"])
        if times:
            passes.append(
                {
                    "times": np.concatenate(times),
                    "ids": np.concatenate(ids),
                    "predictions": np.concatenate(preds),
                    "cursor": 0,
                }
            )
        else:
            passes.append(None)
    if passes[0] is None:
        raise FileNotFoundError(f"no pass_00 chunks under {run}")
    return passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-bin", type=int, default=5000)
    parser.add_argument("--first-chunk", type=int, default=0)
    parser.add_argument("--last-chunk", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.samples_per_bin < 1:
        raise ValueError("samples per bin must be positive")

    metadata = json.loads((args.run / "config.json").read_text())
    fs = float(metadata["fs"])
    n_channels = int(metadata["n_channels"])
    config = metadata["config"]
    chunk_samples = max(1, int(round(config["chunk_seconds"] * fs)))
    n_before = int(round(config["ms_before"] * fs / 1000))
    n_after = int(round(config["ms_after"] * fs / 1000))
    margin = max(
        int(round(config["read_margin_ms"] * fs / 1000)), n_before + n_after, 128
    )
    bar = float(config["all_channel_min_fraction"])
    recording_passes = int(metadata["passes"]["recording_passes"])
    pass_bars = "/".join(f"{bar + 0.1 * index:.0%}" for index in range(recording_passes))

    passes = load_pass_events(args.run, recording_passes)
    pass_totals = [len(stage["times"]) if stage else 0 for stage in passes]

    recording_path = Path(metadata["recording_path"])
    meta = dict(
        line.split("=", 1)
        for line in recording_path.with_suffix(".meta").read_text().splitlines()
        if "=" in line
    )
    ap, lf, sy = (int(part) for part in meta["snsApLfSy"].split(","))
    total_samples = recording_path.stat().st_size // (2 * (ap + lf + sy))
    raw = np.memmap(
        recording_path, dtype="<i2", mode="r", shape=(total_samples, ap + lf + sy)
    )

    contacts = np.load(args.run / "channel_positions.npy")
    channel_order = np.lexsort((contacts[:, 0], contacts[:, 1]))
    spike_times = np.load(args.run / "spike_times.npy", mmap_mode="r")
    spike_channels = np.load(args.run / "spike_channels.npy", mmap_mode="r")
    recording_pass = np.load(args.run / "recording_pass.npy", mmap_mode="r")
    global_sources = np.load(args.run / "global_sources.npy", mmap_mode="r")
    event_x = (np.asarray(spike_times, dtype=np.float64)
               - int(metadata["first_sample"])) / (60.0 * fs)
    event_depth = np.asarray(global_sources, dtype=np.float64)[:, 1]
    depth_low, depth_high = np.quantile(event_depth, (0.002, 0.998))
    cumulative = [
        np.flatnonzero(np.asarray(recording_pass) < recording_pass_limit)
        for recording_pass_limit in range(recording_passes + 1)
    ]

    n_chunks = (int(metadata["stop_sample"]) - int(metadata["first_sample"])
                + chunk_samples - 1) // chunk_samples
    last_chunk = n_chunks - 1 if args.last_chunk < 0 else min(args.last_chunk, n_chunks - 1)
    stages = [[] for _ in range(recording_passes + 1)]

    for index in range(args.first_chunk, last_chunk + 1):
        core_start = int(metadata["first_sample"]) + index * chunk_samples
        core_stop = min(core_start + chunk_samples, int(metadata["stop_sample"]))
        read_start = max(0, core_start - margin)
        read_stop = min(total_samples, core_stop + margin)
        data = preprocess_voltage(
            raw[read_start:read_stop, :n_channels].astype(np.float32) * 2.34375e-06,
            fs,
        ).astype(np.float32)
        noise_path = args.run / "pass_00" / f"chunk_{index:06d}.npz"
        with np.load(noise_path, allow_pickle=False) as archive:
            noise = np.asarray(archive["noise"], dtype=np.float32)
        residual = np.array(data, copy=True)
        local = slice(core_start - read_start, core_stop - read_start)
        stages[0].append(
            signed_block_extrema(residual[local] / noise[None], args.samples_per_bin)
        )
        for pass_index, stage in enumerate(passes, start=1):
            if stage is None:
                stages[pass_index].append(stages[pass_index - 1][-1])
                continue
            times = stage["times"]
            lo = stage["cursor"]
            hi = int(np.searchsorted(times, core_stop + n_after, side="right"))
            stage["cursor"] = hi
            for event in range(lo, hi):
                time = int(times[event])
                row_ids = stage["ids"][event]
                row_mask = row_ids >= 0
                start = time - read_start - n_before
                stop = time - read_start + n_after
                if start < 0 or stop > len(residual):
                    continue
                residual[start:stop, row_ids[row_mask]] -= (
                    stage["predictions"][event][row_mask].T
                )
            stages[pass_index].append(
                signed_block_extrema(residual[local] / noise[None], args.samples_per_bin)
            )

    stages = [
        np.concatenate(blocks, axis=0) if blocks else None for blocks in stages
    ]
    input_rms = float(np.sqrt(np.nanmean(stages[0].astype(np.float64) ** 2)))
    rms_percent = [
        100 * float(np.sqrt(np.nanmean(stage.astype(np.float64) ** 2))) / input_rms
        for stage in stages
    ]
    vlim = float(np.nanquantile(np.abs(stages[0]), 0.995))
    vlim = min(12.0, max(4.0, vlim))
    duration_minutes = (core_stop - int(metadata["first_sample"])) / (60.0 * fs)
    extent = (0.0, duration_minutes, 0, n_channels - 1)

    figure, axes = plt.subplots(
        len(stages),
        3,
        figsize=(24, 14),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1.0, 1.0]},
    )
    image = None
    for index, columns in enumerate(axes):
        left_axis, raster_axis, single_axis = columns
        image = left_axis.imshow(
            stages[index][:, channel_order].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        if index == 0:
            title = f"preprocessed input · RMS {rms_percent[index]:.1f}%"
        else:
            title = (
                f"after recording pass {index - 1} · RMS {rms_percent[index]:.1f}% · "
                f"{pass_totals[index - 1]:,} pass events"
            )
        left_axis.set_title(title, fontsize=10)
        left_axis.set_ylabel("probe depth")
        if index == 0:
            raster_axis.set_title("spikes added by passes (cumulative)", fontsize=10)
            raster_axis.text(
                0.5, 0.5, "no events yet",
                transform=raster_axis.transAxes, ha="center", va="center",
                color="#888888", fontsize=12,
            )
            single_axis.set_title("each pass's own events", fontsize=10)
            single_axis.text(
                0.5, 0.5, "—",
                transform=single_axis.transAxes, ha="center", va="center",
                color="#888888", fontsize=12,
            )
        else:
            rows = cumulative[index]
            raster = pass_colour_raster(
                event_x[rows],
                event_depth[rows],
                np.asarray(recording_pass)[rows],
                (0.0, duration_minutes),
                (depth_low, depth_high),
                1750,
                960,
            )
            raster_axis.imshow(
                raster,
                origin="lower",
                extent=(0.0, duration_minutes, depth_low, depth_high),
                aspect="auto",
                interpolation="nearest",
            )
            added = pass_totals[index - 1]
            raster_axis.set_title(f"+ pass {index - 1} events ({added:,})", fontsize=10)
            own = np.flatnonzero(np.asarray(recording_pass) == index - 1)
            if len(own):
                own_raster = pass_colour_raster(
                    event_x[own],
                    event_depth[own],
                    np.asarray(recording_pass)[own],
                    (0.0, duration_minutes),
                    (depth_low, depth_high),
                    1750,
                    960,
                )
                single_axis.imshow(
                    own_raster,
                    origin="lower",
                    extent=(0.0, duration_minutes, depth_low, depth_high),
                    aspect="auto",
                    interpolation="nearest",
                )
            else:
                single_axis.text(
                    0.5, 0.5, "nothing accepted",
                    transform=single_axis.transAxes, ha="center", va="center",
                    color="#888888", fontsize=12,
                )
            single_axis.set_title(
                f"pass {index - 1} events ({pass_totals[index - 1]:,})", fontsize=10
            )
        raster_axis.set_xlim(extent[0], extent[1])
        raster_axis.set_ylim(depth_low, depth_high)
        raster_axis.set_ylabel("fitted depth (µm)")
        single_axis.set_xlim(extent[0], extent[1])
        single_axis.set_ylim(depth_low, depth_high)
    tick_rows = np.linspace(0, n_channels - 1, 6).astype(int)
    depth_ticks = np.linspace(depth_low, depth_high, 6)
    for left, raster, single in axes:
        left.set_yticks(
            tick_rows, [f"{contacts[channel_order[row], 1]:.0f}" for row in tick_rows]
        )
        raster.set_yticks(depth_ticks, [f"{tick:.0f}" for tick in depth_ticks])
        single.set_yticks(depth_ticks, [f"{tick:.0f}" for tick in depth_ticks])
    axes[-1, 0].set_xlabel("recording time (min)")
    axes[-1, 1].set_xlabel("recording time (min)")
    axes[-1, 2].set_xlabel("recording time (min)")
    figure.colorbar(image, ax=axes[:, 0], label="voltage / robust channel noise", pad=0.01)
    figure.suptitle(
        f"0019 full-recording replay · chunks {args.first_chunk}–{last_chunk} · "
        f"{args.samples_per_bin}-sample signed extrema · pass bars {pass_bars}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)
    print(f"pass events {pass_totals}", flush=True)
    print(f"RMS percent by stage: {rms_percent}", flush=True)


if __name__ == "__main__":
    main()
