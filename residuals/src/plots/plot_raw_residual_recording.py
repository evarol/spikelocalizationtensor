"""Replay saved fits and plot a raw recording chunk through residual passes."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from maths import reconstruct_spike_fits
from preprocessing.raw_residual import (
    ResidualConfig,
    preprocess_voltage,
    robust_channel_noise,
    subtract_predictions,
)


def load_config(run):
    metadata = json.loads((run / "config.json").read_text())
    valid = {item.name for item in fields(ResidualConfig)}
    config = ResidualConfig(**{key: value for key, value in metadata["config"].items() if key in valid})
    return metadata, config


def choose_window(times, passes, core_start, core_stop, width):
    eligible_start = core_start + width
    eligible_stop = core_stop - 2 * width
    if eligible_stop <= eligible_start:
        return core_start
    starts = np.arange(eligible_start, eligible_stop + 1, max(1, width // 4))
    base_times = np.sort(times[passes == 0])
    if not len(base_times):
        base_times = np.sort(times)
    counts = np.searchsorted(base_times, starts + width) - np.searchsorted(base_times, starts)
    return int(starts[int(np.argmax(counts))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--reconstruction-batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import spikeglx

    metadata, config = load_config(args.run)
    fs = float(metadata["fs"])
    n_channels = int(metadata["n_channels"])
    chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
    core_start = int(metadata["first_sample"]) + args.chunk_index * chunk_samples
    core_stop = min(core_start + chunk_samples, int(metadata["stop_sample"]))
    n_before = int(round(config.ms_before * fs / 1000))
    n_after = int(round(config.ms_after * fs / 1000))
    margin = max(int(round(config.read_margin_ms * fs / 1000)), n_before + n_after, 128)

    chunk_path = args.run / "chunks" / f"chunk_{args.chunk_index:06d}.npz"
    with np.load(chunk_path, allow_pickle=False) as archive:
        keys = (
            "spike_times",
            "neighbor_ids",
            "local_coords",
            "sources",
            "profile_idx",
            "temporal_idx",
            "alpha",
            "residual_pass",
        )
        saved = {key: archive[key] for key in keys}

    reader = spikeglx.Reader(metadata["recording_path"])
    try:
        read_start = max(0, core_start - margin)
        read_stop = min(reader.ns, core_stop + margin)
        raw = reader[read_start:read_stop, :n_channels]
    finally:
        reader.close()
    data = preprocess_voltage(
        raw,
        fs,
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        order=config.filter_order,
    )
    noise = robust_channel_noise(data)
    residual = np.array(data, dtype=np.float32, copy=True)
    window_samples = max(1, int(round(args.window_ms * fs / 1000)))
    display_start = choose_window(
        saved["spike_times"], saved["residual_pass"], core_start, core_stop, window_samples
    )
    display_stop = min(display_start + window_samples, core_stop)
    local_slice = slice(display_start - read_start, display_stop - read_start)
    histories = [residual[local_slice] / noise[None]]
    pass_counts = []
    window_counts = []
    omega = np.load(args.run / "omega.npy")

    for residual_pass in range(config.max_residual_passes):
        rows = np.flatnonzero(saved["residual_pass"] == residual_pass)
        pass_counts.append(len(rows))
        window_counts.append(
            int(np.sum((saved["spike_times"][rows] >= display_start) & (saved["spike_times"][rows] < display_stop)))
        )
        for offset in range(0, len(rows), args.reconstruction_batch_size):
            batch = rows[offset:offset + args.reconstruction_batch_size]
            mask = saved["neighbor_ids"][batch] >= 0
            prediction = reconstruct_spike_fits(
                saved["local_coords"][batch],
                mask,
                saved["sources"][batch],
                saved["profile_idx"][batch],
                omega,
                saved["temporal_idx"][batch],
                saved["alpha"][batch],
                kernels=tuple(part.strip() for part in config.kernel.split(",")),
                n_scales=config.n_scales,
                device=args.device,
            )
            subtract_predictions(
                residual,
                saved["spike_times"][batch] - read_start,
                saved["neighbor_ids"][batch],
                mask,
                prediction,
                n_before,
                n_after,
            )
        histories.append(residual[local_slice] / noise[None])

    contacts = np.load(args.run / "channel_positions.npy")
    channel_order = np.lexsort((contacts[:, 0], contacts[:, 1]))
    input_rms = float(np.sqrt(np.mean(histories[0] ** 2)))
    rms_percent = [100 * float(np.sqrt(np.mean(stage ** 2))) / input_rms for stage in histories]
    vlim = float(np.percentile(np.abs(histories[0]), 99.5))
    vlim = min(12.0, max(4.0, vlim))
    extent = (display_start / fs, display_stop / fs, 0, n_channels - 1)

    figure, axes = plt.subplots(
        len(histories), 1, figsize=(13, 12), sharex=True, constrained_layout=True
    )
    image = None
    for index, (axis, stage) in enumerate(zip(axes, histories)):
        image = axis.imshow(
            stage[:, channel_order].T,
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
                f"after residual pass {index} · RMS {rms_percent[index]:.1f}% · "
                f"{pass_counts[index - 1]:,} chunk fits, {window_counts[index - 1]:,} in view"
            )
        axis.set_title(title, fontsize=9)
        axis.set_ylabel("probe depth")
    tick_rows = np.linspace(0, n_channels - 1, 6).astype(int)
    for axis in axes:
        axis.set_yticks(tick_rows, [f"{contacts[channel_order[row], 1]:.0f}" for row in tick_rows])
    axes[-1].set_xlabel("recording time (s)")
    figure.colorbar(image, ax=axes, label="voltage / robust channel noise", pad=0.01)
    figure.suptitle(
        f"raw recording residual replay · chunk {args.chunk_index} · "
        f"{1000 * display_start / fs:.1f}–{1000 * display_stop / fs:.1f} ms"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)
    print(f"display samples [{display_start}, {display_stop})", flush=True)
    print(f"RMS percent by stage: {rms_percent}", flush=True)


if __name__ == "__main__":
    main()
