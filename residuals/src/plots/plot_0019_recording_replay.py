"""Replay saved 0019 fits and plot a continuous recording window through passes."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from preprocessing.raw_residual import preprocess_voltage, robust_channel_noise


def load_pass_chunks(run, chunk_index, recording_passes):
    chunks = []
    for recording_pass in range(recording_passes):
        path = run / f"pass_{recording_pass:02d}" / f"chunk_{chunk_index:06d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                chunks.append(
                    {
                        "recording_pass": recording_pass,
                        "spike_times": archive["spike_times"],
                        "neighbor_ids": archive["neighbor_ids"],
                        "predictions": archive["predictions"],
                        "noise": archive["noise"],
                    }
                )
    if not chunks:
        raise FileNotFoundError(f"no saved chunks for chunk {chunk_index} in {run}")
    return chunks


def choose_window(times, core_start, core_stop, width):
    eligible_start = core_start + width
    eligible_stop = core_stop - 2 * width
    if eligible_stop <= eligible_start:
        return core_start
    starts = np.arange(eligible_start, eligible_stop + 1, max(1, width // 4))
    base_times = np.sort(times)
    counts = np.searchsorted(base_times, starts + width) - np.searchsorted(
        base_times, starts
    )
    return int(starts[int(np.argmax(counts))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")

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

    chunks = load_pass_chunks(args.run, args.chunk_index, recording_passes)
    core_start = int(metadata["first_sample"]) + args.chunk_index * chunk_samples
    core_stop = min(core_start + chunk_samples, int(metadata["stop_sample"]))
    window_samples = max(1, int(round(args.window_ms * fs / 1000)))
    display_start = choose_window(
        chunks[0]["spike_times"], core_start, core_stop, window_samples
    )
    display_stop = min(display_start + window_samples, core_stop)

    recording_path = Path(metadata["recording_path"])
    meta_path = recording_path.with_suffix(".meta")
    meta = dict(
        line.split("=", 1)
        for line in meta_path.read_text().splitlines()
        if "=" in line
    )
    ap, lf, sy = (int(part) for part in meta["snsApLfSy"].split(","))
    file_channels = ap + lf + sy
    total_samples = recording_path.stat().st_size // (2 * file_channels)
    read_start = max(0, core_start - margin)
    read_stop = min(total_samples, core_stop + margin)
    raw = np.memmap(
        recording_path, dtype="<i2", mode="r", shape=(total_samples, file_channels)
    )[read_start:read_stop, :n_channels].astype(np.float32)
    raw *= 2.34375e-06

    data = preprocess_voltage(raw, fs)
    noise = np.asarray(chunks[0]["noise"], dtype=np.float32)
    if noise.shape != (n_channels,):
        noise = robust_channel_noise(data)

    residual = np.array(data, dtype=np.float32, copy=True)
    local_slice = slice(display_start - read_start, display_stop - read_start)
    histories = [residual[local_slice] / noise[None]]
    pass_counts = []
    window_counts = []
    for chunk in chunks:
        times = chunk["spike_times"]
        ids = chunk["neighbor_ids"]
        predictions = chunk["predictions"]
        mask = ids >= 0
        pass_counts.append(len(times))
        window_counts.append(
            int(np.sum((times >= display_start) & (times < display_stop)))
        )
        for time, row_ids, row_mask, model in zip(times, ids, mask, predictions):
            residual[time - read_start - n_before:time - read_start + n_after, row_ids[row_mask]] -= model[row_mask].T
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
                f"after recording pass {index - 1} · RMS {rms_percent[index]:.1f}% · "
                f"{pass_counts[index - 1]:,} pass events, {window_counts[index - 1]:,} in view"
            )
        axis.set_title(title, fontsize=9)
        axis.set_ylabel("probe depth")
    tick_rows = np.linspace(0, n_channels - 1, 6).astype(int)
    for axis in axes:
        axis.set_yticks(tick_rows, [f"{contacts[channel_order[row], 1]:.0f}" for row in tick_rows])
    axes[-1].set_xlabel("recording time (s)")
    figure.colorbar(image, ax=axes, label="voltage / robust channel noise", pad=0.01)
    figure.suptitle(
        f"0019 recording replay · chunk {args.chunk_index} · "
        f"{1000 * display_start / fs:.1f}–{1000 * display_stop / fs:.1f} ms · "
        f"pass bars {pass_bars}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {args.out}", flush=True)
    print(f"display samples [{display_start}, {display_stop})", flush=True)
    print(f"pass events {pass_counts}, in view {window_counts}", flush=True)
    print(f"RMS percent by stage: {rms_percent}", flush=True)


if __name__ == "__main__":
    main()
