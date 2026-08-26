"""Compare poor reconstruction examples before and after local whitening."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spikeglx


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "plots"))

from preprocessing.raw_residual import preprocess_voltage, robust_channel_noise
from plot_reconstructions import profile_parameters, reconstruct


EPS = np.finfo(np.float32).tiny


def load_preprocessed_chunk(run, metadata, chunk_index):
    config = metadata["config"]
    fs = float(metadata["fs"])
    first_sample = int(metadata["first_sample"])
    stop_sample = int(metadata["stop_sample"])
    chunk_samples = int(round(float(config["chunk_seconds"]) * fs))
    n_before = int(round(float(config["ms_before"]) * fs / 1000))
    n_after = int(round(float(config["ms_after"]) * fs / 1000))
    margin = max(
        int(round(float(config["read_margin_ms"]) * fs / 1000)),
        n_before + n_after,
        128,
    )
    core_start = first_sample + chunk_index * chunk_samples
    core_stop = min(core_start + chunk_samples, stop_sample)
    read_start = max(0, core_start - margin)
    read_stop = core_stop + margin
    reader = spikeglx.Reader(metadata["recording_path"])
    try:
        read_stop = min(reader.ns, read_stop)
        raw = reader[read_start:read_stop, : int(metadata["n_channels"])]
    finally:
        reader.close()
    data = preprocess_voltage(
        raw,
        fs,
        freq_min=float(config["freq_min"]),
        freq_max=float(config["freq_max"]),
        order=int(config["filter_order"]),
    )
    return data, read_start, core_start, core_stop


def choose_examples(chunk, n_examples, seed):
    passes = np.asarray(chunk["residual_pass"], dtype=np.int64)
    captured = np.asarray(chunk["captured_fraction"], dtype=np.float64)
    input_energy = np.asarray(chunk["input_energy"], dtype=np.float64)
    eligible = np.flatnonzero(
        (passes == 0)
        & np.isfinite(captured)
        & np.isfinite(input_energy)
        & (input_energy >= np.quantile(input_energy[np.isfinite(input_energy)], 0.25))
    )
    if len(eligible) < n_examples:
        raise ValueError(f"only {len(eligible)} eligible pass-0 examples")
    poor_cutoff = np.quantile(captured[eligible], 0.15)
    poor = eligible[captured[eligible] <= poor_cutoff]
    if len(poor) < n_examples:
        poor = eligible[np.argsort(captured[eligible])[: max(n_examples, len(poor))]]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(poor, n_examples, replace=False))


def extract_waveforms(data, chunk, indices, read_start, n_before, n_after):
    times = np.asarray(chunk["spike_times"], dtype=np.int64)[indices] - read_start
    ids = np.asarray(chunk["neighbor_ids"], dtype=np.int64)[indices]
    mask = ids >= 0
    safe_ids = np.maximum(ids, 0)
    width = n_before + n_after
    waveforms = np.zeros((len(indices), ids.shape[1], width), dtype=np.float32)
    for row, time in enumerate(times):
        start = int(time) - n_before
        stop = int(time) + n_after
        if start < 0 or stop > len(data):
            raise ValueError("selected event does not fit in loaded chunk")
        waveforms[row] = data[start:stop, safe_ids[row]].T
        waveforms[row, ~mask[row]] = 0
    return waveforms * mask[:, :, None], mask


def local_whitening_matrix(data, channel_ids, noise, shrinkage):
    standardized = data[:, channel_ids] / noise[channel_ids][None, :]
    standardized = standardized - np.median(standardized, axis=0, keepdims=True)
    covariance = np.cov(standardized, rowvar=False).astype(np.float64)
    diagonal = float(np.trace(covariance) / max(covariance.shape[0], 1))
    covariance = (1 - shrinkage) * covariance + shrinkage * diagonal * np.eye(
        covariance.shape[0]
    )
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, np.median(values[values > 0]) * 1e-4)
    return (vectors / np.sqrt(values)[None, :]) @ vectors.T


def apply_local_whitening(waveform, prediction, data, ids, mask, noise, shrinkage):
    whitened_waveform = np.zeros_like(waveform)
    whitened_prediction = np.zeros_like(prediction)
    for row in range(len(waveform)):
        valid = mask[row]
        channels = ids[row, valid]
        whitening = local_whitening_matrix(data, channels, noise, shrinkage)
        yw = waveform[row, valid] / noise[channels, None]
        pw = prediction[row, valid] / noise[channels, None]
        whitened_waveform[row, valid] = whitening @ yw
        whitened_prediction[row, valid] = whitening @ pw
    return whitened_waveform, whitened_prediction


def relative_error(measured, prediction, mask):
    numerator = np.square((measured - prediction) * mask[:, :, None]).sum((1, 2))
    denominator = np.square(measured * mask[:, :, None]).sum((1, 2))
    return numerator / np.maximum(denominator, EPS)


def plot_panel(axis, coords, measured, predicted, title):
    valid = np.any(measured != 0, axis=1)
    coords = coords[valid]
    measured = measured[valid]
    predicted = predicted[valid]
    residual = measured - predicted
    amplitude = max(
        float(np.max(np.abs(measured))),
        float(np.max(np.abs(predicted))),
        EPS,
    )
    scale = 18.0 / amplitude
    time_offset = (np.arange(measured.shape[1]) - measured.shape[1] / 2) * 0.34
    for channel in range(len(coords)):
        x = coords[channel, 0] + time_offset
        y0 = coords[channel, 1]
        axis.plot(x, y0 + measured[channel] * scale, color="#d62728", linewidth=0.8)
        axis.plot(
            x,
            y0 + predicted[channel] * scale,
            color="#2ca02c",
            linewidth=0.95,
            linestyle="--",
        )
        axis.plot(x, y0 + residual[channel] * scale, color="#1f77b4", linewidth=0.65)
        axis.scatter(coords[channel, 0], y0, s=9, marker="s", color="0.45")
    axis.set_title(title, fontsize=8.3)
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    args = parser.parse_args()

    metadata = json.loads((args.run / "config.json").read_text())
    config = metadata["config"]
    fs = float(metadata["fs"])
    n_before = int(round(float(config["ms_before"]) * fs / 1000))
    n_after = int(round(float(config["ms_after"]) * fs / 1000))
    chunk_path = args.run / "chunks" / f"chunk_{args.chunk_index:06d}.npz"
    with np.load(chunk_path, allow_pickle=False) as archive:
        chunk = {key: archive[key] for key in archive.files}
    data, read_start, core_start, core_stop = load_preprocessed_chunk(
        args.run, metadata, args.chunk_index
    )
    noise = robust_channel_noise(data)
    indices = choose_examples(chunk, args.n_examples, args.seed)
    measured, mask = extract_waveforms(
        data, chunk, indices, read_start, n_before, n_after
    )
    ids = np.asarray(chunk["neighbor_ids"], dtype=np.int64)[indices]
    coords = np.asarray(chunk["local_coords"], dtype=np.float32)[indices]
    parameters = profile_parameters(
        config["kernel"], np.asarray(chunk["profile_idx"])[indices], config["n_scales"]
    )
    _, predicted = reconstruct(
        coords,
        mask,
        np.asarray(chunk["sources"])[indices],
        parameters,
        np.load(args.run / "omega.npy"),
        np.asarray(chunk["temporal_idx"])[indices],
        np.asarray(chunk["alpha"])[indices],
        config["kernel"],
    )
    predicted = predicted.astype(np.float32)
    whitened_measured, whitened_predicted = apply_local_whitening(
        measured, predicted, data, ids, mask, noise, args.shrinkage
    )
    unwhitened_error = relative_error(measured, predicted, mask)
    whitened_error = relative_error(whitened_measured, whitened_predicted, mask)
    captured = np.asarray(chunk["captured_fraction"], dtype=np.float64)[indices]
    spike_times = np.asarray(chunk["spike_times"], dtype=np.int64)[indices]
    relative_ms = 1000 * (spike_times - core_start) / fs

    figure, axes = plt.subplots(
        args.n_examples,
        2,
        figsize=(11.5, 2.8 * args.n_examples),
        constrained_layout=True,
        squeeze=False,
    )
    for row in range(args.n_examples):
        base = (
            f"event {indices[row]} at {relative_ms[row]:.1f} ms, "
            f"capture {100 * captured[row]:.1f}%"
        )
        plot_panel(
            axes[row, 0],
            coords[row],
            measured[row],
            predicted[row],
            f"unwhitened\n{base}\nrel. error {unwhitened_error[row]:.3f}",
        )
        plot_panel(
            axes[row, 1],
            coords[row],
            whitened_measured[row],
            whitened_predicted[row],
            f"local covariance whitened\n{base}\nrel. error {whitened_error[row]:.3f}",
        )
        axes[row, 0].set_ylabel("channel geometry", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("local x plus time offset", fontsize=8)
    handles = [
        plt.Line2D([0], [0], color="#d62728", linewidth=1, label="measured"),
        plt.Line2D([0], [0], color="#2ca02c", linewidth=1, linestyle="--", label="model"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1, label="residual"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        "Poor pass-0 reconstructions from 4-second chunk: unwhitened vs local whitening",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    metrics = {
        "run": str(args.run.resolve()),
        "chunk_index": args.chunk_index,
        "selected_chunk_rows": indices.astype(int).tolist(),
        "selected_spike_times": spike_times.astype(int).tolist(),
        "captured_fraction": captured.astype(float).tolist(),
        "unwhitened_relative_error": unwhitened_error.astype(float).tolist(),
        "whitened_relative_error": whitened_error.astype(float).tolist(),
        "shrinkage": args.shrinkage,
        "output": str(args.out.resolve()),
    }
    metrics_path = args.out.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
