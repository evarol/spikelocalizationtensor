"""Run-level diagnostics for session-0010 whitened dense pursuit."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_localizations(out, source, positions, temporal_idx):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    color = temporal_idx
    axes[0].scatter(source[:, 0], source[:, 1], c=color, s=2, cmap="turbo", rasterized=True)
    axes[0].scatter(positions[:, 0], positions[:, 1], s=5, marker="s", c="0.35", alpha=.5)
    axes[0].set(title="localized spikes on probe", xlabel="x (µm)", ylabel="depth (µm)")
    axes[1].scatter(source[:, 0], source[:, 2], c=color, s=2, cmap="turbo", rasterized=True)
    axes[1].set(title="localized lateral position and source depth", xlabel="x (µm)", ylabel="source z (µm)")
    figure.savefig(out / "localizations.png", dpi=800)
    plt.close(figure)


def save_raster(out, times, source, temporal_idx, fs):
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    image = axis.scatter(times / fs, source[:, 1], c=temporal_idx, s=2, cmap="turbo", rasterized=True)
    figure.colorbar(image, ax=axis, label="temporal codebook row")
    axis.set(title="spike depth × time raster", xlabel="time (s)", ylabel="localized depth (µm)")
    figure.savefig(out / "spike_raster.png", dpi=800)
    plt.close(figure)


def save_codebook(out, omega, temporal_idx, fs):
    columns = min(4, len(omega))
    rows = int(np.ceil(len(omega) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 1.8 * rows), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    limit = np.abs(omega).max()
    usage = np.bincount(temporal_idx, minlength=len(omega))
    time_ms = 1000 * np.arange(omega.shape[1]) / fs
    for row, axis in enumerate(axes.flat):
        if row >= len(omega):
            axis.set_visible(False)
            continue
        axis.plot(time_ms, omega[row], color=plt.cm.turbo(row / max(1, len(omega) - 1)))
        axis.set(title=f"row {row} · n={usage[row]:,}", ylim=(-1.05 * limit, 1.05 * limit))
        axis.grid(alpha=.15)
    figure.savefig(out / "temporal_codebook.png", dpi=800)
    plt.close(figure)


def save_usage(out, temporal_idx, sigma):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    usage = np.bincount(temporal_idx)
    axes[0].bar(np.arange(len(usage)), usage, color=plt.cm.turbo(np.linspace(.05, .95, len(usage))))
    axes[0].set(title="temporal codebook usage", xlabel="row", ylabel="accepted events")
    values, counts = np.unique(sigma, return_counts=True)
    axes[1].bar(np.arange(len(values)), counts, color="#482878")
    axes[1].set(title="refitted spatial-scale usage", xlabel="sigma (µm)", ylabel="accepted events")
    axes[1].set_xticks(np.arange(len(values)), [f"{value:g}" for value in values])
    figure.savefig(out / "codebook_usage.png", dpi=800)
    plt.close(figure)


def save_reconstruction_examples(run, out, omega, whitening):
    chunk_path = next(path for path in sorted((run / "chunks").glob("chunk_*.npz")) if len(np.load(path, allow_pickle=False)["spike_times"]))
    with np.load(chunk_path, allow_pickle=False) as chunk:
        order = np.argsort(chunk["detection_score"])[-4:]
        waveform = chunk["residual_waveforms"][order]
        coords = chunk["local_coords"][order]
        ids = chunk["neighbor_ids"][order]
        source = chunk["sources"][order]
        sigma = chunk["sigma"][order]
        alpha = chunk["alpha"][order]
        temporal_idx = chunk["temporal_idx"][order]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, y, xy, row_ids, src, scale, gain, q in zip(axes.flat, waveform, coords, ids, source, sigma, alpha, temporal_idx):
        valid = row_ids >= 0
        raw = scale / np.sqrt((xy[valid, 0] - src[0]) ** 2 + (xy[valid, 1] - src[1]) ** 2 + src[2] ** 2 + scale ** 2)
        footprint = raw @ whitening[np.ix_(row_ids[valid], row_ids[valid])]
        footprint /= max(np.linalg.norm(footprint), np.finfo(np.float32).tiny)
        predicted = gain * footprint[:, None] * omega[q]
        amplitude = max(np.abs(y[valid]).max(), np.abs(predicted).max(), 1e-6)
        time = (np.arange(y.shape[1]) - y.shape[1] / 2) * .34
        for channel, (coord, observed, model) in enumerate(zip(xy[valid], y[valid], predicted)):
            axis.plot(coord[0] + time, coord[1] + 18 * observed / amplitude, color="#d62728", lw=.7)
            axis.plot(coord[0] + time, coord[1] + 18 * model / amplitude, color="#2ca02c", lw=.8, ls="--")
        axis.set(title=f"q={q}, sigma={scale:g} µm", xlabel="local x (µm)", ylabel="local y (µm)")
        axis.grid(alpha=.15)
    figure.savefig(out / "reconstruction_examples.png", dpi=800)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run = args.run
    out = args.out.parent
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run / "config.json").read_text())
    rounds = np.load(run / "residual_pass.npy")
    scores = np.load(run / "detection_score.npy")
    sigma = np.load(run / "sigma.npy")
    rmse = np.load(run / "channel_normalized_rmse.npy")
    ids = np.load(run / "neighbor_ids.npy")
    source = np.load(run / "global_sources.npy")
    times = np.load(run / "spike_times.npy")
    temporal_idx = np.load(run / "temporal_idx.npy")
    positions = np.load(run / "channel_positions.npy")
    omega = np.load(run / "omega.npy")
    whitening = np.load(run / "whitening_matrix.npy")
    valid_rmse = rmse[ids >= 0]
    if not len(rounds):
        raise RuntimeError("run contains no accepted events")

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    pass_number = np.arange(rounds.max() + 1) + 1
    counts = np.bincount(rounds, minlength=len(pass_number))
    axes[0, 0].bar(pass_number, counts, color="#31688e")
    axes[0, 0].set(title="accepted events per pursuit round", xlabel="round", ylabel="events")

    axes[0, 1].hist(scores[np.isfinite(scores)], bins=60, color="#35b779")
    axes[0, 1].set(title="dense proposal score", xlabel="score", ylabel="events")

    values, counts = np.unique(sigma[np.isfinite(sigma)], return_counts=True)
    axes[1, 0].bar(np.arange(len(values)), counts, color="#482878")
    axes[1, 0].set(title="full-window refitted spatial scale", xlabel="sigma (µm)", ylabel="events")
    axes[1, 0].set_xticks(np.arange(len(values)), [f"{value:g}" for value in values])

    axes[1, 1].hist(valid_rmse[np.isfinite(valid_rmse)], bins=60, color="#21918c")
    axes[1, 1].axvline(3.0, color="#d62728", linestyle="--", linewidth=1, label="default gate")
    axes[1, 1].set(title="per-channel normalized reconstruction RMSE", xlabel="normalized RMSE", ylabel="channels")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    figure.suptitle(f"Session 0010 whitened dense pursuit — {len(rounds):,} accepted events")
    save_localizations(out, source, positions, temporal_idx)
    save_raster(out, times, source, temporal_idx, float(metadata["fs"]))
    save_codebook(out, omega, temporal_idx, float(metadata["fs"]))
    save_usage(out, temporal_idx, sigma)
    save_reconstruction_examples(run, out, omega, whitening)
    figure.savefig(args.out, dpi=800)
    plt.close(figure)


if __name__ == "__main__":
    main()
