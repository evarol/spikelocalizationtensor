"""Residual-pursuit panels adapted from spiketensor's visualization conventions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.ndimage import gaussian_filter

from plot_raw_residual_reconstructions import (
    EPS,
    reconstruct_saved_fit,
    spatial_width_image,
)


BACKGROUND = np.asarray((0.05, 0.05, 0.05), dtype=np.float32)


def colour_raster(x, y, label, palette, xlim, ylim, nx, ny):
    ix = np.floor((x - xlim[0]) * nx / (xlim[1] - xlim[0])).astype(np.int64)
    iy = np.floor((y - ylim[0]) * ny / (ylim[1] - ylim[0])).astype(np.int64)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[keep] * nx + ix[keep]
    count = np.bincount(flat, minlength=nx * ny).reshape(ny, nx).astype(np.float32)
    rgb = np.zeros((ny, nx, 3), dtype=np.float32)
    for channel in range(3):
        rgb[..., channel] = np.bincount(
            flat, weights=palette[label[keep], channel], minlength=nx * ny
        ).reshape(ny, nx)
    mass = gaussian_filter(count, 0.5)
    for channel in range(3):
        rgb[..., channel] = gaussian_filter(rgb[..., channel], 0.5) / np.maximum(mass, 1e-6)
    intensity = np.clip(1.35 * (1 - np.exp(-mass / 1.4)), 0, 1)
    return BACKGROUND * (1 - intensity[..., None]) + rgb * intensity[..., None]


def load_examples(run, count=4):
    path = sorted((run / "chunks").glob("chunk_*.npz"))[0]
    with np.load(path, allow_pickle=False) as archive:
        rows = np.argsort(archive["input_energy"])[-count:]
        return {key: np.asarray(archive[key][rows]) for key in archive.files}, rows


def plot_spikes(run, out, metadata):
    chunk, rows = load_examples(run)
    omega = np.load(run / "omega.npy")
    config = metadata["config"]
    mask = chunk["neighbor_ids"] >= 0
    measured = chunk["residual_waveforms"] * mask[:, :, None]
    footprint, predicted = reconstruct_saved_fit(
        run, config, chunk["local_coords"], chunk["neighbor_ids"], mask,
        chunk["sources"], chunk["profile_idx"], omega, chunk["temporal_idx"],
        chunk["alpha"], chunk.get("rho"),
    )
    figure, axes = plt.subplots(4, len(rows), figsize=(3.2 * len(rows), 11.5),
                                constrained_layout=True, squeeze=False,
                                gridspec_kw={"height_ratios": (1.25, .6, 1.15, .65)})
    for column in range(len(rows)):
        valid = mask[column]
        coords = chunk["local_coords"][column, valid]
        scale = 16 / max(float(np.abs(measured[column, valid]).max()), EPS)
        time = np.arange(measured.shape[2]) * .34
        for channel, coordinate in enumerate(coords):
            axes[0, column].plot(coordinate[0] + time, coordinate[1] + measured[column, valid][channel] * scale, color="#e03131", lw=.8)
            axes[0, column].plot(coordinate[0] + time, coordinate[1] + predicted[column, valid][channel] * scale, color="#2f9e44", lw=.95, ls="--")
        axes[0, column].set_title(f"fit {rows[column]:,} · pass {chunk['residual_pass'][column] + 1}", fontsize=8)
        observed_ptp = np.ptp(measured[column, valid], axis=1)
        predicted_ptp = np.ptp(predicted[column, valid], axis=1)
        channel = np.arange(valid.sum())
        axes[1, column].bar(channel - .19, observed_ptp, .38, color="#e03131")
        axes[1, column].bar(channel + .19, predicted_ptp, .38, color="#2f9e44")
        axes[1, column].set_title("per-channel peak-to-peak", fontsize=8)
        rho = chunk.get("rho", np.sqrt(chunk["sources"][:, 2] ** 2 + chunk["sigma"] ** 2))[column]
        grid, image = spatial_width_image(chunk["sources"][column], rho)
        axes[2, column].imshow(image, origin="lower", extent=(grid[0], grid[-1], grid[0], grid[-1]), cmap="magma", aspect="equal")
        axes[2, column].scatter(coords[:, 0], coords[:, 1], s=12, marker="s", facecolors="none", edgecolors="white", linewidths=.6)
        axes[2, column].plot(chunk["sources"][column, 0], chunk["sources"][column, 1], "o", mfc="none", mec="#4c8dff", ms=10, mew=1.5)
        error = chunk.get("channel_normalized_rmse")
        if error is None:
            error = np.sqrt(np.mean((measured[column, valid] - predicted[column, valid]) ** 2, axis=1))
        else:
            error = error[column, valid]
        axes[3, column].bar(channel, error, color="#845ef7")
        axes[3, column].axhline(3, color="#e03131", ls="--", lw=.8)
        axes[3, column].set_title("per-channel normalized error", fontsize=8)
        for axis in axes[:, column]:
            axis.tick_params(labelsize=6)
            axis.grid(alpha=.18)
    figure.suptitle("example residual-pursuit fits — spiketensor panel convention", fontsize=11)
    figure.savefig(out / "spiketensor_spikes.png", dpi=800, bbox_inches="tight")
    plt.close(figure)


def plot_density_and_raster(run, out, metadata):
    source = np.load(run / "global_sources.npy", mmap_mode="r")
    times = np.load(run / "spike_times.npy", mmap_mode="r")
    alpha = np.abs(np.load(run / "alpha.npy", mmap_mode="r"))
    temporal = np.load(run / "temporal_idx.npy", mmap_mode="r")
    contacts = np.load(run / "channel_positions.npy")
    omega = np.load(run / "omega.npy")
    xlim = (contacts[:, 0].min() - 80, contacts[:, 0].max() + 80)
    ylim = tuple(np.quantile(source[:, 1], (.002, .998)))
    hist, _, _ = np.histogram2d(source[:, 1], source[:, 0], bins=(960, 160), range=(ylim, xlim), weights=alpha)
    hist = gaussian_filter(hist.astype(np.float32), 1.0)
    vmax = float(np.percentile(hist[hist > 0], 99.7))
    figure, axis = plt.subplots(figsize=(7.5, 10), constrained_layout=True)
    image = axis.imshow(hist, origin="lower", extent=(*xlim, *ylim), cmap="magma", aspect="equal", norm=PowerNorm(.45, vmin=0, vmax=vmax))
    axis.scatter(contacts[:, 0], contacts[:, 1], s=5, marker="s", facecolors="none", edgecolors="white", linewidths=.35)
    figure.colorbar(image, ax=axis, label="summed fitted amplitude")
    axis.set(xlabel="lateral x (µm)", ylabel="probe depth y (µm)", title="amplitude-weighted localization density")
    figure.savefig(out / "spiketensor_localization_density.png", dpi=800, bbox_inches="tight")
    plt.close(figure)
    palette = plt.colormaps["rainbow"](np.linspace(0, 1, len(omega)))[:, :3]
    minutes = np.asarray(times, dtype=np.float64) / (60 * float(metadata["fs"]))
    image = colour_raster(minutes, source[:, 1], temporal, palette, (0, minutes.max()), ylim, 1750, 960)
    figure, axis = plt.subplots(figsize=(15, 7.5), constrained_layout=True, facecolor=BACKGROUND)
    axis.imshow(image, origin="lower", extent=(0, minutes.max(), *ylim), aspect="auto", interpolation="nearest")
    axis.set(xlabel="recording time (min)", ylabel="probe depth (µm)", title="depth × time categorical localization raster")
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors="#dddddd")
    for spine in axis.spines.values(): spine.set_color("#444444")
    figure.savefig(out / "spiketensor_depth_time_basis.png", dpi=800, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.run / "config.json").read_text())
    plot_spikes(args.run, args.out, metadata)
    plot_density_and_raster(args.run, args.out, metadata)


if __name__ == "__main__":
    main()
