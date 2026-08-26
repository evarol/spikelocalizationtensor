"""Plot continuous raw-residual localizations using identifiable monopole width."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def padded_limits(values, quantiles=(0.0, 1.0), fraction=0.03):
    finite = np.asarray(values)[np.isfinite(values)]
    low, high = np.quantile(finite, quantiles)
    padding = fraction * max(float(high - low), 1.0)
    return float(low - padding), float(high + padding)


def scatter_projection(axis, horizontal, vertical, color, limits, labels, title):
    artist = axis.scatter(
        horizontal,
        vertical,
        c=color,
        cmap="inferno",
        norm=limits,
        s=0.35,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    axis.set(
        xlabel=labels[0],
        ylabel=labels[1],
        title=title,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="white", alpha=0.08, linewidth=0.5)
    return artist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    metadata = json.loads((args.run / "config.json").read_text())
    config = metadata["config"]
    if config["kernel"] != "monopole":
        raise ValueError("identifiable-width projections require the monopole kernel")

    sources = np.load(args.run / "global_sources.npy", mmap_mode="r")
    alpha = np.load(args.run / "alpha.npy", mmap_mode="r")
    profile_idx = np.load(args.run / "profile_idx.npy", mmap_mode="r")
    residual_pass = np.load(args.run / "residual_pass.npy", mmap_mode="r")
    contacts = np.load(args.run / "channel_positions.npy")
    lengths = {len(sources), len(alpha), len(profile_idx), len(residual_pass)}
    if len(lengths) != 1 or sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("residual localization arrays have inconsistent shapes")

    n_scales = int(config["n_scales"])
    sigmas = np.geomspace(2.0, 512.0, n_scales)
    profile_idx = np.asarray(profile_idx, dtype=np.int64)
    if np.any((profile_idx < 0) | (profile_idx >= n_scales)):
        raise ValueError("profile_idx contains an index outside the monopole scale bank")

    rho = np.sqrt(np.asarray(sources[:, 2], dtype=np.float64) ** 2 + sigmas[profile_idx] ** 2)
    amplitude = np.log10(np.maximum(np.abs(np.asarray(alpha)), np.finfo(np.float32).tiny))
    rng = np.random.default_rng(args.seed)
    keep = np.sort(
        rng.choice(len(sources), min(args.max_points, len(sources)), replace=False)
    )
    order = np.argsort(amplitude[keep], kind="stable")
    keep = keep[order]

    x = np.asarray(sources[keep, 0])
    depth = np.asarray(sources[keep, 1])
    rho_sample = rho[keep]
    color = amplitude[keep]
    color_limits = Normalize(*np.quantile(amplitude, (0.01, 0.995)), clip=True)
    depth_limits = padded_limits(
        np.concatenate((np.asarray(sources[:, 1]), contacts[:, 1]))
    )
    x_limits = padded_limits(
        np.concatenate((np.asarray(sources[:, 0]), contacts[:, 0]))
    )
    rho_limits = (0.0, padded_limits(rho, quantiles=(0.0, 0.999), fraction=0.02)[1])

    plt.style.use("dark_background")
    figure = plt.figure(figsize=(20, 8.5))
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=(1.0, 1.35, 1.35),
        left=0.055,
        right=0.91,
        bottom=0.07,
        top=0.84,
        hspace=0.42,
    )
    axes = [figure.add_subplot(grid[index, 0]) for index in range(3)]

    artist = scatter_projection(
        axes[0],
        depth,
        x,
        color,
        color_limits,
        ("y / probe depth (µm)", "x / lateral (µm)"),
        "continuous residual localizations · x–y",
    )
    axes[0].scatter(
        contacts[:, 1],
        contacts[:, 0],
        s=5,
        marker="s",
        facecolors="none",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.8,
    )
    scatter_projection(
        axes[1],
        depth,
        rho_sample,
        color,
        color_limits,
        ("y / probe depth (µm)", "ρ / effective width (µm)"),
        "identifiable monopole width · ρ–y",
    )
    scatter_projection(
        axes[2],
        x,
        rho_sample,
        color,
        color_limits,
        ("x / lateral (µm)", "ρ / effective width (µm)"),
        "identifiable monopole width · ρ–x",
    )
    axes[0].set(xlim=depth_limits, ylim=x_limits)
    axes[1].set(xlim=depth_limits, ylim=rho_limits)
    axes[2].set(xlim=x_limits, ylim=rho_limits)
    axes[2].set_anchor("W")

    pass_counts = np.bincount(np.asarray(residual_pass, dtype=np.int64))
    figure.suptitle(
        "Continuous Q8 residual smoke · 10 s · "
        f"{len(sources):,} accepted fits across {len(pass_counts)} passes",
        fontsize=18,
        y=0.97,
    )
    figure.text(
        0.01,
        0.91,
        f"{len(keep):,} fits displayed · colour = log10 fitted |α| · "
        "ρ = √(z² + σ²); z and σ are not separately identifiable · "
        "white squares are probe contacts",
        color="0.65",
        fontsize=10,
    )
    colorbar_axis = figure.add_axes((0.935, 0.22, 0.012, 0.5))
    colorbar = figure.colorbar(artist, cax=colorbar_axis)
    colorbar.set_label("log10 fitted |α|")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, bbox_inches="tight")
    plt.close(figure)
    print(f"pass counts: {pass_counts.tolist()}", flush=True)
    print(
        f"rho: median={np.median(rho):.3f} um, "
        f"p99={np.quantile(rho, 0.99):.3f} um, "
        f"p99.9={np.quantile(rho, 0.999):.3f} um",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
