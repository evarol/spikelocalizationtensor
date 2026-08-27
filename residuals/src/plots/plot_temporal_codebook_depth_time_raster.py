"""Plot spike depth over time, colored by selected temporal-codebook row."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from scipy.ndimage import gaussian_filter


BACKGROUND = "#0d0d0d"
FONT = "#d7d7d7"
GRID = "#292929"


def categorical_raster(x, y, labels, palette, xlim, ylim, nx, ny):
    ix = np.floor((x - xlim[0]) * nx / (xlim[1] - xlim[0])).astype(np.int64)
    iy = np.floor((y - ylim[0]) * ny / (ylim[1] - ylim[0])).astype(np.int64)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[keep] * nx + ix[keep]
    count = np.bincount(flat, minlength=nx * ny).reshape(ny, nx).astype(np.float32)
    rgb = np.zeros((ny, nx, 3), dtype=np.float32)
    for channel in range(3):
        rgb[..., channel] = np.bincount(
            flat, weights=palette[labels[keep], channel], minlength=nx * ny
        ).reshape(ny, nx)
    mass = gaussian_filter(count, 0.5)
    for channel in range(3):
        rgb[..., channel] = gaussian_filter(rgb[..., channel], 0.5) / np.maximum(mass, 1e-6)
    intensity = np.clip(1.35 * (1 - np.exp(-mass / 1.4)), 0, 1)
    return np.asarray([0.05, 0.05, 0.05], dtype=np.float32) * (1 - intensity[..., None]) + rgb * intensity[..., None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampling-frequency", type=float, default=30_000.0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--marker-size", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run is None and (args.session is None or args.fit is None):
        parser.error("provide --run or both --session and --fit")
    if args.run is not None and (args.session is not None or args.fit is not None):
        parser.error("--run cannot be combined with --session or --fit")

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.sampling_frequency <= 0:
        raise ValueError("sampling frequency must be positive")
    if args.max_points < 0:
        raise ValueError("max points must be nonnegative")
    if args.marker_size <= 0:
        raise ValueError("marker size must be positive")
    if not 0 < args.alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")

    if args.run is None:
        spike_times = np.load(args.session / "spike_times.npy", mmap_mode="r")
        centroids = np.load(args.session / "centroids.npy", mmap_mode="r")
        with np.load(args.fit, allow_pickle=False) as fit:
            sources = np.asarray(fit["sources"], dtype=np.float32)
            temporal_idx = np.asarray(fit["temporal_idx"], dtype=np.int64)
            omega = np.asarray(fit["omega"], dtype=np.float32)
        depth = np.asarray(centroids[:, 1], dtype=np.float32) + sources[:, 1]
    else:
        metadata = json.loads((args.run / "config.json").read_text())
        args.sampling_frequency = float(metadata["fs"])
        spike_times = np.load(args.run / "spike_times.npy", mmap_mode="r")
        sources = np.load(args.run / "global_sources.npy", mmap_mode="r")
        temporal_idx = np.load(args.run / "temporal_idx.npy", mmap_mode="r")
        omega = np.load(args.run / "omega.npy")
        depth = np.asarray(sources[:, 1], dtype=np.float32)

    event_count = len(spike_times)
    if sources.shape != (event_count, 3):
        raise ValueError(
            f"expected sources with shape {(event_count, 3)}, got {sources.shape}"
        )
    if temporal_idx.shape != (event_count,):
        raise ValueError(
            f"expected temporal_idx with shape {(event_count,)}, got {temporal_idx.shape}"
        )
    if omega.ndim != 2 or len(omega) < 2:
        raise ValueError(f"expected at least two temporal rows, got {omega.shape}")
    if event_count == 0:
        raise ValueError("cannot plot an empty fit")
    if temporal_idx.min() < 0 or temporal_idx.max() >= len(omega):
        raise ValueError("temporal_idx contains a row outside the codebook")

    time_minutes = np.asarray(spike_times, dtype=np.float64) / (
        60.0 * args.sampling_frequency
    )
    finite = np.isfinite(time_minutes) & np.isfinite(depth)
    rows = np.flatnonzero(finite)
    if args.max_points and len(rows) > args.max_points:
        rng = np.random.default_rng(args.seed)
        rows = np.sort(rng.choice(rows, args.max_points, replace=False))

    rgb = plt.colormaps["rainbow"](
        np.linspace(0.0, 1.0, len(omega))
    )[:, :3]
    colormap = ListedColormap(rgb, name=f"q{len(omega)}_rgb")
    boundaries = np.arange(len(omega) + 1, dtype=np.float64) - 0.5
    normalization = BoundaryNorm(boundaries, len(omega))

    figure, axis = plt.subplots(
        figsize=(15, 7.5), constrained_layout=True, facecolor=BACKGROUND
    )
    if args.run is None:
        artist = axis.scatter(
            time_minutes[rows], depth[rows], c=temporal_idx[rows], cmap=colormap,
            norm=normalization, marker=".", s=args.marker_size, linewidths=0,
            alpha=args.alpha, rasterized=True,
        )
    else:
        time_limit = max(float(time_minutes[finite].max()), 1e-9)
        depth_low, depth_high = np.quantile(depth[finite], (0.002, 0.998))
        rgb_image = categorical_raster(
            time_minutes[rows], depth[rows], temporal_idx[rows], rgb,
            (0.0, time_limit), (depth_low, depth_high), 1750,
            max(256, int((depth_high - depth_low) / 3)),
        )
        axis.imshow(
            rgb_image, origin="lower", extent=(0.0, time_limit, depth_low, depth_high),
            aspect="auto", interpolation="nearest",
        )
        artist = plt.cm.ScalarMappable(norm=normalization, cmap=colormap)
    axis.set(
        xlabel="recording time (min)",
        ylabel="probe depth (µm)",
        title=(
            f"Q={len(omega)} temporal-codebook selections · "
            f"{len(rows):,} / {event_count:,} spikes displayed"
        ),
    )
    axis.set_xlim(0.0, max(float(time_minutes[finite].max()), 1e-9))
    depth_low, depth_high = np.quantile(depth[finite], (0.002, 0.998))
    axis.set_ylim(depth_low, depth_high)
    axis.set_facecolor(BACKGROUND)
    axis.grid(color=GRID, alpha=0.8, linewidth=0.45)
    axis.set_axisbelow(True)
    axis.tick_params(colors=FONT)
    axis.xaxis.label.set_color(FONT)
    axis.yaxis.label.set_color(FONT)
    axis.title.set_color("#eeeeee")
    for spine in axis.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.5)

    colorbar = figure.colorbar(
        artist,
        ax=axis,
        boundaries=boundaries,
        ticks=np.arange(len(omega)),
        pad=0.015,
        fraction=0.035,
        aspect=32,
    )
    colorbar.set_label("selected temporal codebook row", color=FONT)
    colorbar.ax.tick_params(colors=FONT)
    colorbar.ax.set_yticklabels([rf"$\Omega_{{{row}}}$" for row in range(len(omega))])
    colorbar.outline.set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, facecolor=BACKGROUND)
    plt.close(figure)
    print(
        f"wrote {args.out} with {len(rows):,}/{event_count:,} spikes and "
        f"{len(omega)} linspaced RGB colors",
        flush=True,
    )


if __name__ == "__main__":
    main()
