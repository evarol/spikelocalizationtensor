"""Amplitude-coloured probe-global localization projections."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from maths import load_channel_map


COORDINATE = {"x": 0, "y": 1, "z": 2}
LABEL = {
    "x": "x / lateral (µm)",
    "y": "y / depth (µm)",
    "z": "z / distance (µm)",
}
BACKGROUND = "#0d0d0d"
FONT = "#d7d7d7"
GRID = "#292929"
UM_PER_INCH = 300.0


def absolute_localizations(session, fit_path):
    centroids = np.load(session / "centroids.npy", mmap_mode="r")
    with np.load(fit_path) as fit:
        sources = np.asarray(fit["sources"], dtype=np.float64)
        alpha = np.abs(np.asarray(fit["alpha"], dtype=np.float64))
    absolute = np.column_stack(
        (
            np.asarray(centroids[:, 0], dtype=np.float64) + sources[:, 0],
            np.asarray(centroids[:, 1], dtype=np.float64) + sources[:, 1],
            sources[:, 2],
        )
    )
    return absolute, alpha


def robust_limits(locations, contacts):
    limits = {}
    contact_coordinates = {
        "x": contacts[:, 0],
        "y": contacts[:, 1],
        "z": np.zeros(len(contacts)),
    }
    for name, column in COORDINATE.items():
        values = locations[:, column]
        low, high = np.quantile(values[np.isfinite(values)], (0.002, 0.998))
        low = min(float(low), float(contact_coordinates[name].min()))
        high = max(float(high), float(contact_coordinates[name].max()))
        margin = 0.025 * max(high - low, 1.0)
        limits[name] = (low - margin, high + margin)
    return limits


def style_axis(axis):
    axis.set_facecolor(BACKGROUND)
    axis.grid(True, color=GRID, linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=FONT, labelsize=7, length=2.5, width=0.5)
    for spine in axis.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.5)


def projection(
    axis,
    locations,
    colour,
    contacts,
    limits,
    norm,
    horizontal,
    vertical,
    title,
):
    def contact_coordinate(name):
        if name == "x":
            return contacts[:, 0]
        if name == "y":
            return contacts[:, 1]
        return np.zeros(len(contacts))

    style_axis(axis)
    axis.scatter(
        contact_coordinate(horizontal),
        contact_coordinate(vertical),
        s=5.0,
        c="white",
        marker="s",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    artist = axis.scatter(
        locations[:, COORDINATE[horizontal]],
        locations[:, COORDINATE[vertical]],
        s=0.55,
        c=colour,
        cmap="inferno",
        norm=norm,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlim(*limits[horizontal])
    axis.set_ylim(*limits[vertical])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(LABEL[horizontal], fontsize=8, color=FONT, labelpad=2)
    axis.set_ylabel(LABEL[vertical], fontsize=8, color=FONT, labelpad=2)
    axis.set_title(title, fontsize=9, color="#eeeeee", loc="left", pad=4)
    return artist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--amplitude", type=Path)
    parser.add_argument("--sample", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method-label", default="masked one-hot")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    locations, fitted_amplitude = absolute_localizations(args.session, args.fit)
    if args.amplitude is None:
        amplitude = fitted_amplitude
        amplitude_label = "fitted |alpha|"
    else:
        amplitude = np.abs(np.asarray(np.load(args.amplitude), dtype=np.float64))
        amplitude_label = args.amplitude.stem.replace("_", " ")
    if len(amplitude) != len(locations):
        raise ValueError(
            f"amplitude has {len(amplitude):,} rows but fit has {len(locations):,}"
        )

    rng = np.random.default_rng(args.seed)
    total_count = len(locations)
    count = min(args.sample, total_count)
    selected = np.sort(rng.choice(total_count, count, replace=False))
    locations = locations[selected]
    amplitude = amplitude[selected]
    positive = amplitude[amplitude > 0]
    floor = float(np.quantile(positive, 0.001)) if positive.size else 1.0
    log_amplitude = np.log10(np.maximum(amplitude, floor))
    colour_limits = np.quantile(log_amplitude, (0.01, 0.99))
    norm = Normalize(float(colour_limits[0]), float(colour_limits[1]))

    order = np.argsort(log_amplitude, kind="stable")
    locations = locations[order]
    log_amplitude = log_amplitude[order]

    probe = load_channel_map(args.recording)
    contacts = np.asarray(probe.contact_positions, dtype=np.float64)
    limits = robust_limits(locations, contacts)
    span = {name: high - low for name, (low, high) in limits.items()}

    left, right, top, bottom = 1.15, 1.15, 1.00, 0.80
    row_gap, group_gap = 0.48, 0.70
    depth_width = span["y"] / UM_PER_INCH
    lateral_height = span["x"] / UM_PER_INCH
    distance_height = span["z"] / UM_PER_INCH
    lateral_width = span["x"] / UM_PER_INCH
    figure_width = left + depth_width + right
    figure_height = (
        top
        + lateral_height
        + row_gap
        + distance_height
        + group_gap
        + distance_height
        + bottom
    )
    figure = plt.figure(figsize=(figure_width, figure_height), facecolor=BACKGROUND)

    def place(x, y, width, height):
        return figure.add_axes(
            (
                x / figure_width,
                1.0 - (y + height) / figure_height,
                width / figure_width,
                height / figure_height,
            )
        )

    cursor = top
    axis_xy = place(left, cursor, depth_width, lateral_height)
    artist = projection(
        axis_xy,
        locations,
        log_amplitude,
        contacts,
        limits,
        norm,
        "y",
        "x",
        f"{args.method_label} · x–y",
    )
    cursor += lateral_height + row_gap
    axis_zy = place(left, cursor, depth_width, distance_height)
    projection(
        axis_zy,
        locations,
        log_amplitude,
        contacts,
        limits,
        norm,
        "y",
        "z",
        f"{args.method_label} · z–y",
    )
    cursor += distance_height + group_gap
    axis_zx = place(left, cursor, lateral_width, distance_height)
    projection(
        axis_zx,
        locations,
        log_amplitude,
        contacts,
        limits,
        norm,
        "x",
        "z",
        f"{args.method_label} · z–x",
    )

    colorbar_axis = place(
        figure_width - right + 0.35,
        top,
        0.09,
        lateral_height + row_gap + distance_height,
    )
    colorbar = figure.colorbar(artist, cax=colorbar_axis)
    colorbar.set_label(f"log10 {amplitude_label}", fontsize=8, color=FONT)
    colorbar.ax.tick_params(colors=FONT, labelsize=7, length=2)
    colorbar.outline.set_visible(False)

    figure.suptitle(
        f"Spike localization on dataset1_p1 — {args.method_label}",
        fontsize=12,
        color="#eeeeee",
        x=left / figure_width,
        ha="left",
        y=1.0 - 0.28 / figure_height,
    )
    figure.text(
        left / figure_width,
        1.0 - 0.62 / figure_height,
        f"{count:,} of {total_count:,} spikes sampled · "
        f"colour = log10 {amplitude_label} · white squares are probe contacts · "
        f"every panel is isotropic at {UM_PER_INCH:g} µm per inch",
        fontsize=7.5,
        color="#8f8f8f",
        ha="left",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=800, facecolor=BACKGROUND)
    plt.close(figure)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
