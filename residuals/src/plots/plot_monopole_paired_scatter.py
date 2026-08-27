"""Paired amplitude scatter for three monopole localization methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


BASE_DIRECTORY = Path(
    "/scratch/ap7151/pre-summer-archive/neurips-week/09_compares/out/"
    "dataset1_p1/monopolar"
)
AM15577_FIT = Path(
    "/scratch/ap7151/_SYMLINKS/am15577-paths/UnitMatch/SLT/"
    "Basic_implementation/one_basis_alpha/results/"
    "pi_scaledonebasis_lat64_monopole_Q8.npz"
)
CHANNEL_LOCATIONS = Path(
    "/scratch/ap7151/_SYMLINKS/am15577-paths/UnitMatch/Data/"
    "postneurips_sln_datasets/Steinmetz/dataset1_p1/"
    "channel_locations.npy"
)

COORDINATE = {"x": 0, "y": 1, "z": 2}
LABEL = {
    "x": "x / lateral (µm)",
    "y": "y / depth (µm)",
    "z": "z / distance (µm)",
}
BACKGROUND = "#0d0d0d"
FONT = "#d7d7d7"
GRID = "#292929"
UM_PER_INCH = 400.0
DOT_SIZE = 0.55
DOT_OPACITY = 0.48


def event_keys(times: np.ndarray, channels: np.ndarray, stride: int) -> np.ndarray:
    times = np.asarray(times, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    if np.any(times < 0) or np.any(channels < 0) or np.any(channels >= stride):
        raise ValueError("sample indices and channels must define nonnegative keys")
    return times * stride + channels


def match_events(
    our_times: np.ndarray,
    our_channels: np.ndarray,
    base_times: np.ndarray,
    base_channels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(int(np.max(our_channels)), int(np.max(base_channels))) + 1
    our_keys = event_keys(our_times, our_channels, stride)
    base_keys = event_keys(base_times, base_channels, stride)
    if len(np.unique(our_keys)) != len(our_keys):
        raise ValueError("our detection list contains duplicate sample/channel pairs")
    if len(np.unique(base_keys)) != len(base_keys):
        raise ValueError("base detection list contains duplicate sample/channel pairs")
    _, our_index, base_index = np.intersect1d(
        our_keys,
        base_keys,
        assume_unique=True,
        return_indices=True,
    )
    return our_index, base_index


def load_base_locations(path: Path, index: np.ndarray) -> np.ndarray:
    locations = np.load(path, mmap_mode="r")
    return np.asarray(locations[index], dtype=np.float64)


def load_am15577_locations(path: Path, index: np.ndarray) -> np.ndarray:
    with np.load(path) as fit:
        site = np.asarray(fit["k"][index], dtype=np.int64) // int(fit["S"])
        anchor = np.asarray(fit["anchor"][index], dtype=np.float64)
        local = np.asarray(fit["mu_site"][site], dtype=np.float64)
    return anchor + local


def load_our_locations(session: Path, fit_path: Path, index: np.ndarray) -> np.ndarray:
    centroids = np.load(session / "centroids.npy", mmap_mode="r")
    with np.load(fit_path) as fit:
        sources = np.asarray(fit["sources"][index], dtype=np.float64)
    selected_centroids = np.asarray(centroids[index], dtype=np.float64)
    return np.column_stack(
        (
            selected_centroids[:, 0] + sources[:, 0],
            selected_centroids[:, 1] + sources[:, 1],
            sources[:, 2],
        )
    )


def shared_limits(
    methods: list[np.ndarray], contacts: np.ndarray
) -> dict[str, tuple[float, float]]:
    combined = np.concatenate(methods, axis=0)
    contact_coordinates = {
        "x": contacts[:, 0],
        "y": contacts[:, 1],
        "z": np.zeros(len(contacts)),
    }
    limits = {}
    for name, column in COORDINATE.items():
        values = combined[:, column]
        finite = values[np.isfinite(values)]
        low, high = np.quantile(finite, (0.01, 0.99))
        low = min(float(low), float(np.min(contact_coordinates[name])))
        high = max(float(high), float(np.max(contact_coordinates[name])))
        margin = 0.035 * max(high - low, 1.0)
        limits[name] = (low - margin, high + margin)
    return limits


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.grid(True, color=GRID, linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=FONT, labelsize=7, length=2.5, width=0.5)
    for spine in axis.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.5)


def projection(
    axis: plt.Axes,
    locations: np.ndarray,
    colour: np.ndarray,
    contacts: np.ndarray,
    limits: dict[str, tuple[float, float]],
    norm: Normalize,
    horizontal: str,
    vertical: str,
    title: str,
) -> None:
    def contact_coordinate(name: str) -> np.ndarray:
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
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        locations[:, COORDINATE[horizontal]],
        locations[:, COORDINATE[vertical]],
        s=DOT_SIZE,
        c=colour,
        cmap="inferno",
        norm=norm,
        alpha=DOT_OPACITY,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlim(*limits[horizontal])
    axis.set_ylim(*limits[vertical])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(LABEL[horizontal], fontsize=8, color=FONT, labelpad=2)
    axis.set_ylabel(LABEL[vertical], fontsize=8, color=FONT, labelpad=2)
    axis.set_title(title, fontsize=9, color="#eeeeee", loc="left", pad=4)


def render(
    methods: list[tuple[str, np.ndarray]],
    log_amplitude: np.ndarray,
    contacts: np.ndarray,
    limits: dict[str, tuple[float, float]],
    matched_count: int,
    sampled_count: int,
    output: Path,
) -> None:
    corner_name = {
        "Base monopole triangulation": "Base monopole",
        "am15577 free-amplitude one-hot monopole": "am15577 monopole",
        "Our masked one-hot monopole": "Our monopole",
    }
    colour_limits = np.quantile(log_amplitude, (0.01, 0.99))
    norm = Normalize(float(colour_limits[0]), float(colour_limits[1]))
    span = {name: high - low for name, (low, high) in limits.items()}

    left, right, top, bottom = 1.10, 1.15, 1.10, 0.75
    row_gap, group_gap = 0.38, 0.62
    depth_width = span["y"] / UM_PER_INCH
    lateral_height = span["x"] / UM_PER_INCH
    distance_height = span["z"] / UM_PER_INCH
    corner_width = span["x"] / UM_PER_INCH
    figure_width = left + depth_width + right
    figure_height = (
        top
        + len(methods)
        * (lateral_height + row_gap + distance_height + group_gap)
        + distance_height
        + bottom
    )
    figure = plt.figure(figsize=(figure_width, figure_height), facecolor=BACKGROUND)

    def place(x: float, y: float, width: float, height: float) -> plt.Axes:
        return figure.add_axes(
            (
                x / figure_width,
                1.0 - (y + height) / figure_height,
                width / figure_width,
                height / figure_height,
            )
        )

    cursor = top
    for name, locations in methods:
        axis_xy = place(left, cursor, depth_width, lateral_height)
        projection(
            axis_xy,
            locations,
            log_amplitude,
            contacts,
            limits,
            norm,
            "y",
            "x",
            f"{name}  ·  x–y",
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
            f"{name}  ·  z–y",
        )
        cursor += distance_height + group_gap

    available_gap = max(
        0.45,
        (depth_width - len(methods) * corner_width) / max(len(methods) - 1, 1),
    )
    for column, (name, locations) in enumerate(methods):
        axis_zx = place(
            left + column * (corner_width + available_gap),
            cursor,
            corner_width,
            distance_height,
        )
        projection(
            axis_zx,
            locations,
            log_amplitude,
            contacts,
            limits,
            norm,
            "x",
            "z",
            f"{corner_name.get(name, name)}  ·  z–x",
        )

    colorbar_axis = place(
        figure_width - right + 0.35,
        top,
        0.09,
        2 * lateral_height + distance_height,
    )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="inferno"), cax=colorbar_axis
    )
    colorbar.set_label("log10 detector amplitude", fontsize=8, color=FONT)
    colorbar.ax.tick_params(colors=FONT, labelsize=7, length=2)
    colorbar.outline.set_visible(False)

    figure.suptitle(
        "Paired monopole localizations on dataset1_p1",
        fontsize=12,
        color="#eeeeee",
        x=left / figure_width,
        ha="left",
        y=1.0 - 0.25 / figure_height,
    )
    figure.text(
        left / figure_width,
        1.0 - 0.60 / figure_height,
        f"{sampled_count:,} uniformly sampled from {matched_count:,} exact "
        "sample-index + channel matches · each matched spike has the same "
        "colour in all methods · shared axes and identical markers · "
        f"isotropic at {UM_PER_INCH:g} µm per inch",
        fontsize=7.5,
        color="#8f8f8f",
        ha="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=800, facecolor=BACKGROUND)
    plt.close(figure)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--our-fit", type=Path, required=True)
    parser.add_argument("--base-directory", type=Path, default=BASE_DIRECTORY)
    parser.add_argument("--am15577-fit", type=Path, default=AM15577_FIT)
    parser.add_argument(
        "--channel-locations", type=Path, default=CHANNEL_LOCATIONS
    )
    parser.add_argument("--sample", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    our_times = np.load(args.session / "spike_times.npy", mmap_mode="r")
    our_channels = np.load(args.session / "spike_channels.npy", mmap_mode="r")
    base_times = np.load(
        args.base_directory / "peak_sample_index.npy", mmap_mode="r"
    )
    base_channels = np.load(
        args.base_directory / "peak_channel.npy", mmap_mode="r"
    )
    our_match, base_match = match_events(
        our_times, our_channels, base_times, base_channels
    )
    if not len(our_match):
        raise ValueError("the detection stores have no matching events")

    rng = np.random.default_rng(args.seed)
    sampled_count = min(args.sample, len(our_match))
    selected = np.sort(rng.choice(len(our_match), sampled_count, replace=False))
    our_index = our_match[selected]
    base_index = base_match[selected]

    base_locations = load_base_locations(
        args.base_directory / "peak_locations.npy", base_index
    )
    am15577_locations = load_am15577_locations(args.am15577_fit, base_index)
    our_locations = load_our_locations(args.session, args.our_fit, our_index)
    amplitude_store = np.load(
        args.base_directory / "peak_amplitude.npy", mmap_mode="r"
    )
    amplitude = np.abs(np.asarray(amplitude_store[base_index], dtype=np.float64))

    positive = amplitude[amplitude > 0]
    floor = float(np.quantile(positive, 0.001)) if positive.size else 1.0
    log_amplitude = np.log10(np.maximum(amplitude, floor))
    order = np.argsort(log_amplitude, kind="stable")
    log_amplitude = log_amplitude[order]
    base_locations = base_locations[order]
    am15577_locations = am15577_locations[order]
    our_locations = our_locations[order]

    contacts = np.asarray(np.load(args.channel_locations), dtype=np.float64)
    methods = [
        ("Base monopole triangulation", base_locations),
        ("am15577 free-amplitude one-hot monopole", am15577_locations),
        ("Our masked one-hot monopole", our_locations),
    ]
    limits = shared_limits([locations for _, locations in methods], contacts)
    print(
        f"matched {len(our_match):,} events; sampled {sampled_count:,}; "
        + "  ".join(
            f"{name}[{low:.1f}, {high:.1f}]"
            for name, (low, high) in limits.items()
        )
    )
    render(
        methods,
        log_amplitude,
        contacts,
        limits,
        len(our_match),
        sampled_count,
        args.out,
    )


if __name__ == "__main__":
    main()
