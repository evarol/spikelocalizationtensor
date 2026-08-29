"""Plot complete-recording diagnostics for one-hot lattice peeling."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import PercentFormatter
import numpy as np


COORDINATE_LABELS = {
    "x": "global lateral x (µm)",
    "y": "probe depth y (µm)",
    "z": "distance from probe z (µm)",
}
PROJECTIONS = (("x", "y"), ("x", "z"), ("y", "z"))


def round_quantiles(values, rounds, n_rounds, points=(0.1, 0.5, 0.9)):
    result = np.empty((len(points), n_rounds), dtype=np.float64)
    for round_index in range(n_rounds):
        result[:, round_index] = np.quantile(
            np.asarray(values[rounds == round_index]), points
        )
    return result


def round_fractions(labels, rounds, n_rounds, n_labels):
    counts = np.zeros((n_rounds, n_labels), dtype=np.int64)
    for round_index in range(n_rounds):
        counts[round_index] = np.bincount(
            np.asarray(labels[rounds == round_index], dtype=np.int64),
            minlength=n_labels,
        )
    return counts, counts / counts.sum(axis=1, keepdims=True)


def plot_peeling_overview(run, output):
    rounds = np.load(run / "peeling_round.npy", mmap_mode="r")
    fitted_score = np.load(run / "fitted_projection_score.npy", mmap_mode="r")
    maximum_rmse = np.load(run / "maximum_channel_normalized_rmse.npy", mmap_mode="r")
    captured_fraction = np.load(run / "captured_fraction.npy", mmap_mode="r")
    sigma_index = np.load(run / "sigma_index.npy", mmap_mode="r")
    temporal_index = np.load(run / "temporal_idx.npy", mmap_mode="r")
    omega = np.load(run / "omega.npy")
    metadata = json.loads((run / "config.json").read_text())
    sigma_values = np.asarray(metadata["sigma_values_um"])
    n_rounds = int(rounds.max()) + 1
    round_numbers = np.arange(1, n_rounds + 1)
    counts = np.bincount(rounds, minlength=n_rounds)
    fitted_quantiles = round_quantiles(fitted_score, rounds, n_rounds)
    rmse_quantiles = round_quantiles(maximum_rmse, rounds, n_rounds)
    capture_quantiles = round_quantiles(captured_fraction, rounds, n_rounds)
    _, sigma_fraction = round_fractions(
        sigma_index, rounds, n_rounds, len(sigma_values)
    )
    _, temporal_fraction = round_fractions(
        temporal_index, rounds, n_rounds, len(omega)
    )
    display_ticks = np.unique(
        np.rint(np.linspace(1, n_rounds, min(10, n_rounds))).astype(np.int64)
    )

    figure, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(round_numbers, counts, color="#31688e", marker=".", linewidth=1.3)
    axis.set_yscale("log")
    axis.set(
        title="accepted events by peeling round",
        xlabel="peeling round",
        ylabel="accepted events",
        xticks=display_ticks,
    )

    axis = axes[0, 1]
    axis.fill_between(
        round_numbers, fitted_quantiles[0], fitted_quantiles[2],
        color="#35b779", alpha=0.25, label="10th–90th percentile",
    )
    axis.plot(round_numbers, fitted_quantiles[1], color="#238443", label="median")
    axis.axhline(
        metadata["config"]["min_fitted_projection"], color="#d94801",
        linestyle="--", linewidth=1, label="configured minimum",
    )
    axis.set(
        title="fitted projection score by round",
        xlabel="peeling round",
        ylabel="projection score",
        xticks=display_ticks,
    )
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    axis.fill_between(
        round_numbers, rmse_quantiles[0], rmse_quantiles[2],
        color="#8073ac", alpha=0.25, label="10th–90th percentile",
    )
    axis.plot(round_numbers, rmse_quantiles[1], color="#542788", label="median")
    axis.axhline(
        metadata["config"]["max_channel_normalized_rmse"], color="#d94801",
        linestyle="--", linewidth=1, label="configured maximum",
    )
    axis.set(
        title="maximum channel-normalized RMSE by round",
        xlabel="peeling round",
        ylabel="maximum normalized RMSE",
        xticks=display_ticks,
    )
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.fill_between(
        round_numbers, capture_quantiles[0], capture_quantiles[2],
        color="#2b8cbe", alpha=0.25, label="10th–90th percentile",
    )
    axis.plot(round_numbers, capture_quantiles[1], color="#045a8d", label="median")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set(
        title="captured local waveform energy by round",
        xlabel="peeling round",
        ylabel="captured fraction",
        xticks=display_ticks,
    )
    axis.legend(fontsize=8)

    axis = axes[2, 0]
    image = axis.imshow(
        sigma_fraction, origin="upper", aspect="auto", cmap="inferno",
        vmin=0, vmax=max(0.01, float(sigma_fraction.max())),
        interpolation="nearest",
    )
    axis.set(
        title="spatial-spread selection within each round",
        xlabel="sigma (µm)",
        ylabel="peeling round",
        xticks=np.arange(len(sigma_values)),
        xticklabels=[f"{value:g}" for value in sigma_values],
        yticks=display_ticks - 1,
        yticklabels=display_ticks,
    )
    figure.colorbar(image, ax=axis, label="fraction of accepted events")

    axis = axes[2, 1]
    image = axis.imshow(
        temporal_fraction, origin="upper", aspect="auto", cmap="magma",
        vmin=0, vmax=max(0.01, float(temporal_fraction.max())),
        interpolation="nearest",
    )
    axis.set(
        title="temporal-codebook selection within each round",
        xlabel="Omega row",
        ylabel="peeling round",
        xticks=np.arange(len(omega)),
        yticks=display_ticks - 1,
        yticklabels=display_ticks,
    )
    figure.colorbar(image, ax=axis, label="fraction of accepted events")

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle(
        f"one-hot lattice full recording · {len(rounds):,} accepted events · "
        f"{n_rounds} peeling rounds",
        fontsize=14,
    )
    figure.savefig(output / "peeling_overview.png", dpi=800, bbox_inches="tight")
    plt.close(figure)


def load_round_summaries(run):
    summaries = []
    terminal = []
    stopping_reasons = Counter()
    for path in sorted((run / "chunks").glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            chunk_summaries = json.loads(str(archive["pass_summaries_json"].item()))
            stopping_reason = str(archive["stopping_reason"].item())
        summaries.extend(chunk_summaries)
        terminal.append((int(chunk_summaries[-1]["peeling_round"]), stopping_reason))
        stopping_reasons[stopping_reason] += 1
    return summaries, terminal, stopping_reasons


def plot_stopping_diagnostics(run, output):
    summaries, terminal, stopping_reasons = load_round_summaries(run)
    n_rounds = max(int(row["peeling_round"]) for row in summaries) + 1
    round_numbers = np.arange(1, n_rounds + 1)
    active = np.zeros(n_rounds, dtype=np.int64)
    proposed = np.zeros(n_rounds, dtype=np.int64)
    before_merge = np.zeros(n_rounds, dtype=np.int64)
    duplicate = np.zeros(n_rounds, dtype=np.int64)
    accepted = np.zeros(n_rounds, dtype=np.int64)
    energy_drop = [[] for _ in range(n_rounds)]
    for row in summaries:
        index = int(row["peeling_round"])
        active[index] += 1
        proposed[index] += int(row["proposed"])
        before_merge[index] += int(row["accepted_before_merge"])
        duplicate[index] += int(row["duplicate_rejected"])
        accepted[index] += int(row["accepted"])
        energy_drop[index].append(float(row["full_energy_drop_fraction"]))
    stop_labels = sorted(stopping_reasons)
    stops = np.zeros((len(stop_labels), n_rounds), dtype=np.int64)
    for round_index, reason in terminal:
        stops[stop_labels.index(reason), round_index] += 1
    drop_quantiles = np.asarray(
        [np.quantile(values, (0.1, 0.5, 0.9)) for values in energy_drop]
    ).T
    display_ticks = np.unique(
        np.rint(np.linspace(1, n_rounds, min(10, n_rounds))).astype(np.int64)
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(round_numbers, proposed, label="proposed", color="#756bb1")
    axis.plot(round_numbers, before_merge, label="passed fit gates", color="#31a354")
    axis.plot(round_numbers, accepted, label="accepted after merge", color="#3182bd")
    axis.set_yscale("log")
    axis.set(
        title="aggregate event flow across active chunks",
        xlabel="peeling round",
        ylabel="events",
        xticks=display_ticks,
    )
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    rejected_fraction = np.divide(
        duplicate, before_merge, out=np.zeros_like(duplicate, dtype=np.float64),
        where=before_merge > 0,
    )
    axis.plot(round_numbers, rejected_fraction, color="#d95f0e")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set(
        title="fit-gate events rejected by duplicate merging",
        xlabel="peeling round",
        ylabel="fraction rejected",
        xticks=display_ticks,
        ylim=(0, 1),
    )

    axis = axes[1, 0]
    axis.plot(round_numbers, active, color="#252525", label="chunks entering round")
    bottom = np.zeros(n_rounds, dtype=np.int64)
    colors = plt.colormaps["Set2"](np.linspace(0, 1, max(3, len(stop_labels))))
    for label, values, color in zip(stop_labels, stops, colors):
        axis.bar(round_numbers, values, bottom=bottom, color=color, label=label)
        bottom += values
    axis.set(
        title="active chunks and recorded terminal reasons",
        xlabel="peeling round",
        ylabel="chunks",
        xticks=display_ticks,
    )
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.fill_between(
        round_numbers, drop_quantiles[0], drop_quantiles[2],
        color="#74a9cf", alpha=0.3, label="10th–90th percentile",
    )
    axis.plot(round_numbers, drop_quantiles[1], color="#0570b0", label="median")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set(
        title="full residual-energy drop per active chunk and round",
        xlabel="peeling round",
        ylabel="energy drop",
        xticks=display_ticks,
    )
    axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    reason_text = " · ".join(
        f"{label}: {stopping_reasons[label]:,}" for label in stop_labels
    )
    figure.suptitle(f"one-hot lattice stopping diagnostics · {reason_text}", fontsize=13)
    figure.savefig(output / "stopping_diagnostics.png", dpi=800, bbox_inches="tight")
    plt.close(figure)


def plot_localization_cohorts(run, output):
    sources = np.load(run / "global_sources.npy", mmap_mode="r")
    rounds = np.load(run / "peeling_round.npy", mmap_mode="r")
    contacts = np.load(run / "channel_positions.npy")
    n_rounds = int(rounds.max()) + 1
    cohorts = [
        ("all rounds", None),
        ("round 1", (0, 1)),
        ("rounds 2–5", (1, 5)),
        ("rounds 6–15", (5, 15)),
        ("rounds 16–30", (15, 30)),
        (f"rounds 31–{n_rounds}", (30, n_rounds)),
    ]
    x_edges = np.linspace(contacts[:, 0].min() - 170, contacts[:, 0].max() + 170, 321)
    depth_edges = np.linspace(
        contacts[:, 1].min() - 170, contacts[:, 1].max() + 170, 769
    )
    counts = []
    sizes = []
    for _, bounds in cohorts:
        if bounds is None:
            keep = np.ones(len(rounds), dtype=bool)
        else:
            keep = (rounds >= bounds[0]) & (rounds < bounds[1])
        count, _, _ = np.histogram2d(
            np.asarray(sources[keep, 1]), np.asarray(sources[keep, 0]),
            bins=(depth_edges, x_edges),
        )
        counts.append(count)
        sizes.append(int(keep.sum()))
    nonzero = np.concatenate([count[count > 0] for count in counts])
    norm = LogNorm(vmin=1, vmax=max(1.0, float(np.quantile(nonzero, 0.997))))

    figure, axes = plt.subplots(2, 3, figsize=(15, 12), constrained_layout=True)
    image = None
    for axis, (label, _), count, size in zip(axes.flat, cohorts, counts, sizes):
        image = axis.imshow(
            count, origin="lower", aspect="auto", cmap="inferno", norm=norm,
            interpolation="nearest",
            extent=(x_edges[0], x_edges[-1], depth_edges[0], depth_edges[-1]),
        )
        axis.scatter(
            contacts[:, 0], contacts[:, 1], s=3, marker="s",
            facecolors="none", edgecolors="#55d6d6", linewidths=0.25,
        )
        axis.set(
            title=f"{label} · {size:,} events",
            xlabel="global lateral position (µm)",
            ylabel="probe depth (µm)",
        )
    figure.colorbar(image, ax=axes, label="events per spatial bin", pad=0.01)
    figure.suptitle("one-hot lattice localization density by peeling-round cohort", fontsize=14)
    figure.savefig(
        output / "localization_by_round_cohort.png", dpi=800, bbox_inches="tight"
    )
    plt.close(figure)


def localization_coordinates(run):
    sources = np.load(run / "global_sources.npy", mmap_mode="r")
    return {
        "x": np.asarray(sources[:, 0]),
        "y": np.asarray(sources[:, 1]),
        "z": np.asarray(sources[:, 2]),
    }


def localization_edges(coordinates, contacts):
    return {
        "x": np.linspace(
            min(float(coordinates["x"].min()), float(contacts[:, 0].min()) - 150),
            max(float(coordinates["x"].max()), float(contacts[:, 0].max()) + 150),
            321,
        ),
        "y": np.linspace(
            min(float(coordinates["y"].min()), float(contacts[:, 1].min()) - 150),
            max(float(coordinates["y"].max()), float(contacts[:, 1].max()) + 150),
            769,
        ),
        "z": np.linspace(0, max(300.0, float(coordinates["z"].max())), 301),
    }


def contact_coordinate(contacts, name):
    if name == "x":
        return contacts[:, 0]
    if name == "y":
        return contacts[:, 1]
    return np.zeros(len(contacts), dtype=np.float32)


def projection_histogram(coordinates, keep, horizontal, vertical, edges):
    counts, _, _ = np.histogram2d(
        coordinates[horizontal][keep],
        coordinates[vertical][keep],
        bins=(edges[horizontal], edges[vertical]),
    )
    return counts.T


def draw_xyz_projection(
    axis,
    counts,
    horizontal,
    vertical,
    edges,
    contacts,
    norm,
    title,
):
    image = axis.imshow(
        counts,
        origin="lower",
        aspect="auto",
        cmap="inferno",
        norm=norm,
        interpolation="nearest",
        extent=(
            edges[horizontal][0],
            edges[horizontal][-1],
            edges[vertical][0],
            edges[vertical][-1],
        ),
    )
    axis.scatter(
        contact_coordinate(contacts, horizontal),
        contact_coordinate(contacts, vertical),
        s=3,
        marker="s",
        facecolors="none",
        edgecolors="#55d6d6",
        linewidths=0.25,
    )
    axis.set(
        title=title,
        xlabel=COORDINATE_LABELS[horizontal],
        ylabel=COORDINATE_LABELS[vertical],
    )
    return image


def plot_xyz_localizations(run, output):
    coordinates = localization_coordinates(run)
    contacts = np.load(run / "channel_positions.npy")
    rounds = np.load(run / "peeling_round.npy", mmap_mode="r")
    edges = localization_edges(coordinates, contacts)
    all_events = np.ones(len(rounds), dtype=bool)
    full_counts = {
        projection: projection_histogram(
            coordinates, all_events, *projection, edges
        )
        for projection in PROJECTIONS
    }

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, projection in zip(axes, PROJECTIONS):
        counts = full_counts[projection]
        nonzero = counts[counts > 0]
        norm = LogNorm(
            vmin=1,
            vmax=max(1.0, float(np.quantile(nonzero, 0.997))),
        )
        image = draw_xyz_projection(
            axis,
            counts,
            *projection,
            edges,
            contacts,
            norm,
            f"{projection[0].upper()}–{projection[1].upper()} · {len(rounds):,} events",
        )
        figure.colorbar(image, ax=axis, label="events per spatial bin", pad=0.01)
    figure.suptitle("one-hot lattice full-recording XYZ localization density", fontsize=14)
    figure.savefig(
        output / "xyz_localization_density.png", dpi=800, bbox_inches="tight"
    )
    plt.close(figure)

    n_rounds = int(rounds.max()) + 1
    cohorts = [
        ("round 1", (0, 1)),
        ("rounds 2–5", (1, 5)),
        ("rounds 6–15", (5, 15)),
        ("rounds 16–30", (15, 30)),
        (f"rounds 31–{n_rounds}", (30, n_rounds)),
    ]
    cohort_counts = []
    for label, bounds in cohorts:
        keep = (rounds >= bounds[0]) & (rounds < bounds[1])
        cohort_counts.append(
            (
                label,
                int(keep.sum()),
                {
                    projection: projection_histogram(
                        coordinates, keep, *projection, edges
                    )
                    for projection in PROJECTIONS
                },
            )
        )
    norms = {}
    for projection in PROJECTIONS:
        nonzero = np.concatenate(
            [counts[projection][counts[projection] > 0] for _, _, counts in cohort_counts]
        )
        norms[projection] = LogNorm(
            vmin=1,
            vmax=max(1.0, float(np.quantile(nonzero, 0.997))),
        )

    figure, axes = plt.subplots(
        len(cohorts), 3, figsize=(18, 20), constrained_layout=True
    )
    images = {}
    for row, (label, size, counts) in enumerate(cohort_counts):
        for column, projection in enumerate(PROJECTIONS):
            images[projection] = draw_xyz_projection(
                axes[row, column],
                counts[projection],
                *projection,
                edges,
                contacts,
                norms[projection],
                f"{label} · {size:,} events",
            )
    for column, projection in enumerate(PROJECTIONS):
        figure.colorbar(
            images[projection],
            ax=axes[:, column],
            label="events per spatial bin",
            pad=0.01,
        )
    figure.suptitle(
        "one-hot lattice XYZ localization density by peeling-round cohort", fontsize=15
    )
    figure.savefig(
        output / "xyz_localization_by_round.png", dpi=800, bbox_inches="tight"
    )
    plt.close(figure)


def plot_xyzsigma_scatter(run, output, max_points=500_000, seed=16016):
    coordinates = localization_coordinates(run)
    sigma_index = np.load(run / "sigma_index.npy", mmap_mode="r")
    sigma = np.load(run / "sigma.npy", mmap_mode="r")
    contacts = np.load(run / "channel_positions.npy")
    rng = np.random.default_rng(seed)
    keep = np.sort(
        rng.choice(len(sigma_index), min(max_points, len(sigma_index)), replace=False)
    )
    sigma_values = np.unique(np.asarray(sigma))
    colormap = plt.colormaps["turbo"].resampled(len(sigma_values))
    normalization = BoundaryNorm(
        np.arange(len(sigma_values) + 1) - 0.5, len(sigma_values)
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    artist = None
    for axis, projection in zip(axes.flat[:3], PROJECTIONS):
        horizontal, vertical = projection
        artist = axis.scatter(
            coordinates[horizontal][keep],
            coordinates[vertical][keep],
            c=np.asarray(sigma_index[keep]),
            cmap=colormap,
            norm=normalization,
            s=0.25,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            contact_coordinate(contacts, horizontal),
            contact_coordinate(contacts, vertical),
            s=4,
            marker="s",
            facecolors="none",
            edgecolors="white",
            linewidths=0.3,
        )
        axis.set(
            title=f"{horizontal.upper()}–{vertical.upper()} localization",
            xlabel=COORDINATE_LABELS[horizontal],
            ylabel=COORDINATE_LABELS[vertical],
        )
        axis.grid(alpha=0.12)

    counts = np.bincount(np.asarray(sigma_index, dtype=np.int64), minlength=len(sigma_values))
    fractions = counts / counts.sum()
    axes[1, 1].bar(
        np.arange(len(sigma_values)), fractions,
        color=colormap(np.arange(len(sigma_values))),
    )
    axes[1, 1].set(
        title="sigma selection across all accepted events",
        xlabel="sigma (µm)",
        ylabel="fraction of accepted events",
        xticks=np.arange(len(sigma_values)),
        xticklabels=[f"{value:g}" for value in sigma_values],
    )
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(alpha=0.18, axis="y")
    colorbar = figure.colorbar(
        artist,
        ax=axes.flat[:3].tolist(),
        boundaries=np.arange(len(sigma_values) + 1) - 0.5,
        ticks=np.arange(len(sigma_values)),
        pad=0.01,
    )
    colorbar.set_label("selected sigma (µm)")
    colorbar.ax.set_yticklabels([f"{value:g}" for value in sigma_values])
    figure.suptitle(
        f"one-hot lattice XYZ-sigma localizations · {len(keep):,} sampled of "
        f"{len(sigma_index):,} accepted events",
        fontsize=14,
    )
    figure.savefig(
        output / "xyzsigma_localization_scatter.png", dpi=800, bbox_inches="tight"
    )
    plt.close(figure)


def geometry_waveforms(axis, coordinates, waveforms, scale, color, linestyle="-"):
    offsets = (np.arange(waveforms.shape[1]) - waveforms.shape[1] / 2) * 0.34
    for coordinate, waveform in zip(coordinates, waveforms):
        axis.plot(
            coordinate[0] + offsets,
            coordinate[1] + scale * waveform,
            color=color,
            linewidth=0.85,
            linestyle=linestyle,
        )


def select_round_examples(rounds, scores, count):
    available = np.unique(rounds)
    selected_rounds = available[
        np.unique(
            np.rint(np.linspace(0, len(available) - 1, min(count, len(available)))).astype(int)
        )
    ]
    selected = []
    for round_index in selected_rounds:
        rows = np.flatnonzero(rounds == round_index)
        target = np.median(scores[rows])
        selected.append(rows[np.argmin(np.abs(scores[rows] - target))])
    return np.asarray(selected, dtype=np.int64)


def select_boundary_examples(times, scores, count):
    edges = np.linspace(float(times.min()), float(times.max()) + 1, count + 1)
    selected = []
    for low, high in zip(edges[:-1], edges[1:]):
        rows = np.flatnonzero((times >= low) & (times < high))
        if len(rows):
            selected.append(rows[np.argmin(scores[rows])])
    return np.asarray(selected, dtype=np.int64)


def plot_reconstruction_grid(values, output, title):
    measured = values["residual_waveforms"]
    predicted = values["predictions"]
    after = measured - predicted
    n_examples = len(measured)
    figure, axes = plt.subplots(
        5,
        n_examples,
        figsize=(3.3 * n_examples, 15),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"height_ratios": (1.15, 1.0, 0.72, 0.9, 0.72)},
    )
    for column in range(n_examples):
        valid = values["neighbor_ids"][column] >= 0
        coordinates = values["local_coords"][column, valid]
        observed = measured[column, valid]
        model = predicted[column, valid]
        residual = after[column, valid]
        scale = 17.0 / max(float(np.abs(observed).max()), np.finfo(np.float32).tiny)

        geometry_waveforms(axes[0, column], coordinates, observed, scale, "#e03131")
        geometry_waveforms(
            axes[0, column], coordinates, model, scale, "#2f9e44", "--"
        )
        axes[0, column].scatter(
            coordinates[:, 0], coordinates[:, 1], s=7, marker="s", color="0.55"
        )
        axes[0, column].set_title(
            f"round {int(values['peeling_round'][column]) + 1} · "
            f"score {values['fitted_projection_score'][column]:.2f}\n"
            f"capture {100 * values['captured_fraction'][column]:.1f}% · "
            f"q={int(values['temporal_idx'][column])}",
            fontsize=8,
        )
        if column == 0:
            axes[0, column].set_ylabel("input residual (red)\nsaved prediction (green)")

        geometry_waveforms(axes[1, column], coordinates, residual, scale, "#3182bd")
        axes[1, column].scatter(
            coordinates[:, 0], coordinates[:, 1], s=7, marker="s", color="0.55"
        )
        if column == 0:
            axes[1, column].set_ylabel("post-subtraction residual")

        peak_channel = int(np.argmax(np.square(model).sum(axis=1)))
        time_ms = 1000 * np.arange(observed.shape[1]) / 30_000.0
        axes[2, column].plot(time_ms, observed[peak_channel], color="#e03131")
        axes[2, column].plot(time_ms, model[peak_channel], color="#2f9e44", linestyle="--")
        axes[2, column].plot(time_ms, residual[peak_channel], color="#3182bd")
        axes[2, column].set(xlabel="waveform time (ms)", title="strongest model channel")
        if column == 0:
            axes[2, column].set_ylabel("voltage")

        source = values["sources"][column]
        sigma = float(values["sigma"][column])
        footprint = sigma / np.sqrt(
            np.square(coordinates[:, 0] - source[0])
            + np.square(coordinates[:, 1] - source[1])
            + source[2] ** 2
            + sigma**2
        )
        artist = axes[3, column].scatter(
            coordinates[:, 0], coordinates[:, 1], c=footprint,
            cmap="magma", s=42, marker="s",
        )
        axes[3, column].plot(
            source[0], source[1], "o", markerfacecolor="none",
            markeredgecolor="#4c8dff", markersize=11, markeredgewidth=1.6,
        )
        axes[3, column].set(
            title=f"x={source[0]:.0f}, y={source[1]:.0f}, z={source[2]:.0f}, sigma={sigma:.0f}",
            xlabel="local x (µm)",
        )
        if column == 0:
            axes[3, column].set_ylabel("local y (µm)")
        figure.colorbar(artist, ax=axes[3, column], fraction=0.045, pad=0.02)

        channel_error = values["channel_normalized_rmse"][column, valid]
        axes[4, column].bar(np.arange(len(channel_error)), channel_error, color="#845ef7")
        axes[4, column].axhline(3, color="#e03131", linestyle="--", linewidth=0.8)
        axes[4, column].set(xlabel="local channel", title="channel-normalized RMSE")
        if column == 0:
            axes[4, column].set_ylabel("RMSE")

        for axis in axes[:, column]:
            axis.grid(alpha=0.15)
            axis.tick_params(labelsize=7)
    figure.suptitle(title, fontsize=14)
    figure.savefig(output, dpi=800, bbox_inches="tight")
    plt.close(figure)


def plot_reconstruction_examples(run, output, chunk_index=0, count=6):
    path = run / "chunks" / f"chunk_{chunk_index:06d}.npz"
    fields = (
        "spike_times",
        "residual_waveforms",
        "predictions",
        "local_coords",
        "neighbor_ids",
        "sources",
        "sigma",
        "temporal_idx",
        "alpha",
        "channel_normalized_rmse",
        "peeling_round",
        "fitted_projection_score",
        "captured_fraction",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = set(fields).difference(archive.files)
        if missing:
            raise KeyError(f"{path} is missing reconstruction fields: {sorted(missing)}")
        rounds = np.asarray(archive["peeling_round"])
        scores = np.asarray(archive["fitted_projection_score"])
        times = np.asarray(archive["spike_times"])
        selections = {
            "reconstruction_examples_by_round.png": select_round_examples(
                rounds, scores, count
            ),
            "reconstruction_examples_score_boundary.png": select_boundary_examples(
                times, scores, count
            ),
        }
        for name, indices in selections.items():
            values = {field: np.asarray(archive[field][indices]) for field in fields}
            plot_reconstruction_grid(
                values,
                output / name,
                (
                    f"one-hot lattice saved reconstruction examples · chunk {chunk_index} · "
                    + ("round progression" if "by_round" in name else "lowest fitted score in each time segment")
                ),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_peeling_overview(args.run, args.out)
    print(f"wrote {args.out / 'peeling_overview.png'}", flush=True)
    plot_stopping_diagnostics(args.run, args.out)
    print(f"wrote {args.out / 'stopping_diagnostics.png'}", flush=True)
    plot_localization_cohorts(args.run, args.out)
    print(f"wrote {args.out / 'localization_by_round_cohort.png'}", flush=True)
    plot_xyz_localizations(args.run, args.out)
    print(f"wrote {args.out / 'xyz_localization_density.png'}", flush=True)
    print(f"wrote {args.out / 'xyz_localization_by_round.png'}", flush=True)
    plot_xyzsigma_scatter(args.run, args.out)
    print(f"wrote {args.out / 'xyzsigma_localization_scatter.png'}", flush=True)
    plot_reconstruction_examples(args.run, args.out)
    print(f"wrote reconstruction example panels under {args.out}", flush=True)


if __name__ == "__main__":
    main()
