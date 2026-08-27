"""Plot residual localizations and exact two-detection collision examples."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree

from plot_reconstructions import profile_parameters, reconstruct


EPS = np.finfo(np.float32).tiny
COMPONENT_COLORS = ("#ff9f1c", "#2ec4b6")


def padded_limits(values, quantiles=(0.0, 1.0), fraction=0.04):
    finite = np.asarray(values)[np.isfinite(values)]
    low, high = np.quantile(finite, quantiles)
    padding = fraction * max(float(high - low), 1.0)
    return float(low - padding), float(high + padding)


def validate_run(run):
    metadata = json.loads((run / "config.json").read_text())
    config = metadata["config"]
    if config["kernel"] != "monopole":
        raise ValueError("residual collision plots currently require monopole fits")
    if not config.get("save_waveforms", False):
        raise ValueError("collision examples require residual_waveforms in chunk files")
    chunk_paths = sorted((run / "chunks").glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"no chunk archives found in {run / 'chunks'}")
    return metadata, chunk_paths


def effective_width(sources, profile_idx, n_scales):
    sigma = np.geomspace(2.0, 512.0, n_scales)[
        np.asarray(profile_idx, dtype=np.int64)
    ]
    return np.sqrt(np.asarray(sources[:, 2], dtype=np.float64) ** 2 + sigma ** 2)


def plot_localizations(run, metadata, output, max_points_per_pass, seed):
    config = metadata["config"]
    sources = np.load(run / "global_sources.npy", mmap_mode="r")
    alpha = np.load(run / "alpha.npy", mmap_mode="r")
    profile_idx = np.load(run / "profile_idx.npy", mmap_mode="r")
    residual_pass = np.load(run / "residual_pass.npy", mmap_mode="r")
    captured_fraction = np.load(run / "captured_fraction.npy", mmap_mode="r")
    contacts = np.load(run / "channel_positions.npy")
    lengths = {
        len(sources),
        len(alpha),
        len(profile_idx),
        len(residual_pass),
        len(captured_fraction),
    }
    if len(lengths) != 1:
        raise ValueError("consolidated residual arrays have inconsistent lengths")

    rho = effective_width(sources, profile_idx, int(config["n_scales"]))
    log_amplitude = np.log10(
        np.maximum(np.abs(np.asarray(alpha)), np.finfo(np.float32).tiny)
    )
    color_limits = Normalize(
        *np.quantile(log_amplitude[np.isfinite(log_amplitude)], (0.01, 0.995)),
        clip=True,
    )
    n_passes = int(np.max(residual_pass)) + 1
    n_columns = 2
    n_rows = int(np.ceil(n_passes / n_columns))
    rng = np.random.default_rng(seed)

    source_x_limits = padded_limits(np.asarray(sources[:, 0]), (0.001, 0.999))
    source_y_limits = padded_limits(np.asarray(sources[:, 1]), (0.002, 0.998))
    x_limits = (
        min(source_x_limits[0], float(contacts[:, 0].min()) - 4),
        max(source_x_limits[1], float(contacts[:, 0].max()) + 4),
    )
    y_limits = (
        min(source_y_limits[0], float(contacts[:, 1].min()) - 10),
        max(source_y_limits[1], float(contacts[:, 1].max()) + 10),
    )

    plt.style.use("dark_background")
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(11.5, 5.2 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    artist = None
    counts = []
    for pass_index, axis in enumerate(axes.flat[:n_passes]):
        rows = np.flatnonzero(residual_pass == pass_index)
        counts.append(int(len(rows)))
        keep = np.sort(
            rng.choice(rows, min(max_points_per_pass, len(rows)), replace=False)
        )
        keep = keep[np.argsort(log_amplitude[keep], kind="stable")]
        artist = axis.scatter(
            np.asarray(sources[keep, 0]),
            np.asarray(sources[keep, 1]),
            c=log_amplitude[keep],
            cmap="inferno",
            norm=color_limits,
            s=0.45,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            contacts[:, 0],
            contacts[:, 1],
            s=4,
            marker="s",
            facecolors="none",
            edgecolors="white",
            linewidths=0.3,
            alpha=0.7,
        )
        axis.set(
            title=(
                f"pass {pass_index + 1} · {len(rows):,} fits\n"
                f"median capture {100 * np.median(captured_fraction[rows]):.1f}% · "
                f"median ρ {np.median(rho[rows]):.1f} µm"
            ),
            xlim=x_limits,
            ylim=y_limits,
            xlabel="lateral x (µm)",
            ylabel="probe depth y (µm)",
        )
        axis.title.set_fontsize(10)
        axis.grid(color="white", alpha=0.07, linewidth=0.5)
    for axis in axes.flat[n_passes:]:
        axis.set_visible(False)
    colorbar = figure.colorbar(artist, ax=list(axes.flat[:n_passes]), shrink=0.82)
    colorbar.set_label("log10 fitted |α|")
    figure.suptitle(
        f"Continuous residual localizations by subtraction pass · {len(sources):,} fits",
        fontsize=15,
    )
    figure.savefig(output, dpi=800, bbox_inches="tight")
    plt.close(figure)
    return counts


def collision_candidates(path, waveform_length, radius_um, min_separation_um):
    with np.load(path, allow_pickle=False) as archive:
        times = np.asarray(archive["spike_times"], dtype=np.int64)
        passes = np.asarray(archive["residual_pass"], dtype=np.int64)
        positions = np.asarray(archive["global_sources"][:, :2], dtype=np.float64)
        neighbor_ids = np.asarray(archive["neighbor_ids"], dtype=np.int32)
        captured = np.asarray(archive["captured_fraction"], dtype=np.float64)
        energy = np.asarray(archive["input_energy"], dtype=np.float64)

    scaled = np.column_stack(
        (
            times / max(waveform_length - 1, 1),
            positions[:, 0] / radius_um,
            positions[:, 1] / radius_um,
        )
    )
    pairs = cKDTree(scaled).query_pairs(1.0, p=np.inf, output_type="ndarray")
    if not len(pairs):
        return []
    left, right = pairs.T
    delta_samples = np.abs(times[left] - times[right])
    separation = np.linalg.norm(positions[left] - positions[right], axis=1)
    shared_channel = np.any(
        (neighbor_ids[left, :, None] == neighbor_ids[right, None, :])
        & (neighbor_ids[left, :, None] >= 0),
        axis=(1, 2),
    )
    is_edge = (
        (delta_samples < waveform_length)
        & (separation <= radius_um)
        & shared_channel
    )
    left = left[is_edge]
    right = right[is_edge]
    delta_samples = delta_samples[is_edge]
    separation = separation[is_edge]
    degree = np.bincount(
        np.concatenate((left, right)), minlength=len(times)
    )
    exact_pair = (
        (degree[left] == 1)
        & (degree[right] == 1)
        & (passes[left] != passes[right])
        & (separation >= min_separation_um)
    )
    chunk_index = int(path.stem.split("_")[-1])
    result = []
    for first, second, dt, distance in zip(
        left[exact_pair],
        right[exact_pair],
        delta_samples[exact_pair],
        separation[exact_pair],
    ):
        quality = min(captured[first], captured[second]) * np.sqrt(
            max(energy[first], 0) * max(energy[second], 0)
        )
        result.append(
            {
                "chunk": chunk_index,
                "first": int(first),
                "second": int(second),
                "delta_samples": int(dt),
                "separation_um": float(distance),
                "quality": float(quality),
                "passes": [int(passes[first]), int(passes[second])],
                "times": [int(times[first]), int(times[second])],
                "captured_fraction": [
                    float(captured[first]),
                    float(captured[second]),
                ],
            }
        )
    return result


def choose_examples(candidates, n_examples):
    best_by_chunk = {}
    for candidate in candidates:
        current = best_by_chunk.get(candidate["chunk"])
        if current is None or candidate["quality"] > current["quality"]:
            best_by_chunk[candidate["chunk"]] = candidate
    chosen = sorted(
        best_by_chunk.values(), key=lambda item: item["quality"], reverse=True
    )[:n_examples]
    if len(chosen) < n_examples:
        keys = {
            (item["chunk"], item["first"], item["second"]) for item in chosen
        }
        remaining = sorted(
            (
                item
                for item in candidates
                if (item["chunk"], item["first"], item["second"]) not in keys
            ),
            key=lambda item: item["quality"],
            reverse=True,
        )
        chosen.extend(remaining[: n_examples - len(chosen)])
    return chosen


def load_example(run, candidate):
    path = run / "chunks" / f"chunk_{candidate['chunk']:06d}.npz"
    required = {
        "alpha",
        "captured_fraction",
        "global_sources",
        "local_coords",
        "neighbor_ids",
        "profile_idx",
        "residual_pass",
        "residual_waveforms",
        "sources",
        "spike_times",
        "temporal_idx",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{path} is missing fields: {sorted(missing)}")
        rows = np.asarray([candidate["first"], candidate["second"]])
        result = {key: np.asarray(archive[key][rows]) for key in required}
    order = np.argsort(result["residual_pass"], kind="stable")
    return {key: value[order] for key, value in result.items()}


def geometry_waveforms(axis, coordinates, measured, predicted, source, color):
    scale = 13.0 / max(float(np.abs(measured).max()), EPS)
    time_offset = np.linspace(-16.0, 16.0, measured.shape[1])
    for channel in range(len(coordinates)):
        axis.plot(
            coordinates[channel, 0] + time_offset,
            coordinates[channel, 1] + scale * measured[channel],
            color="0.78",
            linewidth=0.8,
        )
        axis.plot(
            coordinates[channel, 0] + time_offset,
            coordinates[channel, 1] + scale * predicted[channel],
            color=color,
            linewidth=1.0,
            linestyle="--",
        )
    axis.scatter(
        coordinates[:, 0], coordinates[:, 1], s=8, marker="s", color="0.5"
    )
    axis.plot(
        source[0],
        source[1],
        marker="o",
        markerfacecolor="none",
        markeredgecolor=color,
        markersize=10,
        markeredgewidth=1.5,
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(color="white", alpha=0.07, linewidth=0.5)


def plot_collision_examples(run, metadata, examples, output):
    config = metadata["config"]
    fs = float(metadata["fs"])
    n_before = int(round(float(config["ms_before"]) * fs / 1000))
    omega = np.load(run / "omega.npy")
    channel_positions = np.load(run / "channel_positions.npy")
    n_examples = len(examples)

    plt.style.use("dark_background")
    figure, axes = plt.subplots(
        4,
        n_examples,
        figsize=(4.0 * n_examples, 12.2),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"height_ratios": (0.85, 1.1, 1.1, 0.85)},
    )
    for column, candidate in enumerate(examples):
        chunk = load_example(run, candidate)
        mask = chunk["neighbor_ids"] >= 0
        measured = chunk["residual_waveforms"] * mask[:, :, None]
        parameters = profile_parameters(
            config["kernel"], chunk["profile_idx"], int(config["n_scales"])
        )
        footprint, predicted = reconstruct(
            chunk["local_coords"],
            mask,
            chunk["sources"],
            parameters,
            omega,
            chunk["temporal_idx"],
            chunk["alpha"],
            config["kernel"],
        )
        after = measured - predicted
        rho = effective_width(
            chunk["sources"], chunk["profile_idx"], int(config["n_scales"])
        )
        union_ids = np.unique(chunk["neighbor_ids"][mask])
        local_contacts = channel_positions[union_ids]

        axis = axes[0, column]
        axis.scatter(
            local_contacts[:, 0],
            local_contacts[:, 1],
            s=30,
            marker="s",
            facecolors="none",
            edgecolors="0.75",
            linewidths=0.8,
        )
        axis.plot(
            chunk["global_sources"][:, 0],
            chunk["global_sources"][:, 1],
            color="0.45",
            linestyle=":",
            linewidth=1.0,
        )
        for component in range(2):
            axis.plot(
                chunk["global_sources"][component, 0],
                chunk["global_sources"][component, 1],
                marker="o",
                markerfacecolor="none",
                markeredgecolor=COMPONENT_COLORS[component],
                markersize=12,
                markeredgewidth=2,
                label=f"pass {int(chunk['residual_pass'][component]) + 1}",
            )
        axis.set(
            title=(
                f"case {column + 1} · chunk {candidate['chunk']} · "
                f"dt={candidate['delta_samples']} samples\n"
                f"localized separation {candidate['separation_um']:.1f} µm"
            ),
            xlabel="global lateral x (µm)",
            ylabel="global depth y (µm)",
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(color="white", alpha=0.08, linewidth=0.5)
        axis.legend(frameon=False, fontsize=7)

        for component in range(2):
            valid = mask[component]
            ids = chunk["neighbor_ids"][component, valid]
            coordinates = channel_positions[ids]
            axis = axes[component + 1, column]
            geometry_waveforms(
                axis,
                coordinates,
                measured[component, valid],
                predicted[component, valid],
                chunk["global_sources"][component],
                COMPONENT_COLORS[component],
            )
            axis.set_title(
                f"pass {int(chunk['residual_pass'][component]) + 1} input residual vs fit · "
                f"capture {100 * chunk['captured_fraction'][component]:.1f}%\n"
                f"temporal row {int(chunk['temporal_idx'][component])} · "
                f"ρ={rho[component]:.1f} µm",
                fontsize=8,
            )
            axis.set_xlabel("lateral x + waveform time offset")
            if column == 0:
                axis.set_ylabel("probe depth + voltage")

        axis = axes[3, column]
        reference_time = int(np.min(chunk["spike_times"]))
        for component in range(2):
            projected = np.einsum(
                "c,ct->t", footprint[component], measured[component]
            )
            projected_after = np.einsum(
                "c,ct->t", footprint[component], after[component]
            )
            model = chunk["alpha"][component] * omega[
                chunk["temporal_idx"][component]
            ]
            scale = max(
                float(np.max(np.abs(projected))),
                float(np.max(np.abs(model))),
                EPS,
            )
            baseline = 1.15 - 2.3 * component
            sample_axis = (
                int(chunk["spike_times"][component])
                + np.arange(len(projected))
                - n_before
                - reference_time
            )
            time_ms = 1000 * sample_axis / fs
            axis.plot(
                time_ms,
                baseline + projected / scale,
                color="0.88",
                linewidth=1.0,
            )
            axis.plot(
                time_ms,
                baseline + model / scale,
                color=COMPONENT_COLORS[component],
                linewidth=1.15,
                linestyle="--",
            )
            axis.plot(
                time_ms,
                baseline + projected_after / scale,
                color="#4c78a8",
                linewidth=0.9,
                alpha=0.9,
            )
            axis.axhline(baseline, color="0.4", linewidth=0.5)
            axis.text(
                0.01,
                0.91 - 0.48 * component,
                f"P{int(chunk['residual_pass'][component]) + 1}",
                color=COMPONENT_COLORS[component],
                transform=axis.transAxes,
                fontsize=8,
                va="top",
            )
        axis.set(
            title="projections · input / fit / after",
            xlabel="time from first detection (ms)",
            yticks=[],
        )
        if column == 0:
            axis.set_ylabel("normalized sequential residuals")
        axis.grid(color="white", alpha=0.07, linewidth=0.5)

    figure.suptitle(
        "Exact two-detection residual collisions · different passes, overlapping "
        "waveforms, shared electrodes",
        fontsize=14,
    )
    figure.savefig(output, dpi=800, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--max-points-per-pass", type=int, default=100_000)
    parser.add_argument("--collision-radius-um", type=float, default=80.0)
    parser.add_argument("--min-source-separation-um", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    metadata, chunk_paths = validate_run(args.run)
    config = metadata["config"]
    fs = float(metadata["fs"])
    waveform_length = int(
        round((float(config["ms_before"]) + float(config["ms_after"])) * fs / 1000)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    localization_output = args.out_dir / "residual_localizations_by_pass.png"
    collision_output = args.out_dir / "two_detection_collision_examples.png"
    selection_output = args.out_dir / "collision_selection.json"

    pass_counts = plot_localizations(
        args.run,
        metadata,
        localization_output,
        args.max_points_per_pass,
        args.seed,
    )
    candidates = []
    for path in chunk_paths:
        candidates.extend(
            collision_candidates(
                path,
                waveform_length,
                args.collision_radius_um,
                args.min_source_separation_um,
            )
        )
    examples = choose_examples(candidates, args.n_examples)
    if len(examples) < args.n_examples:
        raise RuntimeError(
            f"found only {len(examples)} exact collision pairs, requested {args.n_examples}"
        )
    plot_collision_examples(args.run, metadata, examples, collision_output)
    selection = {
        "run": str(args.run),
        "definition": {
            "different_residual_passes": True,
            "maximum_time_separation_samples": waveform_length - 1,
            "maximum_source_separation_um": args.collision_radius_um,
            "minimum_source_separation_um": args.min_source_separation_um,
            "requires_shared_electrode": True,
            "requires_degree_one_for_both_detections": True,
        },
        "n_candidates": len(candidates),
        "selected": examples,
    }
    selection_output.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"pass counts: {pass_counts}", flush=True)
    print(f"exact two-detection candidates: {len(candidates)}", flush=True)
    for index, example in enumerate(examples, start=1):
        print(
            f"case {index}: chunk={example['chunk']} rows="
            f"({example['first']}, {example['second']}) "
            f"dt={example['delta_samples']} samples "
            f"distance={example['separation_um']:.2f} um "
            f"passes={[value + 1 for value in example['passes']]}",
            flush=True,
        )
    print(f"localizations: {localization_output}", flush=True)
    print(f"collisions: {collision_output}", flush=True)
    print(f"selection: {selection_output}", flush=True)


if __name__ == "__main__":
    main()
