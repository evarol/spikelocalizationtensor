"""Plot the fresh-raw global-codebook pursuit experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np


Q_COLORS = {
    8: "#4c78a8",
    16: "#59a14f",
    24: "#f28e2b",
    32: "#e15759",
}


def load_json(path):
    return json.loads(path.read_text())


def plot_codebooks(comparison, codebook_summary, output):
    q_values = sorted(int(value) for value in comparison["runs"])
    reference_run = Path(comparison["runs"][str(q_values[0])]["path"])
    metadata = load_json(reference_run / "config.json")
    fs = float(metadata["fs"])
    n_before = int(round(metadata["config"]["ms_before"] * fs / 1000))
    banks = {}
    for q in q_values:
        path = Path(codebook_summary[str(q)]["result_path"])
        with np.load(path, allow_pickle=False) as archive:
            banks[q] = np.asarray(archive["omega"], dtype=np.float32)
    waveform_length = next(iter(banks.values())).shape[1]
    time_ms = 1000 * (np.arange(waveform_length) - n_before) / fs
    limit = float(
        np.quantile(
            np.abs(np.concatenate([bank.ravel() for bank in banks.values()])),
            0.995,
        )
    )

    figure, axes = plt.subplots(
        len(q_values),
        1,
        figsize=(12.5, 10.5),
        constrained_layout=True,
        sharex=True,
    )
    artist = None
    for axis, q in zip(axes, q_values):
        bank = banks[q]
        artist = axis.imshow(
            bank,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(time_ms[0], time_ms[-1], q - 0.5, -0.5),
        )
        nmse = float(codebook_summary[str(q)]["nmse"])
        minimum_count = min(codebook_summary[str(q)]["temporal_count"])
        axis.set(
            title=(
                f"Q={q} global temporal bank · training nMSE={nmse:.4f} · "
                f"least-used row={minimum_count:,} assignments"
            ),
            ylabel="row",
            yticks=np.linspace(0, q - 1, min(q, 5), dtype=int),
        )
        axis.axvline(0, color="black", linewidth=0.6, alpha=0.55)
    axes[-1].set_xlabel("time from detection (ms)")
    colorbar = figure.colorbar(artist, ax=axes, pad=0.015, fraction=0.025)
    colorbar.set_label("normalized temporal amplitude")
    figure.suptitle("Global codebooks learned from fresh raw-recording detections")
    figure.savefig(output, dpi=800, bbox_inches="tight")
    plt.close(figure)


def plot_q_ablation(comparison, codebook_summary, thresholds, output):
    q_values = np.asarray(sorted(int(value) for value in comparison["runs"]))
    runs = [comparison["runs"][str(q)] for q in q_values]
    colors = [Q_COLORS[int(q)] for q in q_values]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)

    refined_nmse = [codebook_summary[str(q)]["nmse"] for q in q_values]
    coarse_nmse = [codebook_summary[str(q)]["nmse_coarse"] for q in q_values]
    axes[0, 0].plot(q_values, coarse_nmse, "o--", color="0.55", label="coarse")
    axes[0, 0].plot(q_values, refined_nmse, "o-", color="#4c78a8", label="refined")
    axes[0, 0].set(title="fresh-raw codebook fit", ylabel="training nMSE")
    axes[0, 0].legend(frameon=False)

    means = [run["captured_fraction_mean"] for run in runs]
    medians = [run["captured_fraction_median"] for run in runs]
    axes[0, 1].plot(q_values, means, "o-", color="#e15759", label="mean")
    axes[0, 1].plot(q_values, medians, "o--", color="#f28e2b", label="median")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].set(title="accepted-event reconstruction", ylabel="captured fraction")
    axes[0, 1].legend(frameon=False)

    remaining = [run["remaining_core_energy_fraction_mean"] for run in runs]
    bars = axes[0, 2].bar(q_values, remaining, width=5.0, color=colors)
    axes[0, 2].bar_label(bars, labels=[f"{value:.3f}" for value in remaining])
    axes[0, 2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 2].set(
        title="residual after 60 pursuit rounds",
        ylabel="core-energy fraction remaining",
        ylim=(0.65, 0.73),
    )

    rates = [run["events_per_second"] for run in runs]
    bars = axes[1, 0].bar(q_values, rates, width=5.0, color=colors)
    labels = [
        f"{rate:,.0f}\nth={thresholds['results'][str(q)]['matched_threshold']:.2f}"
        for q, rate in zip(q_values, rates)
    ]
    axes[1, 0].bar_label(bars, labels=labels, padding=3, fontsize=8)
    axes[1, 0].set(
        title="matched-threshold event rate",
        ylabel="events / second",
        ylim=(min(rates) - 120, max(rates) + 150),
    )

    runtimes = [run["runtime"]["elapsed_seconds"] for run in runs]
    bars = axes[1, 1].bar(q_values, runtimes, width=5.0, color=colors)
    axes[1, 1].bar_label(bars, labels=[f"{value:.0f}s" for value in runtimes])
    axes[1, 1].set(title="pursuit runtime for 8.748 s", ylabel="wall time (seconds)")

    compared_q = q_values[1:]
    time_overlap = [
        comparison["overlap_vs_reference"][str(q)]["time_only"]["reference_fraction"]
        for q in compared_q
    ]
    anchor_overlap = [
        comparison["overlap_vs_reference"][str(q)]["same_anchor_channel"][
            "reference_fraction"
        ]
        for q in compared_q
    ]
    axes[1, 2].plot(compared_q, time_overlap, "o-", color="#4c78a8", label="time")
    axes[1, 2].plot(
        compared_q,
        anchor_overlap,
        "o-",
        color="#b279a2",
        label="time + same anchor",
    )
    axes[1, 2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 2].set(
        title="one-to-one overlap with Q8",
        ylabel="fraction of Q8 events matched",
        ylim=(0.3, 0.95),
    )
    axes[1, 2].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xlabel("temporal codebook size Q")
        axis.set_xticks(q_values)
        axis.grid(alpha=0.2, linewidth=0.6)
    figure.suptitle("Fresh-raw global-codebook pursuit ablation")
    figure.savefig(output, dpi=800, bbox_inches="tight")
    plt.close(figure)


def plot_superbatch(superbatch, output):
    batch_sizes = np.asarray(sorted(int(value) for value in superbatch["results"]))
    results = [superbatch["results"][str(value)] for value in batch_sizes]
    rates = np.asarray([result["events_per_second"] for result in results])
    memory_gib = np.asarray([result["peak_allocated_bytes"] / 2**30 for result in results])
    source_difference = np.asarray(
        [
            result["comparison_to_reference"]["source_max_absolute_difference_um"]
            for result in results
        ]
    )
    row_agreement = np.asarray(
        [
            result["comparison_to_reference"]["temporal_row_agreement"]
            for result in results
        ]
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    axes[0].plot(batch_sizes, rates, "o-", color="#4c78a8", linewidth=2)
    for batch, rate in zip(batch_sizes, rates):
        axes[0].annotate(f"{rate:,.0f}/s", (batch, rate), xytext=(0, 7), textcoords="offset points", ha="center")
    axes[0].set(title="localization throughput", ylabel="events / second")

    axes[1].plot(batch_sizes, memory_gib, "o-", color="#f28e2b", linewidth=2)
    for batch, value in zip(batch_sizes, memory_gib):
        axes[1].annotate(f"{value:.1f} GiB", (batch, value), xytext=(0, 7), textcoords="offset points", ha="center")
    axes[1].set(title="peak allocated GPU memory", ylabel="GiB")

    axes[2].plot(batch_sizes, source_difference, "o-", color="#e15759", linewidth=2)
    axes[2].set_yscale("symlog", linthresh=0.1)
    axes[2].set(title="worst source-coordinate change", ylabel="maximum |Δsource| (µm)")
    for batch, difference, agreement in zip(batch_sizes, source_difference, row_agreement):
        axes[2].annotate(
            f"{difference:.3g} µm\nrows {100 * agreement:.0f}%",
            (batch, difference),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    for axis in axes:
        axis.set_xlabel("localization batch size")
        axis.set_xticks(batch_sizes)
        axis.grid(alpha=0.2, linewidth=0.6)
    figure.suptitle(
        f"Q32 localization superbatch benchmark · {superbatch['n_events']:,} events"
    )
    figure.savefig(output, dpi=800, bbox_inches="tight")
    plt.close(figure)


def chunk_round_metrics(run):
    chunk_paths = sorted((run / "chunks").glob("chunk_*.npz"))
    count_parts = []
    captured_parts = []
    drop_parts = []
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as archive:
            rounds = np.asarray(archive["residual_pass"], dtype=np.int64)
            captured = np.asarray(archive["captured_fraction"], dtype=np.float64)
            drops = np.asarray(archive["pass_energy_drop_fraction"], dtype=np.float64)
        unique_rounds = np.unique(rounds)
        count_parts.append([np.count_nonzero(rounds == value) for value in unique_rounds])
        captured_parts.append(
            [np.median(captured[rounds == value]) for value in unique_rounds]
        )
        drop_parts.append([np.median(drops[rounds == value]) for value in unique_rounds])
    return (
        np.asarray(count_parts),
        np.asarray(captured_parts),
        np.asarray(drop_parts),
    )


def densest_local_window(run, fs, duration_ms=3.0, depth_um=100.0):
    times = np.asarray(np.load(run / "spike_times.npy", mmap_mode="r"), dtype=np.int64)
    sources = np.asarray(np.load(run / "global_sources.npy", mmap_mode="r"))
    rounds = np.asarray(np.load(run / "residual_pass.npy", mmap_mode="r"), dtype=np.int64)
    width = int(round(duration_ms * fs / 1000))
    first_time = int(times.min())
    first_depth = depth_um * np.floor(float(sources[:, 1].min()) / depth_um)
    time_bins = (times - first_time) // width
    depth_bins = np.floor((sources[:, 1] - first_depth) / depth_um).astype(np.int64)
    order = np.lexsort((depth_bins, time_bins))
    ordered_keys = np.column_stack((time_bins[order], depth_bins[order]))
    boundaries = np.flatnonzero(
        np.r_[True, np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1), True]
    )
    best = None
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        rows = order[start:stop]
        score = (len(np.unique(rounds[rows])), len(rows))
        if best is None or score > best[0]:
            best = (score, rows)
    rows = best[1]
    time_start = first_time + int(time_bins[rows[0]]) * width
    return times[rows], sources[rows], rounds[rows], time_start, depth_um


def plot_pursuit_dynamics(run, output):
    metadata = load_json(run / "config.json")
    fs = float(metadata["fs"])
    counts, captured, drops = chunk_round_metrics(run)
    rounds = np.arange(1, counts.shape[1] + 1)
    remaining = np.cumprod(1 - drops, axis=1)
    dense_times, dense_sources, dense_rounds, time_start, depth_um = (
        densest_local_window(run, fs)
    )
    relative_ms = 1000 * (dense_times - time_start) / fs

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    metrics = (
        (counts, axes[0, 0], "accepted events per chunk", "events"),
        (drops, axes[0, 1], "core energy removed per round", "fraction removed"),
        (remaining, axes[1, 0], "cumulative residual core energy", "fraction remaining"),
    )
    for values, axis, title, ylabel in metrics:
        axis.fill_between(
            rounds,
            values.min(axis=0),
            values.max(axis=0),
            color="#4c78a8",
            alpha=0.18,
            linewidth=0,
            label="chunk range",
        )
        axis.plot(rounds, values.mean(axis=0), color="#4c78a8", linewidth=1.8, label="mean")
        axis.set(title=title, xlabel="pursuit round", ylabel=ylabel)
        axis.grid(alpha=0.2, linewidth=0.6)
    axes[0, 1].set_yscale("log")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].legend(frameon=False)

    artist = axes[1, 1].scatter(
        relative_ms,
        dense_sources[:, 1],
        c=dense_rounds + 1,
        cmap="turbo",
        norm=Normalize(1, counts.shape[1]),
        s=13,
        linewidths=0,
        alpha=0.8,
        rasterized=True,
    )
    axes[1, 1].set(
        title=(
            f"local 3 ms × {depth_um:.0f} µm bin · {len(dense_times)} events "
            f"across {len(np.unique(dense_rounds))} rounds"
        ),
        xlabel="time within window (ms)",
        ylabel="localized probe depth (µm)",
    )
    axes[1, 1].grid(alpha=0.15, linewidth=0.5)
    colorbar = figure.colorbar(artist, ax=axes[1, 1], pad=0.02)
    colorbar.set_label("pursuit round")
    figure.suptitle("Q32 greedy residual-pursuit dynamics")
    figure.savefig(output, dpi=800, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--codebook-summary", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--superbatch", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    comparison = load_json(args.comparison)
    codebook_summary = load_json(args.codebook_summary)
    thresholds = load_json(args.thresholds)
    superbatch = load_json(args.superbatch)
    q32_run = Path(comparison["runs"]["32"]["path"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "codebooks": args.out_dir / "learned_global_codebooks.png",
        "q_ablation": args.out_dir / "global_codebook_q_ablation.png",
        "superbatch": args.out_dir / "localization_superbatch.png",
        "pursuit": args.out_dir / "q32_pursuit_dynamics.png",
    }
    plot_codebooks(comparison, codebook_summary, outputs["codebooks"])
    plot_q_ablation(comparison, codebook_summary, thresholds, outputs["q_ablation"])
    plot_superbatch(superbatch, outputs["superbatch"])
    plot_pursuit_dynamics(q32_run, outputs["pursuit"])
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
