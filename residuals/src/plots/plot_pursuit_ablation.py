import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


COLORS = {
    "frozen": "#4c78a8",
    "learned": "#e45756",
    "learned_stopped": "#72b7b2",
}
LABELS = {
    "frozen": "frozen initial",
    "learned": "learned, 60 rounds",
    "learned_stopped": "learned, energy stop",
}


def _chunk_metrics(run_path, chunk_indices):
    metrics = []
    for chunk_index in chunk_indices:
        path = run_path / "chunks" / f"chunk_{chunk_index:06d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            rounds = np.asarray(archive["residual_pass"], dtype=np.int64)
            drops = np.asarray(
                archive["pass_energy_drop_fraction"], dtype=np.float64
            )
            captured = np.asarray(archive["captured_fraction"], dtype=np.float64)
        values = []
        remaining = 1.0
        for residual_round in np.unique(rounds):
            keep = rounds == residual_round
            drop = float(np.median(drops[keep]))
            remaining *= 1 - drop
            values.append(
                {
                    "round": int(residual_round) + 1,
                    "count": int(keep.sum()),
                    "energy_drop": drop,
                    "remaining_energy": remaining,
                    "captured_fraction": float(np.median(captured[keep])),
                }
            )
        metrics.append(values)
    return metrics


def _trajectory(chunk_metrics, key):
    rounds = sorted({item["round"] for chunk in chunk_metrics for item in chunk})
    centers = []
    lower = []
    upper = []
    for residual_round in rounds:
        values = [
            item[key]
            for chunk in chunk_metrics
            for item in chunk
            if item["round"] == residual_round
        ]
        centers.append(float(np.mean(values)))
        lower.append(float(np.min(values)))
        upper.append(float(np.max(values)))
    return np.asarray(rounds), np.asarray(centers), np.asarray(lower), np.asarray(upper)


def plot_codebook(learned_path, output_path):
    metadata = json.loads((learned_path / "config.json").read_text())
    config = metadata["config"]
    fs = float(metadata["fs"])
    initial = np.load(learned_path / "omega_initial.npy")
    learned = np.load(learned_path / "omega_learned.npy")
    n_before = int(round(float(config["ms_before"]) * fs / 1000))
    time_ms = 1000 * (np.arange(initial.shape[1]) - n_before) / fs
    cosine = np.einsum("qt,qt->q", initial, learned) / (
        np.linalg.norm(initial, axis=1) * np.linalg.norm(learned, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(np.abs(cosine), -1, 1)))

    figure, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True, constrained_layout=True)
    for row, axis in enumerate(axes.flat):
        axis.plot(time_ms, initial[row], color="0.45", linewidth=1.4, label="initial")
        axis.plot(
            time_ms,
            learned[row],
            color=COLORS["learned"],
            linewidth=1.4,
            label="learned",
        )
        axis.fill_between(
            time_ms,
            initial[row],
            learned[row],
            color=COLORS["learned"],
            alpha=0.16,
            linewidth=0,
        )
        axis.axhline(0, color="0.8", linewidth=0.6)
        axis.set_title(f"temporal row {row} · {angles[row]:.2f}° movement")
        axis.set_ylabel("normalized amplitude")
    for axis in axes[-1]:
        axis.set_xlabel("time from detection (ms)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Q8 temporal codebook update")
    figure.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close(figure)


def plot_trajectories(run_paths, heldout_indices, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    specifications = (
        ("count", "accepted events", "accepted core events"),
        ("energy_drop", "energy removed per round", "core-energy fraction"),
        ("remaining_energy", "cumulative residual energy", "fraction remaining"),
        ("captured_fraction", "median captured fraction", "local-energy fraction"),
    )
    for name, run_path in run_paths.items():
        metrics = _chunk_metrics(run_path, heldout_indices)
        for axis, (key, title, ylabel) in zip(axes.flat, specifications):
            rounds, center, lower, upper = _trajectory(metrics, key)
            axis.plot(
                rounds,
                center,
                color=COLORS[name],
                linewidth=1.8,
                label=LABELS[name],
            )
            axis.fill_between(
                rounds, lower, upper, color=COLORS[name], alpha=0.13, linewidth=0
            )
            axis.set(title=title, xlabel="pursuit round", ylabel=ylabel)
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(0.005, color="0.25", linestyle="--", linewidth=0.9)
    axes[0, 1].text(60, 0.0052, "0.5% stop", ha="right", va="bottom", fontsize=8)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Held-out pursuit trajectories · shaded range across held-out chunks"
    )
    figure.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close(figure)


def plot_summary(comparison, output_path):
    names = ["frozen", "learned", "learned_stopped"]
    summaries = [comparison["runs"][name]["heldout"] for name in names]
    labels = [LABELS[name] for name in names]
    colors = [COLORS[name] for name in names]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    values = [item["events_per_second"] for item in summaries]
    bars = axes[0, 0].bar(labels, values, color=colors)
    axes[0, 0].bar_label(bars, fmt="%.0f", padding=3)
    axes[0, 0].set(title="held-out event rate", ylabel="events / second")

    values = [item["remaining_core_energy_fraction_mean"] for item in summaries]
    bars = axes[0, 1].bar(labels, values, color=colors)
    axes[0, 1].bar_label(bars, fmt="%.3f", padding=3)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].set(title="residual core energy", ylabel="fraction remaining", ylim=(0, 1))

    x = np.arange(len(names))
    width = 0.36
    means = [item["captured_fraction_mean"] for item in summaries]
    medians = [item["captured_fraction_median"] for item in summaries]
    axes[1, 0].bar(x - width / 2, means, width, color=colors, alpha=0.55, label="mean")
    axes[1, 0].bar(x + width / 2, medians, width, color=colors, label="median")
    axes[1, 0].set(
        title="held-out captured fraction",
        ylabel="local-energy fraction",
        xticks=x,
        xticklabels=labels,
        ylim=(0, max(means + medians) * 1.2),
    )
    axes[1, 0].legend(frameon=False)

    pair_keys = [
        "frozen_vs_learned",
        "learned_vs_learned_stopped",
        "frozen_vs_learned_stopped",
    ]
    pair_labels = ["frozen ↔ learned", "learned ↔ stopped", "frozen ↔ stopped"]
    overlap = comparison["event_overlap"]
    time_jaccard = [overlap[key]["heldout"]["time_only"]["jaccard"] for key in pair_keys]
    anchor_jaccard = [
        overlap[key]["heldout"]["same_anchor_channel"]["jaccard"]
        for key in pair_keys
    ]
    x = np.arange(len(pair_keys))
    axes[1, 1].bar(x - width / 2, time_jaccard, width, color="#f2cf5b", label="time ±3 samples")
    axes[1, 1].bar(x + width / 2, anchor_jaccard, width, color="#8c6bb1", label="time + same anchor")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].set(
        title="held-out one-to-one event overlap",
        ylabel="Jaccard overlap",
        xticks=x,
        xticklabels=pair_labels,
        ylim=(0, 1),
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=12)
    figure.suptitle("Frozen vs learned temporal-codebook ablation")
    figure.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--stopped", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_paths = {
        "frozen": args.frozen,
        "learned": args.learned,
        "learned_stopped": args.stopped,
    }
    outputs = {
        "codebook": args.out_dir / "temporal_codebook_update.png",
        "trajectories": args.out_dir / "heldout_pursuit_trajectories.png",
        "summary": args.out_dir / "heldout_ablation_summary.png",
    }
    plot_codebook(args.learned, outputs["codebook"])
    plot_trajectories(
        run_paths, comparison["heldout_chunk_indices"], outputs["trajectories"]
    )
    plot_summary(comparison, outputs["summary"])
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
