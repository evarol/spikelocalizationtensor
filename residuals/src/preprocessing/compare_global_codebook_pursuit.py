"""Compare frozen global-codebook pursuit runs across temporal dictionary sizes."""

import argparse
import json
from pathlib import Path

import numpy as np


def chunk_paths(run):
    paths = sorted((run / "chunks").glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no chunk checkpoints in {run}")
    return paths


def round_drops(archive):
    rounds = np.asarray(archive["residual_pass"], dtype=np.int64)
    drops = np.asarray(archive["pass_energy_drop_fraction"], dtype=np.float64)
    return np.asarray(
        [np.median(drops[rounds == value]) for value in np.unique(rounds)],
        dtype=np.float64,
    )


def codebook_diagnostics(omega):
    normalized = omega / np.linalg.norm(omega, axis=1, keepdims=True)
    cosine = np.abs(normalized @ normalized.T)
    off_diagonal = cosine[~np.eye(len(omega), dtype=bool)]
    nearest = np.max(cosine - np.eye(len(omega)), axis=1)
    return {
        "shape": list(omega.shape),
        "maximum_pairwise_absolute_cosine": float(np.max(off_diagonal)),
        "median_pairwise_absolute_cosine": float(np.median(off_diagonal)),
        "minimum_nearest_row_angle_degrees": float(
            np.degrees(np.arccos(np.clip(np.max(nearest), -1, 1)))
        ),
    }


def summarize_run(run):
    metadata = json.loads((run / "config.json").read_text())
    omega = np.load(run / "omega.npy")
    q = len(omega)
    fs = float(metadata["fs"])
    captured = []
    temporal = []
    times = []
    channels = []
    remaining = []
    rounds = []
    total_samples = 0
    for path in chunk_paths(run):
        chunk_index = int(path.stem.split("_")[-1])
        with np.load(path, allow_pickle=False) as archive:
            captured.append(np.asarray(archive["captured_fraction"], dtype=np.float64))
            temporal.append(np.asarray(archive["temporal_idx"], dtype=np.int64))
            times.append(np.asarray(archive["spike_times"], dtype=np.int64))
            channels.append(np.asarray(archive["spike_channels"], dtype=np.int32))
            drops = round_drops(archive)
            remaining.append(float(np.prod(1 - drops)))
            rounds.append(len(drops))
        chunk_samples = int(round(metadata["config"]["chunk_seconds"] * fs))
        core_start = int(metadata["first_sample"]) + chunk_index * chunk_samples
        core_stop = min(core_start + chunk_samples, int(metadata["stop_sample"]))
        total_samples += core_stop - core_start
    captured = np.concatenate(captured)
    temporal = np.concatenate(temporal)
    times = np.concatenate(times)
    channels = np.concatenate(channels)
    counts = np.bincount(temporal, minlength=q)
    fractions = counts / max(int(counts.sum()), 1)
    nonzero = fractions > 0
    entropy = -float(np.sum(fractions[nonzero] * np.log(fractions[nonzero])))
    runtime_path = run / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    return {
        "path": str(run.resolve()),
        "q": q,
        "n_events": int(len(times)),
        "duration_seconds": total_samples / fs,
        "events_per_second": len(times) / (total_samples / fs),
        "captured_fraction_mean": float(np.mean(captured)),
        "captured_fraction_median": float(np.median(captured)),
        "remaining_core_energy_fraction_mean": float(np.mean(remaining)),
        "rounds_completed_mean": float(np.mean(rounds)),
        "row_counts": counts.tolist(),
        "row_fractions": fractions.tolist(),
        "row_usage_entropy": entropy,
        "row_usage_effective_count": float(np.exp(entropy)),
        "runtime": runtime,
        "codebook": codebook_diagnostics(omega),
        "times": times,
        "channels": channels,
    }


def match_sorted_times(left, right, tolerance):
    left = np.sort(np.asarray(left, dtype=np.int64))
    right = np.sort(np.asarray(right, dtype=np.int64))
    i = 0
    j = 0
    matches = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j] - tolerance:
            i += 1
        elif right[j] < left[i] - tolerance:
            j += 1
        else:
            matches += 1
            i += 1
            j += 1
    return matches


def overlap(reference, candidate, tolerance):
    time_matches = match_sorted_times(reference["times"], candidate["times"], tolerance)
    anchor_matches = 0
    for channel in np.intersect1d(reference["channels"], candidate["channels"]):
        anchor_matches += match_sorted_times(
            reference["times"][reference["channels"] == channel],
            candidate["times"][candidate["channels"] == channel],
            tolerance,
        )

    def metrics(matches):
        union = len(reference["times"]) + len(candidate["times"]) - matches
        return {
            "matches": int(matches),
            "reference_fraction": matches / len(reference["times"]),
            "candidate_fraction": matches / len(candidate["times"]),
            "jaccard": matches / union,
        }

    return {
        "tolerance_samples": tolerance,
        "time_only": metrics(time_matches),
        "same_anchor_channel": metrics(anchor_matches),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference-q", type=int, default=8)
    parser.add_argument("--tolerance-samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = {item["q"]: item for item in map(summarize_run, args.runs)}
    if len(summaries) != len(args.runs):
        raise ValueError("run list contains duplicate Q values")
    if args.reference_q not in summaries:
        raise ValueError(f"missing reference Q={args.reference_q} run")
    reference = summaries[args.reference_q]
    overlaps = {
        str(q): overlap(reference, candidate, args.tolerance_samples)
        for q, candidate in summaries.items()
        if q != args.reference_q
    }
    for summary in summaries.values():
        summary.pop("times")
        summary.pop("channels")
    result = {
        "reference_q": args.reference_q,
        "runs": {str(q): summaries[q] for q in sorted(summaries)},
        "overlap_vs_reference": overlaps,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
