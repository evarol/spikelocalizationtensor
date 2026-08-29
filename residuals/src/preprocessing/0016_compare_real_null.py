"""Compare full-recording 0016 detections with their matched empirical null."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def validate_pair(real, null):
    real_metadata = read_json(real / "metadata.json")
    null_metadata = read_json(null / "metadata.json")
    if real_metadata["config"].get("empirical_null"):
        raise ValueError(f"{real} is marked as an empirical null")
    if not null_metadata["config"].get("empirical_null"):
        raise ValueError(f"{null} is not marked as an empirical null")
    keys = ("fs", "n_channels", "first_sample", "stop_sample")
    mismatch = {
        key: (real_metadata[key], null_metadata[key])
        for key in keys
        if real_metadata[key] != null_metadata[key]
    }
    if mismatch:
        raise ValueError(f"real/null recording coverage differs: {mismatch}")
    real_omega = np.load(real / "omega.npy", mmap_mode="r")
    null_omega = np.load(null / "omega.npy", mmap_mode="r")
    if real_omega.shape != null_omega.shape or not np.allclose(real_omega, null_omega):
        raise ValueError("real and null runs did not use the same Omega")
    ignored = {
        "empirical_null",
        "omega_prior",
        "null_shift_min_ms",
        "null_shift_max_ms",
        "null_seed",
        "save_waveforms",
    }
    real_config = {
        key: value for key, value in real_metadata["config"].items() if key not in ignored
    }
    null_config = {
        key: value for key, value in null_metadata["config"].items() if key not in ignored
    }
    if real_config != null_config:
        changed = {
            key: (real_config.get(key), null_config.get(key))
            for key in sorted(set(real_config) | set(null_config))
            if real_config.get(key) != null_config.get(key)
        }
        raise ValueError(f"real/null model configurations differ: {changed}")
    return real_metadata


def counts_by_round(path, score, threshold):
    rounds = np.load(path / "peeling_round.npy", mmap_mode="r")
    selected = score >= threshold
    values, counts = np.unique(rounds[selected], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def compare(real, null, thresholds):
    metadata = validate_pair(real, null)
    real_score = np.load(real / "fitted_projection_score.npy", mmap_mode="r")
    null_score = np.load(null / "fitted_projection_score.npy", mmap_mode="r")
    rows = []
    for threshold in thresholds:
        real_count = int(np.count_nonzero(real_score >= threshold))
        null_count = int(np.count_nonzero(null_score >= threshold))
        rows.append(
            {
                "fitted_projection_threshold": float(threshold),
                "real_events": real_count,
                "null_events": null_count,
                "null_to_real_ratio": (
                    float(null_count / real_count) if real_count else None
                ),
                "real_excess_over_null": max(0, real_count - null_count),
                "real_by_peeling_round": counts_by_round(real, real_score, threshold),
                "null_by_peeling_round": counts_by_round(null, null_score, threshold),
            }
        )
    duration_seconds = (
        metadata["stop_sample"] - metadata["first_sample"]
    ) / metadata["fs"]
    return {
        "real_run": str(real.resolve()),
        "null_run": str(null.resolve()),
        "duration_seconds": duration_seconds,
        "interpretation": (
            "The null count estimates detections explainable without coherent cross-channel "
            "spike spread under the identical maximized bank, fitting, merging, and peeling path."
        ),
        "thresholds": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("real_run", type=Path)
    parser.add_argument("null_run", type=Path)
    parser.add_argument("--thresholds", default="8,9,10,11,12,14,16,20,24,32,40")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    report = compare(args.real_run, args.null_run, thresholds)
    output = args.output or args.real_run / "empirical_null_comparison.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
