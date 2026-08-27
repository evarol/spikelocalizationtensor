import argparse
import json
from pathlib import Path

import numpy as np


def _chunk_index(path):
    return int(path.stem.split("_")[-1])


def _chunk_paths(run_path):
    paths = sorted((run_path / "chunks").glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no chunk checkpoints found in {run_path}")
    return {_chunk_index(path): path for path in paths}


def _load_config(run_path):
    return json.loads((run_path / "config.json").read_text())


def _round_drops(archive):
    rounds = np.asarray(archive["residual_pass"], dtype=np.int64)
    drops = np.asarray(archive["pass_energy_drop_fraction"], dtype=np.float64)
    return np.asarray(
        [np.median(drops[rounds == value]) for value in np.unique(rounds)],
        dtype=np.float64,
    )


def summarize_run(run_path, chunk_indices):
    paths = _chunk_paths(run_path)
    config = _load_config(run_path)
    fs = float(config["fs"])
    chunk_samples = int(round(config["config"]["chunk_seconds"] * fs))
    first_sample = int(config["first_sample"])
    stop_sample = int(config["stop_sample"])
    captured = []
    chunk_metrics = []
    total_events = 0
    total_samples = 0
    for index in chunk_indices:
        path = paths[index]
        with np.load(path) as archive:
            n_events = int(len(archive["spike_times"]))
            fractions = np.asarray(archive["captured_fraction"], dtype=np.float64)
            drops = _round_drops(archive)
            rounds = int(len(drops))
        core_start = first_sample + index * chunk_samples
        core_stop = min(core_start + chunk_samples, stop_sample)
        total_samples += core_stop - core_start
        total_events += n_events
        captured.append(fractions)
        chunk_metrics.append(
            {
                "chunk_index": index,
                "n_events": n_events,
                "rounds_completed": rounds,
                "remaining_core_energy_fraction": float(np.prod(1 - drops)),
            }
        )
    captured = np.concatenate(captured) if captured else np.empty(0)
    remaining = [item["remaining_core_energy_fraction"] for item in chunk_metrics]
    runtime_path = run_path / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    return {
        "n_events": total_events,
        "duration_seconds": total_samples / fs,
        "events_per_second": total_events / (total_samples / fs),
        "captured_fraction_mean": float(np.mean(captured)),
        "captured_fraction_median": float(np.median(captured)),
        "remaining_core_energy_fraction_mean": float(np.mean(remaining)),
        "rounds_completed_mean": float(
            np.mean([item["rounds_completed"] for item in chunk_metrics])
        ),
        "runtime": runtime,
        "chunks": chunk_metrics,
    }


def _event_arrays(run_path, chunk_indices):
    paths = _chunk_paths(run_path)
    times = []
    channels = []
    for index in chunk_indices:
        with np.load(paths[index]) as archive:
            times.append(np.asarray(archive["spike_times"], dtype=np.int64))
            channels.append(np.asarray(archive["spike_channels"], dtype=np.int32))
    return np.concatenate(times), np.concatenate(channels)


def _match_sorted_times(left, right, tolerance_samples):
    left = np.sort(np.asarray(left, dtype=np.int64))
    right = np.sort(np.asarray(right, dtype=np.int64))
    i = 0
    j = 0
    matches = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j] - tolerance_samples:
            i += 1
        elif right[j] < left[i] - tolerance_samples:
            j += 1
        else:
            matches += 1
            i += 1
            j += 1
    return matches


def event_overlap(left_path, right_path, chunk_indices, tolerance_samples):
    left_times, left_channels = _event_arrays(left_path, chunk_indices)
    right_times, right_channels = _event_arrays(right_path, chunk_indices)
    time_matches = _match_sorted_times(
        left_times, right_times, tolerance_samples
    )
    anchor_matches = 0
    for channel in np.intersect1d(left_channels, right_channels):
        anchor_matches += _match_sorted_times(
            left_times[left_channels == channel],
            right_times[right_channels == channel],
            tolerance_samples,
        )

    def metrics(matches):
        union = len(left_times) + len(right_times) - matches
        return {
            "matches": int(matches),
            "left_fraction": matches / len(left_times),
            "right_fraction": matches / len(right_times),
            "jaccard": matches / union,
        }

    return {
        "tolerance_samples": tolerance_samples,
        "time_only": metrics(time_matches),
        "same_anchor_channel": metrics(anchor_matches),
    }


def codebook_change(learned_path):
    initial = np.load(learned_path / "omega_initial.npy")
    learned = np.load(learned_path / "omega_learned.npy")
    cosine = np.einsum("qt,qt->q", initial, learned) / (
        np.linalg.norm(initial, axis=1) * np.linalg.norm(learned, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(np.abs(cosine), -1, 1)))
    return {
        "row_l2_change": np.linalg.norm(learned - initial, axis=1).tolist(),
        "row_angle_degrees": angles.tolist(),
        "global_l2_change": float(np.linalg.norm(learned - initial)),
    }


def compare_runs(
    frozen_path,
    learned_path,
    stopped_path,
    tolerance_samples=3,
    expected_chunk_samples=None,
):
    paths = {
        "frozen": Path(frozen_path),
        "learned": Path(learned_path),
        "learned_stopped": Path(stopped_path),
    }
    indices = [set(_chunk_paths(path)) for path in paths.values()]
    if not indices[0] == indices[1] == indices[2]:
        raise ValueError("ablation runs do not contain identical chunk indices")
    all_indices = sorted(indices[0])
    history = json.loads(
        (paths["learned"] / "codebook_learning_history.json").read_text()
    )
    learning_indices = sorted({int(item["chunk_index"]) for item in history})
    heldout_indices = sorted(set(all_indices) - set(learning_indices))
    if not heldout_indices:
        raise ValueError("the learned run has no held-out chunks")
    configs = {name: _load_config(path) for name, path in paths.items()}
    chunk_samples = {
        name: int(round(value["config"]["chunk_seconds"] * value["fs"]))
        for name, value in configs.items()
    }
    if len(set(chunk_samples.values())) != 1:
        raise ValueError(f"ablation chunk sizes differ: {chunk_samples}")
    if (
        expected_chunk_samples is not None
        and next(iter(chunk_samples.values())) != expected_chunk_samples
    ):
        raise ValueError(
            f"expected {expected_chunk_samples} samples per chunk, got "
            f"{chunk_samples}"
        )
    if not np.array_equal(
        np.load(paths["frozen"] / "omega.npy"),
        np.load(paths["learned"] / "omega_initial.npy"),
    ):
        raise ValueError("frozen codebook differs from learned-run initialization")
    if not np.allclose(
        np.load(paths["learned_stopped"] / "omega.npy"),
        np.load(paths["learned"] / "omega_learned.npy"),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("stopped run did not use the learned frozen codebook")

    scopes = {"all": all_indices, "heldout": heldout_indices}
    summaries = {
        name: {
            scope: summarize_run(path, selected)
            for scope, selected in scopes.items()
        }
        for name, path in paths.items()
    }
    pair_names = (
        ("frozen_vs_learned", "frozen", "learned"),
        ("learned_vs_learned_stopped", "learned", "learned_stopped"),
        ("frozen_vs_learned_stopped", "frozen", "learned_stopped"),
    )
    overlaps = {
        label: {
            scope: event_overlap(
                paths[left], paths[right], selected, tolerance_samples
            )
            for scope, selected in scopes.items()
        }
        for label, left, right in pair_names
    }
    return {
        "chunk_samples": next(iter(chunk_samples.values())),
        "all_chunk_indices": all_indices,
        "learning_chunk_indices": learning_indices,
        "heldout_chunk_indices": heldout_indices,
        "runs": summaries,
        "event_overlap": overlaps,
        "codebook_change": codebook_change(paths["learned"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_path", type=Path)
    parser.add_argument("learned_path", type=Path)
    parser.add_argument("stopped_path", type=Path)
    parser.add_argument("--tolerance-samples", type=int, default=3)
    parser.add_argument("--expected-chunk-samples", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_runs(
        args.frozen_path,
        args.learned_path,
        args.stopped_path,
        tolerance_samples=args.tolerance_samples,
        expected_chunk_samples=args.expected_chunk_samples,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
