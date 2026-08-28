"""Deterministic session-0013 rho-localizer fixture and timing harness.

This launcher compares an explicit-identity reference with a candidate
identity fast path on fixed inputs from a saved pursuit chunk.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maths_0010 import localize_spikes_fixed_codebook


OUTPUT_FIELDS = (
    "sources",
    "sources_grid",
    "profile_idx",
    "sigma",
    "rho",
    "temporal_idx",
    "alpha",
    "captured_energy",
    "prediction",
    "continuous_displacement_um",
    "continuous_energy_gain",
)


def load_fixture(run, chunk, events):
    """Load a deterministic fit batch and its frozen codebook."""
    run = Path(run)
    chunk_path = run / "chunks" / f"chunk_{chunk:06d}.npz"
    with np.load(chunk_path, allow_pickle=False) as archive:
        required = ("local_coords", "neighbor_ids", "residual_waveforms")
        missing = set(required).difference(archive.files)
        if missing:
            raise KeyError(f"{chunk_path} is missing {sorted(missing)}")
        count = min(events, len(archive["local_coords"]))
        if count != events:
            raise ValueError(f"{chunk_path} contains {count}, not {events}, events")
        coords = np.asarray(archive["local_coords"][:count], dtype=np.float32)
        waveforms = np.asarray(archive["residual_waveforms"][:count], dtype=np.float32)
        mask = np.asarray(archive["neighbor_ids"][:count] >= 0, dtype=bool)
    omega = np.load(run / "omega.npy").astype(np.float32, copy=False)
    metadata = json.loads((run / "config.json").read_text())
    return coords, waveforms, mask, omega, metadata["config"], chunk_path


def identity_transforms(events, channels):
    return np.broadcast_to(
        np.eye(channels, dtype=np.float32), (events, channels, channels)
    ).copy()


def localize(coords, waveforms, mask, omega, config, identity_fast_path):
    transforms = None if identity_fast_path else identity_transforms(
        len(waveforms), waveforms.shape[1]
    )
    return localize_spikes_fixed_codebook(
        coords,
        waveforms,
        omega,
        kernels=tuple(config["kernel"].split(",")),
        n_scales=int(config["n_scales"]),
        n_sites=int(config["n_sites"]),
        refine_levels=int(config["refine_levels"]),
        device="cuda",
        mask=mask,
        spatial_transform=transforms,
        continuous=bool(config["continuous_refine"]),
        continuous_max_iterations=int(config["continuous_max_iterations"]),
        continuous_backtracks=int(config["continuous_backtracks"]),
        identifiable_rho=bool(config["identifiable_rho"]),
    )


def compare(reference, candidate):
    result = {}
    for field in OUTPUT_FIELDS:
        left, right = reference[field], candidate[field]
        if np.issubdtype(left.dtype, np.integer):
            result[field] = {"equal": bool(np.array_equal(left, right))}
        else:
            result[field] = {
                "equal": bool(np.array_equal(left, right)),
                "max_abs_difference": float(np.max(np.abs(left - right))),
            }
    return result


def assert_under_runs(path):
    runs = Path(__file__).resolve().parents[3] / "runs"
    try:
        path.resolve().relative_to(runs.resolve())
    except ValueError as error:
        raise ValueError(f"output must be under {runs}") from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--events", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.events <= 0 or args.warmups < 0 or args.repeats <= 0:
        parser.error("events and repeats must be positive; warmups must be nonnegative")
    assert_under_runs(args.output)

    coords, waveforms, mask, omega, config, chunk_path = load_fixture(
        args.run, args.chunk, args.events
    )
    reference = localize(coords, waveforms, mask, omega, config, False)
    for _ in range(args.warmups):
        localize(coords, waveforms, mask, omega, config, True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    timings = []
    comparisons = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = perf_counter()
        output = localize(coords, waveforms, mask, omega, config, True)
        torch.cuda.synchronize()
        timings.append(perf_counter() - started)
        comparisons.append(compare(reference, output))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        local_coords=coords,
        residual_waveforms=waveforms,
        mask=mask,
        omega=omega,
        **{field: reference[field] for field in OUTPUT_FIELDS},
    )
    result = {
        "source_run": str(args.run.resolve()),
        "source_chunk": str(chunk_path.resolve()),
        "events": len(waveforms),
        "timings_seconds": timings,
        "median_seconds": float(np.median(timings)),
        "range_seconds": [float(min(timings)), float(max(timings))],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reference": "explicit per-event identity transforms",
        "candidate": "spatial_transform=None identity fast path",
        "repeat_comparisons": comparisons,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
