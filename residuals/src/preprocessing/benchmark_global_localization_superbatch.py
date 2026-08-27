"""Benchmark larger localization superbatches assembled from saved pursuit events."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maths import localize_spikes_fixed_codebook


def load_events(run, max_events):
    waveforms = []
    coordinates = []
    masks = []
    remaining = max_events
    for path in sorted((run / "chunks").glob("chunk_*.npz")):
        if remaining <= 0:
            break
        with np.load(path, allow_pickle=False) as archive:
            if "residual_waveforms" not in archive.files:
                raise KeyError(f"{path} has no residual_waveforms; rerun with --save-waveforms")
            count = min(remaining, len(archive["spike_times"]))
            waveforms.append(np.asarray(archive["residual_waveforms"][:count]))
            coordinates.append(np.asarray(archive["local_coords"][:count]))
            masks.append(np.asarray(archive["neighbor_ids"][:count]) >= 0)
            remaining -= count
    if not waveforms:
        raise FileNotFoundError(f"no saved pursuit events in {run}")
    return np.concatenate(coordinates), np.concatenate(waveforms), np.concatenate(masks)


def localize_batches(coordinates, waveforms, mask, omega, config, batch_size):
    cache = {}
    outputs = {key: [] for key in ("sources", "temporal_idx", "alpha", "captured_energy")}
    warm_stop = min(batch_size, len(waveforms))
    localize_spikes_fixed_codebook(
        coordinates[:warm_stop],
        waveforms[:warm_stop],
        omega,
        kernels=("monopole",),
        n_scales=int(config["n_scales"]),
        n_sites=int(config["n_sites"]),
        refine_levels=int(config["refine_levels"]),
        continuous=bool(config["continuous_refine"]),
        continuous_max_iterations=int(config["continuous_max_iterations"]),
        continuous_backtracks=int(config["continuous_backtracks"]),
        device="cuda",
        mask=mask[:warm_stop],
        coarse_footprint_cache=cache,
        config_batch_size=int(config["localization_config_batch_size"]),
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for start in range(0, len(waveforms), batch_size):
        stop = min(start + batch_size, len(waveforms))
        fit = localize_spikes_fixed_codebook(
            coordinates[start:stop],
            waveforms[start:stop],
            omega,
            kernels=("monopole",),
            n_scales=int(config["n_scales"]),
            n_sites=int(config["n_sites"]),
            refine_levels=int(config["refine_levels"]),
            continuous=bool(config["continuous_refine"]),
            continuous_max_iterations=int(config["continuous_max_iterations"]),
            continuous_backtracks=int(config["continuous_backtracks"]),
            device="cuda",
            mask=mask[start:stop],
            coarse_footprint_cache=cache,
            config_batch_size=int(config["localization_config_batch_size"]),
        )
        for key in outputs:
            outputs[key].append(np.asarray(fit[key]))
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    return (
        {key: np.concatenate(parts) for key, parts in outputs.items()},
        elapsed,
        int(torch.cuda.max_memory_allocated()),
    )


def compare(reference, candidate):
    return {
        "temporal_row_agreement": float(
            np.mean(reference["temporal_idx"] == candidate["temporal_idx"])
        ),
        "source_max_absolute_difference_um": float(
            np.max(np.abs(reference["sources"] - candidate["sources"]))
        ),
        "alpha_max_absolute_difference": float(
            np.max(np.abs(reference["alpha"] - candidate["alpha"]))
        ),
        "captured_energy_max_absolute_difference": float(
            np.max(
                np.abs(reference["captured_energy"] - candidate["captured_energy"])
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-events", type=int, default=2048)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(256, 512, 1024, 2048))
    args = parser.parse_args()

    metadata = json.loads((args.run / "config.json").read_text())
    config = metadata["config"]
    omega = np.load(args.run / "omega.npy")
    coordinates, waveforms, mask = load_events(args.run, args.max_events)
    results = {}
    reference = None
    reference_batch_size = None
    for batch_size in args.batch_sizes:
        outputs, elapsed, peak_memory = localize_batches(
            coordinates, waveforms, mask, omega, config, batch_size
        )
        if reference is None:
            reference = outputs
            reference_batch_size = batch_size
        results[str(batch_size)] = {
            "elapsed_seconds": elapsed,
            "events_per_second": len(waveforms) / elapsed,
            "peak_allocated_bytes": peak_memory,
            "comparison_to_reference": compare(reference, outputs),
        }
        print(
            f"batch={batch_size} elapsed={elapsed:.3f}s "
            f"rate={len(waveforms) / elapsed:.1f} events/s "
            f"peak={peak_memory / 2**30:.2f} GiB",
            flush=True,
        )
    result = {
        "run": str(args.run.resolve()),
        "q": int(len(omega)),
        "n_events": int(len(waveforms)),
        "reference_batch_size": reference_batch_size,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
