import argparse
import json
from pathlib import Path

import numpy as np


def quantiles(values):
    levels = np.asarray((0, .01, .1, .25, .5, .75, .9, .99, 1))
    return {str(level): float(value) for level, value in zip(levels, np.quantile(values, levels))}


def recurrence(times, channels, passes, positions, fs, radius_um, milliseconds):
    distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    nearby = [np.flatnonzero(row <= radius_um) for row in distance]
    result = {}
    for residual_pass in range(1, int(passes.max()) + 1):
        prior = passes < residual_pass
        lookup = {
            channel: np.sort(times[prior & (channels == channel)])
            for channel in range(len(positions))
        }
        rows = np.flatnonzero(passes == residual_pass)
        for milliseconds_value in milliseconds:
            radius = int(round(milliseconds_value * fs / 1000))
            hits = 0
            for row in rows:
                event_time = times[row]
                found = False
                for channel in nearby[channels[row]]:
                    candidates = lookup[channel]
                    insertion = np.searchsorted(candidates, event_time)
                    after = insertion < len(candidates) and candidates[insertion] - event_time <= radius
                    before = insertion and event_time - candidates[insertion - 1] <= radius
                    if after or before:
                        found = True
                        break
                hits += found
            result[f"pass_{residual_pass}_within_{milliseconds_value:g}ms"] = hits / max(len(rows), 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--radius-um", type=float, default=48.0)
    args = parser.parse_args()
    metadata = json.loads((args.run / "config.json").read_text())
    path = args.run / "chunks" / f"chunk_{args.chunk_index:06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    mask = saved["neighbor_ids"] >= 0
    noise = saved["noise"]
    local_noise = noise[np.maximum(saved["neighbor_ids"], 0)]
    waveform = saved["residual_waveforms"].astype(np.float64)
    prediction = saved["predictions"].astype(np.float64)
    input_energy = ((waveform / local_noise[:, :, None]) ** 2 * mask[:, :, None]).sum((1, 2))
    residual_energy = (((waveform - prediction) / local_noise[:, :, None]) ** 2 * mask[:, :, None]).sum((1, 2))
    direct_capture = (input_energy - residual_energy) / np.maximum(input_energy, 1e-12)
    thresholds = (.05, .1, .15, .2, .25, .3, .4, .5)
    output = {
        "chunk": args.chunk_index,
        "events": len(saved["spike_times"]),
        "maximum_saved_direct_capture_error": float(np.max(np.abs(direct_capture - saved["captured_fraction"]))),
        "quantiles": {
            key: quantiles(values)
            for key, values in {
                "captured_fraction": saved["captured_fraction"],
                "captured_energy": saved["captured_energy"],
                "fitted_projection_score": np.sqrt(np.maximum(saved["captured_energy"], 0)),
                "detection_score": saved["detection_score"],
                "maximum_channel_normalized_rmse": saved["maximum_channel_normalized_rmse"],
                "sigma": saved["sigma"],
                "rho": saved["rho"],
            }.items()
        },
        "capture_gate_retention": {
            str(threshold): float(np.mean(saved["captured_fraction"] >= threshold))
            for threshold in thresholds
        },
        "fitted_projection_retention": {
            str(threshold): float(np.mean(np.sqrt(np.maximum(saved["captured_energy"], 0)) >= threshold))
            for threshold in (3, 4, 5, 6, 7, 8)
        },
        "near_prior_event_fraction": recurrence(
            saved["spike_times"], saved["spike_channels"], saved["residual_pass"],
            np.load(args.run / "channel_positions.npy"), float(metadata["fs"]),
            args.radius_um, (.5, 1.5, 3.0),
        ),
    }
    destination = args.run / f"audit_chunk_{args.chunk_index:06d}.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
