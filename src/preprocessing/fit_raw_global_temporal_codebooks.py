"""Detect fresh raw-recording peaks and fit global temporal codebooks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maths import fit_spike_model
from preprocessing.raw_residual import (
    channel_neighborhoods,
    detect_residual_peaks,
    extract_waveforms,
    preprocess_voltage,
    robust_channel_noise,
)


def save_training_sample(output_dir, sample):
    np.save(output_dir / "training_waveforms.npy", sample["waveforms"])
    np.save(output_dir / "training_local_coords.npy", sample["local_coords"])
    np.save(output_dir / "training_mask.npy", sample["mask"])
    np.save(output_dir / "training_spike_times.npy", sample["spike_times"])
    np.save(output_dir / "training_spike_channels.npy", sample["spike_channels"])
    np.save(output_dir / "training_detection_scores.npy", sample["detection_scores"])


def collect_training_sample(reader, output_dir, args):
    fs = float(reader.fs)
    n_channels = len(reader.geometry["x"])
    channel_positions = np.column_stack(
        (reader.geometry["x"], reader.geometry["y"])
    ).astype(np.float32)
    neighborhood_ids, local_coords, _, _, spatial_neighbors = channel_neighborhoods(
        channel_positions, args.radius_um
    )
    n_before = int(round(args.ms_before * fs / 1000))
    n_after = int(round(args.ms_after * fs / 1000))
    waveform_length = n_before + n_after
    temporal_radius = max(1, int(round(args.temporal_radius_ms * fs / 1000)))
    first_sample = max(0, int(round(args.start_seconds * fs)))
    stop_sample = reader.ns
    if args.duration_seconds is not None:
        stop_sample = min(
            stop_sample,
            first_sample + int(round(args.duration_seconds * fs)),
        )
    chunk_samples = max(1, int(round(args.chunk_seconds * fs)))
    margin = max(
        int(round(args.read_margin_ms * fs / 1000)),
        waveform_length,
        128,
    )
    starts = list(range(first_sample, stop_sample, chunk_samples))
    if not starts:
        raise ValueError("raw training interval is empty")
    events_per_chunk = max(1, int(np.ceil(args.max_events / len(starts))))
    rng = np.random.default_rng(args.seed)
    waveform_parts = []
    coordinate_parts = []
    mask_parts = []
    time_parts = []
    channel_parts = []
    score_parts = []
    detected_total = 0

    for chunk_index, core_global_start in enumerate(starts):
        core_global_stop = min(core_global_start + chunk_samples, stop_sample)
        read_start = max(0, core_global_start - margin)
        read_stop = min(reader.ns, core_global_stop + margin)
        raw = reader[read_start:read_stop, :n_channels]
        data = preprocess_voltage(
            raw,
            fs,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            order=args.filter_order,
        )
        noise = robust_channel_noise(data)
        times, channels, scores = detect_residual_peaks(
            data,
            noise,
            spatial_neighbors,
            threshold=args.detection_threshold,
            temporal_radius=temporal_radius,
            valid_start=core_global_start - read_start,
            valid_stop=core_global_stop - read_start,
            max_peaks=None,
        )
        valid = (
            (times >= n_before)
            & (times + n_after <= len(data))
            & (times >= core_global_start - read_start)
            & (times < core_global_stop - read_start)
        )
        times = times[valid]
        channels = channels[valid]
        scores = scores[valid]
        detected_total += len(times)
        if len(times):
            keep = rng.choice(
                len(times), min(events_per_chunk, len(times)), replace=False
            )
            keep.sort()
            selected_times = times[keep]
            selected_channels = channels[keep]
            waveforms, _, coordinates, mask = extract_waveforms(
                data,
                selected_times,
                selected_channels,
                neighborhood_ids,
                local_coords,
                n_before,
                n_after,
            )
            waveform_parts.append(waveforms)
            coordinate_parts.append(coordinates)
            mask_parts.append(mask)
            time_parts.append((read_start + selected_times).astype(np.int64))
            channel_parts.append(selected_channels.astype(np.int32))
            score_parts.append(scores[keep].astype(np.float32))
        if (chunk_index + 1) % 25 == 0 or chunk_index + 1 == len(starts):
            sampled = sum(len(part) for part in time_parts)
            print(
                f"raw scan {chunk_index + 1}/{len(starts)}: "
                f"detected={detected_total:,} sampled={sampled:,}",
                flush=True,
            )

    if not waveform_parts:
        raise RuntimeError("fresh raw detector selected no training events")
    sample = {
        "waveforms": np.concatenate(waveform_parts),
        "local_coords": np.concatenate(coordinate_parts),
        "mask": np.concatenate(mask_parts),
        "spike_times": np.concatenate(time_parts),
        "spike_channels": np.concatenate(channel_parts),
        "detection_scores": np.concatenate(score_parts),
    }
    if len(sample["spike_times"]) > args.max_events:
        keep = np.sort(
            rng.choice(len(sample["spike_times"]), args.max_events, replace=False)
        )
        sample = {key: value[keep] for key, value in sample.items()}
    save_training_sample(output_dir, sample)
    metadata = {
        "recording_path": str(args.recording_path.resolve()),
        "fs": fs,
        "n_channels": n_channels,
        "first_sample": first_sample,
        "stop_sample": stop_sample,
        "chunk_samples": chunk_samples,
        "n_chunks": len(starts),
        "detected_events": int(detected_total),
        "sampled_events": int(len(sample["spike_times"])),
        "events_per_chunk_target": events_per_chunk,
        "waveform_length": waveform_length,
        "parameters": {
            "start_seconds": args.start_seconds,
            "duration_seconds": args.duration_seconds,
            "chunk_seconds": args.chunk_seconds,
            "detection_threshold": args.detection_threshold,
            "radius_um": args.radius_um,
            "ms_before": args.ms_before,
            "ms_after": args.ms_after,
            "temporal_radius_ms": args.temporal_radius_ms,
            "max_events": args.max_events,
            "seed": args.seed,
        },
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return sample


def fit_codebooks(sample, output_dir, args):
    summaries = {}
    for q in args.q_values:
        print(f"=== fitting global Q={q} codebook ===", flush=True)
        fit = fit_spike_model(
            sample["local_coords"],
            sample["waveforms"],
            Q=q,
            kernels=("monopole",),
            n_scales=args.n_scales,
            n_sites=args.n_sites,
            n_iters=args.n_iters,
            tol=args.tolerance,
            refine_levels=args.refine_levels,
            device=args.device,
            mask=sample["mask"],
        )
        counts = np.bincount(fit["temporal_idx"], minlength=q)
        result_path = output_dir / f"global_codebook_q{q}.npz"
        np.savez_compressed(
            result_path,
            omega=fit["omega"],
            temporal_count=counts,
            nmse=np.asarray(fit["nmse"]),
            nmse_coarse=np.asarray(fit["nmse_coarse"]),
            q=np.asarray(q),
        )
        history = [
            {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in step.items()
            }
            for step in fit["history"]
        ]
        summary = {
            "q": q,
            "result_path": str(result_path.resolve()),
            "nmse": float(fit["nmse"]),
            "nmse_coarse": float(fit["nmse_coarse"]),
            "temporal_count": counts.tolist(),
            "history": history,
        }
        (output_dir / f"global_codebook_q{q}.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        summaries[str(q)] = summary
        del fit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (output_dir / "codebook_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--q-values", type=int, nargs="+", default=(8, 16, 24, 32))
    parser.add_argument("--start-seconds", type=float, default=60.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--detection-threshold", type=float, default=6.0)
    parser.add_argument("--radius-um", type=float, default=48.0)
    parser.add_argument("--ms-before", type=float, default=1.5)
    parser.add_argument("--ms-after", type=float, default=1.5)
    parser.add_argument("--temporal-radius-ms", type=float, default=0.5)
    parser.add_argument("--read-margin-ms", type=float, default=20.0)
    parser.add_argument("--freq-min", type=float, default=300.0)
    parser.add_argument("--freq-max", type=float, default=6000.0)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--max-events", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-scales", type=int, default=10)
    parser.add_argument("--n-sites", type=int, default=16)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--refine-levels", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(set(args.q_values)) != len(args.q_values):
        raise ValueError("Q values must be unique")
    if min(args.q_values) < 1:
        raise ValueError("Q values must be positive")
    if args.max_events < max(args.q_values):
        raise ValueError("max events must be at least the largest Q")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import spikeglx

    reader = spikeglx.Reader(args.recording_path)
    try:
        sample = collect_training_sample(reader, args.output_dir, args)
    finally:
        reader.close()
    fit_codebooks(sample, args.output_dir, args)


if __name__ == "__main__":
    main()
