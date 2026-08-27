"""Calibrate matched-candidate thresholds for global temporal codebooks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maths import build_codebook_detection_footprints
from preprocessing.raw_residual import (
    channel_neighborhoods,
    full_template_scores,
    load_omega,
    preprocess_voltage,
    robust_channel_noise,
    select_template_peaks_torch,
)


def load_codebooks(paths):
    codebooks = {}
    for path in paths:
        omega = load_omega(path)
        q = len(omega)
        if q in codebooks:
            raise ValueError(f"duplicate Q={q} codebooks: {codebooks[q][0]} and {path}")
        codebooks[q] = (Path(path), omega)
    return dict(sorted(codebooks.items()))


def matched_threshold(scores, target_count):
    values = np.asarray(scores, dtype=np.float32)
    if target_count < 1:
        raise ValueError("reference threshold selected no calibration candidates")
    if len(values) < target_count:
        raise ValueError(
            f"only {len(values)} candidates are available for target count {target_count}"
        )
    descending = np.sort(values)[::-1]
    threshold = float(descending[target_count - 1])
    return threshold, int(np.count_nonzero(values >= threshold))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("codebooks", nargs="+", type=Path)
    parser.add_argument("--reference-q", type=int, default=8)
    parser.add_argument("--reference-threshold", type=float, default=6.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--radius-um", type=float, default=48.0)
    parser.add_argument("--ms-before", type=float, default=1.5)
    parser.add_argument("--ms-after", type=float, default=1.5)
    parser.add_argument("--temporal-radius-ms", type=float, default=0.5)
    parser.add_argument("--read-margin-ms", type=float, default=20.0)
    parser.add_argument("--freq-min", type=float, default=300.0)
    parser.add_argument("--freq-max", type=float, default=6000.0)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--n-scales", type=int, default=10)
    parser.add_argument("--template-time-batch", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import spikeglx

    codebooks = load_codebooks(args.codebooks)
    if args.reference_q not in codebooks:
        raise ValueError(f"reference Q={args.reference_q} is not among the codebooks")
    lengths = {omega.shape[1] for _, omega in codebooks.values()}
    if len(lengths) != 1:
        raise ValueError(f"codebook waveform lengths differ: {sorted(lengths)}")
    waveform_length = lengths.pop()

    reader = spikeglx.Reader(args.recording_path)
    try:
        fs = float(reader.fs)
        n_channels = len(reader.geometry["x"])
        channel_positions = np.column_stack(
            (reader.geometry["x"], reader.geometry["y"])
        ).astype(np.float32)
        neighborhood_ids, local_coords, centroids, _, _ = channel_neighborhoods(
            channel_positions, args.radius_um
        )
        footprints, _ = build_codebook_detection_footprints(
            local_coords,
            neighborhood_ids >= 0,
            channel_positions - centroids,
            kernels=("monopole",),
            n_scales=args.n_scales,
            device="cpu",
        )
        n_before = int(round(args.ms_before * fs / 1000))
        n_after = int(round(args.ms_after * fs / 1000))
        if n_before + n_after != waveform_length:
            raise ValueError(
                f"codebooks have {waveform_length} samples but extraction uses "
                f"{n_before + n_after}"
            )
        temporal_radius = max(
            1, int(round(args.temporal_radius_ms * fs / 1000))
        )
        first_sample = max(0, int(round(args.start_seconds * fs)))
        stop_sample = min(
            reader.ns,
            first_sample + int(round(args.duration_seconds * fs)),
        )
        chunk_samples = max(1, int(round(args.chunk_seconds * fs)))
        margin = max(
            int(round(args.read_margin_ms * fs / 1000)),
            waveform_length,
            128,
        )
        chunks = []
        for core_start in range(first_sample, stop_sample, chunk_samples):
            core_stop = min(core_start + chunk_samples, stop_sample)
            read_start = max(0, core_start - margin)
            read_stop = min(reader.ns, core_stop + margin)
            raw = reader[read_start:read_stop, :n_channels]
            data = preprocess_voltage(
                raw,
                fs,
                freq_min=args.freq_min,
                freq_max=args.freq_max,
                order=args.filter_order,
            )
            chunks.append(
                (
                    data,
                    robust_channel_noise(data),
                    core_start - read_start,
                    core_stop - read_start,
                )
            )
    finally:
        reader.close()

    candidate_scores = {}
    for q, (path, omega) in codebooks.items():
        parts = []
        for chunk_index, (data, noise, core_start, core_stop) in enumerate(chunks):
            scores = full_template_scores(
                data,
                noise,
                omega,
                footprints,
                neighborhood_ids,
                device=args.device,
                time_batch=args.template_time_batch,
                return_torch=True,
            )
            times, _, selected_scores = select_template_peaks_torch(
                scores,
                neighborhood_ids,
                threshold=float("-inf"),
                temporal_radius=temporal_radius,
                n_before=n_before,
                max_peaks=None,
            )
            in_core = (times >= core_start) & (times < core_stop)
            parts.append(selected_scores[in_core])
            del scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"Q={q} calibration chunk {chunk_index + 1}/{len(chunks)}: "
                f"local_maxima={int(in_core.sum())}",
                flush=True,
            )
        candidate_scores[q] = np.concatenate(parts)
        print(
            f"Q={q} codebook={path} total_local_maxima={len(candidate_scores[q])}",
            flush=True,
        )

    reference_scores = candidate_scores[args.reference_q]
    target_count = int(
        np.count_nonzero(reference_scores >= args.reference_threshold)
    )
    results = {}
    for q, (path, omega) in codebooks.items():
        threshold, achieved_count = matched_threshold(candidate_scores[q], target_count)
        results[str(q)] = {
            "codebook_path": str(path.resolve()),
            "shape": list(omega.shape),
            "local_maxima": int(len(candidate_scores[q])),
            "fixed_threshold": float(args.reference_threshold),
            "fixed_threshold_count": int(
                np.count_nonzero(candidate_scores[q] >= args.reference_threshold)
            ),
            "matched_threshold": threshold,
            "matched_threshold_count": achieved_count,
        }

    output = {
        "recording_path": str(args.recording_path.resolve()),
        "start_seconds": args.start_seconds,
        "duration_seconds": args.duration_seconds,
        "chunk_seconds": args.chunk_seconds,
        "reference_q": args.reference_q,
        "reference_threshold": args.reference_threshold,
        "target_candidate_count": target_count,
        "results": results,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
