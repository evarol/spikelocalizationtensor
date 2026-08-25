"""Fit global temporal codebooks from fresh peak-channel waveforms."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.raw_residual import (
    channel_neighborhoods,
    detect_residual_peaks,
    preprocess_voltage,
    robust_channel_noise,
)


def isolated_peak_mask(times, channels, spatial_neighbors, radius_samples):
    times = np.asarray(times, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int32)
    if radius_samples <= 0 or not len(times):
        return np.ones(len(times), dtype=bool)
    channel_times = [np.sort(times[channels == channel]) for channel in range(len(spatial_neighbors))]
    isolated = np.ones(len(times), dtype=bool)
    for index, (time, channel) in enumerate(zip(times, channels)):
        count = 0
        for neighbor in spatial_neighbors[channel]:
            values = channel_times[neighbor]
            count += np.searchsorted(values, time + radius_samples, side="right")
            count -= np.searchsorted(values, time - radius_samples, side="left")
            if count > 1:
                isolated[index] = False
                break
    return isolated


def extract_peak_channel_waveforms(data, times, channels, n_before, n_after):
    offsets = np.arange(-n_before, n_after, dtype=np.int64)
    return np.asarray(data[times[:, None] + offsets, channels[:, None]], dtype=np.float32)


def collect_training_sample(reader, output_dir, args):
    fs = float(reader.fs)
    n_channels = len(reader.geometry["x"])
    channel_positions = np.column_stack(
        (reader.geometry["x"], reader.geometry["y"])
    ).astype(np.float32)
    _, _, _, _, spatial_neighbors = channel_neighborhoods(
        channel_positions, args.radius_um
    )
    n_before = int(round(args.ms_before * fs / 1000))
    n_after = int(round(args.ms_after * fs / 1000))
    waveform_length = n_before + n_after
    temporal_radius = max(1, int(round(args.temporal_radius_ms * fs / 1000)))
    isolation_radius = max(0, int(round(args.isolation_ms * fs / 1000)))
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
        isolation_radius,
        128,
    )
    starts = np.arange(first_sample, stop_sample, chunk_samples, dtype=np.int64)
    if not len(starts):
        raise ValueError("raw training interval is empty")
    rng = np.random.default_rng(args.seed)
    scan_order = rng.permutation(len(starts))
    waveform_parts = []
    time_parts = []
    channel_parts = []
    score_parts = []
    scanned_chunk_indices = []
    detected_total = 0
    isolated_total = 0
    sampled_total = 0

    for scan_number, chunk_index in enumerate(scan_order, start=1):
        core_global_start = int(starts[chunk_index])
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
            valid_start=n_before,
            valid_stop=min(len(data), len(data) - n_after + 1),
            max_peaks=None,
        )
        valid_waveform = (
            (times >= n_before)
            & (times + n_after <= len(data))
        )
        keep_isolated = isolated_peak_mask(
            times, channels, spatial_neighbors, isolation_radius
        )
        in_core = (
            (times >= core_global_start - read_start)
            & (times < core_global_stop - read_start)
        )
        detected_total += int(np.count_nonzero(valid_waveform & in_core))
        keep = valid_waveform & keep_isolated & in_core
        times = times[keep]
        channels = channels[keep]
        scores = scores[keep]
        isolated_total += len(times)
        remaining = args.max_events - sampled_total
        take = min(len(times), args.max_events_per_chunk, remaining)
        if take:
            keep = np.sort(rng.choice(len(times), take, replace=False))
            selected_times = times[keep]
            selected_channels = channels[keep]
            waveform_parts.append(
                extract_peak_channel_waveforms(
                    data,
                    selected_times,
                    selected_channels,
                    n_before,
                    n_after,
                )
            )
            time_parts.append((read_start + selected_times).astype(np.int64))
            channel_parts.append(selected_channels.astype(np.int32))
            score_parts.append(scores[keep].astype(np.float32))
            sampled_total += take
        scanned_chunk_indices.append(int(chunk_index))
        if scan_number % 10 == 0 or sampled_total >= args.max_events:
            print(
                f"peak-channel scan {scan_number}/{len(starts)}: "
                f"detected={detected_total:,} isolated={isolated_total:,} "
                f"sampled={sampled_total:,}",
                flush=True,
            )
        if sampled_total >= args.max_events:
            break

    if not waveform_parts:
        raise RuntimeError("fresh raw detector selected no peak-channel waveforms")
    sample = {
        "waveforms": np.concatenate(waveform_parts),
        "spike_times": np.concatenate(time_parts),
        "spike_channels": np.concatenate(channel_parts),
        "detection_scores": np.concatenate(score_parts),
    }
    np.save(output_dir / "training_peak_channel_waveforms.npy", sample["waveforms"])
    np.save(output_dir / "training_spike_times.npy", sample["spike_times"])
    np.save(output_dir / "training_spike_channels.npy", sample["spike_channels"])
    np.save(output_dir / "training_detection_scores.npy", sample["detection_scores"])
    metadata = {
        "recording_path": str(args.recording_path.resolve()),
        "fs": fs,
        "n_channels": n_channels,
        "first_sample": first_sample,
        "stop_sample": stop_sample,
        "chunk_samples": chunk_samples,
        "available_chunks": int(len(starts)),
        "scanned_chunks": len(scanned_chunk_indices),
        "scanned_chunk_indices": scanned_chunk_indices,
        "detected_events_in_scanned_chunks": int(detected_total),
        "isolated_events_in_scanned_chunks": int(isolated_total),
        "sampled_events": int(len(sample["spike_times"])),
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
            "isolation_ms": args.isolation_ms,
            "max_events": args.max_events,
            "max_events_per_chunk": args.max_events_per_chunk,
            "seed": args.seed,
        },
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return sample


def assign_waveforms(waveforms, omega, batch_size):
    labels = torch.empty(len(waveforms), dtype=torch.long, device=waveforms.device)
    response = torch.empty(len(waveforms), dtype=waveforms.dtype, device=waveforms.device)
    for start in range(0, len(waveforms), batch_size):
        stop = min(start + batch_size, len(waveforms))
        scores = waveforms[start:stop] @ omega.T
        _, selected = scores.abs().max(dim=1)
        labels[start:stop] = selected
        response[start:stop] = scores.gather(1, selected[:, None]).squeeze(1)
    return labels, response


def fit_peak_channel_codebook(
    waveforms,
    q,
    initial_indices,
    n_iters,
    tolerance,
    assignment_batch_size,
    device,
):
    values = torch.as_tensor(waveforms, dtype=torch.float32, device=device)
    values = values - values.mean(dim=1, keepdim=True)
    energy = values.square().sum(dim=1)
    valid = torch.isfinite(energy) & (energy > torch.finfo(values.dtype).tiny)
    values = values[valid]
    if len(values) < q:
        raise ValueError(f"only {len(values)} finite waveforms are available for Q={q}")
    selected_initial = initial_indices[initial_indices < len(values)][:q]
    if len(selected_initial) != q:
        raise ValueError("shared initialization permutation is shorter than Q")
    selected_initial = torch.as_tensor(
        selected_initial, dtype=torch.long, device=values.device
    )
    omega = torch.nn.functional.normalize(values[selected_initial], dim=1)
    history = []

    for iteration in range(1, n_iters + 1):
        labels, response = assign_waveforms(values, omega, assignment_batch_size)
        numerator = torch.zeros_like(omega)
        numerator.index_add_(0, labels, response[:, None] * values)
        counts = torch.bincount(labels, minlength=q)
        updated = omega.clone()
        used = counts > 0
        updated[used] = torch.nn.functional.normalize(numerator[used], dim=1)
        if not bool(used.all()):
            worst = torch.argsort(response.abs())
            cursor = 0
            for row in torch.nonzero(~used, as_tuple=False).flatten().tolist():
                updated[row] = torch.nn.functional.normalize(
                    values[worst[cursor]][None], dim=1
                )[0]
                cursor += 1
        alignment = (updated * omega).sum(dim=1)
        updated[alignment < 0] *= -1
        row_change = torch.linalg.vector_norm(updated - omega, dim=1)
        residual_energy = torch.clamp(energy[valid] - response.square(), min=0).sum()
        nmse = float((residual_energy / energy[valid].sum()).item())
        step = {
            "iteration": iteration,
            "nmse": nmse,
            "used_rows": int(torch.count_nonzero(counts).item()),
            "maximum_row_change": float(row_change.max().item()),
            "minimum_row_count": int(counts.min().item()),
            "maximum_row_count": int(counts.max().item()),
        }
        history.append(step)
        print(
            f"Q={q} iteration {iteration}: nMSE={nmse:.6f} "
            f"used={step['used_rows']}/{q} "
            f"max_change={step['maximum_row_change']:.6f}",
            flush=True,
        )
        omega = updated
        if step["maximum_row_change"] < tolerance:
            break

    labels, response = assign_waveforms(values, omega, assignment_batch_size)
    counts = torch.bincount(labels, minlength=q)
    residual_energy = torch.clamp(energy[valid] - response.square(), min=0).sum()
    nmse = float((residual_energy / energy[valid].sum()).item())
    center = omega.shape[1] // 2
    omega[omega[:, center] > 0] *= -1
    return {
        "omega": omega.to("cpu").numpy(),
        "temporal_count": counts.to("cpu").numpy(),
        "nmse": nmse,
        "history": history,
        "valid_waveforms": int(len(values)),
    }


def codebook_diagnostics(omega):
    cosine = np.abs(omega @ omega.T)
    if len(omega) == 1:
        return {
            "maximum_pairwise_absolute_cosine": 0.0,
            "median_pairwise_absolute_cosine": 0.0,
        }
    values = cosine[~np.eye(len(omega), dtype=bool)]
    return {
        "maximum_pairwise_absolute_cosine": float(values.max()),
        "median_pairwise_absolute_cosine": float(np.median(values)),
    }


def fit_codebooks(sample, output_dir, args):
    rng = np.random.default_rng(args.seed)
    initial_indices = rng.permutation(len(sample["waveforms"]))
    summaries = {}
    for q in args.q_values:
        print(f"=== fitting peak-channel Q={q} codebook ===", flush=True)
        fit = fit_peak_channel_codebook(
            sample["waveforms"],
            q,
            initial_indices,
            args.n_iters,
            args.tolerance,
            args.assignment_batch_size,
            args.device,
        )
        result_path = output_dir / f"global_codebook_q{q}.npz"
        np.savez_compressed(
            result_path,
            omega=fit["omega"],
            temporal_count=fit["temporal_count"],
            nmse=np.asarray(fit["nmse"]),
            q=np.asarray(q),
            initialization=np.asarray("peak_channel_correlation"),
        )
        summary = {
            "q": q,
            "result_path": str(result_path.resolve()),
            "nmse": fit["nmse"],
            "valid_waveforms": fit["valid_waveforms"],
            "temporal_count": fit["temporal_count"].tolist(),
            "history": fit["history"],
            "diagnostics": codebook_diagnostics(fit["omega"]),
        }
        (output_dir / f"global_codebook_q{q}.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        summaries[str(q)] = summary
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
    parser.add_argument("--q-values", type=int, nargs="+", default=(4, 8, 16, 32, 64))
    parser.add_argument("--start-seconds", type=float, default=60.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--detection-threshold", type=float, default=6.0)
    parser.add_argument("--radius-um", type=float, default=48.0)
    parser.add_argument("--ms-before", type=float, default=1.5)
    parser.add_argument("--ms-after", type=float, default=1.5)
    parser.add_argument("--temporal-radius-ms", type=float, default=0.5)
    parser.add_argument("--isolation-ms", type=float, default=1.0)
    parser.add_argument("--read-margin-ms", type=float, default=20.0)
    parser.add_argument("--freq-min", type=float, default=300.0)
    parser.add_argument("--freq-max", type=float, default=6000.0)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--max-events", type=int, default=100000)
    parser.add_argument("--max-events-per-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--assignment-batch-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(set(args.q_values)) != len(args.q_values):
        raise ValueError("Q values must be unique")
    if min(args.q_values) < 1:
        raise ValueError("Q values must be positive")
    if args.max_events < max(args.q_values):
        raise ValueError("max events must be at least the largest Q")
    if args.max_events_per_chunk < 1:
        raise ValueError("max events per chunk must be positive")
    if args.n_iters < 1 or args.assignment_batch_size < 1:
        raise ValueError("iteration count and assignment batch size must be positive")
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
