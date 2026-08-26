"""Save unmodified-first-round Q8 fits for waveform-level diagnosis."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

import spikeglx
from raw_residual import (
    ResidualConfig,
    build_codebook_detection_footprints,
    channel_neighborhoods,
    extract_waveforms,
    full_template_scores,
    load_omega,
    preprocess_voltage,
    robust_channel_noise,
    select_conflict_free_peaks,
    select_template_peaks_torch,
)
from maths import localize_spikes_fixed_codebook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("codebook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=8.748)
    parser.add_argument("--chunk-seconds", type=float, default=2.187)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def save_chunk(path, data):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(path)


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    chunks_dir = args.output / "chunks"
    chunks_dir.mkdir()

    config = ResidualConfig(
        threshold=args.threshold,
        chunk_seconds=args.chunk_seconds,
        pursuit_rounds=60,
        codebook_learning_chunks=0,
        kernel="monopole",
        n_scales=10,
        n_sites=16,
        refine_levels=6,
        continuous_refine=True,
        device=args.device,
        fit_batch_size=args.fit_batch_size,
    )
    omega = load_omega(args.codebook)
    reader = spikeglx.Reader(args.recording)
    try:
        fs = float(reader.fs)
        channel_positions = np.column_stack(
            (reader.geometry["x"], reader.geometry["y"])
        ).astype(np.float32)
        (
            neighborhood_ids,
            local_coordinates,
            channel_centroids,
            _,
            _,
        ) = channel_neighborhoods(channel_positions, config.radius_um)
        anchor_xy = channel_positions - channel_centroids
        detection_footprints, _ = build_codebook_detection_footprints(
            local_coordinates,
            neighborhood_ids >= 0,
            anchor_xy,
            kernels=(config.kernel,),
            n_scales=config.n_scales,
            device="cpu",
        )
        n_before = int(round(config.ms_before * fs / 1000))
        n_after = int(round(config.ms_after * fs / 1000))
        if omega.shape[1] != n_before + n_after:
            raise ValueError("codebook length does not match extraction window")
        first_sample = int(round(args.start_seconds * fs))
        stop_sample = min(
            reader.ns, first_sample + int(round(args.duration_seconds * fs))
        )
        chunk_samples = int(round(args.chunk_seconds * fs))
        margin = max(
            int(round(config.read_margin_ms * fs / 1000)), n_before + n_after, 128
        )
        metadata = {
            "recording": str(args.recording.resolve()),
            "codebook": str(args.codebook.resolve()),
            "fs": fs,
            "first_sample": first_sample,
            "stop_sample": stop_sample,
            "config": asdict(config),
            "interpretation": (
                "Every row is a threshold-6 first-round pursuit candidate fit on "
                "the unmodified residual; no subtraction or rescoring was performed."
            ),
            "per_channel_delta_chi2": (
                "2 * sum(Y_c * Yhat_c / sigma_c^2) - "
                "sum(Yhat_c^2 / sigma_c^2); values sum to the weighted residual "
                "improvement of the saved fitted reconstruction."
            ),
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        np.save(args.output / "channel_positions.npy", channel_positions)

        for chunk_index, core_global_start in enumerate(
            range(first_sample, stop_sample, chunk_samples)
        ):
            core_global_stop = min(core_global_start + chunk_samples, stop_sample)
            read_start = max(0, core_global_start - margin)
            read_stop = min(reader.ns, core_global_stop + margin)
            raw = reader[read_start:read_stop, : len(channel_positions)]
            data = preprocess_voltage(
                raw,
                fs,
                freq_min=config.freq_min,
                freq_max=config.freq_max,
                order=config.filter_order,
            )
            noise = robust_channel_noise(data)
            scores = full_template_scores(
                data,
                noise,
                omega,
                detection_footprints,
                neighborhood_ids,
                device=config.device,
                time_batch=config.template_time_batch,
                return_torch=True,
            )
            times, channels, detection_score = select_template_peaks_torch(
                scores,
                neighborhood_ids,
                threshold=config.threshold,
                temporal_radius=max(1, int(round(config.temporal_radius_ms * fs / 1000))),
                n_before=n_before,
            )
            times, channels, detection_score = select_conflict_free_peaks(
                times,
                channels,
                detection_score,
                lockout_samples=n_before + n_after - 1,
                max_peaks=config.max_peaks_per_round,
            )
            in_core = (times >= core_global_start - read_start) & (
                times < core_global_stop - read_start
            )
            times = times[in_core]
            channels = channels[in_core]
            detection_score = detection_score[in_core]
            parts = []
            for start in range(0, len(times), config.fit_batch_size):
                stop = min(start + config.fit_batch_size, len(times))
                waveforms, ids, local_coords, mask = extract_waveforms(
                    data,
                    times[start:stop],
                    channels[start:stop],
                    neighborhood_ids,
                    local_coordinates,
                    n_before,
                    n_after,
                )
                fit = localize_spikes_fixed_codebook(
                    local_coords,
                    waveforms,
                    omega,
                    kernels=(config.kernel,),
                    n_scales=config.n_scales,
                    n_sites=config.n_sites,
                    refine_levels=config.refine_levels,
                    continuous=config.continuous_refine,
                    continuous_max_iterations=config.continuous_max_iterations,
                    continuous_backtracks=config.continuous_backtracks,
                    device=config.device,
                    mask=mask,
                    config_batch_size=config.localization_config_batch_size,
                )
                predicted = np.asarray(fit["prediction"], dtype=np.float32)
                noise_local = noise[np.maximum(ids, 0)] * mask
                inverse_variance = np.divide(
                    1.0,
                    np.square(noise_local),
                    out=np.zeros_like(noise_local, dtype=np.float32),
                    where=mask,
                )
                per_channel_delta = (
                    2 * np.sum(waveforms * predicted * inverse_variance[:, :, None], axis=2)
                    - np.sum(np.square(predicted) * inverse_variance[:, :, None], axis=2)
                ).astype(np.float32)
                parts.append(
                    {
                        "spike_times": (read_start + times[start:stop]).astype(np.int64),
                        "spike_channels": channels[start:stop].astype(np.int32),
                        "detection_score": detection_score[start:stop].astype(np.float32),
                        "neighbor_ids": ids.astype(np.int32),
                        "neighbor_mask": mask,
                        "noise": noise_local.astype(np.float32),
                        "observed": waveforms.astype(np.float32),
                        "reconstructed": predicted,
                        "residual": (waveforms - predicted).astype(np.float32),
                        "per_channel_delta_chi2": per_channel_delta,
                        "temporal_idx": fit["temporal_idx"].astype(np.int16),
                        "sources": fit["sources"].astype(np.float32),
                    }
                )
            if parts:
                result = {
                    key: np.concatenate([part[key] for part in parts])
                    for key in parts[0]
                }
            else:
                width = neighborhood_ids.shape[1]
                result = {
                    "spike_times": np.empty(0, dtype=np.int64),
                    "spike_channels": np.empty(0, dtype=np.int32),
                    "detection_score": np.empty(0, dtype=np.float32),
                    "neighbor_ids": np.empty((0, width), dtype=np.int32),
                    "neighbor_mask": np.empty((0, width), dtype=bool),
                    "noise": np.empty((0, width), dtype=np.float32),
                    "observed": np.empty((0, width, len(omega[0])), dtype=np.float32),
                    "reconstructed": np.empty((0, width, len(omega[0])), dtype=np.float32),
                    "residual": np.empty((0, width, len(omega[0])), dtype=np.float32),
                    "per_channel_delta_chi2": np.empty((0, width), dtype=np.float32),
                    "temporal_idx": np.empty(0, dtype=np.int16),
                    "sources": np.empty((0, 3), dtype=np.float32),
                }
            save_chunk(chunks_dir / f"chunk_{chunk_index:06d}.npz", result)
            print(
                f"chunk {chunk_index + 1}: saved {len(result['spike_times'])} first fits",
                flush=True,
            )
    finally:
        reader.close()


if __name__ == "__main__":
    main()
