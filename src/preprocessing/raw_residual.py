"""Detect, localize, reconstruct, and peel spikes directly from SpikeGLX data."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn.functional as torch_functional


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maths import (
    build_codebook_detection_footprints,
    localize_spikes_fixed_codebook,
)


MAD_SCALE = 0.6744897501960817


@dataclass(frozen=True)
class ResidualConfig:
    threshold: float = 6.0
    freq_min: float = 300.0
    freq_max: float = 6000.0
    filter_order: int = 3
    radius_um: float = 48.0
    ms_before: float = 1.5
    ms_after: float = 1.5
    temporal_radius_ms: float = 0.5
    chunk_seconds: float = 1.0
    read_margin_ms: float = 20.0
    max_residual_passes: int = 4
    min_captured_fraction: float = 0.05
    min_pass_energy_drop_fraction: float = 0.01
    max_peaks_per_round: int = 10000
    fit_batch_size: int = 1024
    localization_config_batch_size: int = 32
    template_time_batch: int = 4096
    kernel: str = "monopole"
    n_scales: int = 10
    n_sites: int = 16
    refine_levels: int = 6
    continuous_refine: bool = True
    continuous_max_iterations: int = 80
    continuous_backtracks: int = 30
    device: str = "cuda"
    save_waveforms: bool = False
    profile_stages: bool = False


def channel_neighborhoods(channel_positions, radius_um):
    positions = np.asarray(channel_positions, dtype=np.float32)
    distance = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=2
    )
    neighbors = [np.flatnonzero(row <= radius_um) for row in distance]
    width = max(map(len, neighbors))
    ids = np.full((len(positions), width), -1, dtype=np.int32)
    local_coords = np.zeros((len(positions), width, 2), dtype=np.float32)
    centroids = np.zeros((len(positions), 2), dtype=np.float32)
    counts = np.empty(len(positions), dtype=np.int16)
    for channel, row in enumerate(neighbors):
        counts[channel] = len(row)
        ids[channel, :len(row)] = row
        centroids[channel] = positions[row].mean(axis=0)
        local_coords[channel, :len(row)] = positions[row] - centroids[channel]
    return ids, local_coords, centroids, counts, neighbors


def preprocess_voltage(raw, fs, freq_min=300.0, freq_max=6000.0, order=3):
    data = np.asarray(raw, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"raw data must have shape (time, channels), got {data.shape}")
    if not 0 < freq_min < freq_max < fs / 2:
        raise ValueError("filter frequencies must satisfy 0 < min < max < Nyquist")
    sos = butter(order, (freq_min, freq_max), btype="bandpass", fs=fs, output="sos")
    padlen = min(3 * (2 * len(sos) + 1), len(data) - 1)
    filtered = sosfiltfilt(sos, data, axis=0, padlen=padlen).astype(
        np.float32, copy=False
    )
    filtered -= np.median(filtered, axis=1, keepdims=True)
    return filtered


def robust_channel_noise(data):
    centered = data - np.median(data, axis=0, keepdims=True)
    noise = np.median(np.abs(centered), axis=0) / MAD_SCALE
    positive = noise[np.isfinite(noise) & (noise > 0)]
    floor = np.median(positive) * 1e-3 if len(positive) else 1.0
    return np.maximum(noise, floor).astype(np.float32)


def detect_residual_peaks(
    residual,
    noise,
    spatial_neighbors,
    threshold,
    temporal_radius,
    valid_start,
    valid_stop,
    max_peaks=None,
):
    score = -np.asarray(residual, dtype=np.float32) / noise[None]
    candidate = score >= threshold
    spatial_max = np.empty_like(score)
    for channel, neighbors in enumerate(spatial_neighbors):
        spatial_max[:, channel] = np.max(
            maximum_filter1d(
                score[:, neighbors],
                size=2 * temporal_radius + 1,
                axis=0,
                mode="nearest",
            ),
            axis=1,
        )
    candidate &= score >= spatial_max
    candidate[:valid_start] = False
    candidate[valid_stop:] = False
    times, channels = np.nonzero(candidate)
    scores = score[times, channels]
    order = np.argsort(-scores, kind="stable")
    if max_peaks is not None:
        order = order[:max_peaks]
    return (
        times[order].astype(np.int64),
        channels[order].astype(np.int32),
        scores[order].astype(np.float32),
    )


def full_template_scores(
    residual,
    noise,
    omega,
    detection_footprints,
    neighborhood_ids,
    device="cuda",
    time_batch=512,
    return_torch=False,
):
    """Evaluate the best separable codebook-template score at every valid center."""
    residual_np = np.asarray(residual, dtype=np.float32)
    omega_np = np.asarray(omega, dtype=np.float32)
    footprints_np = np.asarray(detection_footprints, dtype=np.float32)
    ids_np = np.asarray(neighborhood_ids, dtype=np.int64)
    n_samples, n_channels = residual_np.shape
    n_anchors, _, n_slots = footprints_np.shape
    if n_anchors != n_channels or ids_np.shape != (n_anchors, n_slots):
        raise ValueError("detection footprints and neighborhood ids are inconsistent")
    if time_batch < 1:
        raise ValueError("time_batch must be positive")
    torch_device = torch.device(device)
    standardized = torch.as_tensor(
        residual_np / noise[None], dtype=torch.float32, device=torch_device
    ).T[None]
    omega_t = torch.as_tensor(omega_np, device=torch_device)
    omega_t = omega_t / omega_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    weights = omega_t.repeat(n_channels, 1).unsqueeze(1)
    projection = torch_functional.conv1d(
        standardized, weights, groups=n_channels
    )[0]
    n_windows = projection.shape[1]
    projection = projection.reshape(
        n_channels, len(omega_np), n_windows
    ).permute(2, 0, 1)

    safe_ids_np = np.maximum(ids_np, 0)
    mask_np = ids_np >= 0
    neighbor_noise = noise[safe_ids_np]
    whitened_footprints = footprints_np / neighbor_noise[:, None, :]
    whitened_footprints *= mask_np[:, None, :]
    footprint_norm = np.linalg.norm(whitened_footprints, axis=2, keepdims=True)
    whitened_footprints /= np.maximum(footprint_norm, 1e-12)
    safe_ids = torch.as_tensor(safe_ids_np, dtype=torch.long, device=torch_device)
    footprints_t = torch.as_tensor(whitened_footprints, device=torch_device)
    scores = torch.empty(
        (n_windows, n_anchors), dtype=torch.float32, device=torch_device
    )
    for start in range(0, n_windows, time_batch):
        stop = min(start + time_batch, n_windows)
        local_projection = projection[start:stop, safe_ids]
        response = torch.einsum(
            "tacq,apc->tapq", local_projection, footprints_t
        )
        scores[start:stop] = response.abs().flatten(2).amax(2)
    if return_torch:
        return scores
    return scores.to("cpu").numpy()


def select_template_peaks(
    scores,
    spatial_neighbors,
    threshold,
    temporal_radius,
    n_before,
    max_peaks=None,
):
    candidate = scores >= threshold
    spatial_max = np.empty_like(scores)
    for channel, neighbors in enumerate(spatial_neighbors):
        spatial_max[:, channel] = np.max(
            maximum_filter1d(
                scores[:, neighbors],
                size=2 * temporal_radius + 1,
                axis=0,
                mode="nearest",
            ),
            axis=1,
        )
    candidate &= scores >= spatial_max
    windows, channels = np.nonzero(candidate)
    selected_scores = scores[windows, channels]
    order = np.argsort(-selected_scores, kind="stable")
    if max_peaks is not None:
        order = order[:max_peaks]
    return (
        (windows[order] + n_before).astype(np.int64),
        channels[order].astype(np.int32),
        selected_scores[order].astype(np.float32),
    )


def select_template_peaks_torch(
    scores,
    neighborhood_ids,
    threshold,
    temporal_radius,
    n_before,
    max_peaks=None,
):
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (windows, channels), got {scores.shape}")
    if temporal_radius < 0:
        raise ValueError("temporal_radius must be nonnegative")
    ids_np = np.asarray(neighborhood_ids, dtype=np.int64)
    if ids_np.ndim != 2 or ids_np.shape[0] != scores.shape[1]:
        raise ValueError("neighborhood ids and score channels are inconsistent")
    safe_ids = torch.as_tensor(
        np.maximum(ids_np, 0), dtype=torch.long, device=scores.device
    )
    neighbor_mask = torch.as_tensor(ids_np >= 0, device=scores.device)
    temporal_max = torch_functional.max_pool1d(
        scores.T[None],
        kernel_size=2 * temporal_radius + 1,
        stride=1,
        padding=temporal_radius,
    )[0].T
    spatial_max = temporal_max[:, safe_ids].masked_fill(
        ~neighbor_mask[None], float("-inf")
    ).amax(dim=2)
    windows, channels = torch.nonzero(
        (scores >= threshold) & (scores >= spatial_max), as_tuple=True
    )
    selected_scores = scores[windows, channels]
    windows_np = windows.to("cpu").numpy()
    channels_np = channels.to("cpu").numpy()
    selected_scores_np = selected_scores.to("cpu").numpy()
    order = np.argsort(-selected_scores_np, kind="stable")
    if max_peaks is not None:
        order = order[:max_peaks]
    return (
        (windows_np[order] + n_before).astype(np.int64),
        channels_np[order].astype(np.int32),
        selected_scores_np[order].astype(np.float32),
    )


def extract_waveforms(
    data,
    times,
    channels,
    neighborhood_ids,
    channel_local_coords,
    n_before,
    n_after,
):
    width = neighborhood_ids.shape[1]
    length = n_before + n_after
    waveforms = np.zeros((len(times), width, length), dtype=np.float32)
    ids = neighborhood_ids[channels]
    mask = ids >= 0
    for index, (time, row_ids, row_mask) in enumerate(zip(times, ids, mask)):
        waveforms[index, row_mask] = data[
            time - n_before:time + n_after, row_ids[row_mask]
        ].T
    return waveforms, ids, channel_local_coords[channels], mask


def subtract_predictions(residual, times, ids, mask, prediction, n_before, n_after):
    for time, row_ids, row_mask, model in zip(times, ids, mask, prediction):
        residual[
            time - n_before:time + n_after, row_ids[row_mask]
        ] -= model[row_mask].T


def subtract_predictions_monotone(
    residual,
    times,
    ids,
    mask,
    prediction,
    n_before,
    n_after,
    min_captured_fraction=0.0,
):
    accepted = np.zeros(len(times), dtype=bool)
    scale = np.zeros(len(times), dtype=np.float32)
    input_energy = np.zeros(len(times), dtype=np.float32)
    captured_energy = np.zeros(len(times), dtype=np.float32)
    captured_fraction = np.zeros(len(times), dtype=np.float32)
    waveforms = np.zeros_like(prediction, dtype=np.float32)
    tiny = np.finfo(np.float32).tiny
    for index, (time, row_ids, row_mask, model) in enumerate(
        zip(times, ids, mask, prediction)
    ):
        current = residual[
            time - n_before:time + n_after, row_ids[row_mask]
        ].T
        atom = model[row_mask]
        model_energy = float(np.square(atom).sum())
        energy = float(np.square(current).sum())
        if not np.isfinite(model_energy) or not np.isfinite(energy) or model_energy <= tiny:
            continue
        response = float(np.sum(current * atom))
        fitted_scale = response / model_energy
        adjusted = fitted_scale * atom
        candidate = current - adjusted
        reduction = energy - float(np.square(candidate).sum())
        fraction = reduction / max(energy, tiny)
        if (
            not np.isfinite(fitted_scale)
            or not np.isfinite(reduction)
            or reduction <= 0
            or fraction < min_captured_fraction
        ):
            continue
        waveforms[index, row_mask] = current
        residual[
            time - n_before:time + n_after, row_ids[row_mask]
        ] = candidate.T
        accepted[index] = True
        scale[index] = fitted_scale
        input_energy[index] = energy
        captured_energy[index] = reduction
        captured_fraction[index] = fraction
    return {
        "accepted": accepted,
        "scale": scale,
        "input_energy": input_energy,
        "captured_energy": captured_energy,
        "captured_fraction": captured_fraction,
        "waveforms": waveforms,
    }


def _empty_result(width, waveform_length, save_waveforms):
    result = {
        "spike_times": np.empty(0, dtype=np.int64),
        "spike_channels": np.empty(0, dtype=np.int32),
        "sources": np.empty((0, 3), dtype=np.float32),
        "sources_grid": np.empty((0, 3), dtype=np.float32),
        "global_sources": np.empty((0, 3), dtype=np.float32),
        "centroids": np.empty((0, 2), dtype=np.float32),
        "neighbor_ids": np.empty((0, width), dtype=np.int32),
        "neighbor_counts": np.empty(0, dtype=np.int16),
        "local_coords": np.empty((0, width, 2), dtype=np.float32),
        "profile_idx": np.empty(0, dtype=np.int16),
        "temporal_idx": np.empty(0, dtype=np.int16),
        "alpha": np.empty(0, dtype=np.float32),
        "detection_score": np.empty(0, dtype=np.float32),
        "input_energy": np.empty(0, dtype=np.float32),
        "captured_energy": np.empty(0, dtype=np.float32),
        "captured_fraction": np.empty(0, dtype=np.float32),
        "continuous_displacement_um": np.empty(0, dtype=np.float32),
        "continuous_energy_gain": np.empty(0, dtype=np.float32),
        "pass_energy_drop_fraction": np.empty(0, dtype=np.float32),
        "residual_pass": np.empty(0, dtype=np.int8),
    }
    if save_waveforms:
        result["residual_waveforms"] = np.empty(
            (0, width, waveform_length), dtype=np.float32
        )
    return result


def _concatenate_event_parts(parts, width, waveform_length, save_waveforms):
    if not parts:
        return _empty_result(width, waveform_length, save_waveforms)
    keys = parts[0]
    result = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    order = np.lexsort((result["residual_pass"], result["spike_times"]))
    return {key: value[order] for key, value in result.items()}


def peel_preprocessed_chunk(
    data,
    global_start,
    core_start,
    core_stop,
    channel_positions,
    neighborhood_ids,
    channel_local_coords,
    channel_centroids,
    neighbor_counts,
    spatial_neighbors,
    detection_footprints,
    omega,
    fs,
    config,
    coarse_footprint_cache=None,
):
    n_before = int(round(config.ms_before * fs / 1000))
    n_after = int(round(config.ms_after * fs / 1000))
    waveform_length = n_before + n_after
    if omega.shape[1] != waveform_length:
        raise ValueError(
            f"omega has {omega.shape[1]} samples but extraction uses {waveform_length}"
        )
    temporal_radius = max(1, int(round(config.temporal_radius_ms * fs / 1000)))
    noise = robust_channel_noise(data)
    residual = np.array(data, dtype=np.float32, copy=True)
    parts = []
    kernels = tuple(part.strip() for part in config.kernel.split(","))
    if coarse_footprint_cache is None:
        coarse_footprint_cache = {}

    def profile_start():
        if not config.profile_stages:
            return None
        if torch.device(config.device).type == "cuda":
            torch.cuda.synchronize(torch.device(config.device))
        return perf_counter()

    def profile_stop(timings, stage, started):
        if started is None:
            return
        if torch.device(config.device).type == "cuda":
            torch.cuda.synchronize(torch.device(config.device))
        timings[stage] += perf_counter() - started

    for residual_pass in range(config.max_residual_passes):
        pass_started = profile_start()
        timings = {
            "pass_setup": 0.0,
            "template_scoring": 0.0,
            "peak_selection": 0.0,
            "waveform_extraction": 0.0,
            "localization": 0.0,
            "subtraction": 0.0,
            "result_assembly": 0.0,
            "pass_finalize": 0.0,
        }
        started = profile_start()
        residual_before_pass = residual.copy()
        pass_energy_before = float(
            np.square(residual[core_start:core_stop], dtype=np.float64).sum()
        )
        profile_stop(timings, "pass_setup", started)
        started = profile_start()
        template_scores = full_template_scores(
            residual,
            noise,
            omega,
            detection_footprints,
            neighborhood_ids,
            device=config.device,
            time_batch=config.template_time_batch,
            return_torch=True,
        )
        profile_stop(timings, "template_scoring", started)
        started = profile_start()
        times, channels, detection_score = select_template_peaks_torch(
            template_scores,
            neighborhood_ids,
            threshold=config.threshold,
            temporal_radius=temporal_radius,
            n_before=n_before,
            max_peaks=config.max_peaks_per_round,
        )
        profile_stop(timings, "peak_selection", started)
        if not len(times):
            if config.profile_stages:
                total = perf_counter() - pass_started
                detail = " ".join(
                    f"{name}={seconds:.3f}s" for name, seconds in timings.items()
                )
                print(
                    f"profile pass {residual_pass + 1}: {detail} total={total:.3f}s",
                    flush=True,
                )
            break
        pass_parts = []
        accepted_for_subtraction = 0
        for start in range(0, len(times), config.fit_batch_size):
            stop = min(start + config.fit_batch_size, len(times))
            batch_times = times[start:stop]
            batch_channels = channels[start:stop]
            started = profile_start()
            waveforms, ids, local_coords, mask = extract_waveforms(
                residual,
                batch_times,
                batch_channels,
                neighborhood_ids,
                channel_local_coords,
                n_before,
                n_after,
            )
            profile_stop(timings, "waveform_extraction", started)
            started = profile_start()
            fit = localize_spikes_fixed_codebook(
                local_coords,
                waveforms,
                omega,
                kernels=kernels,
                n_scales=config.n_scales,
                n_sites=config.n_sites,
                refine_levels=config.refine_levels,
                continuous=config.continuous_refine,
                continuous_max_iterations=config.continuous_max_iterations,
                continuous_backtracks=config.continuous_backtracks,
                device=config.device,
                mask=mask,
                coarse_footprint_cache=coarse_footprint_cache,
                config_batch_size=config.localization_config_batch_size,
            )
            profile_stop(timings, "localization", started)
            captured_fraction = fit["captured_energy"] / np.maximum(
                fit["input_energy"], np.finfo(np.float32).tiny
            )
            accepted = (
                np.isfinite(captured_fraction)
                & np.isfinite(fit["alpha"])
                & (fit["captured_energy"] > 0)
                & (captured_fraction >= config.min_captured_fraction)
            )
            if not np.any(accepted):
                continue
            selected = np.flatnonzero(accepted)
            started = profile_start()
            subtraction = subtract_predictions_monotone(
                residual,
                batch_times[selected],
                ids[selected],
                mask[selected],
                fit["prediction"][selected],
                n_before,
                n_after,
                min_captured_fraction=config.min_captured_fraction,
            )
            profile_stop(timings, "subtraction", started)
            accepted[:] = False
            accepted[selected[subtraction["accepted"]]] = True
            if not np.any(accepted):
                continue
            started = profile_start()
            accepted_for_subtraction += int(accepted.sum())
            fitted_alpha = np.asarray(fit["alpha"]).copy()
            fitted_input_energy = np.zeros(len(fitted_alpha), dtype=np.float32)
            fitted_captured_energy = np.zeros(len(fitted_alpha), dtype=np.float32)
            fitted_captured_fraction = np.zeros(len(fitted_alpha), dtype=np.float32)
            fitted_waveforms = np.zeros_like(waveforms, dtype=np.float32)
            fitted_alpha[selected] *= subtraction["scale"]
            fitted_input_energy[selected] = subtraction["input_energy"]
            fitted_captured_energy[selected] = subtraction["captured_energy"]
            fitted_captured_fraction[selected] = subtraction["captured_fraction"]
            fitted_waveforms[selected] = subtraction["waveforms"]
            in_core = (
                (batch_times >= core_start)
                & (batch_times < core_stop)
                & accepted
            )
            if not np.any(in_core):
                continue
            anchor = batch_channels[in_core]
            sources = fit["sources"][in_core].astype(np.float32)
            sources_grid = fit["sources_grid"][in_core].astype(np.float32)
            centroids = channel_centroids[anchor]
            global_sources = np.column_stack(
                (centroids + sources[:, :2], sources[:, 2])
            ).astype(np.float32)
            part = {
                "spike_times": (
                    global_start + batch_times[in_core]
                ).astype(np.int64),
                "spike_channels": anchor.astype(np.int32),
                "sources": sources,
                "sources_grid": sources_grid,
                "global_sources": global_sources,
                "centroids": centroids.astype(np.float32),
                "neighbor_ids": ids[in_core].astype(np.int32),
                "neighbor_counts": neighbor_counts[anchor].astype(np.int16),
                "local_coords": local_coords[in_core].astype(np.float32),
                "profile_idx": fit["profile_idx"][in_core].astype(np.int16),
                "temporal_idx": fit["temporal_idx"][in_core].astype(np.int16),
                "alpha": fitted_alpha[in_core].astype(np.float32),
                "detection_score": detection_score[start:stop][in_core].astype(
                    np.float32
                ),
                "input_energy": fitted_input_energy[in_core],
                "captured_energy": fitted_captured_energy[in_core],
                "captured_fraction": fitted_captured_fraction[in_core],
                "continuous_displacement_um": fit[
                    "continuous_displacement_um"
                ][in_core].astype(np.float32),
                "continuous_energy_gain": fit["continuous_energy_gain"][
                    in_core
                ].astype(np.float32),
                "pass_energy_drop_fraction": np.zeros(
                    in_core.sum(), dtype=np.float32
                ),
                "residual_pass": np.full(
                    in_core.sum(), residual_pass, dtype=np.int8
                ),
            }
            if config.save_waveforms:
                part["residual_waveforms"] = fitted_waveforms[in_core]
            pass_parts.append(part)
            profile_stop(timings, "result_assembly", started)
        if accepted_for_subtraction == 0:
            break
        started = profile_start()
        pass_energy_after = float(
            np.square(residual[core_start:core_stop], dtype=np.float64).sum()
        )
        pass_energy_drop_fraction = (
            pass_energy_before - pass_energy_after
        ) / max(pass_energy_before, np.finfo(np.float64).tiny)
        if pass_energy_drop_fraction < config.min_pass_energy_drop_fraction:
            residual[...] = residual_before_pass
            print(
                f"residual pass {residual_pass + 1}: rollback "
                f"{accepted_for_subtraction} fits; energy drop "
                f"{pass_energy_drop_fraction:.6f} < "
                f"{config.min_pass_energy_drop_fraction:.6f}",
                flush=True,
            )
            break
        for part in pass_parts:
            part["pass_energy_drop_fraction"].fill(pass_energy_drop_fraction)
        parts.extend(pass_parts)
        profile_stop(timings, "pass_finalize", started)
        print(
            f"residual pass {residual_pass + 1}: accepted "
            f"{accepted_for_subtraction} fits; energy drop "
            f"{pass_energy_drop_fraction:.6f}",
            flush=True,
        )
        if config.profile_stages:
            total = perf_counter() - pass_started
            detail = " ".join(
                f"{name}={seconds:.3f}s" for name, seconds in timings.items()
            )
            print(
                f"profile pass {residual_pass + 1}: {detail} total={total:.3f}s",
                flush=True,
            )

    return _concatenate_event_parts(
        parts, neighborhood_ids.shape[1], waveform_length, config.save_waveforms
    )


def load_omega(path):
    path = Path(path)
    if path.suffix == ".npy":
        omega = np.load(path)
    else:
        with np.load(path, allow_pickle=False) as archive:
            if "omega" not in archive.files:
                raise KeyError(f"{path} does not contain an omega array")
            omega = archive["omega"]
    omega = np.asarray(omega, dtype=np.float32)
    if omega.ndim != 2 or not np.isfinite(omega).all():
        raise ValueError(f"omega must be a finite two-dimensional array, got {omega.shape}")
    norm = np.linalg.norm(omega, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("omega contains an all-zero row")
    return omega / norm


def _write_chunk(path, result):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **result)
    temporary.replace(path)


def consolidate_chunks(chunk_dir, output_path, save_waveforms):
    chunk_paths = sorted(chunk_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise RuntimeError("no completed chunks found")
    keys = None
    arrays = {}
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as archive:
            current = [key for key in archive.files if key != "residual_waveforms"]
            if keys is None:
                keys = current
                arrays = {key: [] for key in keys}
            elif current != keys:
                raise RuntimeError(f"inconsistent fields in {path}")
            for key in keys:
                arrays[key].append(archive[key])
    count = 0
    for key, pieces in arrays.items():
        combined = np.concatenate(pieces)
        np.save(output_path / f"{key}.npy", combined)
        if key == "spike_times":
            count = len(combined)
    summary = {
        "n_events": count,
        "n_chunks": len(chunk_paths),
        "waveforms": "sharded in chunks" if save_waveforms else "not saved",
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_recording(
    recording_path,
    omega_path,
    output_path,
    config,
    start_seconds=0.0,
    duration_seconds=None,
    resume=False,
):
    import spikeglx

    recording_path = Path(recording_path)
    output_path = Path(output_path)
    if output_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_path / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    omega = load_omega(omega_path)

    reader = spikeglx.Reader(recording_path)
    try:
        fs = float(reader.fs)
        n_channels = len(reader.geometry["x"])
        channel_positions = np.column_stack(
            (reader.geometry["x"], reader.geometry["y"])
        ).astype(np.float32)
        (
            neighborhood_ids,
            channel_local_coords,
            channel_centroids,
            neighbor_counts,
            spatial_neighbors,
        ) = channel_neighborhoods(channel_positions, config.radius_um)
        anchor_xy = channel_positions - channel_centroids
        detection_footprints, detection_profiles = (
            build_codebook_detection_footprints(
                channel_local_coords,
                neighborhood_ids >= 0,
                anchor_xy,
                kernels=tuple(
                    part.strip() for part in config.kernel.split(",")
                ),
                n_scales=config.n_scales,
                device="cpu",
            )
        )
        n_before = int(round(config.ms_before * fs / 1000))
        n_after = int(round(config.ms_after * fs / 1000))
        if omega.shape[1] != n_before + n_after:
            raise ValueError(
                f"omega length {omega.shape[1]} does not match the "
                f"{n_before + n_after}-sample extraction window"
            )
        first_sample = max(0, int(round(start_seconds * fs)))
        requested_stop = reader.ns
        if duration_seconds is not None:
            requested_stop = min(
                requested_stop,
                first_sample + int(round(duration_seconds * fs)),
            )
        chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
        margin = max(
            int(round(config.read_margin_ms * fs / 1000)),
            n_before + n_after,
            128,
        )
        metadata = {
            "recording_path": str(recording_path.resolve()),
            "omega_path": str(Path(omega_path).resolve()),
            "fs": fs,
            "n_channels": n_channels,
            "n_detection_spatial_profiles": len(detection_profiles),
            "first_sample": first_sample,
            "stop_sample": requested_stop,
            "config": asdict(config),
        }
        (output_path / "config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        np.save(output_path / "omega.npy", omega)
        np.save(output_path / "channel_positions.npy", channel_positions)
        np.save(output_path / "detection_footprints.npy", detection_footprints)

        starts = list(range(first_sample, requested_stop, chunk_samples))
        coarse_footprint_cache = {}
        for chunk_index, core_global_start in enumerate(starts):
            chunk_path = chunk_dir / f"chunk_{chunk_index:06d}.npz"
            if resume and chunk_path.exists():
                print(f"chunk {chunk_index + 1}/{len(starts)} already complete", flush=True)
                continue
            core_global_stop = min(core_global_start + chunk_samples, requested_stop)
            read_start = max(0, core_global_start - margin)
            read_stop = min(reader.ns, core_global_stop + margin)
            raw = reader[read_start:read_stop, :n_channels]
            data = preprocess_voltage(
                raw,
                fs,
                freq_min=config.freq_min,
                freq_max=config.freq_max,
                order=config.filter_order,
            )
            result = peel_preprocessed_chunk(
                data,
                read_start,
                core_global_start - read_start,
                core_global_stop - read_start,
                channel_positions,
                neighborhood_ids,
                channel_local_coords,
                channel_centroids,
                neighbor_counts,
                spatial_neighbors,
                detection_footprints,
                omega,
                fs,
                config,
                coarse_footprint_cache=coarse_footprint_cache,
            )
            _write_chunk(chunk_path, result)
            print(
                f"chunk {chunk_index + 1}/{len(starts)} "
                f"samples [{core_global_start}, {core_global_stop}) "
                f"events={len(result['spike_times'])}",
                flush=True,
            )
    finally:
        reader.close()
    summary = consolidate_chunks(chunk_dir, output_path, config.save_waveforms)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", type=Path)
    parser.add_argument("omega_fit", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--freq-min", type=float, default=300.0)
    parser.add_argument("--freq-max", type=float, default=6000.0)
    parser.add_argument("--radius-um", type=float, default=48.0)
    parser.add_argument("--ms-before", type=float, default=1.5)
    parser.add_argument("--ms-after", type=float, default=1.5)
    parser.add_argument("--temporal-radius-ms", type=float, default=0.5)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--read-margin-ms", type=float, default=20.0)
    parser.add_argument("--max-residual-passes", type=int, default=4)
    parser.add_argument("--min-captured-fraction", type=float, default=0.05)
    parser.add_argument("--min-pass-energy-drop-fraction", type=float, default=0.01)
    parser.add_argument("--max-peaks-per-round", type=int, default=10000)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument("--localization-config-batch-size", type=int, default=32)
    parser.add_argument("--template-time-batch", type=int, default=4096)
    parser.add_argument("--kernel", default="monopole")
    parser.add_argument("--n-scales", type=int, default=10)
    parser.add_argument("--n-sites", type=int, default=16)
    parser.add_argument("--refine-levels", type=int, default=6)
    parser.add_argument(
        "--no-continuous-refine", action="store_false", dest="continuous_refine"
    )
    parser.add_argument("--continuous-max-iterations", type=int, default=80)
    parser.add_argument("--continuous-backtracks", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-waveforms", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = ResidualConfig(
        threshold=args.threshold,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        radius_um=args.radius_um,
        ms_before=args.ms_before,
        ms_after=args.ms_after,
        temporal_radius_ms=args.temporal_radius_ms,
        chunk_seconds=args.chunk_seconds,
        read_margin_ms=args.read_margin_ms,
        max_residual_passes=args.max_residual_passes,
        min_captured_fraction=args.min_captured_fraction,
        min_pass_energy_drop_fraction=args.min_pass_energy_drop_fraction,
        max_peaks_per_round=args.max_peaks_per_round,
        fit_batch_size=args.fit_batch_size,
        localization_config_batch_size=args.localization_config_batch_size,
        template_time_batch=args.template_time_batch,
        kernel=args.kernel,
        n_scales=args.n_scales,
        n_sites=args.n_sites,
        refine_levels=args.refine_levels,
        continuous_refine=args.continuous_refine,
        continuous_max_iterations=args.continuous_max_iterations,
        continuous_backtracks=args.continuous_backtracks,
        device=args.device,
        save_waveforms=args.save_waveforms,
        profile_stages=args.profile_stages,
    )
    run_recording(
        args.recording_path,
        args.omega_fit,
        args.output_path,
        config,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
