"""Standalone temporal-codebook residual pursuit for SpikeGLX recordings."""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
from time import perf_counter

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn.functional as F


MAD_SCALE = 0.6744897501960817
EPS = 1e-12
XYZ_LO = (-150.0, -150.0, 1.0)
XYZ_HI = (150.0, 150.0, 300.0)


@dataclass(frozen=True)
class Config:
    q: int = 8
    threshold: float = 6.0
    freq_min: float = 300.0
    freq_max: float = 6000.0
    filter_order: int = 3
    radius_um: float = 48.0
    merge_radius_um: float = 48.0
    ms_before: float = 1.5
    ms_after: float = 1.5
    merge_ms: float = 0.5
    chunk_seconds: float = 4.0
    read_margin_ms: float = 20.0
    outer_passes: int = 4
    n_scales: int = 9
    sigma_min_um: float = 2.0
    sigma_max_um: float = 512.0
    lattice_size: int = 16
    refine_levels: int = 6
    fit_batch_size: int = 1024
    site_block_size: int = 64
    template_time_batch: int = 2048
    max_events_per_pass: int = 40000
    max_channel_normalized_rmse: float = 3.0
    min_captured_fraction: float = 0.05
    min_fitted_projection: float = 0.0
    cross_pass_lockout_ms: float = 0.0
    min_pass_energy_drop_fraction: float = 0.0
    spatial_score: str = "max-channel-rmse"
    codebook_chunks: int = 32
    codebook_max_events: int = 100000
    codebook_events_per_chunk: int = 4096
    codebook_isolation_ms: float = 1.0
    codebook_iterations: int = 10
    codebook_tolerance: float = 1e-5
    codebook_assignment_batch_size: int = 65536
    seed: int = 42
    device: str = "cuda"
    save_waveforms: bool = False


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npy(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def atomic_npz(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def robust_channel_noise(data):
    data = np.asarray(data, dtype=np.float32)
    centered = data - np.median(data, axis=0, keepdims=True)
    noise = np.median(np.abs(centered), axis=0) / MAD_SCALE
    positive = noise[np.isfinite(noise) & (noise > 0)]
    floor = float(np.median(positive) * 1e-3) if len(positive) else 1.0
    return np.maximum(noise, floor).astype(np.float32)


def make_filter(fs, config):
    if not 0 < config.freq_min < config.freq_max < fs / 2:
        raise ValueError("filter frequencies must satisfy 0 < min < max < Nyquist")
    return butter(
        config.filter_order,
        (config.freq_min, config.freq_max),
        btype="bandpass",
        fs=fs,
        output="sos",
    )


def preprocess_voltage(raw, sos):
    data = np.asarray(raw, dtype=np.float32)
    if data.ndim != 2 or len(data) < 2:
        raise ValueError(f"raw data must have shape (time, channels), got {data.shape}")
    padlen = min(3 * (2 * len(sos) + 1), len(data) - 1)
    filtered = sosfiltfilt(sos, data, axis=0, padlen=padlen).astype(
        np.float32, copy=False
    )
    filtered -= np.median(filtered, axis=1, keepdims=True)
    return filtered


def build_neighborhoods(channel_positions, radius_um):
    positions = np.asarray(channel_positions, dtype=np.float32)
    distance = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=2
    )
    rows = [np.flatnonzero(distance[channel] <= radius_um) for channel in range(len(positions))]
    width = max(map(len, rows))
    ids = np.full((len(positions), width), -1, dtype=np.int32)
    offsets = np.zeros((len(positions), width, 2), dtype=np.float32)
    counts = np.empty(len(positions), dtype=np.int16)
    for channel, neighbors in enumerate(rows):
        counts[channel] = len(neighbors)
        ids[channel, : len(neighbors)] = neighbors
        offsets[channel, : len(neighbors)] = positions[neighbors] - positions[channel]
    return ids, offsets, counts


def sigma_bank(config):
    return np.geomspace(
        config.sigma_min_um, config.sigma_max_um, config.n_scales
    ).astype(np.float32)


def coarse_lattice(config):
    if config.lattice_size < 2:
        raise ValueError("lattice size must be at least two")
    axes = [
        np.rint(np.linspace(lo, hi, config.lattice_size)).astype(np.float32)
        for lo, hi in zip(XYZ_LO, XYZ_HI)
    ]
    if any(len(np.unique(axis)) != config.lattice_size for axis in axes):
        raise ValueError("coarse lattice contains repeated integer sites")
    mesh = np.meshgrid(*axes, indexing="ij")
    sites = np.stack([part.ravel() for part in mesh], axis=1)
    return sites.astype(np.float32), axes


def monopole_footprint(offsets, sites, sigmas, mask):
    dxy2 = (
        (offsets[:, None, None, :, 0] - sites[None, :, None, None, 0]).square()
        + (offsets[:, None, None, :, 1] - sites[None, :, None, None, 1]).square()
    )
    sigma = sigmas[None, None, :, None]
    denominator = torch.sqrt(
        dxy2 + sites[None, :, None, None, 2].square() + sigma.square()
    )
    return sigma / denominator.clamp_min(EPS) * mask[:, None, None, :]


def detection_footprints(offsets, ids, sigmas, noise, device):
    mask = ids >= 0
    safe_ids = np.maximum(ids, 0)
    offsets_t = torch.as_tensor(offsets, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
    sigma_t = torch.as_tensor(sigmas, device=device)
    dxy2 = offsets_t.square().sum(dim=2)
    raw = sigma_t[None, :, None] / torch.sqrt(
        dxy2[:, None, :] + sigma_t[None, :, None].square()
    ).clamp_min(EPS)
    local_noise = torch.as_tensor(noise[safe_ids], device=device)
    weighted = raw * mask_t[:, None, :] / local_noise[:, None, :]
    weighted = weighted / weighted.square().sum(dim=2, keepdim=True).sqrt().clamp_min(EPS)
    return weighted, torch.as_tensor(safe_ids, dtype=torch.long, device=device)


def gpu_neighborhood(ids, device):
    return (
        torch.as_tensor(np.maximum(ids, 0), dtype=torch.long, device=device),
        torch.as_tensor(ids >= 0, device=device),
    )


def spatiotemporal_nms(
    scores,
    neighborhood_ids,
    threshold,
    temporal_radius,
    valid_start,
    valid_stop,
    max_events,
):
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (time, channels), got {scores.shape}")
    if isinstance(neighborhood_ids, tuple):
        safe_ids, valid_neighbors = neighborhood_ids
    else:
        safe_ids, valid_neighbors = gpu_neighborhood(neighborhood_ids, scores.device)
    temporal = F.max_pool1d(
        scores.T[None],
        kernel_size=2 * temporal_radius + 1,
        stride=1,
        padding=temporal_radius,
    )[0].T
    spatial = temporal[:, safe_ids].masked_fill(
        ~valid_neighbors[None], float("-inf")
    ).amax(dim=2)
    candidate = (scores >= threshold) & (scores >= spatial)
    if valid_start > 0:
        candidate[:valid_start] = False
    if valid_stop < len(candidate):
        candidate[valid_stop:] = False
    times, channels = torch.nonzero(candidate, as_tuple=True)
    selected_scores = scores[times, channels]
    if max_events is not None and len(selected_scores) > max_events:
        selected_scores, order = torch.topk(
            selected_scores, max_events, largest=True, sorted=False
        )
        times = times[order]
        channels = channels[order]
    order = torch.argsort(times, stable=True)
    return times[order], channels[order], selected_scores[order]


def raw_negative_peaks(data, noise, merge_ids, threshold, radius, device):
    score = -torch.as_tensor(data / noise[None], device=device)
    return spatiotemporal_nms(
        score, merge_ids, threshold, radius, 0, len(data), None
    )


def isolated_events(times, channels, merge_ids, radius_samples):
    times = np.asarray(times, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int32)
    if radius_samples <= 0 or not len(times):
        return np.ones(len(times), dtype=bool)
    channel_times = [np.sort(times[channels == channel]) for channel in range(len(merge_ids))]
    isolated = np.ones(len(times), dtype=bool)
    for row, (time, channel) in enumerate(zip(times, channels)):
        count = 0
        for neighbor in merge_ids[channel]:
            if neighbor < 0:
                continue
            values = channel_times[neighbor]
            count += np.searchsorted(values, time + radius_samples, side="right")
            count -= np.searchsorted(values, time - radius_samples, side="left")
            if count > 1:
                isolated[row] = False
                break
    return isolated


def assign_codebook(values, omega, batch_size):
    labels = torch.empty(len(values), dtype=torch.long, device=values.device)
    response = torch.empty(len(values), dtype=values.dtype, device=values.device)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        scores = values[start:stop] @ omega.T
        _, selected = scores.abs().max(dim=1)
        labels[start:stop] = selected
        response[start:stop] = scores.gather(1, selected[:, None]).squeeze(1)
    return labels, response


def fit_codebook(waveforms, config):
    values = torch.as_tensor(waveforms, dtype=torch.float32, device=config.device)
    values -= values.mean(dim=1, keepdim=True)
    energy = values.square().sum(dim=1)
    valid = torch.isfinite(energy) & (energy > torch.finfo(values.dtype).tiny)
    values = values[valid]
    energy = energy[valid]
    if len(values) < config.q:
        raise RuntimeError(f"only {len(values)} valid waveforms are available for Q={config.q}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    initial = torch.randperm(len(values), generator=generator)[: config.q].to(values.device)
    omega = F.normalize(values[initial], dim=1)
    history = []
    for iteration in range(1, config.codebook_iterations + 1):
        labels, response = assign_codebook(
            values, omega, config.codebook_assignment_batch_size
        )
        numerator = torch.zeros_like(omega)
        numerator.index_add_(0, labels, response[:, None] * values)
        counts = torch.bincount(labels, minlength=config.q)
        updated = omega.clone()
        used = counts > 0
        updated[used] = F.normalize(numerator[used], dim=1)
        if not bool(used.all()):
            worst = torch.argsort(response.abs())
            cursor = 0
            for row in torch.nonzero(~used, as_tuple=False).flatten().tolist():
                updated[row] = F.normalize(values[worst[cursor]][None], dim=1)[0]
                cursor += 1
        alignment = (updated * omega).sum(dim=1)
        updated[alignment < 0] *= -1
        change = torch.linalg.vector_norm(updated - omega, dim=1)
        nmse = torch.clamp(energy - response.square(), min=0).sum() / energy.sum()
        step = {
            "iteration": iteration,
            "nmse": float(nmse.item()),
            "maximum_row_change": float(change.max().item()),
            "row_counts": counts.to("cpu").tolist(),
        }
        history.append(step)
        print(
            f"codebook iteration {iteration}: nMSE={step['nmse']:.6f} "
            f"max_change={step['maximum_row_change']:.6f}",
            flush=True,
        )
        omega = updated
        if step["maximum_row_change"] < config.codebook_tolerance:
            break
    center = omega.shape[1] // 2
    omega[omega[:, center] > 0] *= -1
    return omega.to("cpu").numpy().astype(np.float32), history


def collect_codebook_waveforms(
    reader, first_sample, stop_sample, fs, n_channels, sos, merge_ids, config
):
    n_before = int(round(config.ms_before * fs / 1000))
    n_after = int(round(config.ms_after * fs / 1000))
    chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
    margin = max(
        int(round(config.read_margin_ms * fs / 1000)), n_before + n_after, 128
    )
    starts = np.arange(first_sample, stop_sample, chunk_samples, dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    selected_chunks = rng.permutation(len(starts))[: min(config.codebook_chunks, len(starts))]
    isolation = max(0, int(round(config.codebook_isolation_ms * fs / 1000)))
    peak_radius = max(1, int(round(config.merge_ms * fs / 1000)))
    offsets = np.arange(-n_before, n_after, dtype=np.int64)
    pieces = []
    total = 0
    for scan_number, chunk_index in enumerate(selected_chunks, start=1):
        core_start = int(starts[chunk_index])
        core_stop = min(core_start + chunk_samples, stop_sample)
        read_start = max(0, core_start - margin)
        read_stop = min(reader.ns, core_stop + margin)
        data = preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
        noise = robust_channel_noise(data)
        times_t, channels_t, _ = raw_negative_peaks(
            data, noise, merge_ids, config.threshold, peak_radius, config.device
        )
        times = times_t.to("cpu").numpy()
        channels = channels_t.to("cpu").numpy()
        keep = (
            (times >= core_start - read_start)
            & (times < core_stop - read_start)
            & (times >= n_before)
            & (times + n_after <= len(data))
        )
        keep &= isolated_events(times, channels, merge_ids, isolation)
        times = times[keep]
        channels = channels[keep]
        remaining = config.codebook_max_events - total
        take = min(len(times), config.codebook_events_per_chunk, remaining)
        if take:
            chosen = np.sort(rng.choice(len(times), take, replace=False))
            waveforms = data[
                times[chosen, None] + offsets[None], channels[chosen, None]
            ]
            pieces.append(np.asarray(waveforms, dtype=np.float32))
            total += take
        print(
            f"codebook scan {scan_number}/{len(selected_chunks)} sampled={total:,}",
            flush=True,
        )
        if total >= config.codebook_max_events:
            break
    if total < config.q:
        raise RuntimeError(f"codebook scan produced only {total} waveforms for Q={config.q}")
    return np.concatenate(pieces, axis=0)


def full_template_scores(residual, noise, omega, footprints, safe_ids, config):
    n_samples, n_channels = residual.shape
    omega_t = F.normalize(omega, dim=1)
    standardized = (residual / noise[None]).T[None]
    weights = omega_t.repeat(n_channels, 1).unsqueeze(1)
    projection = F.conv1d(standardized, weights, groups=n_channels)[0]
    n_windows = projection.shape[1]
    projection = projection.reshape(n_channels, len(omega), n_windows).permute(2, 0, 1)
    scores = torch.empty((n_windows, n_channels), device=config.device)
    choices = torch.empty(
        (n_windows, n_channels), dtype=torch.int16, device=config.device
    )
    for start in range(0, n_windows, config.template_time_batch):
        stop = min(start + config.template_time_batch, n_windows)
        local = projection[start:stop, safe_ids]
        response = torch.einsum("tacq,asc->tasq", local, footprints)
        flat = response.abs().flatten(2)
        values, selected = flat.max(dim=2)
        scores[start:stop] = values
        choices[start:stop] = selected.to(torch.int16)
    return scores, choices


def detect_events(
    residual,
    noise,
    omega,
    footprints,
    safe_fit_ids,
    merge_neighborhood,
    config,
    fs,
    valid_start,
    valid_stop,
):
    scores, choices = full_template_scores(
        residual, noise, omega, footprints, safe_fit_ids, config
    )
    n_before = int(round(config.ms_before * fs / 1000))
    temporal_radius = max(1, int(round(config.merge_ms * fs / 1000)))
    window_start = max(0, valid_start - n_before)
    window_stop = min(len(scores), valid_stop - n_before)
    windows, channels, selected_scores = spatiotemporal_nms(
        scores,
        merge_neighborhood,
        config.threshold,
        temporal_radius,
        window_start,
        window_stop,
        config.max_events_per_pass,
    )
    initial = choices[windows, channels].long()
    initial_sigma = initial // omega.shape[0]
    initial_temporal = initial % omega.shape[0]
    return (
        windows + n_before,
        channels,
        selected_scores,
        initial_sigma,
        initial_temporal,
    )


def exclude_prior_detections(detected, prior_times, prior_channels, neighborhood,
                             temporal_radius, n_samples, n_channels):
    if not len(prior_times) or temporal_radius <= 0:
        return detected
    times, channels, *values = detected
    occupied = torch.zeros((n_samples, n_channels), dtype=torch.float32, device=times.device)
    occupied[prior_times, prior_channels] = 1
    temporal = F.max_pool1d(
        occupied.T[None], kernel_size=2 * temporal_radius + 1,
        stride=1, padding=temporal_radius,
    )[0].T.bool()
    safe_ids, valid = neighborhood
    blocked = temporal[:, safe_ids].masked_fill(~valid[None], False).any(dim=2)
    keep = ~blocked[times, channels]
    return (times[keep], channels[keep], *(value[keep] for value in values))


def extract_waveforms_torch(
    residual,
    times,
    channels,
    safe_fit_ids,
    fit_mask,
    offsets,
    noise,
    n_before,
    n_after,
):
    safe_ids = safe_fit_ids[channels]
    mask = fit_mask[channels]
    time_offsets = torch.arange(-n_before, n_after, device=residual.device)
    sample_index = times[:, None, None] + time_offsets[None, None, :]
    channel_index = safe_ids[:, :, None]
    waveforms = residual[sample_index, channel_index]
    waveforms = waveforms.masked_fill(~mask[:, :, None], 0)
    local_noise = noise[safe_ids]
    local_offsets = offsets[channels]
    return waveforms, safe_ids, local_offsets, mask, local_noise


def score_candidates(
    projected,
    channel_energy,
    offsets,
    mask,
    local_noise,
    sites,
    sigmas,
    spatial_score,
):
    raw = monopole_footprint(offsets, sites, sigmas, mask)
    weighted = raw / local_noise[:, None, None, :]
    response_channel = weighted[..., None] * projected[:, None, None, :, :]
    response = response_channel.sum(dim=3)
    denominator_channel = weighted.square()
    denominator = denominator_channel.sum(dim=3).clamp_min(EPS)
    alpha = response / denominator[..., None]
    channel_sse = (
        channel_energy[:, None, None, :, None]
        - 2 * alpha[..., None, :] * response_channel
        + alpha[..., None, :].square() * denominator_channel[..., None]
    ).clamp_min(0)
    valid = mask[:, None, None, :, None]
    if spatial_score == "max-channel-rmse":
        objective = channel_sse.masked_fill(~valid, float("-inf")).amax(dim=3)
    elif spatial_score == "mean-channel-rmse":
        count = mask.sum(dim=1).clamp_min(1)[:, None, None, None]
        objective = channel_sse.masked_fill(~valid, 0).sum(dim=3) / count
    else:
        raise ValueError(f"unknown spatial score {spatial_score!r}")
    return objective, alpha, raw


def score_event_candidates(
    projected,
    channel_energy,
    offsets,
    mask,
    local_noise,
    sites,
    sigmas,
    spatial_score,
):
    dxy2 = (
        (offsets[:, None, None, :, 0] - sites[:, :, None, None, 0]).square()
        + (offsets[:, None, None, :, 1] - sites[:, :, None, None, 1]).square()
    )
    sigma = sigmas[None, None, :, None]
    raw = sigma / torch.sqrt(
        dxy2 + sites[:, :, None, None, 2].square() + sigma.square()
    ).clamp_min(EPS)
    raw *= mask[:, None, None, :]
    weighted = raw / local_noise[:, None, None, :]
    response_channel = weighted[..., None] * projected[:, None, None, :, :]
    response = response_channel.sum(dim=3)
    denominator_channel = weighted.square()
    denominator = denominator_channel.sum(dim=3).clamp_min(EPS)
    alpha = response / denominator[..., None]
    channel_sse = (
        channel_energy[:, None, None, :, None]
        - 2 * alpha[..., None, :] * response_channel
        + alpha[..., None, :].square() * denominator_channel[..., None]
    ).clamp_min(0)
    valid = mask[:, None, None, :, None]
    if spatial_score == "max-channel-rmse":
        objective = channel_sse.masked_fill(~valid, float("-inf")).amax(dim=3)
    elif spatial_score == "mean-channel-rmse":
        count = mask.sum(dim=1).clamp_min(1)[:, None, None, None]
        objective = channel_sse.masked_fill(~valid, 0).sum(dim=3) / count
    else:
        raise ValueError(f"unknown spatial score {spatial_score!r}")
    return objective, alpha, raw


def choose_best_coarse(projected, channel_energy, offsets, mask, local_noise, sites, sigmas, config):
    n_events = len(projected)
    best_objective = torch.full((n_events,), float("inf"), device=projected.device)
    best_site = torch.zeros(n_events, dtype=torch.long, device=projected.device)
    best_sigma = torch.zeros(n_events, dtype=torch.long, device=projected.device)
    best_temporal = torch.zeros(n_events, dtype=torch.long, device=projected.device)
    best_alpha = torch.zeros(n_events, device=projected.device)
    for start in range(0, len(sites), config.site_block_size):
        stop = min(start + config.site_block_size, len(sites))
        objective, alpha, _ = score_candidates(
            projected,
            channel_energy,
            offsets,
            mask,
            local_noise,
            sites[start:stop],
            sigmas,
            config.spatial_score,
        )
        flat_objective = objective.flatten(1)
        block_value, flat = flat_objective.min(dim=1)
        update = block_value < best_objective
        profile_temporal = len(sigmas) * projected.shape[2]
        local_site = flat // profile_temporal
        remainder = flat % profile_temporal
        sigma_index = remainder // projected.shape[2]
        temporal_index = remainder % projected.shape[2]
        chosen_alpha = alpha[
            torch.arange(n_events, device=projected.device),
            local_site,
            sigma_index,
            temporal_index,
        ]
        best_objective = torch.where(update, block_value, best_objective)
        best_site = torch.where(update, local_site + start, best_site)
        best_sigma = torch.where(update, sigma_index, best_sigma)
        best_temporal = torch.where(update, temporal_index, best_temporal)
        best_alpha = torch.where(update, chosen_alpha, best_alpha)
    return best_objective, best_site, best_sigma, best_temporal, best_alpha


def refine_sites(
    projected,
    channel_energy,
    offsets,
    mask,
    local_noise,
    sites,
    axes,
    sigmas,
    site_index,
    config,
):
    n_events = len(projected)
    rows = torch.arange(n_events, device=projected.device)
    current = sites[site_index].clone()
    coarse = current.clone()
    grid_index = torch.stack(
        (
            site_index // (config.lattice_size * config.lattice_size),
            (site_index // config.lattice_size) % config.lattice_size,
            site_index % config.lattice_size,
        ),
        dim=1,
    )
    steps = []
    for dimension, axis in enumerate(axes):
        index = grid_index[:, dimension]
        left = axis[index] - axis[(index - 1).clamp_min(0)]
        right = axis[(index + 1).clamp_max(config.lattice_size - 1)] - axis[index]
        steps.append(torch.ceil(0.5 * torch.maximum(left, right)))
    step = torch.stack(steps, dim=1).clamp_min(1)
    delta = torch.cartesian_prod(
        torch.tensor([-1.0, 0.0, 1.0], device=projected.device),
        torch.tensor([-1.0, 0.0, 1.0], device=projected.device),
        torch.tensor([-1.0, 0.0, 1.0], device=projected.device),
    )
    best_sigma = torch.zeros(n_events, dtype=torch.long, device=projected.device)
    best_temporal = torch.zeros(n_events, dtype=torch.long, device=projected.device)
    best_alpha = torch.zeros(n_events, device=projected.device)
    best_objective = torch.full((n_events,), float("inf"), device=projected.device)
    levels = 0
    for _ in range(config.refine_levels):
        candidates = current[:, None, :] + step[:, None, :] * delta[None]
        for dimension, (lo, hi) in enumerate(zip(XYZ_LO, XYZ_HI)):
            candidates[:, :, dimension].clamp_(lo, hi)
        objective, alpha, _ = score_event_candidates(
            projected,
            channel_energy,
            offsets,
            mask,
            local_noise,
            candidates,
            sigmas,
            config.spatial_score,
        )
        flat_value, flat = objective.flatten(1).min(dim=1)
        profile_temporal = len(sigmas) * projected.shape[2]
        candidate_index = flat // profile_temporal
        remainder = flat % profile_temporal
        sigma_index = remainder // projected.shape[2]
        temporal_index = remainder % projected.shape[2]
        current = candidates[rows, candidate_index]
        best_objective = flat_value
        best_sigma = sigma_index
        best_temporal = temporal_index
        best_alpha = alpha[rows, candidate_index, sigma_index, temporal_index]
        levels += 1
        step = torch.floor(step / 2).clamp_min(1)
    return current, coarse, best_sigma, best_temporal, best_alpha, best_objective, levels


def fit_spatial_batch(waveforms, offsets, mask, local_noise, omega, sites, axes, sigmas, config):
    normalized = waveforms / local_noise[:, :, None]
    omega_t = F.normalize(torch.as_tensor(omega, device=waveforms.device), dim=1)
    projected = torch.einsum("nct,qt->ncq", normalized, omega_t)
    channel_energy = normalized.square().sum(dim=2)
    coarse_objective, site_index, _, _, _ = choose_best_coarse(
        projected,
        channel_energy,
        offsets,
        mask,
        local_noise,
        sites,
        sigmas,
        config,
    )
    source, coarse_source, sigma_index, temporal_index, alpha, objective, levels = refine_sites(
        projected,
        channel_energy,
        offsets,
        mask,
        local_noise,
        sites,
        axes,
        sigmas,
        site_index,
        config,
    )
    rows = torch.arange(len(waveforms), device=waveforms.device)
    selected_sigma = sigmas[sigma_index]
    dxy2 = (offsets - source[:, None, :2]).square().sum(dim=2)
    raw = selected_sigma[:, None] / torch.sqrt(
        dxy2 + source[:, 2, None].square() + selected_sigma[:, None].square()
    ).clamp_min(EPS)
    raw *= mask
    prediction = (
        alpha[:, None, None]
        * raw[:, :, None]
        * omega_t[temporal_index, None, :]
    )
    residual = waveforms - prediction
    channel_rmse = residual.square().mean(dim=2).sqrt()
    channel_normalized_rmse = channel_rmse / local_noise
    channel_normalized_rmse = channel_normalized_rmse.masked_fill(~mask, 0)
    maximum_channel_rmse = channel_normalized_rmse.masked_fill(
        ~mask, float("-inf")
    ).amax(dim=1)
    mean_channel_rmse = channel_normalized_rmse.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    input_energy = (normalized.square() * mask[:, :, None]).sum(dim=(1, 2))
    residual_energy = ((residual / local_noise[:, :, None]).square() * mask[:, :, None]).sum(dim=(1, 2))
    captured_energy = (input_energy - residual_energy).clamp_min(0)
    fitted_projection_score = torch.sqrt(captured_energy)
    captured_fraction = captured_energy / input_energy.clamp_min(EPS)
    rho = torch.sqrt(source[:, 2].square() + selected_sigma.square())
    return {
        "sources": source,
        "coarse_sources": coarse_source,
        "sigma_index": sigma_index,
        "sigma": selected_sigma,
        "rho": rho,
        "temporal_index": temporal_index,
        "alpha": alpha,
        "prediction": prediction,
        "channel_rmse": channel_rmse.masked_fill(~mask, 0),
        "channel_normalized_rmse": channel_normalized_rmse,
        "maximum_channel_normalized_rmse": maximum_channel_rmse,
        "mean_channel_normalized_rmse": mean_channel_rmse,
        "input_energy": input_energy,
        "captured_energy": captured_energy,
        "fitted_projection_score": fitted_projection_score,
        "captured_fraction": captured_fraction,
        "coarse_objective": coarse_objective,
        "objective": objective,
        "refinement_levels": torch.full(
            (len(waveforms),), levels, dtype=torch.uint8, device=waveforms.device
        ),
    }


def subtract_predictions(residual, times, ids, mask, prediction, n_before):
    time_offsets = torch.arange(prediction.shape[2], device=residual.device) - n_before
    sample_index = times[:, None, None] + time_offsets[None, None, :]
    channel_index = ids[:, :, None].expand_as(sample_index.expand(-1, ids.shape[1], -1))
    sample_index = sample_index.expand(-1, ids.shape[1], -1)
    valid = mask[:, :, None].expand_as(prediction)
    residual.index_put_(
        (sample_index[valid], channel_index[valid]),
        -prediction[valid],
        accumulate=True,
    )


def tensor_numpy(value, rows):
    return value[rows].detach().to("cpu").numpy()


def empty_chunk(width, waveform_length, save_waveforms):
    result = {
        "spike_times": np.empty(0, dtype=np.int64),
        "spike_channels": np.empty(0, dtype=np.int32),
        "sources": np.empty((0, 3), dtype=np.float32),
        "global_sources": np.empty((0, 3), dtype=np.float32),
        "coarse_sources": np.empty((0, 3), dtype=np.float32),
        "sigma_index": np.empty(0, dtype=np.int16),
        "sigma": np.empty(0, dtype=np.float32),
        "rho": np.empty(0, dtype=np.float32),
        "temporal_index": np.empty(0, dtype=np.int16),
        "alpha": np.empty(0, dtype=np.float32),
        "detection_score": np.empty(0, dtype=np.float32),
        "initial_sigma_index": np.empty(0, dtype=np.int16),
        "initial_temporal_index": np.empty(0, dtype=np.int16),
        "neighbor_ids": np.empty((0, width), dtype=np.int32),
        "neighbor_counts": np.empty(0, dtype=np.int16),
        "channel_rmse": np.empty((0, width), dtype=np.float32),
        "channel_normalized_rmse": np.empty((0, width), dtype=np.float32),
        "maximum_channel_normalized_rmse": np.empty(0, dtype=np.float32),
        "mean_channel_normalized_rmse": np.empty(0, dtype=np.float32),
        "input_energy": np.empty(0, dtype=np.float32),
        "captured_energy": np.empty(0, dtype=np.float32),
        "fitted_projection_score": np.empty(0, dtype=np.float32),
        "captured_fraction": np.empty(0, dtype=np.float32),
        "refinement_levels": np.empty(0, dtype=np.uint8),
        "residual_pass": np.empty(0, dtype=np.int8),
        "pass_energy_drop_fraction": np.empty(0, dtype=np.float32),
    }
    if save_waveforms:
        result["waveforms"] = np.empty((0, width, waveform_length), dtype=np.float32)
        result["predictions"] = np.empty((0, width, waveform_length), dtype=np.float32)
    return result


def concatenate_parts(parts, width, waveform_length, save_waveforms):
    if not parts:
        return empty_chunk(width, waveform_length, save_waveforms)
    keys = parts[0].keys()
    result = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    order = np.lexsort((result["residual_pass"], result["spike_times"]))
    return {key: value[order] for key, value in result.items()}


def process_chunk(
    data,
    read_start,
    core_start,
    core_stop,
    channel_positions,
    fit_ids,
    fit_offsets,
    fit_counts,
    merge_ids,
    omega,
    sites,
    axes,
    sigmas,
    fs,
    config,
):
    n_before = int(round(config.ms_before * fs / 1000))
    n_after = int(round(config.ms_after * fs / 1000))
    waveform_length = n_before + n_after
    residual = torch.as_tensor(data, dtype=torch.float32, device=config.device)
    noise_np = robust_channel_noise(data)
    noise = torch.as_tensor(noise_np, device=config.device)
    omega_t = F.normalize(torch.as_tensor(omega, device=config.device), dim=1)
    safe_fit_ids, fit_mask = gpu_neighborhood(fit_ids, config.device)
    merge_neighborhood = gpu_neighborhood(merge_ids, config.device)
    fit_offsets_t = torch.as_tensor(fit_offsets, device=config.device)
    detection_bank, safe_detection_ids = detection_footprints(
        fit_offsets, fit_ids, sigmas, noise_np, config.device
    )
    parts = []
    pass_summaries = []
    prior_times = torch.empty(0, dtype=torch.long, device=config.device)
    prior_channels = torch.empty(0, dtype=torch.long, device=config.device)
    local_core_start = core_start - read_start
    local_core_stop = core_stop - read_start
    valid_start = max(n_before, local_core_start - n_after)
    valid_stop = min(len(data) - n_after + 1, local_core_stop + n_before)
    for residual_pass in range(config.outer_passes):
        started = perf_counter()
        before = residual.clone()
        energy_before = residual[local_core_start:local_core_stop].square().sum()
        detected = detect_events(
            residual,
            noise,
            omega_t,
            detection_bank,
            safe_detection_ids,
            merge_neighborhood,
            config,
            fs,
            valid_start,
            valid_stop,
        )
        detected = exclude_prior_detections(
            detected, prior_times, prior_channels, merge_neighborhood,
            int(round(config.cross_pass_lockout_ms * fs / 1000)),
            len(residual), residual.shape[1],
        )
        times, channels, detection_score, initial_sigma, initial_temporal = detected
        batch_results = []
        for start in range(0, len(times), config.fit_batch_size):
            stop = min(start + config.fit_batch_size, len(times))
            batch_times = times[start:stop]
            batch_channels = channels[start:stop]
            extracted = extract_waveforms_torch(
                residual,
                batch_times,
                batch_channels,
                safe_fit_ids,
                fit_mask,
                fit_offsets_t,
                noise,
                n_before,
                n_after,
            )
            waveforms, ids, local_offsets, mask, local_noise = extracted
            fit = fit_spatial_batch(
                waveforms,
                local_offsets,
                mask,
                local_noise,
                omega_t,
                sites,
                axes,
                sigmas,
                config,
            )
            accepted = (
                torch.isfinite(fit["alpha"])
                & torch.isfinite(fit["maximum_channel_normalized_rmse"])
                & (fit["maximum_channel_normalized_rmse"] <= config.max_channel_normalized_rmse)
                & (fit["captured_fraction"] >= config.min_captured_fraction)
                & (fit["fitted_projection_score"] >= config.min_fitted_projection)
            )
            batch_results.append(
                {
                    "times": batch_times,
                    "channels": batch_channels,
                    "detection_score": detection_score[start:stop],
                    "initial_sigma": initial_sigma[start:stop],
                    "initial_temporal": initial_temporal[start:stop],
                    "waveforms": waveforms,
                    "ids": ids,
                    "mask": mask,
                    "fit": fit,
                    "accepted": accepted,
                }
            )
        for batch in batch_results:
            selected = batch["accepted"]
            subtract_predictions(
                residual,
                batch["times"][selected],
                batch["ids"][selected],
                batch["mask"][selected],
                batch["fit"]["prediction"][selected],
                n_before,
            )
        energy_after = residual[local_core_start:local_core_stop].square().sum()
        drop = float(((energy_before - energy_after) / energy_before.clamp_min(EPS)).item())
        accepted_count = int(
            torch.stack([batch["accepted"].sum() for batch in batch_results]).sum().item()
        ) if batch_results else 0
        rolled_back = accepted_count == 0 or drop <= config.min_pass_energy_drop_fraction
        if rolled_back:
            residual.copy_(before)
        else:
            for batch in batch_results:
                in_core = (
                    batch["accepted"]
                    & (batch["times"] >= local_core_start)
                    & (batch["times"] < local_core_stop)
                )
                if not bool(in_core.any()):
                    continue
                fit = batch["fit"]
                anchor = batch["channels"][in_core]
                sources = tensor_numpy(fit["sources"], in_core).astype(np.float32)
                anchors = channel_positions[anchor.to("cpu").numpy()]
                global_sources = np.column_stack(
                    (anchors + sources[:, :2], sources[:, 2])
                ).astype(np.float32)
                count = len(sources)
                part = {
                    "spike_times": (read_start + tensor_numpy(batch["times"], in_core)).astype(np.int64),
                    "spike_channels": anchor.detach().to("cpu").numpy().astype(np.int32),
                    "sources": sources,
                    "global_sources": global_sources,
                    "coarse_sources": tensor_numpy(fit["coarse_sources"], in_core).astype(np.float32),
                    "sigma_index": tensor_numpy(fit["sigma_index"], in_core).astype(np.int16),
                    "sigma": tensor_numpy(fit["sigma"], in_core).astype(np.float32),
                    "rho": tensor_numpy(fit["rho"], in_core).astype(np.float32),
                    "temporal_index": tensor_numpy(fit["temporal_index"], in_core).astype(np.int16),
                    "alpha": tensor_numpy(fit["alpha"], in_core).astype(np.float32),
                    "detection_score": tensor_numpy(batch["detection_score"], in_core).astype(np.float32),
                    "initial_sigma_index": tensor_numpy(batch["initial_sigma"], in_core).astype(np.int16),
                    "initial_temporal_index": tensor_numpy(batch["initial_temporal"], in_core).astype(np.int16),
                    "neighbor_ids": fit_ids[anchor.to("cpu").numpy()].astype(np.int32),
                    "neighbor_counts": fit_counts[anchor.to("cpu").numpy()].astype(np.int16),
                    "channel_rmse": tensor_numpy(fit["channel_rmse"], in_core).astype(np.float32),
                    "channel_normalized_rmse": tensor_numpy(fit["channel_normalized_rmse"], in_core).astype(np.float32),
                    "maximum_channel_normalized_rmse": tensor_numpy(fit["maximum_channel_normalized_rmse"], in_core).astype(np.float32),
                    "mean_channel_normalized_rmse": tensor_numpy(fit["mean_channel_normalized_rmse"], in_core).astype(np.float32),
                    "input_energy": tensor_numpy(fit["input_energy"], in_core).astype(np.float32),
                    "captured_energy": tensor_numpy(fit["captured_energy"], in_core).astype(np.float32),
                    "fitted_projection_score": tensor_numpy(fit["fitted_projection_score"], in_core).astype(np.float32),
                    "captured_fraction": tensor_numpy(fit["captured_fraction"], in_core).astype(np.float32),
                    "refinement_levels": tensor_numpy(fit["refinement_levels"], in_core).astype(np.uint8),
                    "residual_pass": np.full(count, residual_pass, dtype=np.int8),
                    "pass_energy_drop_fraction": np.full(count, drop, dtype=np.float32),
                }
                if config.save_waveforms:
                    part["waveforms"] = tensor_numpy(batch["waveforms"], in_core).astype(np.float32)
                    part["predictions"] = tensor_numpy(fit["prediction"], in_core).astype(np.float32)
                parts.append(part)
            prior_times = torch.cat([
                prior_times,
                *[batch["times"][batch["accepted"]] for batch in batch_results],
            ])
            prior_channels = torch.cat([
                prior_channels,
                *[batch["channels"][batch["accepted"]] for batch in batch_results],
            ])
        pass_summary = {
            "pass": residual_pass,
            "proposed": int(len(times)),
            "accepted": accepted_count,
            "energy_before": float(energy_before.item()),
            "energy_after": float(energy_after.item()),
            "energy_drop_fraction": drop,
            "rolled_back": rolled_back,
            "seconds": perf_counter() - started,
        }
        pass_summaries.append(pass_summary)
        print(json.dumps(pass_summary), flush=True)
        if rolled_back:
            break
    result = concatenate_parts(parts, fit_ids.shape[1], waveform_length, config.save_waveforms)
    result["noise"] = noise_np
    result["pass_summaries_json"] = np.asarray(json.dumps(pass_summaries))
    return result


def validate_config(config):
    if config.q < 1 or config.outer_passes < 1 or config.n_scales < 1:
        raise ValueError("Q, outer passes, and number of scales must be positive")
    if config.fit_batch_size < 1 or config.site_block_size < 1:
        raise ValueError("fit and site block sizes must be positive")
    if config.min_fitted_projection < 0 or config.cross_pass_lockout_ms < 0:
        raise ValueError("fitted-projection threshold and cross-pass lockout must be nonnegative")
    if config.sigma_min_um <= 0 or config.sigma_max_um < config.sigma_min_um:
        raise ValueError("sigma bounds must be positive and ordered")
    if config.spatial_score not in ("max-channel-rmse", "mean-channel-rmse"):
        raise ValueError("unknown spatial score")
    if config.codebook_max_events < config.q:
        raise ValueError("codebook max events must be at least Q")
    if torch.device(config.device).type != "cuda":
        raise ValueError("recording extraction is CUDA-only; CPU is supported only by --self-test")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def load_or_fit_codebook(reader, output_path, first_sample, stop_sample, fs, n_channels, sos, merge_ids, config, resume):
    omega_path = output_path / "omega.npy"
    history_path = output_path / "codebook_history.json"
    if resume and omega_path.exists():
        omega = np.asarray(np.load(omega_path), dtype=np.float32)
        waveform_length = int(round(config.ms_before * fs / 1000)) + int(
            round(config.ms_after * fs / 1000)
        )
        if omega.shape != (config.q, waveform_length):
            raise ValueError(f"saved codebook has incompatible shape {omega.shape}")
        return omega
    if resume and any((output_path / "chunks").glob("chunk_*.npz")):
        raise RuntimeError("cannot resume saved chunks without their omega.npy")
    waveforms = collect_codebook_waveforms(
        reader,
        first_sample,
        stop_sample,
        fs,
        n_channels,
        sos,
        merge_ids,
        config,
    )
    omega, history = fit_codebook(waveforms, config)
    atomic_npy(omega_path, omega)
    atomic_json(
        history_path,
        {
            "sampled_waveforms": int(len(waveforms)),
            "q": config.q,
            "history": history,
        },
    )
    return omega


def summarize_chunks(chunk_dir, output_path):
    paths = sorted(chunk_dir.glob("chunk_*.npz"))
    event_count = 0
    pass_counts = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            passes = archive["residual_pass"]
            event_count += len(passes)
            unique, counts = np.unique(passes, return_counts=True)
            for value, count in zip(unique, counts):
                key = str(int(value))
                pass_counts[key] = pass_counts.get(key, 0) + int(count)
    summary = {
        "completed_chunks": len(paths),
        "events": event_count,
        "events_by_pass": pass_counts,
        "storage": "chunk-sharded",
    }
    atomic_json(output_path / "summary.json", summary)
    return summary


def run_recording(recording_path, output_path, config, start_seconds, duration_seconds, resume):
    import spikeglx

    validate_config(config)
    recording_path = Path(recording_path)
    output_path = Path(output_path)
    if output_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_path / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"received signal {signum}; stopping after the current chunk", flush=True)

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    reader = spikeglx.Reader(recording_path)
    try:
        fs = float(reader.fs)
        n_channels = len(reader.geometry["x"])
        positions = np.column_stack(
            (reader.geometry["x"], reader.geometry["y"])
        ).astype(np.float32)
        fit_ids, fit_offsets, fit_counts = build_neighborhoods(positions, config.radius_um)
        merge_ids, _, _ = build_neighborhoods(positions, config.merge_radius_um)
        sos = make_filter(fs, config)
        first_sample = max(0, int(round(start_seconds * fs)))
        stop_sample = reader.ns
        if duration_seconds is not None:
            stop_sample = min(
                stop_sample, first_sample + int(round(duration_seconds * fs))
            )
        if stop_sample <= first_sample:
            raise ValueError("requested recording interval is empty")
        omega = load_or_fit_codebook(
            reader,
            output_path,
            first_sample,
            stop_sample,
            fs,
            n_channels,
            sos,
            merge_ids,
            config,
            resume,
        )
        sites_np, axes_np = coarse_lattice(config)
        sites = torch.as_tensor(sites_np, device=config.device)
        axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
        sigmas = torch.as_tensor(sigma_bank(config), device=config.device)
        metadata = {
            "recording_path": str(recording_path.resolve()),
            "fs": fs,
            "n_channels": n_channels,
            "first_sample": first_sample,
            "stop_sample": stop_sample,
            "config": asdict(config),
            "sigma_values_um": sigmas.to("cpu").tolist(),
            "lattice_bounds_um": [list(XYZ_LO), list(XYZ_HI)],
            "model": "alpha * monopole(x, y, z, sigma) * Omega[q]",
            "identifiability": "z and sigma are saved with rho=sqrt(z^2+sigma^2); only rho is identifiable for this model",
        }
        config_path = output_path / "config.json"
        if resume and config_path.exists():
            saved = json.loads(config_path.read_text())
            comparable = (
                "recording_path",
                "fs",
                "n_channels",
                "first_sample",
                "stop_sample",
                "config",
                "sigma_values_um",
                "lattice_bounds_um",
                "model",
            )
            mismatched = [key for key in comparable if saved.get(key) != metadata.get(key)]
            if mismatched:
                raise ValueError(
                    "resume configuration differs in: " + ", ".join(mismatched)
                )
        atomic_json(config_path, metadata)
        atomic_npy(output_path / "channel_positions.npy", positions)
        atomic_npy(output_path / "fit_neighborhood_ids.npy", fit_ids)
        atomic_npy(output_path / "fit_neighborhood_offsets.npy", fit_offsets)
        atomic_npy(output_path / "merge_neighborhood_ids.npy", merge_ids)
        chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
        n_before = int(round(config.ms_before * fs / 1000))
        n_after = int(round(config.ms_after * fs / 1000))
        margin = max(
            int(round(config.read_margin_ms * fs / 1000)),
            n_before + n_after + int(round(config.merge_ms * fs / 1000)),
            128,
        )
        starts = list(range(first_sample, stop_sample, chunk_samples))
        for chunk_index, core_start in enumerate(starts):
            chunk_path = chunk_dir / f"chunk_{chunk_index:06d}.npz"
            if resume and chunk_path.exists():
                print(f"chunk {chunk_index + 1}/{len(starts)} already complete", flush=True)
                continue
            core_stop = min(core_start + chunk_samples, stop_sample)
            read_start = max(0, core_start - margin)
            read_stop = min(reader.ns, core_stop + margin)
            data = preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
            with torch.inference_mode():
                result = process_chunk(
                    data,
                    read_start,
                    core_start,
                    core_stop,
                    positions,
                    fit_ids,
                    fit_offsets,
                    fit_counts,
                    merge_ids,
                    omega,
                    sites,
                    axes,
                    sigmas,
                    fs,
                    config,
                )
            atomic_npz(chunk_path, result)
            print(
                f"chunk {chunk_index + 1}/{len(starts)} "
                f"samples [{core_start}, {core_stop}) events={len(result['spike_times'])}",
                flush=True,
            )
            if stop_requested:
                break
    finally:
        reader.close()
    summary = summarize_chunks(chunk_dir, output_path)
    print(json.dumps(summary, indent=2), flush=True)


def self_test(device):
    config = Config(
        q=2,
        n_scales=4,
        sigma_min_um=4,
        sigma_max_um=32,
        lattice_size=4,
        refine_levels=6,
        fit_batch_size=4,
        site_block_size=16,
        device=device,
    )
    positions = np.asarray(
        [[0, 0], [20, 0], [-20, 0], [0, 20], [0, -20]], dtype=np.float32
    )
    ids, offsets_np, _ = build_neighborhoods(positions, 48)
    mask = torch.as_tensor(ids[:1] >= 0, device=device)
    offsets = torch.as_tensor(offsets_np[:1], device=device)
    local_noise = torch.ones((1, ids.shape[1]), device=device)
    omega = np.zeros((2, 12), dtype=np.float32)
    omega[0, 4:8] = np.asarray([-0.25, -1.0, -0.5, 0.25])
    omega[1, 3:9] = np.asarray([0.1, -0.3, -1.0, -0.6, 0.2, 0.1])
    omega /= np.linalg.norm(omega, axis=1, keepdims=True)
    sites_np, axes_np = coarse_lattice(config)
    sites = torch.as_tensor(sites_np, device=device)
    axes = [torch.as_tensor(axis, device=device) for axis in axes_np]
    sigmas = torch.as_tensor(sigma_bank(config), device=device)
    source = torch.tensor([[[10.0, -5.0, 41.0]]], device=device)
    sigma = torch.tensor([[16.0]], device=device)
    footprint = monopole_footprint(offsets, source[0], sigma[0], mask)[0, 0, 0]
    waveform = 7.0 * footprint[:, None] * torch.as_tensor(omega[1], device=device)[None]
    fit = fit_spatial_batch(
        waveform[None], offsets, mask, local_noise, omega, sites, axes, sigmas, config
    )
    relative_error = (
        (fit["prediction"] - waveform[None]).square().sum()
        / waveform.square().sum().clamp_min(EPS)
    )
    scores = torch.zeros((40, len(positions)), device=device)
    scores[20, 0] = 8
    scores[21, 1] = 7
    merged = spatiotemporal_nms(scores, ids, 6, 2, 0, 40, None)
    if len(merged[0]) != 1:
        raise AssertionError("adjacent channel/time proposals were not merged")
    if float(relative_error.item()) > 1e-3:
        raise AssertionError(f"synthetic reconstruction error is {float(relative_error.item())}")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "relative_reconstruction_error": float(relative_error.item()),
                "source": fit["sources"][0].to("cpu").tolist(),
                "sigma": float(fit["sigma"][0].item()),
                "rho": float(fit["rho"][0].item()),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", type=Path, nargs="?")
    parser.add_argument("output_path", type=Path, nargs="?")
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--freq-min", type=float, default=300.0)
    parser.add_argument("--freq-max", type=float, default=6000.0)
    parser.add_argument("--filter-order", type=int, default=3)
    parser.add_argument("--radius-um", type=float, default=48.0)
    parser.add_argument("--merge-radius-um", type=float, default=48.0)
    parser.add_argument("--ms-before", type=float, default=1.5)
    parser.add_argument("--ms-after", type=float, default=1.5)
    parser.add_argument("--merge-ms", type=float, default=0.5)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--read-margin-ms", type=float, default=20.0)
    parser.add_argument("--outer-passes", type=int, default=4)
    parser.add_argument("--n-scales", type=int, default=9)
    parser.add_argument("--sigma-min-um", type=float, default=2.0)
    parser.add_argument("--sigma-max-um", type=float, default=512.0)
    parser.add_argument("--lattice-size", type=int, default=16)
    parser.add_argument("--refine-levels", type=int, default=6)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument("--site-block-size", type=int, default=64)
    parser.add_argument("--template-time-batch", type=int, default=2048)
    parser.add_argument("--max-events-per-pass", type=int, default=40000)
    parser.add_argument("--max-channel-normalized-rmse", type=float, default=3.0)
    parser.add_argument("--min-captured-fraction", type=float, default=0.05)
    parser.add_argument("--min-pass-energy-drop-fraction", type=float, default=0.0)
    parser.add_argument(
        "--spatial-score",
        choices=("max-channel-rmse", "mean-channel-rmse"),
        default="max-channel-rmse",
    )
    parser.add_argument("--codebook-chunks", type=int, default=32)
    parser.add_argument("--codebook-max-events", type=int, default=100000)
    parser.add_argument("--codebook-events-per-chunk", type=int, default=4096)
    parser.add_argument("--codebook-isolation-ms", type=float, default=1.0)
    parser.add_argument("--codebook-iterations", type=int, default=10)
    parser.add_argument("--codebook-tolerance", type=float, default=1e-5)
    parser.add_argument("--codebook-assignment-batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-waveforms", action="store_true")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test(args.device)
        return
    if args.recording_path is None or args.output_path is None:
        raise SystemExit("recording_path and output_path are required unless --self-test is used")
    config = Config(
        q=args.q,
        threshold=args.threshold,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        filter_order=args.filter_order,
        radius_um=args.radius_um,
        merge_radius_um=args.merge_radius_um,
        ms_before=args.ms_before,
        ms_after=args.ms_after,
        merge_ms=args.merge_ms,
        chunk_seconds=args.chunk_seconds,
        read_margin_ms=args.read_margin_ms,
        outer_passes=args.outer_passes,
        n_scales=args.n_scales,
        sigma_min_um=args.sigma_min_um,
        sigma_max_um=args.sigma_max_um,
        lattice_size=args.lattice_size,
        refine_levels=args.refine_levels,
        fit_batch_size=args.fit_batch_size,
        site_block_size=args.site_block_size,
        template_time_batch=args.template_time_batch,
        max_events_per_pass=args.max_events_per_pass,
        max_channel_normalized_rmse=args.max_channel_normalized_rmse,
        min_captured_fraction=args.min_captured_fraction,
        min_pass_energy_drop_fraction=args.min_pass_energy_drop_fraction,
        spatial_score=args.spatial_score,
        codebook_chunks=args.codebook_chunks,
        codebook_max_events=args.codebook_max_events,
        codebook_events_per_chunk=args.codebook_events_per_chunk,
        codebook_isolation_ms=args.codebook_isolation_ms,
        codebook_iterations=args.codebook_iterations,
        codebook_tolerance=args.codebook_tolerance,
        codebook_assignment_batch_size=args.codebook_assignment_batch_size,
        seed=args.seed,
        device=args.device,
        save_waveforms=args.save_waveforms,
    )
    run_recording(
        args.recording_path,
        args.output_path,
        config,
        args.start_seconds,
        args.duration_seconds,
        args.resume,
    )


if __name__ == "__main__":
    main()
