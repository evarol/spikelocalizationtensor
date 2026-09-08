"""Merged all-channel one-hot peeling chain (0012-0019) for NWB recordings.

Standalone copy-merge of residuals_0012, 0014_xyzsig_residual,
0016_onehot_lattice_peeling, and 0019_allchannel_peeling into a single
namespace: definitions are concatenated in precedence order 0012 -> 0014 ->
0016 -> 0019 and later definitions shadow earlier ones, reproducing what
0019's runtime monkey-patching achieved.  Definitions whose names are
shadowed by a later level carry a ``_0012``/``_0014``/``_0016`` suffix and
callers reference the suffixed name directly.  The SpikeGLX reader is
replaced by ``NwbReader`` (session 0025); no spikeglx import exists anywhere
in this module.
"""

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
class Config_0012:
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


def atomic_json_0012(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npy_0012(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def atomic_npz_0012(path, values):
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


def detect_events_0012(
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


def score_event_candidates_0012(
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


def refine_sites_0012(
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
        objective, alpha, _ = score_event_candidates_0012(
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
    source, coarse_source, sigma_index, temporal_index, alpha, objective, levels = refine_sites_0012(
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


def empty_chunk_0012(width, waveform_length, save_waveforms):
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


def concatenate_parts_0012(parts, width, waveform_length, save_waveforms):
    if not parts:
        return empty_chunk_0012(width, waveform_length, save_waveforms)
    keys = parts[0].keys()
    result = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    order = np.lexsort((result["residual_pass"], result["spike_times"]))
    return {key: value[order] for key, value in result.items()}


def process_chunk_0012(
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
        detected = detect_events_0012(
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
    result = concatenate_parts_0012(parts, fit_ids.shape[1], waveform_length, config.save_waveforms)
    result["noise"] = noise_np
    result["pass_summaries_json"] = np.asarray(json.dumps(pass_summaries))
    return result


def validate_config_0012(config):
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
    atomic_npy_0012(omega_path, omega)
    atomic_json_0012(
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
    atomic_json_0012(output_path / "summary.json", summary)
    return summary


import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import signal

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config_0014:
    q: int = 8
    threshold: float = 9.0
    radius_um: float = 48.0
    merge_radius_um: float = 48.0
    ms_before: float = 1.5
    ms_after: float = 1.5
    merge_ms: float = 0.5
    chunk_seconds: float = 1.0
    read_margin_ms: float = 20.0
    outer_passes: int = 4
    n_scales: int = 9
    sigma_min_um: float = 2.0
    sigma_max_um: float = 512.0
    lattice_size: int = 16
    refine_levels: int = 6
    fit_batch_size: int = 2048
    site_block_size: int = 64
    template_time_batch: int = 2048
    max_events_per_pass: int = 40000
    max_channel_normalized_rmse: float = 3.0
    min_captured_fraction: float = 0.0
    min_fitted_projection: float = 9.0
    cross_pass_lockout_ms: float = 0.5
    min_pass_energy_drop_fraction: float = 0.0
    spatial_score: str = "max-channel-rmse"
    calibration_chunks: int = 32
    calibration_max_events: int = 100000
    calibration_events_per_chunk: int = 4096
    calibration_isolation_ms: float = 1.0
    alternating_iterations: int = 10
    alternating_tolerance: float = 1e-5
    seed: int = 42
    device: str = "cuda"
    save_waveforms: bool = True

    def base(self):
        return Config_0012(
            q=self.q, threshold=self.threshold, radius_um=self.radius_um,
            merge_radius_um=self.merge_radius_um, ms_before=self.ms_before,
            ms_after=self.ms_after, merge_ms=self.merge_ms,
            chunk_seconds=self.chunk_seconds, read_margin_ms=self.read_margin_ms,
            outer_passes=self.outer_passes, n_scales=self.n_scales,
            sigma_min_um=self.sigma_min_um, sigma_max_um=self.sigma_max_um,
            lattice_size=self.lattice_size, refine_levels=self.refine_levels,
            fit_batch_size=self.fit_batch_size, site_block_size=self.site_block_size,
            template_time_batch=self.template_time_batch,
            max_events_per_pass=self.max_events_per_pass,
            max_channel_normalized_rmse=self.max_channel_normalized_rmse,
            min_captured_fraction=self.min_captured_fraction,
            min_fitted_projection=self.min_fitted_projection,
            cross_pass_lockout_ms=self.cross_pass_lockout_ms,
            min_pass_energy_drop_fraction=self.min_pass_energy_drop_fraction,
            spatial_score=self.spatial_score, seed=self.seed, device=self.device,
            save_waveforms=self.save_waveforms,
        )


def atomic_json(path, value):
    atomic_json_0012(path, value)


def atomic_npy(path, value):
    atomic_npy_0012(path, value)


def atomic_npz(path, values):
    atomic_npz_0012(path, values)


def output_metadata_0014(config, recording_path, fs, n_channels, first, stop):
    values = asdict(config)
    values["kernel"] = "monopole"
    values["unnormalized_spatial_footprint"] = True
    return {
        "recording_path": str(recording_path.resolve()),
        "fs": fs,
        "n_channels": n_channels,
        "first_sample": first,
        "stop_sample": stop,
        "config": values,
        "sigma_values_um": sigma_bank(config.base()).tolist(),
        "lattice_bounds_um": [list(XYZ_LO), list(XYZ_HI)],
        "model": "alpha * monopole(x,y,z,sigma) * Omega[q]",
        "identifiability": "rho is diagnostic; discrete z and sigma are dictionary labels.",
    }


def consolidate_chunks(chunk_dir, output):
    paths = sorted(chunk_dir.glob("chunk_*.npz"))
    if not paths:
        raise RuntimeError("no completed chunks")
    excluded = {"noise", "pass_summaries_json", "residual_waveforms", "predictions"}
    with np.load(paths[0], allow_pickle=False) as archive:
        fields = [key for key in archive.files if key not in excluded and archive[key].ndim]
    total = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            total += len(archive["spike_times"])
            if any(key not in archive.files for key in fields):
                raise RuntimeError(f"incompatible chunk schema: {path}")
    arrays = {}
    cursor = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            count = len(archive["spike_times"])
            for key in fields:
                value = archive[key]
                if value.shape[0] != count:
                    raise RuntimeError(f"{path}:{key} is not event-aligned")
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(
                        output / f"{key}.npy", mode="w+", dtype=value.dtype,
                        shape=(total, *value.shape[1:]),
                    )
                arrays[key][cursor:cursor + count] = value
            cursor += count
    for array in arrays.values():
        array.flush()
    return {"n_events": total, "n_chunks": len(paths), "waveforms": "sharded in chunks"}


def geometry_key(offsets, mask):
    """Exact geometry key: no rounded coordinates and no Python hash randomization."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(offsets, dtype=np.float32).view(np.uint8))
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).view(np.uint8))
    return digest.hexdigest()


def grouped_rows(offsets, mask):
    groups = {}
    for row in range(len(offsets)):
        groups.setdefault(geometry_key(offsets[row], mask[row]), []).append(row)
    return [(key, np.asarray(rows, dtype=np.int64)) for key, rows in groups.items()]


class FootprintCache:
    """Immutable raw site-by-sigma dictionaries, one for each local geometry."""

    def __init__(self, sites, sigmas, device):
        self.sites = sites
        self.sigmas = sigmas
        self.device = device
        self.values = {}
        self.hits = 0
        self.misses = 0

    def get(self, key, offsets, mask):
        value = self.values.get(key)
        if value is None:
            off = torch.as_tensor(offsets[None], dtype=torch.float32, device=self.device)
            valid = torch.as_tensor(mask[None], dtype=torch.float32, device=self.device)
            value = monopole_footprint(off, self.sites, self.sigmas, valid)[0]
            self.values[key] = value
            self.misses += 1
        else:
            self.hits += 1
        return value

    def diagnostics(self):
        return {"entries": len(self.values), "hits": self.hits, "misses": self.misses}


def grouped_coarse_assignment(waveforms, offsets, mask, local_noise, omega, sites, sigmas,
                              config, cache):
    """Assign coarse candidates by geometry groups, retaining strict tie ordering."""
    device = waveforms.device
    n_events = len(waveforms)
    q = len(omega)
    normalized = waveforms / local_noise[:, :, None]
    projected = torch.einsum("nct,qt->ncq", normalized, omega)
    energy = (normalized.square() * mask[:, :, None]).sum(dim=(1, 2))
    result = [torch.empty(n_events, dtype=torch.long, device=device) for _ in range(3)]
    alpha_all = torch.empty(n_events, dtype=waveforms.dtype, device=device)
    # site, sigma, temporal, and alpha respectively
    for key, rows_np in grouped_rows(offsets.detach().cpu().numpy(), mask.detach().cpu().numpy()):
        rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
        raw = cache.get(key, offsets[rows[0]].detach().cpu().numpy(),
                        mask[rows[0]].detach().cpu().numpy())
        best = torch.full((len(rows),), float("-inf"), device=device)
        best_site = torch.zeros(len(rows), dtype=torch.long, device=device)
        best_sigma = torch.zeros_like(best_site)
        best_q = torch.zeros_like(best_site)
        best_alpha = torch.zeros(len(rows), dtype=waveforms.dtype, device=device)
        group_noise = local_noise[rows]
        group_projected = projected[rows]
        for start in range(0, len(sites), config.site_block_size):
            stop = min(start + config.site_block_size, len(sites))
            weighted = raw[start:stop][None] / group_noise[:, None, None, :]
            response = torch.einsum("bspc,bcq->bspq", weighted, group_projected)
            denominator = weighted.square().sum(dim=3).clamp_min(EPS)
            score = response.square() / denominator[..., None]
            value, flat = score.flatten(1).max(dim=1)
            update = value > best
            per_site = len(sigmas) * q
            local_site = flat // per_site
            rem = flat % per_site
            sigma_index = rem // q
            temporal_index = rem % q
            alpha = response[torch.arange(len(rows), device=device), local_site,
                             sigma_index, temporal_index] / denominator[
                                 torch.arange(len(rows), device=device), local_site, sigma_index]
            best = torch.where(update, value, best)
            best_site = torch.where(update, local_site + start, best_site)
            best_sigma = torch.where(update, sigma_index, best_sigma)
            best_q = torch.where(update, temporal_index, best_q)
            best_alpha = torch.where(update, alpha, best_alpha)
        result[0][rows] = best_site
        result[1][rows] = best_sigma
        result[2][rows] = best_q
        alpha_all[rows] = best_alpha
    return (*result, alpha_all, projected, energy)


def fit_grouped_0014(waveforms, offsets, mask, local_noise, omega, sites, axes, sigmas, config, cache):
    """Cached coarse scoring plus session-0012's ordered integer refinement."""
    omega = F.normalize(omega, dim=1)
    site_index, _, _, _, projected, channel_energy = grouped_coarse_assignment(
        waveforms, offsets, mask, local_noise, omega, sites, sigmas, config, cache)
    source, coarse, sigma_index, temporal_index, alpha, objective, levels = refine_sites_0012(
        projected, (waveforms / local_noise[:, :, None]).square().sum(dim=2), offsets,
        mask, local_noise, sites, axes, sigmas, site_index,
        config.base() if hasattr(config, "base") else config)
    selected_sigma = sigmas[sigma_index]
    dxy2 = (offsets - source[:, None, :2]).square().sum(dim=2)
    footprint = selected_sigma[:, None] / torch.sqrt(
        dxy2 + source[:, 2, None].square() + selected_sigma[:, None].square()).clamp_min(EPS)
    footprint *= mask
    prediction = alpha[:, None, None] * footprint[:, :, None] * omega[temporal_index, None, :]
    residual = waveforms - prediction
    channel_rmse = residual.square().mean(dim=2).sqrt().masked_fill(~mask, 0)
    normalized_rmse = (channel_rmse / local_noise).masked_fill(~mask, 0)
    maximum = normalized_rmse.masked_fill(~mask, float("-inf")).amax(dim=1)
    mean = normalized_rmse.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    input_energy = ((waveforms / local_noise[:, :, None]).square() * mask[:, :, None]).sum(dim=(1, 2))
    residual_energy = ((residual / local_noise[:, :, None]).square() * mask[:, :, None]).sum(dim=(1, 2))
    captured = (input_energy - residual_energy).clamp_min(0)
    fitted_projection_score = torch.sqrt(captured)
    return {"sources": source, "coarse_sources": coarse, "sigma_index": sigma_index,
            "sigma": selected_sigma, "rho": torch.sqrt(source[:, 2].square() + selected_sigma.square()),
            "temporal_index": temporal_index, "alpha": alpha, "prediction": prediction,
            "channel_rmse": channel_rmse, "channel_normalized_rmse": normalized_rmse,
            "maximum_channel_normalized_rmse": maximum, "mean_channel_normalized_rmse": mean,
            "input_energy": input_energy, "captured_energy": captured,
            "fitted_projection_score": fitted_projection_score,
            "captured_fraction": captured / input_energy.clamp_min(EPS), "objective": objective,
            "refinement_levels": torch.full((len(waveforms),), levels, dtype=torch.uint8, device=waveforms.device)}


def calibration_paths(output):
    root = Path(output) / "calibration"
    return root, root / "shards"


def calibration_detect_0014(reader, output, first, stop, offsets, fit_ids, merge_ids, sos, config, resume):
    root, shard_dir = calibration_paths(output)
    shard_dir.mkdir(parents=True, exist_ok=True)
    fs, n_channels = float(reader.fs), fit_ids.shape[0]
    before, after = (int(round(x * fs / 1000)) for x in (config.ms_before, config.ms_after))
    chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
    margin = max(int(round(config.read_margin_ms * fs / 1000)), before + after, 128)
    starts = np.arange(first, stop, chunk_samples, dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    chosen = np.sort(rng.permutation(len(starts))[:min(config.calibration_chunks, len(starts))])
    remaining = config.calibration_max_events
    isolation = int(round(config.calibration_isolation_ms * fs / 1000))
    peak_radius = max(1, int(round(config.merge_ms * fs / 1000)))
    total = 0
    for ordinal, index in enumerate(chosen):
        path = shard_dir / f"shard_{ordinal:03d}.npz"
        if resume and path.exists():
            with np.load(path) as saved:
                total += len(saved["spike_times"])
            continue
        core_start, core_stop = int(starts[index]), min(int(starts[index]) + chunk_samples, stop)
        read_start, read_stop = max(0, core_start - margin), min(reader.ns, core_stop + margin)
        data = preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
        noise = robust_channel_noise(data)
        times, channels, _ = raw_negative_peaks(data, noise, merge_ids, config.threshold,
                                                      peak_radius, config.device)
        times, channels = times.cpu().numpy(), channels.cpu().numpy()
        keep = ((times >= core_start - read_start) & (times < core_stop - read_start) &
                (times >= before) & (times + after <= len(data)))
        keep &= isolated_events(times, channels, merge_ids, isolation)
        times, channels = times[keep], channels[keep]
        take = min(remaining, config.calibration_events_per_chunk, len(times))
        if take:
            pick = np.sort(rng.choice(len(times), take, replace=False))
            times, channels = times[pick], channels[pick]
        else:
            times, channels = times[:0], channels[:0]
        masks = fit_ids[channels] >= 0
        atomic_npz(path, {"spike_times": (times + read_start).astype(np.int64),
                          "spike_channels": channels.astype(np.int32),
                          "local_offsets": offsets[channels],
                          "mask": masks, "noise": noise})
        total += len(times)
        remaining -= len(times)
        print(f"calibration shard {ordinal + 1}/{len(chosen)} events={total:,}", flush=True)
        if not remaining:
            break
    atomic_json(root / "detect.json", {"events": total, "shards": len(list(shard_dir.glob('*.npz'))),
                                        "seed": config.seed, "first_sample": first, "stop_sample": stop})


def iter_calibration_batches(reader, shard_dir, fs, fit_ids, sos, config):
    before, after = (int(round(x * fs / 1000)) for x in (config.ms_before, config.ms_after))
    sample_offsets = np.arange(-before, after, dtype=np.int64)
    n_channels = fit_ids.shape[0]
    for path in sorted(Path(shard_dir).glob("shard_*.npz")):
        with np.load(path) as shard:
            times, channels = shard["spike_times"], shard["spike_channels"]
        if not len(times):
            continue
        read_start, read_stop = max(0, int(times.min()) - before), min(reader.ns, int(times.max()) + after)
        data = preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
        noise = robust_channel_noise(data)
        safe = np.maximum(fit_ids[channels], 0)
        mask = fit_ids[channels] >= 0
        waveforms = data[times[:, None, None] - read_start + sample_offsets[None, None, :], safe[:, :, None]]
        waveforms *= mask[:, :, None]
        yield times, waveforms.astype(np.float32), channels, noise[safe].astype(np.float32), mask


def initial_omega(reader, shard_dir, fs, fit_ids, sos, config):
    picked = []
    for _, waveforms, _, _, _ in iter_calibration_batches(reader, shard_dir, fs, fit_ids, sos, config):
        values = waveforms.mean(axis=1)
        picked.append(values)
        if sum(map(len, picked)) >= config.q:
            break
    values = np.concatenate(picked) if picked else np.empty((0, 0), np.float32)
    if len(values) < config.q:
        raise RuntimeError(f"calibration contains {len(values)} events, fewer than Q={config.q}")
    rng = np.random.default_rng(config.seed)
    return F.normalize(torch.as_tensor(values[rng.permutation(len(values))[:config.q]], device=config.device), dim=1)


def alternating_fit_0014(reader, output, fs, fit_ids, offsets, sos, config, resume):
    root, shards = calibration_paths(output)
    omega_path, history_path = root / "omega.npy", root / "alternating_history.json"
    if resume and omega_path.exists() and history_path.exists():
        return np.load(omega_path).astype(np.float32)
    omega = initial_omega(reader, shards, fs, fit_ids, sos, config)
    sites_np, axes_np = coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(sigma_bank(config.base()), device=config.device)
    cache, history = FootprintCache(sites, sigmas, config.device), []
    assignment_root = root / "assignments"
    prior_objective = float("inf")
    for iteration in range(1, config.alternating_iterations + 1):
        numerator = torch.zeros_like(omega)
        denominator = torch.zeros(config.q, dtype=omega.dtype, device=config.device)
        counts = torch.zeros(config.q, dtype=torch.long, device=config.device)
        objective = 0.0
        iteration_dir = assignment_root / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        for shard_number, (times_np, waveforms_np, channels_np, noise_np, mask_np) in enumerate(
                iter_calibration_batches(reader, shards, fs, fit_ids, sos, config)):
            shard_parts = []
            for start in range(0, len(waveforms_np), config.fit_batch_size):
                stop = min(start + config.fit_batch_size, len(waveforms_np))
                channels = torch.as_tensor(channels_np[start:stop], dtype=torch.long, device=config.device)
                waveforms = torch.as_tensor(waveforms_np[start:stop], device=config.device)
                local_offsets = torch.as_tensor(offsets[channels_np[start:stop]], device=config.device)
                mask = torch.as_tensor(mask_np[start:stop], dtype=torch.bool, device=config.device)
                local_noise = torch.as_tensor(noise_np[start:stop], device=config.device)
                fit = fit_grouped_0014(waveforms, local_offsets, mask, local_noise, omega,
                                  sites, axes, sigmas, config, cache)
                labels = fit["temporal_index"]
                selected_sigma = fit["sigma"]
                dxy2 = (local_offsets - fit["sources"][:, None, :2]).square().sum(dim=2)
                footprint = selected_sigma[:, None] / torch.sqrt(
                    dxy2 + fit["sources"][:, 2, None].square() + selected_sigma[:, None].square()).clamp_min(EPS)
                spatial = fit["alpha"][:, None] * footprint * mask
                # The prediction is alpha * footprint * Omega; solve each temporal row in closed form.
                numerator.index_add_(0, labels, torch.einsum("bct,bc->bt", waveforms, spatial))
                denominator.index_add_(0, labels, spatial.square().sum(dim=1))
                counts += torch.bincount(labels, minlength=config.q)
                objective += float((waveforms - fit["prediction"]).square().sum().item())
                shard_parts.append({"spike_times": times_np[start:stop],
                                    "spike_channels": channels_np[start:stop],
                                    "site": fit["coarse_sources"].detach().cpu().numpy().astype(np.float32),
                                    "sigma_index": fit["sigma_index"].detach().cpu().numpy().astype(np.int16),
                                    "temporal_index": labels.detach().cpu().numpy().astype(np.int16),
                                    "alpha": fit["alpha"].detach().cpu().numpy().astype(np.float32)})
            if shard_parts:
                atomic_npz(iteration_dir / f"shard_{shard_number:03d}.npz",
                           {key: np.concatenate([part[key] for part in shard_parts]) for key in shard_parts[0]})
        updated = omega.clone()
        used = denominator > EPS
        updated[used] = F.normalize(numerator[used] / denominator[used, None], dim=1)
        alignment = (updated * omega).sum(dim=1)
        updated[alignment < 0] *= -1
        change = float(torch.linalg.vector_norm(updated - omega, dim=1).max().item())
        # Alternating least squares should not worsen this fixed-assignment SSE.  Keep the
        # prior dictionary if finite precision or the normalized gauge violates that invariant.
        accepted_update = objective <= prior_objective + 1e-5 * max(1.0, prior_objective)
        if accepted_update:
            omega, prior_objective = updated, objective
        history.append({"iteration": iteration, "objective": objective, "accepted_update": accepted_update,
                        "maximum_row_change": change, "row_counts": counts.cpu().tolist(),
                        "footprint_cache": cache.diagnostics()})
        atomic_json(history_path, history)
        atomic_npy(omega_path, omega.detach().cpu().numpy().astype(np.float32))
        if change < config.alternating_tolerance:
            break
    atomic_json(root / "footprint_cache.json", cache.diagnostics())
    return omega.detach().cpu().numpy().astype(np.float32)


def pursue_0014(reader, output, first, stop, positions, fit_ids, offsets, counts, merge_ids, sos,
           omega, config, resume):
    """Run one-second fresh residual chunks, replacing only the localizer with the cache."""
    fs, n_channels = float(reader.fs), len(positions)
    chunk_dir = Path(output) / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    sites_np, axes_np = coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(sigma_bank(config.base()), device=config.device)
    cache = FootprintCache(sites, sigmas, config.device)
    original = fit_spatial_batch
    def cached_fit(waveforms, local_offsets, mask, local_noise, omega_t, _sites, _axes, _sigmas, base_config):
        return fit_grouped_0014(waveforms, local_offsets, mask, local_noise, omega_t, sites, axes,
                           sigmas, base_config, cache)
    fit_spatial_batch = cached_fit
    try:
        before, after = (int(round(x * fs / 1000)) for x in (config.ms_before, config.ms_after))
        margin = max(int(round(config.read_margin_ms * fs / 1000)), before + after + int(round(config.merge_ms * fs / 1000)), 128)
        starts = list(range(first, stop, max(1, int(round(config.chunk_seconds * fs)))))
        for number, core_start in enumerate(starts):
            path = chunk_dir / f"chunk_{number:06d}.npz"
            if resume and path.exists():
                continue
            core_stop = min(core_start + max(1, int(round(config.chunk_seconds * fs))), stop)
            read_start, read_stop = max(0, core_start - margin), min(reader.ns, core_stop + margin)
            data = preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
            with torch.inference_mode():
                result = process_chunk_0012(data, read_start, core_start, core_stop, positions, fit_ids,
                                            offsets, counts, merge_ids, omega, sites, axes, sigmas, fs, config.base())
            anchors = result["spike_channels"]
            result["sources_grid"] = result["coarse_sources"]
            result["centroids"] = positions[anchors].astype(np.float32)
            result["local_coords"] = offsets[anchors].astype(np.float32)
            result["profile_idx"] = result["sigma_index"]
            result["temporal_idx"] = result["temporal_index"]
            result["continuous_displacement_um"] = np.zeros(len(anchors), dtype=np.float32)
            result["continuous_energy_gain"] = np.zeros(len(anchors), dtype=np.float32)
            if config.save_waveforms:
                result["residual_waveforms"] = result.pop("waveforms")
            atomic_npz(path, result)
            print(f"pursuit chunk {number + 1}/{len(starts)} events={len(result['spike_times'])}", flush=True)
    finally:
        fit_spatial_batch = original
    summary = consolidate_chunks(chunk_dir, Path(output))
    atomic_json(Path(output) / "summary.json", summary)
    atomic_json(Path(output) / "pursuit_footprint_cache.json", cache.diagnostics())


def validate_config_0014(config):
    if config.calibration_max_events < config.q or config.calibration_chunks < 1:
        raise ValueError("calibration limits must provide at least Q events and one chunk")
    validate_config_0012(config.base())


import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config_0016(Config_0014):
    peeling_rounds: int = 60
    event_merge_ms: float = 0.5
    min_improved_channels: int = 2
    min_channel_improvement: float = 0.0
    min_channel_improvement_fraction: float = 0.05
    min_raw_energy_drop: float = 0.0
    duplicate_temporal_correlation: float = 0.9
    positive_gain: bool = True
    omega_prior: str = ""
    empirical_null: bool = False
    null_shift_min_ms: float = 5.0
    null_shift_max_ms: float = 80.0
    null_seed: int = 16042


def output_metadata_0016(config, recording_path, fs, n_channels, first, stop):
    metadata = output_metadata_0014(config, recording_path, fs, n_channels, first, stop)
    metadata["config"] = asdict(config)
    metadata["config"]["kernel"] = "monopole"
    metadata["config"]["unnormalized_spatial_footprint"] = True
    metadata["model"] = (
        "alpha>=0 * monopole(x,y,z,sigma) * Omega[q], q one-hot, "
        "per-channel minimax lattice assignment"
    )
    metadata["detector"] = (
        "positive one-hot matched filter; global best hypothesis per full waveform support; "
        "subtract and rescore"
    )
    metadata["spatial_objective"] = (
        "closed-form gain followed by maximum per-channel noise-normalized SSE"
    )
    metadata["whitening"] = False
    metadata["empirical_null"] = config.empirical_null
    if config.empirical_null:
        metadata["null_construction"] = (
            "independent within-chunk channel shifts outside waveform support; "
            "the wrap seam is kept outside the scored core"
        )
    return metadata


def score_from_raw(
    projected,
    channel_energy,
    local_noise,
    mask,
    raw,
    spatial_score,
    positive_gain,
):
    weighted = raw[None] / local_noise[:, None, None, :]
    response_channel = weighted[..., None] * projected[:, None, None, :, :]
    response = response_channel.sum(dim=3)
    denominator_channel = weighted.square()
    denominator = denominator_channel.sum(dim=3).clamp_min(EPS)
    alpha = response / denominator[..., None]
    if positive_gain:
        alpha = alpha.clamp_min(0)
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
    return objective, alpha


def score_event_candidates(
    projected,
    channel_energy,
    offsets,
    mask,
    local_noise,
    sites,
    sigmas,
    config,
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
    if config.positive_gain:
        alpha = alpha.clamp_min(0)
    channel_sse = (
        channel_energy[:, None, None, :, None]
        - 2 * alpha[..., None, :] * response_channel
        + alpha[..., None, :].square() * denominator_channel[..., None]
    ).clamp_min(0)
    valid = mask[:, None, None, :, None]
    if config.spatial_score == "max-channel-rmse":
        objective = channel_sse.masked_fill(~valid, float("-inf")).amax(dim=3)
    elif config.spatial_score == "mean-channel-rmse":
        count = mask.sum(dim=1).clamp_min(1)[:, None, None, None]
        objective = channel_sse.masked_fill(~valid, 0).sum(dim=3) / count
    else:
        raise ValueError(f"unknown spatial score {config.spatial_score!r}")
    return objective, alpha


def coherent_coarse_assignment(
    waveforms,
    offsets,
    mask,
    local_noise,
    omega,
    sites,
    sigmas,
    config,
    cache,
):
    normalized = waveforms / local_noise[:, :, None]
    projected = torch.einsum("nct,qt->ncq", normalized, omega)
    channel_energy = normalized.square().sum(dim=2)
    n_events = len(waveforms)
    best_objective = torch.full(
        (n_events,), float("inf"), dtype=waveforms.dtype, device=waveforms.device
    )
    best_site = torch.zeros(n_events, dtype=torch.long, device=waveforms.device)
    offset_rows = offsets.detach().cpu().numpy()
    mask_rows = mask.detach().cpu().numpy()
    q = omega.shape[0]
    for key, rows_np in grouped_rows(offset_rows, mask_rows):
        rows = torch.as_tensor(rows_np, dtype=torch.long, device=waveforms.device)
        raw = cache.get(key, offset_rows[rows_np[0]], mask_rows[rows_np[0]])
        group_best = torch.full(
            (len(rows),), float("inf"), dtype=waveforms.dtype, device=waveforms.device
        )
        group_site = torch.zeros(len(rows), dtype=torch.long, device=waveforms.device)
        for start in range(0, len(sites), config.site_block_size):
            stop = min(start + config.site_block_size, len(sites))
            objective, _ = score_from_raw(
                projected[rows],
                channel_energy[rows],
                local_noise[rows],
                mask[rows],
                raw[start:stop],
                config.spatial_score,
                config.positive_gain,
            )
            value, flat = objective.flatten(1).min(dim=1)
            local_site = flat // (len(sigmas) * q)
            update = value < group_best
            group_best = torch.where(update, value, group_best)
            group_site = torch.where(update, local_site + start, group_site)
        best_objective[rows] = group_best
        best_site[rows] = group_site
    return best_objective, best_site, projected, channel_energy


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
    best_alpha = torch.zeros(n_events, dtype=projected.dtype, device=projected.device)
    best_objective = torch.full(
        (n_events,), float("inf"), dtype=projected.dtype, device=projected.device
    )
    levels = 0
    for _ in range(config.refine_levels):
        candidates = current[:, None, :] + step[:, None, :] * delta[None]
        for dimension, (lo, hi) in enumerate(zip(XYZ_LO, XYZ_HI)):
            candidates[:, :, dimension].clamp_(lo, hi)
        objective, alpha = score_event_candidates(
            projected,
            channel_energy,
            offsets,
            mask,
            local_noise,
            candidates,
            sigmas,
            config,
        )
        value, flat = objective.flatten(1).min(dim=1)
        profiles = len(sigmas) * projected.shape[2]
        candidate_index = flat // profiles
        remainder = flat % profiles
        sigma_index = remainder // projected.shape[2]
        temporal_index = remainder % projected.shape[2]
        current = candidates[rows, candidate_index]
        best_objective = value
        best_sigma = sigma_index
        best_temporal = temporal_index
        best_alpha = alpha[rows, candidate_index, sigma_index, temporal_index]
        levels += 1
        step = torch.floor(step / 2).clamp_min(1)
    return (
        current,
        coarse,
        best_sigma,
        best_temporal,
        best_alpha,
        best_objective,
        levels,
    )


def fit_grouped(
    waveforms,
    offsets,
    mask,
    local_noise,
    omega,
    sites,
    axes,
    sigmas,
    config,
    cache,
):
    omega = F.normalize(torch.as_tensor(omega, device=waveforms.device), dim=1)
    coarse_objective, site_index, projected, channel_energy = coherent_coarse_assignment(
        waveforms,
        offsets,
        mask,
        local_noise,
        omega,
        sites,
        sigmas,
        config,
        cache,
    )
    source, coarse, sigma_index, temporal_index, alpha, objective, levels = refine_sites(
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
    selected_sigma = sigmas[sigma_index]
    dxy2 = (offsets - source[:, None, :2]).square().sum(dim=2)
    footprint = selected_sigma[:, None] / torch.sqrt(
        dxy2 + source[:, 2, None].square() + selected_sigma[:, None].square()
    ).clamp_min(EPS)
    footprint *= mask
    prediction = (
        alpha[:, None, None]
        * footprint[:, :, None]
        * omega[temporal_index, None, :]
    )
    residual = waveforms - prediction
    normalized = waveforms / local_noise[:, :, None]
    normalized_residual = residual / local_noise[:, :, None]
    channel_input_energy = normalized.square().sum(dim=2) * mask
    channel_residual_energy = normalized_residual.square().sum(dim=2) * mask
    channel_improvement = channel_input_energy - channel_residual_energy
    improvement_floor = torch.maximum(
        torch.full_like(channel_improvement, config.min_channel_improvement),
        (channel_improvement.sum(dim=1).clamp_min(0)
         * config.min_channel_improvement_fraction)[:, None],
    )
    improved = (channel_improvement > improvement_floor) & mask
    channel_rmse = residual.square().mean(dim=2).sqrt().masked_fill(~mask, 0)
    normalized_rmse = (channel_rmse / local_noise).masked_fill(~mask, 0)
    maximum = normalized_rmse.masked_fill(~mask, float("-inf")).amax(dim=1)
    mean = normalized_rmse.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    input_energy = channel_input_energy.sum(dim=1)
    residual_energy = channel_residual_energy.sum(dim=1)
    captured = input_energy - residual_energy
    raw_input_energy = waveforms.square().sum(dim=(1, 2))
    raw_residual_energy = residual.square().sum(dim=(1, 2))
    raw_energy_drop = raw_input_energy - raw_residual_energy
    return {
        "sources": source,
        "coarse_sources": coarse,
        "sigma_index": sigma_index,
        "sigma": selected_sigma,
        "rho": torch.sqrt(source[:, 2].square() + selected_sigma.square()),
        "temporal_index": temporal_index,
        "alpha": alpha,
        "prediction": prediction,
        "channel_rmse": channel_rmse,
        "channel_normalized_rmse": normalized_rmse,
        "channel_improvement": channel_improvement,
        "improved_channel_count": improved.sum(dim=1),
        "maximum_channel_normalized_rmse": maximum,
        "mean_channel_normalized_rmse": mean,
        "input_energy": input_energy,
        "captured_energy": captured,
        "fitted_projection_score": torch.sqrt(captured.clamp_min(0)),
        "captured_fraction": captured / input_energy.clamp_min(EPS),
        "raw_energy_drop": raw_energy_drop,
        "coarse_objective": coarse_objective,
        "objective": objective,
        "refinement_levels": torch.full(
            (len(waveforms),), levels, dtype=torch.uint8, device=waveforms.device
        ),
    }


def onehot_template_scores(residual, noise, omega, footprints, safe_ids, config):
    n_samples, n_channels = residual.shape
    omega = F.normalize(omega, dim=1)
    standardized = (residual / noise[None]).T[None]
    weights = omega.repeat(n_channels, 1).unsqueeze(1)
    projection = F.conv1d(standardized, weights, groups=n_channels)[0]
    n_windows = projection.shape[1]
    projection = projection.reshape(
        n_channels, len(omega), n_windows
    ).permute(2, 0, 1)
    scores = torch.empty((n_windows, n_channels), device=config.device)
    choices = torch.empty(
        (n_windows, n_channels), dtype=torch.int16, device=config.device
    )
    for start in range(0, n_windows, config.template_time_batch):
        stop = min(start + config.template_time_batch, n_windows)
        local = projection[start:stop, safe_ids]
        response = torch.einsum("tacq,asc->tasq", local, footprints)
        values, selected = response.flatten(2).max(dim=2)
        scores[start:stop] = values.clamp_min(0)
        choices[start:stop] = selected.to(torch.int16)
    return scores, choices


def global_temporal_nms(
    scores,
    threshold,
    temporal_radius,
    valid_start,
    valid_stop,
    max_events,
):
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (time, channels), got {scores.shape}")
    best_score, best_channel = scores.max(dim=1)
    pooled = F.max_pool1d(
        best_score[None, None],
        kernel_size=2 * temporal_radius + 1,
        stride=1,
        padding=temporal_radius,
    )[0, 0]
    if temporal_radius:
        padded = F.pad(best_score[None, None], (temporal_radius, 0), value=float("-inf"))
        prior = F.max_pool1d(
            padded, kernel_size=temporal_radius, stride=1
        )[0, 0, : len(best_score)]
    else:
        prior = torch.full_like(best_score, float("-inf"))
    candidate = (best_score >= threshold) & (best_score >= pooled) & (best_score > prior)
    candidate[:valid_start] = False
    candidate[valid_stop:] = False
    windows = torch.nonzero(candidate, as_tuple=False).flatten()
    selected_scores = best_score[windows]
    channels = best_channel[windows]
    if max_events is not None and len(windows) > max_events:
        selected_scores, order = torch.topk(
            selected_scores, max_events, largest=True, sorted=False
        )
        windows = windows[order]
        channels = channels[order]
    order = torch.argsort(windows, stable=True)
    return windows[order], channels[order], selected_scores[order]


def detect_events_0016(
    residual,
    noise,
    omega,
    footprints,
    safe_fit_ids,
    config,
    fs,
    valid_start,
    valid_stop,
):
    scores, choices = onehot_template_scores(
        residual, noise, omega, footprints, safe_fit_ids, config
    )
    n_before = int(round(config.ms_before * fs / 1000))
    waveform_length = n_before + int(round(config.ms_after * fs / 1000))
    temporal_radius = waveform_length - 1
    window_start = max(0, valid_start - n_before)
    window_stop = min(len(scores), valid_stop - n_before)
    windows, channels, selected_scores = global_temporal_nms(
        scores,
        config.threshold,
        temporal_radius,
        window_start,
        window_stop,
        config.max_events_per_pass,
    )
    initial = choices[windows, channels].long()
    return (
        windows + n_before,
        channels,
        selected_scores,
        initial // omega.shape[0],
        initial % omega.shape[0],
        {
            "anchor_windows_above_threshold": int((scores >= config.threshold).sum().item()),
            "time_windows_above_threshold": int(
                (scores.amax(dim=1) >= config.threshold).sum().item()
            ),
        },
    )


def merge_adjacency(merge_ids, device):
    adjacency = torch.zeros(
        (len(merge_ids), len(merge_ids)), dtype=torch.bool, device=device
    )
    rows = np.repeat(np.arange(len(merge_ids)), (merge_ids >= 0).sum(axis=1))
    columns = merge_ids[merge_ids >= 0]
    adjacency[
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(columns, dtype=torch.long, device=device),
    ] = True
    return adjacency


def duplicate_mask(
    times,
    channels,
    temporal_index,
    prior_times,
    prior_channels,
    prior_temporal,
    adjacency,
    omega_similarity,
    temporal_radius,
):
    if not len(prior_times) or not len(times):
        return torch.zeros(len(times), dtype=torch.bool, device=times.device)
    close_time = (times[:, None] - prior_times[None]).abs() <= temporal_radius
    close_channel = adjacency[channels[:, None], prior_channels[None]]
    matching_shape = omega_similarity[
        temporal_index[:, None], prior_temporal[None]
    ]
    return (close_time & close_channel & matching_shape).any(dim=1)


def empty_chunk_0016(width, waveform_length, save_waveforms):
    result = empty_chunk_0012(width, waveform_length, save_waveforms)
    result["residual_pass"] = np.empty(0, dtype=np.int16)
    result["channel_improvement"] = np.empty((0, width), dtype=np.float32)
    result["improved_channel_count"] = np.empty(0, dtype=np.int16)
    result["raw_energy_drop"] = np.empty(0, dtype=np.float32)
    result["coarse_objective"] = np.empty(0, dtype=np.float32)
    result["objective"] = np.empty(0, dtype=np.float32)
    result["peeling_round"] = np.empty(0, dtype=np.int16)
    return result


def concatenate_parts_0016(parts, width, waveform_length, save_waveforms):
    if not parts:
        return empty_chunk(width, waveform_length, save_waveforms)
    result = {
        key: np.concatenate([part[key] for part in parts]) for key in parts[0]
    }
    order = np.lexsort((result["peeling_round"], result["spike_times"]))
    return {key: value[order] for key, value in result.items()}


def quantiles_0016(value):
    if not len(value):
        return {}
    points = torch.tensor(
        [0.0, 0.1, 0.5, 0.9, 0.99, 1.0], dtype=value.dtype, device=value.device
    )
    values = torch.quantile(value, points).detach().cpu().tolist()
    return {str(float(point)): float(item) for point, item in zip(points.cpu(), values)}


def process_chunk_0016(
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
    cache,
):
    n_before = int(round(config.ms_before * fs / 1000))
    n_after = int(round(config.ms_after * fs / 1000))
    waveform_length = n_before + n_after
    residual = torch.as_tensor(data, dtype=torch.float32, device=config.device)
    noise_np = robust_channel_noise(data)
    noise = torch.as_tensor(noise_np, device=config.device)
    omega_t = preserve_omega_polarity(torch.as_tensor(omega, device=config.device))
    omega_similarity = (omega_t @ omega_t.T).abs() >= config.duplicate_temporal_correlation
    safe_fit_ids, fit_mask = gpu_neighborhood(fit_ids, config.device)
    fit_offsets_t = torch.as_tensor(fit_offsets, device=config.device)
    detection_bank, safe_detection_ids = detection_footprints(
        fit_offsets, fit_ids, sigmas.detach().cpu().numpy(), noise_np, config.device
    )
    adjacency = merge_adjacency(merge_ids, config.device)
    parts = []
    round_summaries = []
    prior_times = torch.empty(0, dtype=torch.long, device=config.device)
    prior_channels = torch.empty(0, dtype=torch.long, device=config.device)
    prior_temporal = torch.empty(0, dtype=torch.long, device=config.device)
    local_core_start = core_start - read_start
    local_core_stop = core_stop - read_start
    valid_start = max(n_before, local_core_start - n_after)
    valid_stop = min(len(data) - n_after + 1, local_core_stop + n_before)
    merge_samples = int(round(config.event_merge_ms * fs / 1000))
    stopping_reason = "maximum_peeling_rounds"
    for peeling_round in range(config.peeling_rounds):
        started = perf_counter()
        before = residual.clone()
        full_energy_before = residual.square().sum()
        core_energy_before = residual[local_core_start:local_core_stop].square().sum()
        detected = detect_events(
            residual,
            noise,
            omega_t,
            detection_bank,
            safe_detection_ids,
            config,
            fs,
            valid_start,
            valid_stop,
        )
        times, channels, detection_score, initial_sigma, initial_temporal, detector_counts = detected
        batch_results = []
        accepted_before_merge = 0
        duplicate_rejected = 0
        for start in range(0, len(times), config.fit_batch_size):
            stop = min(start + config.fit_batch_size, len(times))
            batch_times = times[start:stop]
            batch_channels = channels[start:stop]
            waveforms, ids, local_offsets, mask, local_noise = extract_waveforms_torch(
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
            fit = fit_grouped(
                waveforms,
                local_offsets,
                mask,
                local_noise,
                omega_t,
                sites,
                axes,
                sigmas,
                config,
                cache,
            )
            accepted = (
                torch.isfinite(fit["alpha"])
                & (fit["alpha"] > 0)
                & torch.isfinite(fit["maximum_channel_normalized_rmse"])
                & (
                    fit["maximum_channel_normalized_rmse"]
                    <= config.max_channel_normalized_rmse
                )
                & (fit["captured_fraction"] >= config.min_captured_fraction)
                & (fit["fitted_projection_score"] >= config.min_fitted_projection)
                & (fit["improved_channel_count"] >= config.min_improved_channels)
                & (fit["raw_energy_drop"] > config.min_raw_energy_drop)
            )
            accepted_before_merge += int(accepted.sum().item())
            duplicate = duplicate_mask(
                batch_times,
                batch_channels,
                fit["temporal_index"],
                prior_times,
                prior_channels,
                prior_temporal,
                adjacency,
                omega_similarity,
                merge_samples,
            )
            duplicate_rejected += int((accepted & duplicate).sum().item())
            accepted &= ~duplicate
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
        accepted_count = sum(int(batch["accepted"].sum().item()) for batch in batch_results)
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
        full_energy_after = residual.square().sum()
        core_energy_after = residual[local_core_start:local_core_stop].square().sum()
        full_drop = float(
            ((full_energy_before - full_energy_after) / full_energy_before.clamp_min(EPS)).item()
        )
        core_drop = float(
            ((core_energy_before - core_energy_after) / core_energy_before.clamp_min(EPS)).item()
        )
        rolled_back = accepted_count == 0 or full_drop <= 0
        if rolled_back:
            residual.copy_(before)
        else:
            accepted_times = []
            accepted_channels = []
            accepted_temporal = []
            for batch in batch_results:
                selected = batch["accepted"]
                accepted_times.append(batch["times"][selected])
                accepted_channels.append(batch["channels"][selected])
                accepted_temporal.append(batch["fit"]["temporal_index"][selected])
                in_core = (
                    selected
                    & (batch["times"] >= local_core_start)
                    & (batch["times"] < local_core_stop)
                )
                if not bool(in_core.any()):
                    continue
                fit = batch["fit"]
                anchor = batch["channels"][in_core]
                sources = tensor_numpy(fit["sources"], in_core).astype(np.float32)
                anchors = channel_positions[anchor.detach().cpu().numpy()]
                global_sources = np.column_stack(
                    (anchors + sources[:, :2], sources[:, 2])
                ).astype(np.float32)
                count = len(sources)
                part = {
                    "spike_times": (
                        read_start + tensor_numpy(batch["times"], in_core)
                    ).astype(np.int64),
                    "spike_channels": anchor.detach().cpu().numpy().astype(np.int32),
                    "sources": sources,
                    "global_sources": global_sources,
                    "coarse_sources": tensor_numpy(
                        fit["coarse_sources"], in_core
                    ).astype(np.float32),
                    "sigma_index": tensor_numpy(fit["sigma_index"], in_core).astype(np.int16),
                    "sigma": tensor_numpy(fit["sigma"], in_core).astype(np.float32),
                    "rho": tensor_numpy(fit["rho"], in_core).astype(np.float32),
                    "temporal_index": tensor_numpy(
                        fit["temporal_index"], in_core
                    ).astype(np.int16),
                    "alpha": tensor_numpy(fit["alpha"], in_core).astype(np.float32),
                    "detection_score": tensor_numpy(
                        batch["detection_score"], in_core
                    ).astype(np.float32),
                    "initial_sigma_index": tensor_numpy(
                        batch["initial_sigma"], in_core
                    ).astype(np.int16),
                    "initial_temporal_index": tensor_numpy(
                        batch["initial_temporal"], in_core
                    ).astype(np.int16),
                    "neighbor_ids": fit_ids[anchor.detach().cpu().numpy()].astype(np.int32),
                    "neighbor_counts": fit_counts[anchor.detach().cpu().numpy()].astype(np.int16),
                    "channel_rmse": tensor_numpy(fit["channel_rmse"], in_core).astype(np.float32),
                    "channel_normalized_rmse": tensor_numpy(
                        fit["channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "channel_improvement": tensor_numpy(
                        fit["channel_improvement"], in_core
                    ).astype(np.float32),
                    "improved_channel_count": tensor_numpy(
                        fit["improved_channel_count"], in_core
                    ).astype(np.int16),
                    "maximum_channel_normalized_rmse": tensor_numpy(
                        fit["maximum_channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "mean_channel_normalized_rmse": tensor_numpy(
                        fit["mean_channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "input_energy": tensor_numpy(fit["input_energy"], in_core).astype(np.float32),
                    "captured_energy": tensor_numpy(
                        fit["captured_energy"], in_core
                    ).astype(np.float32),
                    "fitted_projection_score": tensor_numpy(
                        fit["fitted_projection_score"], in_core
                    ).astype(np.float32),
                    "captured_fraction": tensor_numpy(
                        fit["captured_fraction"], in_core
                    ).astype(np.float32),
                    "raw_energy_drop": tensor_numpy(
                        fit["raw_energy_drop"], in_core
                    ).astype(np.float32),
                    "coarse_objective": tensor_numpy(
                        fit["coarse_objective"], in_core
                    ).astype(np.float32),
                    "objective": tensor_numpy(fit["objective"], in_core).astype(np.float32),
                    "refinement_levels": tensor_numpy(
                        fit["refinement_levels"], in_core
                    ).astype(np.uint8),
                    "residual_pass": np.full(count, peeling_round, dtype=np.int16),
                    "peeling_round": np.full(count, peeling_round, dtype=np.int16),
                    "pass_energy_drop_fraction": np.full(count, full_drop, dtype=np.float32),
                }
                if config.save_waveforms:
                    part["waveforms"] = tensor_numpy(
                        batch["waveforms"], in_core
                    ).astype(np.float32)
                    part["predictions"] = tensor_numpy(
                        fit["prediction"], in_core
                    ).astype(np.float32)
                parts.append(part)
            prior_times = torch.cat((prior_times, *accepted_times))
            prior_channels = torch.cat((prior_channels, *accepted_channels))
            prior_temporal = torch.cat((prior_temporal, *accepted_temporal))
        fitted_scores = torch.cat(
            [batch["fit"]["fitted_projection_score"] for batch in batch_results]
        ) if batch_results else torch.empty(0, device=config.device)
        summary = {
            "peeling_round": peeling_round,
            "proposed": int(len(times)),
            "accepted_before_merge": accepted_before_merge,
            "duplicate_rejected": duplicate_rejected,
            "accepted": accepted_count,
            "full_energy_before": float(full_energy_before.item()),
            "full_energy_after": float(full_energy_after.item()),
            "full_energy_drop_fraction": full_drop,
            "core_energy_drop_fraction": core_drop,
            "proposal_score_quantiles": quantiles_0016(detection_score),
            "fitted_score_quantiles": quantiles_0016(fitted_scores),
            "rolled_back": rolled_back,
            "seconds": perf_counter() - started,
            **detector_counts,
        }
        round_summaries.append(summary)
        print(json.dumps(summary), flush=True)
        if not len(times):
            stopping_reason = "no_proposals"
            break
        if accepted_count == 0:
            stopping_reason = "no_accepted_events"
            break
        if rolled_back:
            stopping_reason = "nonpositive_residual_energy_drop"
            break
    result = concatenate_parts(parts, fit_ids.shape[1], waveform_length, config.save_waveforms)
    result["noise"] = noise_np
    result["pass_summaries_json"] = np.asarray(json.dumps(round_summaries))
    result["stopping_reason"] = np.asarray(stopping_reason)
    return result


def orient_omega(omega):
    omega = F.normalize(omega.float(), dim=1)
    rows = torch.arange(len(omega), device=omega.device)
    extrema = omega.abs().argmax(dim=1)
    flip = omega[rows, extrema] > 0
    omega[flip] *= -1
    return omega


def load_omega_prior(path, q, waveform_length):
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            keys = [key for key in ("omega", "a") if key in archive]
            if not keys:
                if len(archive.files) != 1:
                    raise KeyError(f"{path} has no omega/a array")
                keys = [archive.files[0]]
            value = archive[keys[0]]
    elif path.suffix in (".pt", ".pth"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            keys = [key for key in ("omega", "a") if key in checkpoint]
            if not keys:
                raise KeyError(f"{path} has no omega/a tensor")
            value = checkpoint[keys[0]]
        else:
            value = checkpoint
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    else:
        raise ValueError(f"unsupported Omega prior format: {path.suffix}")
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (q, waveform_length):
        raise ValueError(
            f"Omega prior has shape {value.shape}; expected {(q, waveform_length)}"
        )
    return preserve_omega_polarity(torch.from_numpy(value.copy())).numpy().astype(np.float32)


def shifted_channel_null(
    data,
    fs,
    local_core_start,
    local_core_stop,
    waveform_length,
    chunk_number,
    config,
):
    minimum = max(
        waveform_length,
        int(round(config.null_shift_min_ms * fs / 1000)),
    )
    maximum = int(round(config.null_shift_max_ms * fs / 1000))
    if maximum <= minimum:
        raise ValueError("null shift range must extend beyond one waveform support")
    left_room = local_core_start
    right_room = len(data) - local_core_stop
    if left_room >= maximum + waveform_length:
        direction = 1
    elif right_room >= maximum + waveform_length:
        direction = -1
    else:
        raise ValueError("read margin is too small to keep the null wrap seam out of the core")
    rng = np.random.default_rng(config.null_seed + chunk_number)
    shifts = rng.integers(minimum, maximum + 1, size=data.shape[1])
    shifted = np.empty_like(data)
    for channel, shift in enumerate(shifts):
        shifted[:, channel] = np.roll(data[:, channel], direction * int(shift))
    return shifted, (direction * shifts).astype(np.int32)


def alternating_fit_0016(reader, output, fs, fit_ids, offsets, sos, config, resume):
    original = fit_grouped_0014
    fit_grouped_0014 = fit_grouped
    try:
        omega = alternating_fit_0014(
            reader, output, fs, fit_ids, offsets, sos, config, resume
        )
        return preserve_omega_polarity(torch.from_numpy(omega)).numpy().astype(np.float32)
    finally:
        fit_grouped_0014 = original


def pursue_0016(
    reader,
    output,
    first,
    stop,
    positions,
    fit_ids,
    offsets,
    counts,
    merge_ids,
    sos,
    omega,
    config,
    resume,
):
    fs, n_channels = float(reader.fs), len(positions)
    chunk_dir = Path(output) / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    sites_np, axes_np = coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(sigma_bank(config.base()), device=config.device)
    cache = FootprintCache(sites, sigmas, config.device)
    before, after = (
        int(round(value * fs / 1000)) for value in (config.ms_before, config.ms_after)
    )
    margin = max(
        int(round(config.read_margin_ms * fs / 1000)),
        2 * (before + after),
        128,
    )
    if config.empirical_null:
        margin = max(
            margin,
            int(round(config.null_shift_max_ms * fs / 1000)) + before + after,
        )
    chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
    starts = list(range(first, stop, chunk_samples))
    for number, core_start in enumerate(starts):
        path = chunk_dir / f"chunk_{number:06d}.npz"
        if resume and path.exists():
            continue
        core_stop = min(core_start + chunk_samples, stop)
        read_start = max(0, core_start - margin)
        read_stop = min(reader.ns, core_stop + margin)
        data = preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
        if config.empirical_null:
            data, null_shifts = shifted_channel_null(
                data,
                fs,
                core_start - read_start,
                core_stop - read_start,
                before + after,
                number,
                config,
            )
        with torch.inference_mode():
            result = process_chunk(
                data,
                read_start,
                core_start,
                core_stop,
                positions,
                fit_ids,
                offsets,
                counts,
                merge_ids,
                omega,
                sites,
                axes,
                sigmas,
                fs,
                config,
                cache,
            )
        anchors = result["spike_channels"]
        result["sources_grid"] = result["coarse_sources"]
        result["centroids"] = positions[anchors].astype(np.float32)
        result["local_coords"] = offsets[anchors].astype(np.float32)
        result["profile_idx"] = result["sigma_index"]
        result["temporal_idx"] = result["temporal_index"]
        result["continuous_displacement_um"] = np.zeros(len(anchors), dtype=np.float32)
        result["continuous_energy_gain"] = np.zeros(len(anchors), dtype=np.float32)
        if config.save_waveforms:
            result["residual_waveforms"] = result.pop("waveforms")
        if config.empirical_null:
            result["null_channel_shifts_json"] = np.asarray(
                json.dumps(null_shifts.tolist())
            )
        atomic_npz(path, result)
        print(
            f"pursuit chunk {number + 1}/{len(starts)} events={len(result['spike_times'])}",
            flush=True,
        )
    summary = consolidate_chunks(chunk_dir, Path(output))
    atomic_json(Path(output) / "summary.json", summary)
    atomic_json(Path(output) / "pursuit_footprint_cache.json", cache.diagnostics())


def validate_config_0016(config):
    if config.radius_um != 48.0 or config.merge_radius_um != 48.0:
        raise ValueError("0016 fixes fit and merge neighborhoods at 48 um")
    if config.peeling_rounds < 1:
        raise ValueError("peeling rounds must be positive")
    if config.event_merge_ms < 0 or config.min_improved_channels < 1:
        raise ValueError("merge time must be nonnegative and channel support positive")
    if not 0 <= config.min_channel_improvement_fraction <= 1:
        raise ValueError("minimum channel-improvement fraction must be in [0, 1]")
    if not 0 <= config.duplicate_temporal_correlation <= 1:
        raise ValueError("duplicate temporal correlation must be in [0, 1]")
    if config.null_shift_min_ms <= 0 or config.null_shift_max_ms <= config.null_shift_min_ms:
        raise ValueError("null shift bounds must be positive and ordered")
    validate_config_0014(config)


def self_test_0016(device):
    config = Config(
        device=device,
        lattice_size=4,
        refine_levels=3,
        n_scales=3,
        site_block_size=8,
        positive_gain=False,
    )
    generator = torch.Generator(device=device).manual_seed(17)
    n, channels, time = 5, 4, 12
    offsets = torch.tensor(
        [[[0, 0], [16, 0], [0, 20], [16, 20]]] * n,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.ones(n, channels, dtype=torch.bool, device=device)
    noise = torch.rand(n, channels, generator=generator, device=device) + 0.5
    waveforms = torch.randn(n, channels, time, generator=generator, device=device)
    omega = F.normalize(
        torch.randn(config.q, time, generator=generator, device=device), dim=1
    )
    sites_np, axes_np = coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=device)
    axes = [torch.as_tensor(axis, device=device) for axis in axes_np]
    sigmas = torch.as_tensor(sigma_bank(config.base()), device=device)
    cache = FootprintCache(sites, sigmas, device)
    objective, site_index, projected, channel_energy = coherent_coarse_assignment(
        waveforms, offsets, mask, noise, omega, sites, sigmas, config, cache
    )
    reference = choose_best_coarse(
        projected,
        channel_energy,
        offsets,
        mask,
        noise,
        sites,
        sigmas,
        config.base(),
    )
    if not torch.equal(site_index, reference[1]):
        raise AssertionError("cached coherent coarse sites disagree with the reference")
    if not torch.allclose(objective, reference[0], atol=2e-5, rtol=2e-5):
        raise AssertionError("cached coherent coarse objective disagrees with the reference")
    fitted = fit_grouped(
        waveforms,
        offsets,
        mask,
        noise,
        omega,
        sites,
        axes,
        sigmas,
        config,
        cache,
    )
    if not torch.allclose(
        fitted["captured_energy"], fitted["channel_improvement"].sum(dim=1), atol=2e-4
    ):
        raise AssertionError("per-channel improvements do not sum to captured energy")
    scores = torch.zeros(40, 4, device=device)
    scores[10, 1] = 12
    scores[11, 2] = 11
    scores[30, 3] = 10
    windows, channels_out, _ = global_temporal_nms(scores, 8, 5, 0, 40, None)
    if windows.tolist() != [10, 30] or channels_out.tolist() != [1, 3]:
        raise AssertionError("global temporal collapse did not merge nearby hypotheses")
    adjacency = torch.eye(4, dtype=torch.bool, device=device)
    adjacency[1, 2] = adjacency[2, 1] = True
    duplicate = duplicate_mask(
        torch.tensor([11, 30], device=device),
        torch.tensor([2, 3], device=device),
        torch.tensor([0, 1], device=device),
        torch.tensor([10], device=device),
        torch.tensor([1], device=device),
        torch.tensor([0], device=device),
        adjacency,
        torch.eye(config.q, dtype=torch.bool, device=device),
        2,
    )
    if duplicate.tolist() != [True, False]:
        raise AssertionError("cross-round event merging is incorrect")
    detector_omega = torch.zeros(1, 5, device=device)
    detector_omega[0, 2] = -1
    detector_data = torch.zeros(30, 2, device=device)
    detector_data[10, 0] = -10
    detector_data[20, 1] = 10
    detector_scores, _ = onehot_template_scores(
        detector_data,
        torch.ones(2, device=device),
        detector_omega,
        torch.ones(2, 1, 1, device=device),
        torch.tensor([[0], [1]], device=device),
        config,
    )
    if detector_scores[8, 0] != 10 or detector_scores[18, 1] != 0:
        raise AssertionError("positive-gain detector accepted the inverted polarity")
    null_config = Config(
        device=device,
        null_shift_min_ms=5,
        null_shift_max_ms=20,
    )
    null_input = np.arange(240 * 3, dtype=np.float32).reshape(240, 3)
    null_data, null_shifts = shifted_channel_null(
        null_input, 1000, 40, 200, 5, 0, null_config
    )
    for channel, shift in enumerate(null_shifts):
        expected = np.roll(null_input[:, channel], int(shift))
        if not np.array_equal(null_data[40:200, channel], expected[40:200]):
            raise AssertionError("empirical-null channel shift is inconsistent")
    print("0016 self-test passed", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("calibration-detect", "alternating-fit", "pursue", "all"),
        nargs="?",
    )
    parser.add_argument("recording_path", type=Path, nargs="?")
    parser.add_argument("output_path", type=Path, nargs="?")
    for field in Config.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(
                name, action=argparse.BooleanOptionalAction, default=field.default
            )
        else:
            parser.add_argument(name, type=type(field.default), default=field.default)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--electrical-series-path", type=str, default=None,
                        dest="electrical_series_path",
                        help="explicit NWB electrical series path "
                             "(default: per-filename map, then auto-detect)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config(Config_0016):
    threshold: float = 5.0
    exclude_sweep_ms: float = 1.0
    detection_nms_batch_size: int = 4096
    prototype_count: int = 2
    prototype_cone_deg: float = 35.0
    prototype_kmeans_iterations: int = 25
    peeling_rounds: int = 1
    recording_passes: int = 3
    spatial_score: str = "mean-channel-rmse"
    all_channel_improvement: bool = True
    all_channel_min_fraction: float = 0.2
    pass_fraction_step: float = 0.1
    all_channel_rule: str = "min-channel"
    all_channel_required_share: float = 0.875
    log_rejections: bool = True


def output_metadata(config, recording_path, fs, n_channels, first, stop):
    metadata = output_metadata_0016(
        config, recording_path, fs, n_channels, first, stop
    )
    metadata["model"] = (
        "alpha>=0 * monopole(x,y,z,sigma) * Omega[q], q one-hot; "
        "Omega rows constrained to two learned temporal-prototype cones"
    )
    metadata["temporal_codebook_prior"] = {
        "prototype_count": config.prototype_count,
        "cone_half_angle_degrees": config.prototype_cone_deg,
        "atom_assignment": "q modulo prototype_count",
        "prototype_initialization": (
            "mean peak-aligned maximum-channel waveform within each extremum polarity"
        ),
        "atom_initialization": (
            "spherical k-means of matching-polarity calibration waveforms, "
            "projected into the assigned cone"
        ),
        "calibration_update": (
            "closed-form temporal sufficient statistics, cone projection, "
            "and weighted-SVD prototype refit"
        ),
        "pursuit_update": "frozen",
    }
    metadata["temporal_orientation"] = "prototype polarity preserved"
    metadata["detector"] = (
        "SpikeInterface locally-exclusive semantics with peak_sign='both': "
        "signed immediate extrema at a per-channel noise threshold, followed by "
        "spatiotemporal competition on absolute normalized amplitude"
    )
    metadata["discovery_score"] = "signed voltage / per-channel robust noise"
    metadata["discovery_threshold_units"] = (
        "per-channel robust-noise standard deviations"
    )
    metadata["discovery_peak_sign"] = "both"
    metadata["discovery_exclude_sweep_ms"] = config.exclude_sweep_ms
    metadata["discovery_template_search"] = False
    metadata["detection_score_is_signed"] = True
    metadata["fit_objective"] = (
        "closed-form gain; spatial score = total noise-normalized SSE across all "
        "valid channels (mean-channel-rmse), not the worst channel"
    )
    metadata["acceptance"] = {
        "projection_score_floor": config.min_fitted_projection,
        "max_channel_normalized_rmse": config.max_channel_normalized_rmse,
        "captured_fraction_floor": config.min_captured_fraction,
        "raw_energy_drop_floor": config.min_raw_energy_drop,
        "all_channel_improvement": config.all_channel_improvement,
        "all_channel_rule": (
            {
                "min-channel": (
                    "every valid channel must capture at least the pass fraction of "
                    "its own noise-normalized input energy"
                ),
                "mean-channel": (
                    "the mean captured fraction across valid channels must reach "
                    "the pass fraction"
                ),
                "k-of-n": (
                    f"at least {config.all_channel_required_share:.3f} of valid "
                    "channels must capture the pass fraction"
                ),
            }[config.all_channel_rule]
            if config.all_channel_improvement else
            f"at least {config.min_improved_channels} channels improve"
        ),
        "all_channel_rule_name": config.all_channel_rule,
        **(
            {"all_channel_required_share": config.all_channel_required_share}
            if config.all_channel_rule == "k-of-n" else {}
        ),
        "pass1_fraction": config.all_channel_min_fraction,
        "pass_fraction_step": config.pass_fraction_step,
    }
    metadata["passes"] = {
        "recording_passes": config.recording_passes,
        "peeling_rounds_per_chunk": config.peeling_rounds,
        "detection_threshold_per_pass": [config.threshold] * config.recording_passes,
        "residual_carry": (
            "pass 2+ rebuilds each chunk from the raw chunk minus every saved event "
            "of earlier passes (GPU replay); no residual files"
        ),
        "duplicate_prior": (
            "pass 2+ preloads the chunk-local duplicate prior with the replayed "
            "earlier-pass events"
        ),
        "chunk_exhaustion": (
            "a chunk visit that accepts no events marks the chunk exhausted; later "
            "passes skip exhausted chunks instead of re-detecting an unchanged interior"
        ),
        "rejection_logging": bool(config.log_rejections),
    }
    return metadata


def locally_exclusive_peaks(
    residual,
    noise,
    safe_ids,
    valid_neighbors,
    threshold,
    temporal_radius,
    valid_start,
    valid_stop,
    max_events,
    batch_size,
):
    normalized = residual / noise[None]
    peak_mask = torch.zeros_like(normalized, dtype=torch.bool)
    center = normalized[1:-1]
    positive = (
        (center >= threshold)
        & (center > normalized[:-2])
        & (center >= normalized[2:])
    )
    negative = (
        (center <= -threshold)
        & (center < normalized[:-2])
        & (center <= normalized[2:])
    )
    peak_mask[1:-1] = positive | negative
    exclusive_start = max(valid_start, temporal_radius + 1)
    exclusive_stop = min(
        valid_stop, len(normalized) - temporal_radius - 1
    )
    if exclusive_start > 0:
        peak_mask[:exclusive_start] = False
    if exclusive_stop < len(peak_mask):
        peak_mask[exclusive_stop:] = False
    times, channels = torch.nonzero(peak_mask, as_tuple=True)
    initial_count = len(times)
    absolute_score = normalized.abs().masked_fill(~peak_mask, float("-inf"))
    offsets = torch.arange(
        -temporal_radius,
        temporal_radius + 1,
        device=residual.device,
    )
    kept = []
    for start in range(0, len(times), batch_size):
        stop = min(start + batch_size, len(times))
        batch_times = times[start:stop]
        batch_channels = channels[start:stop]
        sample_grid = batch_times[:, None] + offsets[None]
        in_bounds = (sample_grid >= 0) & (sample_grid < len(normalized))
        samples = sample_grid.clamp(0, len(normalized) - 1)
        neighbors = safe_ids[batch_channels]
        neighbor_valid = valid_neighbors[batch_channels]
        values = absolute_score[
            samples[:, :, None], neighbors[:, None, :]
        ]
        valid = in_bounds[:, :, None] & neighbor_valid[:, None, :]
        values = values.masked_fill(~valid, float("-inf"))
        own = absolute_score[batch_times, batch_channels]
        stronger = values.amax(dim=(1, 2)) > own
        earlier_equal = (
            (values == own[:, None, None])
            & (sample_grid[:, :, None] < batch_times[:, None, None])
            & valid
        ).any(dim=(1, 2))
        kept.append(~stronger & ~earlier_equal)
    if kept:
        keep = torch.cat(kept)
        times = times[keep]
        channels = channels[keep]
    selected_absolute = normalized[times, channels].abs()
    if max_events is not None and len(times) > max_events:
        selected_absolute, selected = torch.topk(
            selected_absolute,
            max_events,
            largest=True,
            sorted=False,
        )
        times = times[selected]
        channels = channels[selected]
    order = torch.argsort(times, stable=True)
    times = times[order]
    channels = channels[order]
    signed_score = normalized[times, channels]
    return times, channels, signed_score, {
        "signed_local_extrema_before_exclusion": initial_count,
        "positive_local_extrema_before_exclusion": int(positive.sum().item()),
        "negative_local_extrema_before_exclusion": int(negative.sum().item()),
        "locally_exclusive_proposals": len(times),
        "positive_proposals": int((signed_score > 0).sum().item()),
        "negative_proposals": int((signed_score < 0).sum().item()),
        "channel_samples_above_threshold": int(
            (normalized.abs() >= threshold).sum().item()
        ),
        "time_samples_above_threshold": int(
            (normalized.abs().amax(dim=1) >= threshold).sum().item()
        ),
    }


def detect_events(
    residual,
    noise,
    omega,
    footprints,
    safe_detection_ids,
    config,
    fs,
    valid_start,
    valid_stop,
):
    del omega
    valid_neighbors = footprints.abs().sum(dim=1) > 0
    temporal_radius = int(config.exclude_sweep_ms * fs / 1000)
    times, channels, signed_score, counts = locally_exclusive_peaks(
        residual,
        noise,
        safe_detection_ids,
        valid_neighbors,
        config.threshold,
        temporal_radius,
        valid_start,
        valid_stop,
        config.max_events_per_pass,
        config.detection_nms_batch_size,
    )
    unavailable = torch.full_like(times, -1)
    return (
        times,
        channels,
        signed_score,
        unavailable,
        unavailable,
        counts,
    )


def quantiles(value):
    if not len(value):
        return {}
    points = torch.tensor(
        [0.0, 0.1, 0.5, 0.9, 0.99, 1.0], dtype=value.dtype, device=value.device
    )
    values = torch.quantile(value, points).detach().cpu().tolist()
    return {str(float(point)): float(item) for point, item in zip(points.cpu(), values)}


# --------------------------------------------------------------------------- #
# 0019: per-channel acceptance, multi-pass pursuit with GPU replay
# --------------------------------------------------------------------------- #
_REJECTED_FIELDS = (
    ("rejected_spike_times", np.int64),
    ("rejected_spike_channels", np.int32),
    ("rejected_detection_score", np.float32),
    ("rejected_sigma_index", np.int16),
    ("rejected_temporal_index", np.int16),
    ("rejected_alpha", np.float32),
    ("rejected_captured_fraction", np.float32),
    ("rejected_projection_score", np.float32),
    ("rejected_max_channel_rmse", np.float32),
    ("rejected_min_channel_fraction", np.float32),
    ("rejected_all_ok", np.int8),
    ("rejected_reason", np.int32),
    ("rejected_peeling_round", np.int16),
    ("rejected_recording_pass", np.int16),
)
_CONSOLIDATE_EXCLUDED = {
    "noise",
    "pass_summaries_json",
    "residual_waveforms",
    "predictions",
    "null_channel_shifts_json",
}


def pass_all_channel_fraction(config, pass_index):
    """Per-channel bar for one recording pass: base fraction plus one step per pass."""
    return float(min(config.all_channel_min_fraction
                     + config.pass_fraction_step * pass_index, 0.9))


def all_channel_acceptance(channel_improvement, channel_input, mask, bar, config):
    """Aggregate per-channel captured fractions into the all-channel gate.

    Returns the pass mask and each event's worst valid-channel captured fraction.
    "min-channel" keeps the historical energy-floor form (improvement >= input*bar)
    so its events stay comparable with the completed fraction sweep; channels with
    near-zero input energy are excluded from the mean and count as passing elsewhere,
    matching how the floor form treats them.
    """
    per_channel_fraction = torch.where(
        channel_input > 1e-8,
        channel_improvement / channel_input.clamp_min(1e-8),
        torch.full_like(channel_input, float("inf")),
    )
    if config.all_channel_rule == "min-channel":
        all_ok = (
            (channel_improvement >= channel_input * bar - 1e-6) | ~mask
        ).all(dim=1)
    elif config.all_channel_rule == "mean-channel":
        informative = mask & (channel_input > 1e-8)
        mean_fraction = (
            torch.where(
                informative,
                per_channel_fraction,
                torch.zeros_like(per_channel_fraction),
            ).sum(dim=1)
            / informative.sum(dim=1).clamp_min(1)
        )
        all_ok = mean_fraction >= bar - 1e-6
    elif config.all_channel_rule == "k-of-n":
        required = torch.ceil(
            config.all_channel_required_share * mask.sum(dim=1).float()
        )
        all_ok = (
            ((per_channel_fraction >= bar - 1e-6) | ~mask).sum(dim=1) >= required
        )
    else:
        raise ValueError(f"unknown all-channel rule: {config.all_channel_rule}")
    return all_ok, per_channel_fraction.amin(dim=1)


def empty_chunk(width, waveform_length, save_waveforms):
    result = empty_chunk_0012(width, waveform_length, save_waveforms)
    result["residual_pass"] = np.empty(0, dtype=np.int16)
    result["channel_improvement"] = np.empty((0, width), dtype=np.float32)
    result["improved_channel_count"] = np.empty(0, dtype=np.int16)
    result["raw_energy_drop"] = np.empty(0, dtype=np.float32)
    result["coarse_objective"] = np.empty(0, dtype=np.float32)
    result["objective"] = np.empty(0, dtype=np.float32)
    result["peeling_round"] = np.empty(0, dtype=np.int16)
    result["recording_pass"] = np.empty(0, dtype=np.int16)
    result["all_channel_ok"] = np.empty(0, dtype=np.int8)
    result["all_channel_fraction"] = np.empty(0, dtype=np.float32)
    result["min_channel_captured_fraction"] = np.empty(0, dtype=np.float32)
    for key, dtype in _REJECTED_FIELDS:
        result[key] = np.empty(0, dtype=dtype)
    return result


def concatenate_parts(parts, width, waveform_length, save_waveforms):
    if not parts:
        return empty_chunk(width, waveform_length, save_waveforms)
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.lexsort((result["peeling_round"], result["spike_times"]))
    return {key: value[order] for key, value in result.items()}


def _consolidate(chunk_paths, out_dir):
    """Consolidate event fields and the rejected/rollback audit; nothing stays sharded-only."""
    paths = sorted(chunk_paths)
    if not paths:
        raise RuntimeError("no completed chunks")
    with np.load(paths[0], allow_pickle=False) as archive:
        fields = [
            key for key in archive.files
            if key not in _CONSOLIDATE_EXCLUDED
            and not key.startswith("rejected_")
            and archive[key].ndim
        ]
        rejected_fields = [
            key for key, _ in _REJECTED_FIELDS if key in archive.files
        ]
    event_total = 0
    rejected_total = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            event_total += len(archive["spike_times"])
            rejected_total += len(archive["rejected_reason"])
            if any(key not in archive.files for key in fields + rejected_fields):
                raise RuntimeError(f"incompatible chunk schema: {path}")
    arrays = {}
    rejected_arrays = {}
    cursor = 0
    rejected_cursor = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            count = len(archive["spike_times"])
            for key in fields:
                value = archive[key]
                if value.shape[0] != count:
                    raise RuntimeError(f"{path}:{key} is not event-aligned")
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(
                        Path(out_dir) / f"{key}.npy", mode="w+", dtype=value.dtype,
                        shape=(event_total, *value.shape[1:]),
                    )
                arrays[key][cursor:cursor + count] = value
            rejected_count = len(archive["rejected_reason"])
            for key in rejected_fields:
                value = archive[key]
                if value.shape[0] != rejected_count:
                    raise RuntimeError(f"{path}:{key} is not rejection-aligned")
                if key not in rejected_arrays:
                    rejected_arrays[key] = np.lib.format.open_memmap(
                        Path(out_dir) / f"{key}.npy", mode="w+",
                        dtype=value.dtype,
                        shape=(rejected_total, *value.shape[1:]),
                    )
                rejected_arrays[key][rejected_cursor:rejected_cursor + rejected_count] = value
            cursor += count
            rejected_cursor += rejected_count
    for array in list(arrays.values()) + list(rejected_arrays.values()):
        array.flush()
    return {
        "n_events": cursor,
        "n_rejected": rejected_total,
        "n_chunks": len(paths),
        "waveforms": "sharded in chunks",
    }


def exhausted_chunks(output, completed_passes, total_chunks):
    """Chunks whose most recent completed visit accepted nothing, plus never-visited ones.

    A chunk absent from the last completed pass directory was skipped there and stays
    exhausted; a chunk present with zero accepted events just exhausted itself. Chunks
    missing from earlier passes were already skipped, so only the last pass matters.
    """
    if completed_passes <= 0:
        return set()
    last_dir = Path(output) / f"pass_{completed_passes - 1:02d}"
    exhausted = set(range(total_chunks))
    for path in last_dir.glob("chunk_*.npz"):
        number = int(path.stem.split("_")[1])
        with np.load(path, allow_pickle=False) as archive:
            if len(archive["spike_times"]):
                exhausted.discard(number)
    return exhausted


def load_prior_events(output, pass_index, fit_ids, fit_offsets, n_channels, device):
    """All accepted events from passes < pass_index, on the GPU, sorted by time."""
    fields = (
        "spike_times", "spike_channels", "sources", "sigma", "alpha", "temporal_index",
    )
    arrays = {key: [] for key in fields}
    for earlier in range(pass_index):
        directory = Path(output) / f"pass_{earlier:02d}"
        for key in fields:
            path = directory / f"{key}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is required for replay; consolidate pass {earlier} first"
                )
            arrays[key].append(np.load(path, mmap_mode="r"))
    if not arrays["spike_times"]:
        return None
    stacked = {key: np.concatenate(arrays[key]) for key in fields}
    order = np.argsort(stacked["spike_times"], kind="stable")
    stacked = {key: np.ascontiguousarray(value[order]) for key, value in stacked.items()}
    return {
        "times": torch.from_numpy(stacked["spike_times"]).to(device),
        "channels": torch.from_numpy(stacked["spike_channels"]).to(device),
        "sources": torch.from_numpy(stacked["sources"]).to(device),
        "sigma": torch.from_numpy(stacked["sigma"]).to(device),
        "alpha": torch.from_numpy(stacked["alpha"]).to(device),
        "temporal": torch.from_numpy(
            stacked["temporal_index"].astype(np.int64)
        ).to(device),
        "neighbor_ids": torch.as_tensor(fit_ids, device=device),
        "neighbor_offsets": torch.as_tensor(fit_offsets, device=device),
        "n_channels": n_channels,
    }


def replay_predictions(
    prior,
    read_start,
    read_stop,
    read_length,
    n_before,
    n_after,
    omega,
    device,
    batch_events=100_000,
):
    """Prediction of every prior-pass event whose 90-sample window touches the chunk.

    Returns (prediction, duplicates): a (read_length, n_channels) tensor to subtract
    from the raw chunk, and the (local times, anchors, temporal atoms) of the selected
    events for preloading the chunk-local duplicate prior.
    """
    if prior is None or not len(prior["times"]):
        return None, None
    first = int(torch.searchsorted(
        prior["times"],
        torch.as_tensor(read_start - n_after, device=device, dtype=torch.int64),
        right=True,
    ).item())
    last = int(torch.searchsorted(
        prior["times"],
        torch.as_tensor(read_stop + n_before, device=device, dtype=torch.int64),
        right=False,
    ).item())
    if first >= last:
        return None, None
    prediction = torch.zeros(
        read_length, prior["n_channels"], dtype=torch.float32, device=device
    )
    sample_offsets = torch.arange(n_before + n_after, device=device) - n_before
    for start in range(first, last, batch_events):
        stop = min(start + batch_events, last)
        times = prior["times"][start:stop]
        anchor = prior["channels"][start:stop]
        source = prior["sources"][start:stop]
        sigma = prior["sigma"][start:stop]
        alpha = prior["alpha"][start:stop]
        q = prior["temporal"][start:stop]
        ids = prior["neighbor_ids"][anchor]
        offsets = prior["neighbor_offsets"][anchor]
        valid = ids >= 0
        dxy2 = (offsets - source[:, None, :2]).square().sum(dim=2)
        dxy2 = dxy2.masked_fill(~valid, 1.0)
        footprint = sigma[:, None] / torch.sqrt(
            dxy2 + source[:, 2, None].square() + sigma[:, None].square()
        ).clamp_min(EPS)
        footprint = footprint * valid
        pred = alpha[:, None, None] * footprint[:, :, None] * omega[q][:, None, :]
        sample_index = (
            (times[:, None, None] - read_start) + sample_offsets[None, None, :]
        ).expand(-1, ids.shape[1], -1)
        channel_index = ids[:, :, None].expand_as(sample_index)
        in_range = (
            valid[:, :, None]
            & (sample_index >= 0)
            & (sample_index < read_length)
        )
        prediction.index_put_(
            (sample_index[in_range], channel_index[in_range]),
            -pred[in_range],
            accumulate=True,
        )
    duplicates = (
        (prior["times"][first:last] - read_start).to(torch.long),
        prior["channels"][first:last].to(torch.long),
        prior["temporal"][first:last].to(torch.long),
    )
    return prediction, duplicates


def pursue(
    reader,
    output,
    first,
    stop,
    positions,
    fit_ids,
    offsets,
    counts,
    merge_ids,
    sos,
    omega,
    config,
    resume,
):
    fs, n_channels = float(reader.fs), len(positions)
    sites_np, axes_np = coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(sigma_bank(config.base()), device=config.device)
    cache = FootprintCache(sites, sigmas, config.device)
    before, after = (
        int(round(value * fs / 1000)) for value in (config.ms_before, config.ms_after)
    )
    margin = max(
        int(round(config.read_margin_ms * fs / 1000)),
        2 * (before + after),
        128,
    )
    if config.empirical_null:
        margin = max(
            margin,
            int(round(config.null_shift_max_ms * fs / 1000)) + before + after,
        )
    chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
    starts = list(range(first, stop, chunk_samples))
    output = Path(output)
    summaries_path = output / "pass_summaries.json"
    summaries = (
        json.loads(summaries_path.read_text()) if summaries_path.exists() else []
    )
    completed = 0
    while (
        completed < config.recording_passes
        and (output / f"pass_{completed:02d}" / "consolidation.json").exists()
    ):
        completed += 1
    exhausted = exhausted_chunks(output, completed, len(starts))
    stopping_reason = "all_passes_complete"
    for pass_index in range(completed, config.recording_passes):
        if len(exhausted) >= len(starts):
            stopping_reason = "all_chunks_exhausted"
            break
        pass_dir = output / f"pass_{pass_index:02d}"
        pass_dir.mkdir(exist_ok=True)
        prior = (
            load_prior_events(
                output, pass_index, fit_ids, offsets, n_channels, config.device
            )
            if pass_index else None
        )
        pass_accepted = 0
        visited = 0
        for number, core_start in enumerate(starts):
            if number in exhausted:
                continue
            path = pass_dir / f"chunk_{number:06d}.npz"
            if resume and path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    visit_accepted = int(len(archive["spike_times"]))
                pass_accepted += visit_accepted
                visited += 1
                if not visit_accepted:
                    exhausted.add(number)
                continue
            core_stop = min(core_start + chunk_samples, stop)
            read_start = max(0, core_start - margin)
            read_stop = min(reader.ns, core_stop + margin)
            data = preprocess_voltage(
                reader[read_start:read_stop, :n_channels], sos
            )
            null_shifts = None
            if config.empirical_null:
                data, null_shifts = shifted_channel_null(
                    data,
                    fs,
                    core_start - read_start,
                    core_stop - read_start,
                    before + after,
                    number,
                    config,
                )
            prior_prediction, prior_duplicates = None, None
            if prior is not None:
                with torch.inference_mode():
                    prior_prediction, prior_duplicates = replay_predictions(
                        prior,
                        read_start,
                        read_stop,
                        read_stop - read_start,
                        before,
                        after,
                        torch.as_tensor(omega, device=config.device),
                        config.device,
                    )
            with torch.inference_mode():
                result = process_chunk(
                    data,
                    prior_prediction,
                    prior_duplicates,
                    pass_index,
                    read_start,
                    core_start,
                    core_stop,
                    positions,
                    fit_ids,
                    offsets,
                    counts,
                    merge_ids,
                    omega,
                    sites,
                    axes,
                    sigmas,
                    fs,
                    config,
                    cache,
                )
            anchors = result["spike_channels"]
            result["sources_grid"] = result["coarse_sources"]
            result["centroids"] = positions[anchors].astype(np.float32)
            result["local_coords"] = offsets[anchors].astype(np.float32)
            result["profile_idx"] = result["sigma_index"]
            result["temporal_idx"] = result["temporal_index"]
            result["continuous_displacement_um"] = np.zeros(
                len(anchors), dtype=np.float32
            )
            result["continuous_energy_gain"] = np.zeros(len(anchors), dtype=np.float32)
            if config.save_waveforms:
                result["residual_waveforms"] = result.pop("waveforms")
            if null_shifts is not None:
                result["null_channel_shifts_json"] = np.asarray(
                    json.dumps(null_shifts.tolist())
                )
            atomic_npz(path, result)
            visit_accepted = int(len(anchors))
            pass_accepted += visit_accepted
            print(
                f"pass {pass_index} chunk {number + 1}/{len(starts)} "
                f"events={len(anchors)}",
                flush=True,
            )
            if not visit_accepted:
                exhausted.add(number)
            visited += 1
        pass_summary = _consolidate(pass_dir.glob("chunk_*.npz"), pass_dir)
        atomic_json(pass_dir / "consolidation.json", pass_summary)
        entry = {
            "recording_pass": pass_index,
            "n_events": pass_summary["n_events"],
            "n_rejected": pass_summary["n_rejected"],
            "n_chunks": pass_summary["n_chunks"],
            "n_chunks_visited": visited,
            "n_chunks_exhausted": len(exhausted),
            "accepted": pass_accepted,
            "channel_fraction": pass_all_channel_fraction(config, pass_index),
        }
        summaries.append(entry)
        atomic_json(summaries_path, summaries)
        print(
            f"pass {pass_index}: accepted {pass_accepted} events, "
            f"per-channel bar {entry['channel_fraction']:.2f}, "
            f"visited {visited}/{len(starts)} chunks, "
            f"{len(exhausted)} exhausted",
            flush=True,
        )
    chunk_paths = sorted(
        path
        for pass_dir in sorted(output.glob("pass_*"))
        if pass_dir.is_dir()
        for path in pass_dir.glob("chunk_*.npz")
    )
    total = _consolidate(chunk_paths, output)
    summary = {
        "n_events": total["n_events"],
        "n_rejected": total["n_rejected"],
        "n_chunks": total["n_chunks"],
        "recording_passes": config.recording_passes,
        "peeling_rounds_per_chunk": config.peeling_rounds,
        "pass_summaries": summaries,
        "stopping_reason": stopping_reason,
        "waveforms": "sharded in chunks",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "pursuit_footprint_cache.json", cache.diagnostics())


def process_chunk(
    data,
    prior_prediction,
    prior_duplicates,
    pass_index,
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
    cache,
):
    n_before = int(round(config.ms_before * fs / 1000))
    n_after = int(round(config.ms_after * fs / 1000))
    waveform_length = n_before + n_after
    data_tensor = torch.as_tensor(data, dtype=torch.float32, device=config.device)
    noise_np = robust_channel_noise(data)
    noise = torch.as_tensor(noise_np, device=config.device)
    residual = data_tensor
    if prior_prediction is not None:
        residual = (data_tensor - prior_prediction).contiguous()
    omega_t = preserve_omega_polarity(torch.as_tensor(omega, device=config.device))
    omega_similarity = (omega_t @ omega_t.T).abs() >= config.duplicate_temporal_correlation
    safe_fit_ids, fit_mask = gpu_neighborhood(fit_ids, config.device)
    fit_offsets_t = torch.as_tensor(fit_offsets, device=config.device)
    detection_bank, safe_detection_ids = detection_footprints(
        fit_offsets, fit_ids, sigmas.detach().cpu().numpy(), noise_np, config.device
    )
    adjacency = merge_adjacency(merge_ids, config.device)
    parts = []
    rejected_parts = []
    round_summaries = []
    prior_times = torch.empty(0, dtype=torch.long, device=config.device)
    prior_channels = torch.empty(0, dtype=torch.long, device=config.device)
    prior_temporal = torch.empty(0, dtype=torch.long, device=config.device)
    if prior_duplicates is not None:
        prior_times, prior_channels, prior_temporal = prior_duplicates
    local_core_start = core_start - read_start
    local_core_stop = core_stop - read_start
    valid_start = max(n_before, local_core_start - n_after)
    valid_stop = min(len(residual) - n_after + 1, local_core_stop + n_before)
    merge_samples = int(round(config.event_merge_ms * fs / 1000))
    channel_fraction = pass_all_channel_fraction(config, pass_index)
    stopping_reason = "maximum_peeling_rounds"
    for peeling_round in range(config.peeling_rounds):
        started = perf_counter()
        before_state = residual.clone()
        full_energy_before = residual.square().sum()
        core_energy_before = residual[local_core_start:local_core_stop].square().sum()
        detected = detect_events(
            residual,
            noise,
            omega_t,
            detection_bank,
            safe_detection_ids,
            config,
            fs,
            valid_start,
            valid_stop,
        )
        times, channels, detection_score, initial_sigma, initial_temporal, detector_counts = detected
        batch_results = []
        accepted_before_merge = 0
        duplicate_rejected = 0
        rejected_count = 0
        reason_totals = {}
        for start in range(0, len(times), config.fit_batch_size):
            stop = min(start + config.fit_batch_size, len(times))
            batch_times = times[start:stop]
            batch_channels = channels[start:stop]
            waveforms, ids, local_offsets, mask, local_noise = extract_waveforms_torch(
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
            fit = fit_grouped(
                waveforms,
                local_offsets,
                mask,
                local_noise,
                omega_t,
                sites,
                axes,
                sigmas,
                config,
                cache,
            )
            normalized_waveform = waveforms / local_noise[:, :, None]
            channel_input = normalized_waveform.square().sum(dim=2) * mask
            all_ok, min_channel_fraction = all_channel_acceptance(
                fit["channel_improvement"],
                channel_input,
                mask,
                channel_fraction,
                config,
            )
            accepted = (
                torch.isfinite(fit["alpha"])
                & (fit["alpha"] > 0)
                & torch.isfinite(fit["maximum_channel_normalized_rmse"])
                & (
                    fit["maximum_channel_normalized_rmse"]
                    <= config.max_channel_normalized_rmse
                )
                & (fit["captured_fraction"] >= config.min_captured_fraction)
                & (fit["fitted_projection_score"] >= config.min_fitted_projection)
                & (fit["raw_energy_drop"] > config.min_raw_energy_drop)
            )
            if config.all_channel_improvement:
                accepted &= all_ok
            else:
                accepted &= fit["improved_channel_count"] >= config.min_improved_channels
            reasons = torch.zeros(len(batch_times), dtype=torch.int32, device=config.device)
            reasons += (~(torch.isfinite(fit["alpha"]) & (fit["alpha"] > 0))).to(torch.int32) * 1
            reasons += (
                (~torch.isfinite(fit["maximum_channel_normalized_rmse"]))
                | (fit["maximum_channel_normalized_rmse"] > config.max_channel_normalized_rmse)
            ).to(torch.int32) * 2
            reasons += (fit["captured_fraction"] < config.min_captured_fraction).to(torch.int32) * 4
            reasons += (fit["fitted_projection_score"] < config.min_fitted_projection).to(torch.int32) * 8
            if config.all_channel_improvement:
                reasons += (~all_ok).to(torch.int32) * 16
            else:
                reasons += (fit["improved_channel_count"] < config.min_improved_channels).to(torch.int32) * 16
            reasons += (fit["raw_energy_drop"] <= config.min_raw_energy_drop).to(torch.int32) * 32
            accepted_before_merge += int(accepted.sum().item())
            duplicate = duplicate_mask(
                batch_times,
                batch_channels,
                fit["temporal_index"],
                prior_times,
                prior_channels,
                prior_temporal,
                adjacency,
                omega_similarity,
                merge_samples,
            )
            duplicate_rejected += int((accepted & duplicate).sum().item())
            reasons += duplicate.to(torch.int32) * 64
            accepted &= ~duplicate
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
                    "all_ok": all_ok,
                    "min_channel_fraction": min_channel_fraction,
                    "reasons": reasons,
                }
            )
        accepted_count = sum(int(batch["accepted"].sum().item()) for batch in batch_results)
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
        full_energy_after = residual.square().sum()
        core_energy_after = residual[local_core_start:local_core_stop].square().sum()
        full_drop = float(
            ((full_energy_before - full_energy_after) / full_energy_before.clamp_min(EPS)).item()
        )
        core_drop = float(
            ((core_energy_before - core_energy_after) / core_energy_before.clamp_min(EPS)).item()
        )
        rolled_back = accepted_count == 0 or full_drop <= 0
        if rolled_back:
            residual.copy_(before_state)
        else:
            accepted_times = []
            accepted_channels = []
            accepted_temporal = []
            for batch in batch_results:
                selected = batch["accepted"]
                accepted_times.append(batch["times"][selected])
                accepted_channels.append(batch["channels"][selected])
                accepted_temporal.append(batch["fit"]["temporal_index"][selected])
                in_core = (
                    selected
                    & (batch["times"] >= local_core_start)
                    & (batch["times"] < local_core_stop)
                )
                if not bool(in_core.any()):
                    continue
                fit = batch["fit"]
                anchor = batch["channels"][in_core]
                sources = tensor_numpy(fit["sources"], in_core).astype(np.float32)
                anchors = channel_positions[anchor.detach().cpu().numpy()]
                global_sources = np.column_stack(
                    (anchors + sources[:, :2], sources[:, 2])
                ).astype(np.float32)
                count = len(sources)
                part = {
                    "spike_times": (
                        read_start + tensor_numpy(batch["times"], in_core)
                    ).astype(np.int64),
                    "spike_channels": anchor.detach().cpu().numpy().astype(np.int32),
                    "sources": sources,
                    "global_sources": global_sources,
                    "coarse_sources": tensor_numpy(
                        fit["coarse_sources"], in_core
                    ).astype(np.float32),
                    "sigma_index": tensor_numpy(fit["sigma_index"], in_core).astype(np.int16),
                    "sigma": tensor_numpy(fit["sigma"], in_core).astype(np.float32),
                    "rho": tensor_numpy(fit["rho"], in_core).astype(np.float32),
                    "temporal_index": tensor_numpy(
                        fit["temporal_index"], in_core
                    ).astype(np.int16),
                    "alpha": tensor_numpy(fit["alpha"], in_core).astype(np.float32),
                    "detection_score": tensor_numpy(
                        batch["detection_score"], in_core
                    ).astype(np.float32),
                    "initial_sigma_index": tensor_numpy(
                        batch["initial_sigma"], in_core
                    ).astype(np.int16),
                    "initial_temporal_index": tensor_numpy(
                        batch["initial_temporal"], in_core
                    ).astype(np.int16),
                    "neighbor_ids": fit_ids[anchor.detach().cpu().numpy()].astype(np.int32),
                    "neighbor_counts": fit_counts[anchor.detach().cpu().numpy()].astype(np.int16),
                    "channel_rmse": tensor_numpy(fit["channel_rmse"], in_core).astype(np.float32),
                    "channel_normalized_rmse": tensor_numpy(
                        fit["channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "channel_improvement": tensor_numpy(
                        fit["channel_improvement"], in_core
                    ).astype(np.float32),
                    "improved_channel_count": tensor_numpy(
                        fit["improved_channel_count"], in_core
                    ).astype(np.int16),
                    "all_channel_ok": tensor_numpy(
                        batch["all_ok"], in_core
                    ).astype(np.int8),
                    "all_channel_fraction": np.full(
                        count, channel_fraction, dtype=np.float32
                    ),
                    "min_channel_captured_fraction": tensor_numpy(
                        batch["min_channel_fraction"], in_core
                    ).astype(np.float32),
                    "maximum_channel_normalized_rmse": tensor_numpy(
                        fit["maximum_channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "mean_channel_normalized_rmse": tensor_numpy(
                        fit["mean_channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "input_energy": tensor_numpy(fit["input_energy"], in_core).astype(np.float32),
                    "captured_energy": tensor_numpy(
                        fit["captured_energy"], in_core
                    ).astype(np.float32),
                    "fitted_projection_score": tensor_numpy(
                        fit["fitted_projection_score"], in_core
                    ).astype(np.float32),
                    "captured_fraction": tensor_numpy(
                        fit["captured_fraction"], in_core
                    ).astype(np.float32),
                    "raw_energy_drop": tensor_numpy(
                        fit["raw_energy_drop"], in_core
                    ).astype(np.float32),
                    "coarse_objective": tensor_numpy(
                        fit["coarse_objective"], in_core
                    ).astype(np.float32),
                    "objective": tensor_numpy(fit["objective"], in_core).astype(np.float32),
                    "refinement_levels": tensor_numpy(
                        fit["refinement_levels"], in_core
                    ).astype(np.uint8),
                    "residual_pass": np.full(count, peeling_round, dtype=np.int16),
                    "peeling_round": np.full(count, peeling_round, dtype=np.int16),
                    "recording_pass": np.full(count, pass_index, dtype=np.int16),
                    "pass_energy_drop_fraction": np.full(count, full_drop, dtype=np.float32),
                }
                if config.save_waveforms:
                    part["waveforms"] = tensor_numpy(
                        batch["waveforms"], in_core
                    ).astype(np.float32)
                    part["predictions"] = tensor_numpy(
                        fit["prediction"], in_core
                    ).astype(np.float32)
                parts.append(part)
            prior_times = torch.cat((prior_times, *accepted_times))
            prior_channels = torch.cat((prior_channels, *accepted_channels))
            prior_temporal = torch.cat((prior_temporal, *accepted_temporal))
        if config.log_rejections:
            for batch in batch_results:
                fit = batch["fit"]
                if rolled_back:
                    reasons = batch["reasons"] | 128
                    logged = torch.ones(
                        len(batch["times"]), dtype=torch.bool, device=config.device
                    )
                else:
                    reasons = batch["reasons"]
                    logged = reasons != 0
                if not bool(logged.any()):
                    continue
                sel = torch.nonzero(logged, as_tuple=False).squeeze(1)
                count = len(sel)
                rejected_parts.append(
                    {
                        "rejected_spike_times": (
                            read_start + tensor_numpy(batch["times"], sel)
                        ).astype(np.int64),
                        "rejected_spike_channels": tensor_numpy(
                            batch["channels"], sel
                        ).astype(np.int32),
                        "rejected_detection_score": tensor_numpy(
                            batch["detection_score"], sel
                        ).astype(np.float32),
                        "rejected_sigma_index": tensor_numpy(
                            fit["sigma_index"], sel
                        ).astype(np.int16),
                        "rejected_temporal_index": tensor_numpy(
                            fit["temporal_index"], sel
                        ).astype(np.int16),
                        "rejected_alpha": tensor_numpy(fit["alpha"], sel).astype(np.float32),
                        "rejected_captured_fraction": tensor_numpy(
                            fit["captured_fraction"], sel
                        ).astype(np.float32),
                        "rejected_projection_score": tensor_numpy(
                            fit["fitted_projection_score"], sel
                        ).astype(np.float32),
                        "rejected_max_channel_rmse": tensor_numpy(
                            fit["maximum_channel_normalized_rmse"], sel
                        ).astype(np.float32),
                        "rejected_min_channel_fraction": tensor_numpy(
                            batch["min_channel_fraction"], sel
                        ).astype(np.float32),
                        "rejected_all_ok": tensor_numpy(
                            batch["all_ok"], sel
                        ).astype(np.int8),
                        "rejected_reason": tensor_numpy(reasons, sel).astype(np.int32),
                        "rejected_peeling_round": np.full(
                            count, peeling_round, dtype=np.int16
                        ),
                        "rejected_recording_pass": np.full(
                            count, pass_index, dtype=np.int16
                        ),
                    }
                )
                rejected_count += count
                values, counts_here = np.unique(
                    tensor_numpy(reasons, sel), return_counts=True
                )
                for value, size in zip(values.tolist(), counts_here.tolist()):
                    reason_totals[value] = reason_totals.get(value, 0) + size
        fitted_scores = torch.cat(
            [batch["fit"]["fitted_projection_score"] for batch in batch_results]
        ) if batch_results else torch.empty(0, device=config.device)
        summary = {
            "recording_pass": pass_index,
            "peeling_round": peeling_round,
            "proposed": int(len(times)),
            "accepted_before_merge": accepted_before_merge,
            "duplicate_rejected": duplicate_rejected,
            "accepted": accepted_count,
            "rejected_logged": rejected_count,
            "reason_counts": reason_totals,
            "channel_fraction": channel_fraction,
            "full_energy_before": float(full_energy_before.item()),
            "full_energy_after": float(full_energy_after.item()),
            "full_energy_drop_fraction": full_drop,
            "core_energy_drop_fraction": core_drop,
            "proposal_score_quantiles": quantiles(detection_score),
            "fitted_score_quantiles": quantiles(fitted_scores),
            "rolled_back": rolled_back,
            "seconds": perf_counter() - started,
            **detector_counts,
        }
        round_summaries.append(summary)
        print(json.dumps(summary), flush=True)
        if not len(times):
            stopping_reason = "no_proposals"
            break
        if accepted_count == 0:
            stopping_reason = "no_accepted_events"
            break
        if rolled_back:
            stopping_reason = "nonpositive_residual_energy_drop"
            break
    result = concatenate_parts(parts, fit_ids.shape[1], waveform_length, config.save_waveforms)
    if rejected_parts:
        rejected = {
            key: np.concatenate([part[key] for part in rejected_parts])
            for key in rejected_parts[0]
        }
        order = np.lexsort(
            (rejected["rejected_peeling_round"], rejected["rejected_spike_times"])
        )
        rejected = {key: value[order] for key, value in rejected.items()}
    else:
        rejected = {key: np.empty(0, dtype=dtype) for key, dtype in _REJECTED_FIELDS}
    result.update(rejected)
    result["noise"] = noise_np
    result["pass_summaries_json"] = np.asarray(json.dumps(round_summaries))
    result["stopping_reason"] = np.asarray(stopping_reason)
    signed_score = result["detection_score"]
    result["absolute_detection_score"] = np.abs(signed_score).astype(np.float32)
    result["detection_polarity"] = np.where(signed_score > 0, 1, -1).astype(np.int8)
    result["detection_amplitude"] = (
        signed_score * result["noise"][result["spike_channels"]]
    ).astype(np.float32)
    result["temporal_prototype_index"] = (result["temporal_index"] % 2).astype(np.int8)
    result["fitted_polarity"] = np.where(
        result["temporal_prototype_index"] == 0, 1, -1
    ).astype(np.int8)
    return result


def preserve_omega_polarity(omega):
    omega = omega.float()
    norms = torch.linalg.vector_norm(omega, dim=1, keepdim=True)
    if bool((~torch.isfinite(norms) | (norms <= EPS)).any()):
        raise ValueError("Omega contains a non-finite or zero temporal atom")
    return omega / norms


def fix_polarity(prototypes):
    result = prototypes.clone()
    for index in range(len(result)):
        extremum = result[index, result[index].abs().argmax()]
        if bool((extremum < 0) == (index % 2 == 0)):
            result[index] = -result[index]
    return F.normalize(result, dim=1)


def project_cone(candidate, prototype, cosine_limit):
    candidate = candidate / candidate.norm().clamp_min(EPS)
    cosine = float(candidate @ prototype)
    if cosine >= cosine_limit:
        return candidate
    perpendicular = candidate - cosine * prototype
    norm = perpendicular.norm()
    if float(norm) <= EPS:
        return prototype.clone()
    sine_limit = float(np.sqrt(max(1.0 - cosine_limit**2, 0.0)))
    return cosine_limit * prototype + sine_limit * perpendicular / norm


def peak_aligned_waveforms(waveforms, mask):
    if not len(waveforms):
        return waveforms.new_empty((0, waveforms.shape[2])), torch.empty(
            0, dtype=torch.long, device=waveforms.device
        )
    channel_peak = waveforms.abs().amax(dim=2).masked_fill(~mask, float("-inf"))
    selected_channel = channel_peak.argmax(dim=1)
    rows = torch.arange(len(waveforms), device=waveforms.device)
    selected = waveforms[rows, selected_channel]
    peak = selected.abs().argmax(dim=1)
    polarity = (selected[rows, peak] <= 0).long()
    shifts = selected.shape[1] // 2 - peak
    samples = torch.arange(selected.shape[1], device=waveforms.device)
    gather = (samples[None] - shifts[:, None]) % selected.shape[1]
    return selected.gather(1, gather), polarity


def spherical_kmeans(values, count, seed, iterations):
    generator = torch.Generator(device=values.device).manual_seed(seed)
    if len(values) <= count:
        return values.clone()
    centers = [values[torch.randint(len(values), (1,), generator=generator).item()]]
    for _ in range(count - 1):
        distance = 1.0 - (values @ torch.stack(centers).T).abs().amax(dim=1)
        probability = distance.clamp_min(0).square()
        chosen = torch.multinomial(
            probability / probability.sum().clamp_min(EPS),
            1,
            generator=generator,
        )
        centers.append(values[int(chosen)])
    centers = torch.stack(centers)
    for _ in range(iterations):
        labels = (values @ centers.T).argmax(dim=1)
        for index in range(count):
            selected = labels == index
            if bool(selected.any()):
                centers[index] = F.normalize(values[selected].mean(dim=0), dim=0)
    return centers


def initialize_codebook(aligned, polarity, q, cosine_limit, seed, iterations):
    groups = [aligned[polarity == index] for index in range(2)]
    counts = [len(group) for group in groups]
    if min(counts) == 0:
        raise RuntimeError(
            "two-prototype calibration requires both positive- and negative-extremum "
            f"waveforms; observed group counts {counts}"
        )
    prototypes = fix_polarity(
        torch.stack([F.normalize(group.mean(dim=0), dim=0) for group in groups])
    )
    assignment = torch.arange(q, device=aligned.device) % len(prototypes)
    atoms = torch.zeros(q, aligned.shape[1], device=aligned.device)
    for prototype_index, group in enumerate(groups):
        rows = torch.nonzero(
            assignment == prototype_index, as_tuple=False
        ).squeeze(1)
        centers = spherical_kmeans(
            group, len(rows), seed + prototype_index, iterations
        )
        for local_index, atom_index in enumerate(rows.tolist()):
            atoms[atom_index] = project_cone(
                centers[local_index % len(centers)],
                prototypes[prototype_index],
                cosine_limit,
            )
    return atoms, prototypes, assignment, counts


def load_calibration_temporal_pool(reader, shard_dir, fs, fit_ids, sos, config):
    aligned_parts = []
    polarity_parts = []
    for _, waveforms_np, _, _, mask_np in iter_calibration_batches(
        reader, shard_dir, fs, fit_ids, sos, config
    ):
        waveforms = torch.as_tensor(waveforms_np)
        mask = torch.as_tensor(mask_np, dtype=torch.bool)
        aligned, polarity = peak_aligned_waveforms(waveforms, mask)
        norm = torch.linalg.vector_norm(aligned, dim=1)
        valid = torch.isfinite(norm) & (norm > EPS)
        if bool(valid.any()):
            aligned_parts.append(aligned[valid] / norm[valid, None])
            polarity_parts.append(polarity[valid])
    if not aligned_parts:
        raise RuntimeError("calibration contains no finite nonzero temporal waveforms")
    return torch.cat(aligned_parts), torch.cat(polarity_parts)


def calibration_detect(
    reader,
    output,
    first,
    stop,
    offsets,
    fit_ids,
    merge_ids,
    sos,
    config,
    resume,
):
    root, shard_dir = calibration_paths(output)
    shard_dir.mkdir(parents=True, exist_ok=True)
    fs = float(reader.fs)
    n_channels = fit_ids.shape[0]
    before, after = (
        int(round(value * fs / 1000))
        for value in (config.ms_before, config.ms_after)
    )
    temporal_radius = int(config.exclude_sweep_ms * fs / 1000)
    chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
    margin = max(
        int(round(config.read_margin_ms * fs / 1000)),
        before + after,
        temporal_radius + 1,
        128,
    )
    starts = np.arange(first, stop, chunk_samples, dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    chosen = np.sort(
        rng.permutation(len(starts))[
            : min(config.calibration_chunks, len(starts))
        ]
    )
    remaining = config.calibration_max_events
    isolation = int(round(config.calibration_isolation_ms * fs / 1000))
    safe_ids, valid_neighbors = gpu_neighborhood(
        merge_ids, config.device
    )
    total = 0
    total_positive = 0
    total_negative = 0
    for ordinal, index in enumerate(chosen):
        path = shard_dir / f"shard_{ordinal:03d}.npz"
        if resume and path.exists():
            with np.load(path) as saved:
                count = len(saved["spike_times"])
                total += count
                remaining -= count
                if "peak_polarity" in saved:
                    total_positive += int((saved["peak_polarity"] > 0).sum())
                    total_negative += int((saved["peak_polarity"] < 0).sum())
            continue
        core_start = int(starts[index])
        core_stop = min(core_start + chunk_samples, stop)
        read_start = max(0, core_start - margin)
        read_stop = min(reader.ns, core_stop + margin)
        data = preprocess_voltage(
            reader[read_start:read_stop, :n_channels], sos
        )
        noise = robust_channel_noise(data)
        residual = torch.as_tensor(
            data, dtype=torch.float32, device=config.device
        )
        noise_t = torch.as_tensor(noise, device=config.device)
        valid_start = max(before, core_start - read_start)
        valid_stop = min(len(data) - after + 1, core_stop - read_start)
        times_t, channels_t, scores_t, _ = locally_exclusive_peaks(
            residual,
            noise_t,
            safe_ids,
            valid_neighbors,
            config.threshold,
            temporal_radius,
            valid_start,
            valid_stop,
            None,
            config.detection_nms_batch_size,
        )
        times = times_t.cpu().numpy()
        channels = channels_t.cpu().numpy()
        scores = scores_t.cpu().numpy()
        keep = isolated_events(
            times, channels, merge_ids, isolation
        )
        times = times[keep]
        channels = channels[keep]
        scores = scores[keep]
        take = min(remaining, config.calibration_events_per_chunk, len(times))
        if take:
            selected = np.sort(rng.choice(len(times), take, replace=False))
            times = times[selected]
            channels = channels[selected]
            scores = scores[selected]
        else:
            times = times[:0]
            channels = channels[:0]
            scores = scores[:0]
        polarity = np.where(scores > 0, 1, -1).astype(np.int8)
        amplitude = (scores * noise[channels]).astype(np.float32)
        masks = fit_ids[channels] >= 0
        atomic_npz(
            path,
            {
                "spike_times": (times + read_start).astype(np.int64),
                "spike_channels": channels.astype(np.int32),
                "local_offsets": offsets[channels],
                "mask": masks,
                "noise": noise,
                "peak_score": scores.astype(np.float32),
                "peak_amplitude": amplitude,
                "peak_polarity": polarity,
            },
        )
        total += len(times)
        total_positive += int((polarity > 0).sum())
        total_negative += int((polarity < 0).sum())
        remaining -= len(times)
        print(
            f"calibration shard {ordinal + 1}/{len(chosen)} events={total:,} "
            f"positive={total_positive:,} negative={total_negative:,}",
            flush=True,
        )
        if not remaining:
            break
    atomic_json(
        root / "detect.json",
        {
            "events": total,
            "positive_events": total_positive,
            "negative_events": total_negative,
            "shards": len(list(shard_dir.glob("*.npz"))),
            "seed": config.seed,
            "first_sample": first,
            "stop_sample": stop,
            "peak_sign": "both",
            "detect_threshold": config.threshold,
            "exclude_sweep_ms": config.exclude_sweep_ms,
        },
    )


def fixed_assignment_objective(omega, numerator, denominator, input_energy):
    cross = (omega * numerator).sum()
    prediction = (denominator[:, None] * omega.square()).sum()
    return input_energy - 2 * cross + prediction


def prototype_cone_proposal(
    omega, prototypes, assignment, numerator, atom_weight, cosine_limit
):
    atoms = omega.clone()
    for atom_index in range(len(atoms)):
        if float(numerator[atom_index].norm()) > EPS:
            atoms[atom_index] = project_cone(
                numerator[atom_index],
                prototypes[assignment[atom_index]],
                cosine_limit,
            )
    updated_prototypes = prototypes.clone()
    for prototype_index in range(len(prototypes)):
        group = torch.nonzero(
            assignment == prototype_index, as_tuple=False
        ).squeeze(1)
        weights = atom_weight[group]
        if not len(group) or float(weights.sum()) <= EPS:
            continue
        weighted = (
            atoms[group] * weights.clamp_min(EPS)[:, None]
        ).double()
        _, _, right = torch.linalg.svd(weighted, full_matrices=False)
        updated_prototypes[prototype_index] = right[0].float()
    updated_prototypes = fix_polarity(updated_prototypes)
    for atom_index in range(len(atoms)):
        atoms[atom_index] = project_cone(
            atoms[atom_index],
            updated_prototypes[assignment[atom_index]],
            cosine_limit,
        )
    return atoms, updated_prototypes


def backtracked_update(
    omega,
    prototypes,
    proposed_omega,
    proposed_prototypes,
    assignment,
    numerator,
    denominator,
    input_energy,
    cosine_limit,
):
    before = fixed_assignment_objective(
        omega, numerator, denominator, input_energy
    )
    tolerance = 1e-5 * before.abs().clamp_min(1)
    for step_size in (1.0, 0.5, 0.25):
        if step_size == 1.0:
            candidate = proposed_omega
        else:
            candidate = torch.stack(
                [
                    project_cone(
                        (1 - step_size) * omega[index]
                        + step_size * proposed_omega[index],
                        proposed_prototypes[assignment[index]],
                        cosine_limit,
                    )
                    for index in range(len(omega))
                ]
            )
        after = fixed_assignment_objective(
            candidate, numerator, denominator, input_energy
        )
        if bool(after <= before + tolerance):
            return candidate, proposed_prototypes, before, after, step_size
    return omega, prototypes, before, before, 0.0


def alternating_fit(reader, output, fs, fit_ids, offsets, sos, config, resume):
    root, shards = calibration_paths(output)
    omega_path = root / "omega.npy"
    prototypes_path = root / "prototypes.npy"
    assignment_path = root / "atom_prototype.npy"
    history_path = root / "alternating_history.json"
    complete_path = root / "prototype_fit_complete.json"
    required = (
        omega_path,
        prototypes_path,
        assignment_path,
        history_path,
        complete_path,
    )
    if resume and all(path.exists() for path in required):
        omega = np.load(omega_path).astype(np.float32)
        prototypes = np.load(prototypes_path).astype(np.float32)
        assignment = np.load(assignment_path).astype(np.int16)
        save_prototype_state(output, omega, prototypes, assignment, config)
        return omega

    aligned, polarity = load_calibration_temporal_pool(
        reader, shards, fs, fit_ids, sos, config
    )
    cosine_limit = float(np.cos(np.radians(config.prototype_cone_deg)))
    omega, prototypes, assignment, polarity_counts = initialize_codebook(
        aligned,
        polarity,
        config.q,
        cosine_limit,
        config.seed,
        config.prototype_kmeans_iterations,
    )
    omega = omega.to(config.device)
    prototypes = prototypes.to(config.device)
    assignment = assignment.to(config.device)
    sites_np, axes_np = coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(sigma_bank(config.base()), device=config.device)
    cache = FootprintCache(sites, sigmas, config.device)
    history = []
    assignment_root = root / "assignments"

    atomic_npy(root / "initial_omega.npy", omega.cpu().numpy().astype(np.float32))
    atomic_npy(
        root / "initial_prototypes.npy",
        prototypes.cpu().numpy().astype(np.float32),
    )
    atomic_json(
        root / "prototype_initialization.json",
        {
            "polarity_group_counts": polarity_counts,
            "cone_half_angle_degrees": config.prototype_cone_deg,
            "atom_assignment": assignment.cpu().tolist(),
        },
    )

    for iteration in range(1, config.alternating_iterations + 1):
        numerator = torch.zeros_like(omega)
        denominator = torch.zeros(config.q, device=config.device)
        atom_weight = torch.zeros(config.q, device=config.device)
        counts = torch.zeros(config.q, dtype=torch.long, device=config.device)
        input_energy = torch.zeros((), device=config.device)
        iteration_dir = assignment_root / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        for shard_number, batch in enumerate(
            iter_calibration_batches(reader, shards, fs, fit_ids, sos, config)
        ):
            times_np, waveforms_np, channels_np, noise_np, mask_np = batch
            shard_parts = []
            for start in range(0, len(waveforms_np), config.fit_batch_size):
                stop = min(start + config.fit_batch_size, len(waveforms_np))
                channels = torch.as_tensor(
                    channels_np[start:stop], dtype=torch.long, device=config.device
                )
                waveforms = torch.as_tensor(
                    waveforms_np[start:stop], device=config.device
                )
                local_offsets = torch.as_tensor(
                    offsets[channels_np[start:stop]], device=config.device
                )
                mask = torch.as_tensor(
                    mask_np[start:stop], dtype=torch.bool, device=config.device
                )
                local_noise = torch.as_tensor(
                    noise_np[start:stop], device=config.device
                )
                fit = fit_grouped(
                    waveforms,
                    local_offsets,
                    mask,
                    local_noise,
                    omega,
                    sites,
                    axes,
                    sigmas,
                    config,
                    cache,
                )
                labels = fit["temporal_index"]
                selected_sigma = fit["sigma"]
                distance_xy = (
                    local_offsets - fit["sources"][:, None, :2]
                ).square().sum(dim=2)
                footprint = selected_sigma[:, None] / torch.sqrt(
                    distance_xy
                    + fit["sources"][:, 2, None].square()
                    + selected_sigma[:, None].square()
                ).clamp_min(EPS)
                spatial = fit["alpha"][:, None] * footprint * mask
                numerator.index_add_(
                    0, labels, torch.einsum("bct,bc->bt", waveforms, spatial)
                )
                denominator.index_add_(0, labels, spatial.square().sum(dim=1))
                atom_weight.index_add_(0, labels, fit["alpha"])
                counts += torch.bincount(labels, minlength=config.q)
                input_energy += waveforms.square().sum()
                shard_parts.append(
                    {
                        "spike_times": times_np[start:stop],
                        "spike_channels": channels_np[start:stop],
                        "site": fit["coarse_sources"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        "sigma_index": fit["sigma_index"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int16),
                        "temporal_index": labels.detach()
                        .cpu()
                        .numpy()
                        .astype(np.int16),
                        "alpha": fit["alpha"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                    }
                )
            if shard_parts:
                atomic_npz(
                    iteration_dir / f"shard_{shard_number:03d}.npz",
                    {
                        key: np.concatenate([part[key] for part in shard_parts])
                        for key in shard_parts[0]
                    },
                )

        proposed_omega, proposed_prototypes = prototype_cone_proposal(
            omega,
            prototypes,
            assignment,
            numerator,
            atom_weight,
            cosine_limit,
        )
        updated, updated_prototypes, before, after, step_size = backtracked_update(
            omega,
            prototypes,
            proposed_omega,
            proposed_prototypes,
            assignment,
            numerator,
            denominator,
            input_energy,
            cosine_limit,
        )
        change = float(
            torch.linalg.vector_norm(updated - omega, dim=1).amax().item()
        )
        accepted = step_size > 0
        omega, prototypes = updated, updated_prototypes
        history.append(
            {
                "iteration": iteration,
                "fixed_assignment_objective": float(before.item()),
                "objective_after_basis": float(after.item()),
                "basis_accepted": accepted,
                "step_size": step_size,
                "maximum_row_change": change,
                "row_counts": counts.cpu().tolist(),
                "footprint_cache": cache.diagnostics(),
            }
        )
        atomic_json(history_path, history)
        atomic_npy(omega_path, omega.cpu().numpy().astype(np.float32))
        atomic_npy(
            prototypes_path, prototypes.cpu().numpy().astype(np.float32)
        )
        atomic_npy(
            assignment_path, assignment.cpu().numpy().astype(np.int16)
        )
        print(json.dumps(history[-1]), flush=True)
        if not accepted or change < config.alternating_tolerance:
            break

    atomic_json(root / "footprint_cache.json", cache.diagnostics())
    atomic_json(
        complete_path,
        {"iterations": len(history), "basis_accepted": history[-1]["basis_accepted"]},
    )
    omega_np = omega.cpu().numpy().astype(np.float32)
    prototypes_np = prototypes.cpu().numpy().astype(np.float32)
    assignment_np = assignment.cpu().numpy().astype(np.int16)
    save_prototype_state(
        output, omega_np, prototypes_np, assignment_np, config
    )
    return omega_np


def save_prototype_state(output, omega, prototypes, assignment, config):
    output = Path(output)
    atomic_npy(output / "omega.npy", omega)
    atomic_npy(output / "prototypes.npy", prototypes)
    atomic_npy(output / "atom_prototype.npy", assignment)
    atomic_json(
        output / "omega_source.json",
        {
            "kind": "two_prototype_cone_calibration",
            "prototype_count": config.prototype_count,
            "cone_half_angle_degrees": config.prototype_cone_deg,
            "orientation": "prototype_polarity_preserved",
            "frozen_during_pursuit": True,
        },
    )


def validate_config(config):
    validate_config_0016(config)
    if config.threshold <= 0:
        raise ValueError("SpikeInterface detection threshold must be positive")
    if config.exclude_sweep_ms <= 0:
        raise ValueError("SpikeInterface exclusion sweep must be positive")
    if config.detection_nms_batch_size < 1:
        raise ValueError("detection NMS batch size must be positive")
    if config.prototype_count != 2:
        raise ValueError("0019 keeps the two-prototype temporal cone")
    if config.q < config.prototype_count:
        raise ValueError(
            "the temporal codebook must contain at least one atom per prototype"
        )
    if not 0 < config.prototype_cone_deg < 90:
        raise ValueError("prototype cone angle must be strictly between 0 and 90 degrees")
    if config.prototype_kmeans_iterations < 1:
        raise ValueError("prototype spherical k-means iterations must be positive")
    if config.omega_prior:
        raise ValueError("0019 learns its constrained codebook from calibration events")
    if not config.positive_gain:
        raise ValueError("0019 requires nonnegative gains so prototype polarity is identifiable")
    if config.spatial_score != "mean-channel-rmse":
        raise ValueError("0019 requires the total-channel (mean-channel-rmse) objective")
    if config.recording_passes < 1:
        raise ValueError("recording passes must be positive")
    if config.peeling_rounds < 1:
        raise ValueError("peeling rounds per chunk visit must be positive")
    if not 0 < config.all_channel_min_fraction < 1:
        raise ValueError("the all-channel fraction must be strictly inside (0, 1)")
    if config.pass_fraction_step < 0:
        raise ValueError("the pass fraction step must be nonnegative")
    if config.all_channel_rule not in ("min-channel", "mean-channel", "k-of-n"):
        raise ValueError(
            "all-channel rule must be min-channel, mean-channel, or k-of-n"
        )
    if not 0 < config.all_channel_required_share <= 1:
        raise ValueError("the k-of-n required channel share must be inside (0, 1]")


def self_test(device):
    residual = torch.zeros(128, 2, device=device)
    residual[40, 0] = 6
    residual[70, 1] = -7
    residual[90, 0] = 6
    residual[91, 1] = -8
    safe_ids = torch.tensor([[0, 1], [0, 1]], device=device)
    valid_neighbors = torch.ones(2, 2, dtype=torch.bool, device=device)
    times, channels, scores, counts_detected = locally_exclusive_peaks(
        residual,
        torch.ones(2, device=device),
        safe_ids,
        valid_neighbors,
        5,
        1,
        0,
        len(residual),
        None,
        16,
    )
    if times.tolist() != [40, 70, 91]:
        raise AssertionError("both-polarity local exclusivity is incorrect")
    if channels.tolist() != [0, 1, 1] or scores.tolist() != [6.0, -7.0, -8.0]:
        raise AssertionError("signed peak output is incorrect")
    if counts_detected["positive_proposals"] != 1:
        raise AssertionError("positive peak count is incorrect")
    if counts_detected["negative_proposals"] != 2:
        raise AssertionError("negative peak count is incorrect")
    positive = torch.tensor(
        [0.0, 0.2, 1.0, 0.2, 0.0], device=device
    )
    negative = torch.tensor(
        [0.0, -0.1, -1.0, -0.3, 0.0], device=device
    )
    aligned = torch.stack(
        (
            positive,
            positive + torch.tensor([0.0, 0.05, 0.0, -0.05, 0.0], device=device),
            positive + torch.tensor([0.0, -0.05, 0.0, 0.05, 0.0], device=device),
            negative,
            negative + torch.tensor([0.0, 0.05, 0.0, -0.05, 0.0], device=device),
            negative + torch.tensor([0.0, -0.05, 0.0, 0.05, 0.0], device=device),
        )
    )
    aligned = F.normalize(aligned, dim=1)
    polarity = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    cosine_limit = float(np.cos(np.radians(35.0)))
    omega, prototypes, assignment, counts = initialize_codebook(
        aligned, polarity, 4, cosine_limit, 42, 5
    )
    similarity = (omega * prototypes[assignment]).sum(dim=1)
    if bool((similarity < cosine_limit - 1e-5).any()):
        raise AssertionError("initialized temporal atoms escaped their prototype cones")
    extrema = prototypes[
        torch.arange(len(prototypes), device=device),
        prototypes.abs().argmax(dim=1),
    ]
    if not (float(extrema[0]) > 0 and float(extrema[1]) < 0):
        raise AssertionError("prototype polarity convention is incorrect")
    if assignment.tolist() != [0, 1, 0, 1] or counts != [3, 3]:
        raise AssertionError("prototype assignment or polarity counts are incorrect")
    if not torch.equal(
        preserve_omega_polarity(omega).argmax(dim=1), omega.argmax(dim=1)
    ):
        raise AssertionError("pursuit normalization changed temporal atom polarity")

    fraction_step = Config()
    if pass_all_channel_fraction(fraction_step, 0) != 0.2:
        raise AssertionError("pass-1 channel fraction is incorrect")
    if abs(pass_all_channel_fraction(fraction_step, 2) - 0.4) > 1e-9:
        raise AssertionError("pass-3 channel fraction escalation is incorrect")

    base_config = Config()
    improvement = torch.tensor(
        [[0.5, 0.05, 0.5], [0.4, 0.4, 0.4]], device=device
    )
    channel_input = torch.ones(2, 3, device=device)
    fit_mask = torch.ones(2, 3, dtype=torch.bool, device=device)
    ok_min, worst = all_channel_acceptance(
        improvement, channel_input, fit_mask, 0.2, base_config
    )
    if ok_min.tolist() != [False, True] or not torch.allclose(
        worst, torch.tensor([0.05, 0.4], device=device)
    ):
        raise AssertionError("min-channel acceptance is incorrect")
    ok_mean, _ = all_channel_acceptance(
        improvement,
        channel_input,
        fit_mask,
        0.2,
        replace(base_config, all_channel_rule="mean-channel"),
    )
    if ok_mean.tolist() != [True, True]:
        raise AssertionError("mean-channel acceptance is incorrect")
    ok_share, _ = all_channel_acceptance(
        improvement,
        channel_input,
        fit_mask,
        0.2,
        replace(
            base_config,
            all_channel_rule="k-of-n",
            all_channel_required_share=0.5,
        ),
    )
    if ok_share.tolist() != [True, True]:
        raise AssertionError("k-of-n acceptance at share 0.5 is incorrect")
    ok_strict, _ = all_channel_acceptance(
        improvement, channel_input, fit_mask, 0.2, replace(
            base_config, all_channel_rule="k-of-n"
        )
    )
    if ok_strict.tolist() != [False, True]:
        raise AssertionError("k-of-n acceptance at the default share is incorrect")
    sparse_input = torch.tensor([[1.0, 1e-9, 1.0]], device=device)
    sparse_improvement = torch.tensor([[0.5, -1e-9, 0.5]], device=device)
    sparse_mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    ok_sparse, worst_sparse = all_channel_acceptance(
        sparse_improvement, sparse_input, sparse_mask, 0.2, base_config
    )
    if ok_sparse.tolist() != [True] or float(worst_sparse[0]) != 0.5:
        raise AssertionError("near-zero-input channels must not fail the min rule")
    ok_sparse_mean, _ = all_channel_acceptance(
        sparse_improvement,
        sparse_input,
        sparse_mask,
        0.2,
        replace(base_config, all_channel_rule="mean-channel"),
    )
    if ok_sparse_mean.tolist() != [True]:
        raise AssertionError("mean-channel must skip near-zero-input channels")

    prediction, duplicates = replay_predictions(
        None, 0, 100, 100, 5, 5, omega, "cpu"
    )
    if prediction is not None or duplicates is not None:
        raise AssertionError("empty prior replay must produce no prediction")

    prior = {
        "times": torch.tensor([50], dtype=torch.long),
        "channels": torch.tensor([0], dtype=torch.long),
        "sources": torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32),
        "sigma": torch.tensor([16.0], dtype=torch.float32),
        "alpha": torch.tensor([2.0], dtype=torch.float32),
        "temporal": torch.tensor([0], dtype=torch.long),
        "neighbor_ids": torch.tensor([[0, 1]]),
        "neighbor_offsets": torch.zeros(1, 2, 2),
        "n_channels": 2,
    }
    prediction, duplicates = replay_predictions(
        prior, 40, 80, 40, 5, 5, torch.ones(2, 10), "cpu"
    )
    if prediction is None or float(prediction.square().sum()) <= 0:
        raise AssertionError("replay lost a prior-pass event")
    if duplicates[0].tolist() != [10] or duplicates[1].tolist() != [0]:
        raise AssertionError("replay duplicate records are incorrect")

    import shutil
    import tempfile

    scratch = Path(tempfile.mkdtemp())
    try:
        for name in ("pass_00", "pass_01"):
            (scratch / name).mkdir(parents=True)

        def write_chunk(pass_name, number, events):
            atomic_npz(
                scratch / pass_name / f"chunk_{number:06d}.npz",
                {
                    "spike_times": np.arange(events, dtype=np.int64),
                    "rejected_reason": np.empty(0, dtype=np.int32),
                },
            )

        write_chunk("pass_00", 0, 5)
        write_chunk("pass_00", 1, 0)
        write_chunk("pass_00", 2, 3)
        write_chunk("pass_01", 2, 1)
        if exhausted_chunks(scratch, 0, 4) != set():
            raise AssertionError("no completed pass must exhaust nothing")
        if exhausted_chunks(scratch, 1, 4) != {1, 3}:
            raise AssertionError("pass-0 exhaustion set is incorrect")
        if exhausted_chunks(scratch, 2, 4) != {0, 1, 3}:
            raise AssertionError("exhaustion must persist across skipped passes")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    try:
        import h5py
    except ImportError:
        raise AssertionError("0025 self-test needs h5py for the synthetic-NWB section")

    def _synthetic_nwb_roundtrip(nwb_path, series_override):
        rdr = build_nwb_reader(nwb_path, series_override)
        try:
            assert abs(rdr.fs - 30000.0) < 1e-3, rdr.fs
            assert rdr.ns == 6000, rdr.ns
            assert rdr.geometry["x"].shape == (8,), rdr.geometry["x"].shape
            assert rdr.geometry["y"].shape == (8,), rdr.geometry["y"].shape
            chunk = rdr[100:200, :8]
            assert chunk.shape == (100, 8) and chunk.dtype == np.float32
            expected = (np.arange(6000, dtype=np.int16) % 777).reshape(-1, 1) * np.ones(8, dtype=np.int16)
            assert np.max(np.abs(chunk - expected[100:200].astype(np.float32) * 2.34375e-06)) < 1e-9
        finally:
            rdr.close()

    scratch = Path(tempfile.mkdtemp())
    try:
        values = (np.arange(6000, dtype=np.int16) % 777).reshape(-1, 1) * np.ones(8, dtype=np.int16)
        for file_name in ("human1_sub-Pt01_ecephys.nwb", "synth_other.nwb"):
            with h5py.File(scratch / file_name, "w") as f:
                acq = f.create_group("acquisition")
                series = acq.create_group("ElectricalSeriesRaw")
                series.attrs["neurodata_type"] = "ElectricalSeries"
                dset = series.create_dataset("data", data=values, dtype="int16")
                dset.attrs["conversion"] = 2.34375e-06
                dset.attrs["offset"] = 0.0
                start = series.create_group("starting_time")
                start.attrs["rate"] = 30000.0
                electrodes = f.create_group("general/extracellular_ephys/electrodes")
                electrodes.attrs["neurodata_type"] = "DynamicTable"
                electrodes.create_dataset("rel_x", data=np.arange(8, dtype=np.float32) * 20.0)
                electrodes.create_dataset("rel_y", data=np.arange(8, dtype=np.float32) * 15.0)
                series.create_dataset("electrodes", data=np.arange(8, dtype=np.int64))
        _synthetic_nwb_roundtrip(scratch / "human1_sub-Pt01_ecephys.nwb", None)
        _synthetic_nwb_roundtrip(scratch / "synth_other.nwb", "acquisition/ElectricalSeriesRaw")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("0025 self-test passed", flush=True)


NWB_SERIES_BY_FILENAME = {
    "human1_sub-Pt01_ecephys.nwb": "acquisition/ElectricalSeriesRaw",
    "macaque1_sub-L11104_ecephys.nwb": "acquisition/ElectricalSeriesAPImec",
    "macaque2_sub-C_20220121_ecephys.nwb": "acquisition/ElectricalSeriesAP",
    "macaque3_sub-C_20220218_ecephys.nwb": "acquisition/ElectricalSeriesAP",
}


class NwbReader:
    """Duck-typed stand-in for spikeglx.Reader over an NWB ElectricalSeries.

    Contract used by the pipeline: ``fs`` (float Hz), ``ns`` (int samples),
    ``geometry["x"/"y"]`` (float32 um, row-aligned to data columns),
    ``reader[start:stop, :n_channels]`` -> float32 volts (time, channels),
    ``close()``.
    """

    def __init__(self, path, electrical_series_path=None, fs=None, ns=None, geometry=None):
        import h5py

        self.path = Path(path)
        self.series_path = electrical_series_path
        self._file = h5py.File(self.path, "r")
        try:
            series_grp = self._resolve_series_group(self._file, electrical_series_path)
            self._data = series_grp["data"]
            if self._data.dtype != np.int16:
                raise ValueError(
                    f"{self.path}: series data is {self._data.dtype}, expected int16"
                )
            self._conversion = float(self._data.attrs.get("conversion", 1.0))
            self._offset = float(self._data.attrs.get("offset", 0.0))
            self.ns = int(self._data.shape[0])
            self._n_channels = int(self._data.shape[1])
            rate = float(series_grp["starting_time"].attrs["rate"])
            self.fs = rate
            if fs is not None:
                if abs(fs - rate) > 1e-6:
                    raise ValueError(f"{self.path}: SI rate {fs} != starting_time.rate {rate}")
            if ns is not None and int(ns) != self.ns:
                raise ValueError(f"{self.path}: SI ns {ns} != data rows {self.ns}")
            if geometry is not None:
                if geometry.shape != (self._n_channels, 2):
                    raise ValueError(
                        f"{self.path}: SI location {geometry.shape} != data columns {self._n_channels}"
                    )
                x, y = geometry[:, 0], geometry[:, 1]
            else:
                x, y = self._h5py_geometry(self._file, series_grp)
            self.geometry = {"x": np.asarray(x, dtype=np.float32),
                             "y": np.asarray(y, dtype=np.float32)}
            if self._n_channels != 384:
                print(
                    f"warning: {self.path} has {self._n_channels} channels "
                    f"(codebook chain assumes an NP1-like 384-channel probe)",
                    flush=True,
                )
        except Exception:
            self._file.close()
            raise

    @staticmethod
    def _resolve_series_group(nwbfile, series_path):
        if series_path is not None:
            grp = nwbfile[series_path]
            if "data" not in grp or "starting_time" not in grp:
                raise ValueError(
                    f"{series_path} is not an electrical series with data + starting_time"
                )
            return grp
        found = []
        acquisition = nwbfile.get("acquisition", {})
        for name in acquisition:
            grp = acquisition[name]
            if ("data" in grp and "starting_time" in grp
                    and grp["data"].ndim == 2 and grp["data"].dtype == np.int16):
                found.append(name)
        if len(found) == 1:
            return acquisition[found[0]]
        if len(found) == 0:
            raise ValueError("no electrical series found under acquisition/")
        raise ValueError(
            f"multiple electrical series found ({found}); "
            f"pass --electrical-series-path with one of "
            f"{[f'acquisition/{n}' for n in found]}"
        )

    @staticmethod
    def _h5py_geometry(nwbfile, series_grp):
        indices = np.asarray(series_grp["electrodes"])[()].ravel()
        table = nwbfile["/general/extracellular_ephys/electrodes"]
        if "rel_x" not in table or "rel_y" not in table:
            raise ValueError("electrodes table lacks rel_x/rel_y columns")
        x = np.asarray(table["rel_x"], dtype=np.float32)[indices]
        y = np.asarray(table["rel_y"], dtype=np.float32)[indices]
        return x, y

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            t_slice, c_slice = key, slice(None)
        else:
            t_slice, c_slice = key
        data = np.asarray(self._data[t_slice], dtype=np.float32)
        data = data * self._conversion - self._offset
        return data[:, c_slice]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def build_nwb_reader(recording_path, electrical_series_path=None):
    """SI-wrapped NWB reader with a pure-h5py fallback.

    Metadata (fs, ns, geometry) come from spikeinterface's read_nwb_recording
    when it can construct the file; trace reads always go through h5py on the
    raw int16 data times the series conversion so volts arrive exactly like
    spikeglx.Reader produced them (SI's uV gain is never applied).
    """
    path = Path(recording_path)
    if path.suffix.lower() != ".nwb":
        raise NotImplementedError(
            f"0025 reads NWB recordings only, got {path}; use 0019 for SpikeGLX .ap.bin"
        )
    series_path = electrical_series_path
    if series_path is None:
        series_path = NWB_SERIES_BY_FILENAME.get(path.name)
    fs = ns = geometry = None
    try:
        from spikeinterface.extractors import read_nwb_recording

        rec = read_nwb_recording(str(path), electrical_series_path=series_path)
        fs = float(rec.get_sampling_frequency())
        ns = int(rec.get_num_samples())
        location = rec.get_property("location")
        if location is not None:
            geometry = np.asarray(location, dtype=np.float32)
    except Exception:
        fs = ns = geometry = None
    return NwbReader(path, series_path, fs, ns, geometry)


def main():
    args = parse_args()
    if args.self_test:
        self_test(args.device)
        return
    if args.stage is None or args.recording_path is None or args.output_path is None:
        raise SystemExit("stage, recording_path, and output_path are required")
    config = Config(
        **{name: getattr(args, name) for name in Config.__dataclass_fields__}
    )
    validate_config(config)
    output = args.output_path
    if output.exists() and not args.resume and args.stage != "calibration-detect":
        raise FileExistsError(f"refusing to overwrite {output}; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    reader = build_nwb_reader(args.recording_path, args.electrical_series_path)
    try:
        fs = float(reader.fs)
        positions = np.column_stack(
            (reader.geometry["x"], reader.geometry["y"])
        ).astype(np.float32)
        fit_ids, offsets, counts = build_neighborhoods(positions, config.radius_um)
        merge_ids, _, _ = build_neighborhoods(positions, config.merge_radius_um)
        sos = make_filter(fs, config.base())
        first = max(0, int(round(args.start_seconds * fs)))
        stop = reader.ns if args.duration_seconds is None else min(
            reader.ns, first + int(round(args.duration_seconds * fs))
        )
        metadata = output_metadata(
            config, args.recording_path, fs, len(positions), first, stop
        )
        metadata_path = output / "metadata.json"
        if args.resume and metadata_path.exists() and any((output / "chunks").glob("chunk_*.npz")):
            existing = json.loads(metadata_path.read_text())
            if existing != metadata:
                raise RuntimeError("resume configuration differs from the saved 0025 run")
        atomic_json(output / "config.json", metadata)
        atomic_json(metadata_path, metadata)
        atomic_npy(output / "channel_positions.npy", positions)
        atomic_npy(output / "fit_neighborhood_ids.npy", fit_ids)
        atomic_npy(output / "fit_neighborhood_offsets.npy", offsets)
        atomic_npy(output / "merge_neighborhood_ids.npy", merge_ids)
        waveform_length = int(round(config.ms_before * fs / 1000)) + int(
            round(config.ms_after * fs / 1000)
        )
        if config.omega_prior:
            omega = load_omega_prior(config.omega_prior, config.q, waveform_length)
            atomic_npy(output / "omega.npy", omega)
            atomic_json(
                output / "omega_source.json",
                {"kind": "external_prior", "path": str(Path(config.omega_prior).resolve())},
            )
        else:
            if args.stage in ("calibration-detect", "all"):
                calibration_detect(
                    reader, output, first, stop, offsets, fit_ids, merge_ids,
                    sos, config, args.resume,
                )
            if args.stage in ("alternating-fit", "all"):
                omega = alternating_fit(
                    reader, output, fs, fit_ids, offsets, sos, config, args.resume
                )
                atomic_npy(output / "omega.npy", omega)
            else:
                omega_path = calibration_paths(output)[0] / "omega.npy"
                if args.stage == "pursue" and not omega_path.exists():
                    raise FileNotFoundError(f"{omega_path} is required for pursue")
                omega = (
                    np.load(omega_path).astype(np.float32)
                    if omega_path.exists()
                    else None
                )
                if omega is not None:
                    omega = preserve_omega_polarity(torch.from_numpy(omega)).numpy().astype(np.float32)
        if args.stage in ("pursue", "all"):
            atomic_npy(output / "omega.npy", omega)
            pursue(
                reader, output, first, stop, positions, fit_ids, offsets, counts,
                merge_ids, sos, omega, config, args.resume,
            )
    finally:
        reader.close()


if __name__ == "__main__":
    main()
