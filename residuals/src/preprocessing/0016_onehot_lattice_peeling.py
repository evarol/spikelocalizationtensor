"""One-hot lattice pursuit with coherent fitting and residual rescoring."""

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "xyzsig_0014", HERE / "0014_xyzsig_residual.py"
)
OLD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OLD
SPEC.loader.exec_module(OLD)
BASE = OLD.BASE
EPS = BASE.EPS


@dataclass(frozen=True)
class Config(OLD.Config):
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


def output_metadata(config, recording_path, fs, n_channels, first, stop):
    metadata = OLD.output_metadata(config, recording_path, fs, n_channels, first, stop)
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
    for key, rows_np in OLD.grouped_rows(offset_rows, mask_rows):
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
        for dimension, (lo, hi) in enumerate(zip(BASE.XYZ_LO, BASE.XYZ_HI)):
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


def detect_events(
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


def empty_chunk(width, waveform_length, save_waveforms):
    result = BASE.empty_chunk(width, waveform_length, save_waveforms)
    result["residual_pass"] = np.empty(0, dtype=np.int16)
    result["channel_improvement"] = np.empty((0, width), dtype=np.float32)
    result["improved_channel_count"] = np.empty(0, dtype=np.int16)
    result["raw_energy_drop"] = np.empty(0, dtype=np.float32)
    result["coarse_objective"] = np.empty(0, dtype=np.float32)
    result["objective"] = np.empty(0, dtype=np.float32)
    result["peeling_round"] = np.empty(0, dtype=np.int16)
    return result


def concatenate_parts(parts, width, waveform_length, save_waveforms):
    if not parts:
        return empty_chunk(width, waveform_length, save_waveforms)
    result = {
        key: np.concatenate([part[key] for part in parts]) for key in parts[0]
    }
    order = np.lexsort((result["peeling_round"], result["spike_times"]))
    return {key: value[order] for key, value in result.items()}


def quantiles(value):
    if not len(value):
        return {}
    points = torch.tensor(
        [0.0, 0.1, 0.5, 0.9, 0.99, 1.0], dtype=value.dtype, device=value.device
    )
    values = torch.quantile(value, points).detach().cpu().tolist()
    return {str(float(point)): float(item) for point, item in zip(points.cpu(), values)}


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
    cache,
):
    n_before = int(round(config.ms_before * fs / 1000))
    n_after = int(round(config.ms_after * fs / 1000))
    waveform_length = n_before + n_after
    residual = torch.as_tensor(data, dtype=torch.float32, device=config.device)
    noise_np = BASE.robust_channel_noise(data)
    noise = torch.as_tensor(noise_np, device=config.device)
    omega_t = orient_omega(torch.as_tensor(omega, device=config.device))
    omega_similarity = (omega_t @ omega_t.T).abs() >= config.duplicate_temporal_correlation
    safe_fit_ids, fit_mask = BASE.gpu_neighborhood(fit_ids, config.device)
    fit_offsets_t = torch.as_tensor(fit_offsets, device=config.device)
    detection_bank, safe_detection_ids = BASE.detection_footprints(
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
            waveforms, ids, local_offsets, mask, local_noise = BASE.extract_waveforms_torch(
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
            BASE.subtract_predictions(
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
                sources = BASE.tensor_numpy(fit["sources"], in_core).astype(np.float32)
                anchors = channel_positions[anchor.detach().cpu().numpy()]
                global_sources = np.column_stack(
                    (anchors + sources[:, :2], sources[:, 2])
                ).astype(np.float32)
                count = len(sources)
                part = {
                    "spike_times": (
                        read_start + BASE.tensor_numpy(batch["times"], in_core)
                    ).astype(np.int64),
                    "spike_channels": anchor.detach().cpu().numpy().astype(np.int32),
                    "sources": sources,
                    "global_sources": global_sources,
                    "coarse_sources": BASE.tensor_numpy(
                        fit["coarse_sources"], in_core
                    ).astype(np.float32),
                    "sigma_index": BASE.tensor_numpy(fit["sigma_index"], in_core).astype(np.int16),
                    "sigma": BASE.tensor_numpy(fit["sigma"], in_core).astype(np.float32),
                    "rho": BASE.tensor_numpy(fit["rho"], in_core).astype(np.float32),
                    "temporal_index": BASE.tensor_numpy(
                        fit["temporal_index"], in_core
                    ).astype(np.int16),
                    "alpha": BASE.tensor_numpy(fit["alpha"], in_core).astype(np.float32),
                    "detection_score": BASE.tensor_numpy(
                        batch["detection_score"], in_core
                    ).astype(np.float32),
                    "initial_sigma_index": BASE.tensor_numpy(
                        batch["initial_sigma"], in_core
                    ).astype(np.int16),
                    "initial_temporal_index": BASE.tensor_numpy(
                        batch["initial_temporal"], in_core
                    ).astype(np.int16),
                    "neighbor_ids": fit_ids[anchor.detach().cpu().numpy()].astype(np.int32),
                    "neighbor_counts": fit_counts[anchor.detach().cpu().numpy()].astype(np.int16),
                    "channel_rmse": BASE.tensor_numpy(fit["channel_rmse"], in_core).astype(np.float32),
                    "channel_normalized_rmse": BASE.tensor_numpy(
                        fit["channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "channel_improvement": BASE.tensor_numpy(
                        fit["channel_improvement"], in_core
                    ).astype(np.float32),
                    "improved_channel_count": BASE.tensor_numpy(
                        fit["improved_channel_count"], in_core
                    ).astype(np.int16),
                    "maximum_channel_normalized_rmse": BASE.tensor_numpy(
                        fit["maximum_channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "mean_channel_normalized_rmse": BASE.tensor_numpy(
                        fit["mean_channel_normalized_rmse"], in_core
                    ).astype(np.float32),
                    "input_energy": BASE.tensor_numpy(fit["input_energy"], in_core).astype(np.float32),
                    "captured_energy": BASE.tensor_numpy(
                        fit["captured_energy"], in_core
                    ).astype(np.float32),
                    "fitted_projection_score": BASE.tensor_numpy(
                        fit["fitted_projection_score"], in_core
                    ).astype(np.float32),
                    "captured_fraction": BASE.tensor_numpy(
                        fit["captured_fraction"], in_core
                    ).astype(np.float32),
                    "raw_energy_drop": BASE.tensor_numpy(
                        fit["raw_energy_drop"], in_core
                    ).astype(np.float32),
                    "coarse_objective": BASE.tensor_numpy(
                        fit["coarse_objective"], in_core
                    ).astype(np.float32),
                    "objective": BASE.tensor_numpy(fit["objective"], in_core).astype(np.float32),
                    "refinement_levels": BASE.tensor_numpy(
                        fit["refinement_levels"], in_core
                    ).astype(np.uint8),
                    "residual_pass": np.full(count, peeling_round, dtype=np.int16),
                    "peeling_round": np.full(count, peeling_round, dtype=np.int16),
                    "pass_energy_drop_fraction": np.full(count, full_drop, dtype=np.float32),
                }
                if config.save_waveforms:
                    part["waveforms"] = BASE.tensor_numpy(
                        batch["waveforms"], in_core
                    ).astype(np.float32)
                    part["predictions"] = BASE.tensor_numpy(
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
    return orient_omega(torch.from_numpy(value.copy())).numpy().astype(np.float32)


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


def alternating_fit(reader, output, fs, fit_ids, offsets, sos, config, resume):
    original = OLD.fit_grouped
    OLD.fit_grouped = fit_grouped
    try:
        omega = OLD.alternating_fit(
            reader, output, fs, fit_ids, offsets, sos, config, resume
        )
        return orient_omega(torch.from_numpy(omega)).numpy().astype(np.float32)
    finally:
        OLD.fit_grouped = original


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
    chunk_dir = Path(output) / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=config.device)
    cache = OLD.FootprintCache(sites, sigmas, config.device)
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
        data = BASE.preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
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
        OLD.atomic_npz(path, result)
        print(
            f"pursuit chunk {number + 1}/{len(starts)} events={len(result['spike_times'])}",
            flush=True,
        )
    summary = OLD.consolidate_chunks(chunk_dir, Path(output))
    OLD.atomic_json(Path(output) / "summary.json", summary)
    OLD.atomic_json(Path(output) / "pursuit_footprint_cache.json", cache.diagnostics())


def validate_config(config):
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
    OLD.validate_config(config)


def self_test(device):
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
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=device)
    axes = [torch.as_tensor(axis, device=device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=device)
    cache = OLD.FootprintCache(sites, sigmas, device)
    objective, site_index, projected, channel_energy = coherent_coarse_assignment(
        waveforms, offsets, mask, noise, omega, sites, sigmas, config, cache
    )
    reference = BASE.choose_best_coarse(
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


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
    import spikeglx

    output = args.output_path
    if output.exists() and not args.resume and args.stage != "calibration-detect":
        raise FileExistsError(f"refusing to overwrite {output}; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    reader = spikeglx.Reader(args.recording_path)
    try:
        fs = float(reader.fs)
        positions = np.column_stack(
            (reader.geometry["x"], reader.geometry["y"])
        ).astype(np.float32)
        fit_ids, offsets, counts = BASE.build_neighborhoods(positions, config.radius_um)
        merge_ids, _, _ = BASE.build_neighborhoods(positions, config.merge_radius_um)
        sos = BASE.make_filter(fs, config.base())
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
                raise RuntimeError("resume configuration differs from the saved 0016 run")
        OLD.atomic_json(output / "config.json", metadata)
        OLD.atomic_json(metadata_path, metadata)
        OLD.atomic_npy(output / "channel_positions.npy", positions)
        OLD.atomic_npy(output / "fit_neighborhood_ids.npy", fit_ids)
        OLD.atomic_npy(output / "fit_neighborhood_offsets.npy", offsets)
        OLD.atomic_npy(output / "merge_neighborhood_ids.npy", merge_ids)
        waveform_length = int(round(config.ms_before * fs / 1000)) + int(
            round(config.ms_after * fs / 1000)
        )
        if config.omega_prior:
            omega = load_omega_prior(config.omega_prior, config.q, waveform_length)
            OLD.atomic_npy(output / "omega.npy", omega)
            OLD.atomic_json(
                output / "omega_source.json",
                {"kind": "external_prior", "path": str(Path(config.omega_prior).resolve())},
            )
        else:
            if args.stage in ("calibration-detect", "all"):
                OLD.calibration_detect(
                    reader,
                    output,
                    first,
                    stop,
                    offsets,
                    fit_ids,
                    merge_ids,
                    sos,
                    config,
                    args.resume,
                )
            if args.stage in ("alternating-fit", "all"):
                omega = alternating_fit(
                    reader, output, fs, fit_ids, offsets, sos, config, args.resume
                )
                OLD.atomic_npy(output / "omega.npy", omega)
            else:
                omega_path = OLD.calibration_paths(output)[0] / "omega.npy"
                if args.stage == "pursue" and not omega_path.exists():
                    raise FileNotFoundError(f"{omega_path} is required for pursue")
                omega = (
                    np.load(omega_path).astype(np.float32)
                    if omega_path.exists()
                    else None
                )
                if omega is not None:
                    omega = orient_omega(torch.from_numpy(omega)).numpy().astype(np.float32)
        if args.stage in ("pursue", "all"):
            OLD.atomic_npy(output / "omega.npy", omega)
            pursue(
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
                args.resume,
            )
    finally:
        reader.close()


if __name__ == "__main__":
    main()
