"""Convolving detection peeling: 0019 with an optional noise-normalized matched-filter detector.

Same model, coherent fit, all-channel acceptance, passes, GPU replay, chunk exhaustion,
and rejection audit as 0019. `--detector bipolar` (the default) routes to 0019's
SpikeInterface-style locally exclusive peak detector unchanged. `--detector convolving`
replaces only the proposal generator: a dense temporal matched filter of the current
residual against every learned Omega row, s[c,t,q] = <r[c], Omega_q> / (sigma_c*||Omega_q||),
with S[c,t] = max_q |s| keeping the signed winning value. A spatial score merge
(--score-merge perchannel | gaussian | growsum) controls duplicate merging, a
spatiotemporal NMS on |M| over the 48 um neighborhood and --proposal-lockout samples
keeps only local maxima with strict > comparisons (earliest max wins ties; this joint
window is also the cleanup-heights cross-channel dedup, ported from IBL template-id
distance to um distances — a batch-wide IBL-style heights pass would suppress
same-channel peaks beyond the lockout), and --proposal-threshold plus
--max-proposals-per-pass bound the proposal count. Proposals carry no temporal or sigma
prior (indices -1, the 0017 convention); the fit searches the full Q x sigma x lattice
product and every downstream gate is untouched. Rejected convolving proposals carry
audit bit 256 in `rejected_reason`.
"""

import importlib.util
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "allchannel_0019_for_024", HERE / "0019_allchannel_peeling.py"
)
ALLCHANNEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ALLCHANNEL
SPEC.loader.exec_module(ALLCHANNEL)
PIPELINE = ALLCHANNEL.PIPELINE
OLD = ALLCHANNEL.OLD
BASE = ALLCHANNEL.BASE
EPS = ALLCHANNEL.EPS
_pass_all_channel_fraction = ALLCHANNEL.pass_all_channel_fraction
_all_channel_acceptance = ALLCHANNEL.all_channel_acceptance
_quantiles = ALLCHANNEL.quantiles
_concatenate_parts = ALLCHANNEL.concatenate_parts
_REJECTED_FIELDS = ALLCHANNEL._REJECTED_FIELDS
_0019_detect_events = ALLCHANNEL.detect_events

CONVOLVING_REJECTION_BIT = 256
_CONVOLVING_ONLY_CONFIG_FIELDS = (
    "detector",
    "score_merge",
    "proposal_lockout",
    "proposal_threshold",
    "gaussian_sigma_um",
    "growsum_channels",
    "max_proposals_per_pass",
)
_CONVOLVING_ONLY_DEFAULTS = {
    "score_merge": "perchannel",
    "proposal_lockout": 60,
    "proposal_threshold": 6.0,
    "gaussian_sigma_um": 30.0,
    "growsum_channels": 7,
    "max_proposals_per_pass": 2_000_000,
}


@dataclass(frozen=True)
class Config(ALLCHANNEL.Config):
    detector: str = "bipolar"
    score_merge: str = "perchannel"
    proposal_lockout: int = 60
    proposal_threshold: float = 6.0
    gaussian_sigma_um: float = 30.0
    growsum_channels: int = 7
    max_proposals_per_pass: int = 2_000_000


def output_metadata(config, recording_path, fs, n_channels, first, stop):
    metadata = ALLCHANNEL.output_metadata(
        config, recording_path, fs, n_channels, first, stop
    )
    if config.detector == "bipolar":
        for name in _CONVOLVING_ONLY_CONFIG_FIELDS:
            metadata["config"].pop(name)
        return metadata
    metadata["detector"] = (
        "dense temporal matched filter of the residual against every learned Omega row "
        "in one batched conv1d, s[c,t,q] = <r[c], Omega_q> / (sigma_c*||Omega_q||), "
        "S[c,t] = max_q |s| with the signed winning value kept, followed by the "
        "configured spatial score merge and spatiotemporal NMS"
    )
    metadata["discovery_score"] = (
        "spatially merged noise-normalized matched-filter score (max over Omega rows)"
    )
    metadata["discovery_threshold_units"] = (
        "merged matched-filter score in per-channel robust-noise standard deviations"
    )
    metadata["discovery_template_search"] = True
    metadata["discovery_template_scope"] = (
        "temporal Omega rows only; proposals carry no sigma or spatial prior "
        "(indices -1) and the fit searches the full Q x sigma x lattice product"
    )
    metadata["proposals"] = {
        "score_merge": {
            "perchannel": "S as computed, no spatial merge",
            "gaussian": (
                "M[c,t] = sum_c' w(c,c') S[c',t] / sqrt(sum w^2) with "
                "w = exp(-d^2/2sigma^2) over the 48 um channel-map neighborhood"
            ),
            "growsum": (
                "growing prefix sum over the nearest channels ranked by um distance, "
                "keeping the best prefix size under count normalization M/sqrt(j)"
            ),
        }[config.score_merge],
        "proposal_lockout_samples": config.proposal_lockout,
        "proposal_threshold": config.proposal_threshold,
        "max_proposals_per_pass": config.max_proposals_per_pass,
        "cross_channel_dedup": (
            "joint with the temporal lockout over the 48 um neighborhood using um "
            "distances from the channel map; a proposal survives only if no sample in "
            "that window is strictly larger in |M| and no equal |M| sits strictly "
            "earlier, so the earliest max wins ties"
        ),
        "rejection_audit_bit": CONVOLVING_REJECTION_BIT,
    }
    if config.score_merge == "gaussian":
        metadata["proposals"]["gaussian_sigma_um"] = config.gaussian_sigma_um
    if config.score_merge == "growsum":
        metadata["proposals"]["growsum_channels"] = config.growsum_channels
    metadata["passes"]["detection_threshold_per_pass"] = [
        config.proposal_threshold
    ] * config.recording_passes
    return metadata


def matched_filter_scores(residual, noise, omega):
    """s[c,t,q] = <r[c], Omega_q> / (sigma_c * ||Omega_q||) for windows starting at t."""
    omega = omega.float()
    norms = torch.linalg.vector_norm(omega, dim=1).clamp_min(EPS)
    standardized = (residual / noise[None]).T[None]
    n_channels = standardized.shape[1]
    weight = (omega / norms[:, None]).repeat(n_channels, 1).unsqueeze(1)
    projection = F.conv1d(standardized, weight, groups=n_channels)
    projection = projection[0].reshape(n_channels, len(omega), -1).permute(0, 2, 1)
    winning = projection.abs().argmax(dim=2)
    signed = torch.gather(projection, 2, winning[:, :, None]).squeeze(2)
    return signed


def gaussian_merge(scores, positions, config):
    delta = positions[:, None, :] - positions[None, :, :]
    d2 = delta.square().sum(dim=2)
    weights = torch.exp(-d2 / (2.0 * config.gaussian_sigma_um**2))
    weights = weights * (d2 <= config.radius_um**2).to(scores.dtype)
    norm = weights.square().sum(dim=1).sqrt().clamp_min(EPS)
    return (weights @ scores) / norm[:, None]


def growsum_merge(scores, positions, config):
    delta = positions[:, None, :] - positions[None, :, :]
    d2 = delta.square().sum(dim=2)
    count = min(config.growsum_channels, len(positions))
    order = torch.argsort(d2, dim=1, stable=True)[:, :count]
    prefix = scores[order].cumsum(dim=1)
    sizes = torch.sqrt(
        torch.arange(1, count + 1, device=scores.device, dtype=scores.dtype)
    )
    normalized = prefix / sizes[None, :, None]
    best = normalized.abs().argmax(dim=1)
    return torch.gather(normalized, 1, best[:, None, :]).squeeze(1)


def merge_scores(scores, positions, config):
    if config.score_merge == "perchannel":
        return scores
    if config.score_merge == "gaussian":
        return gaussian_merge(scores, positions, config)
    if config.score_merge == "growsum":
        return growsum_merge(scores, positions, config)
    raise ValueError(f"unknown score merge: {config.score_merge}")


def select_proposals(
    absolute,
    signed,
    safe_ids,
    valid_neighbors,
    config,
    valid_start,
    valid_stop,
):
    """Spatiotemporal NMS on |M| plus threshold and the per-pass hard cap.

    A candidate survives only if no sample within the 48 um neighborhood and the
    proposal-lockout window is strictly larger in |M| and no equal |M| sits strictly
    earlier (earliest max wins ties, equal-time cross-channel ties survive).
    """
    n_samples = absolute.shape[0]
    crossings = int((absolute >= config.proposal_threshold).sum().item())
    candidate = absolute >= config.proposal_threshold
    exclusive_start = max(valid_start, config.proposal_lockout + 1)
    exclusive_stop = max(
        min(valid_stop, n_samples - config.proposal_lockout - 1), 0
    )
    if exclusive_start > 0:
        candidate[:exclusive_start] = False
    if exclusive_stop < n_samples:
        candidate[exclusive_stop:] = False
    times, channels = torch.nonzero(candidate, as_tuple=True)
    offsets = torch.arange(
        -config.proposal_lockout,
        config.proposal_lockout + 1,
        device=absolute.device,
    )
    kept = []
    for start in range(0, len(times), config.detection_nms_batch_size):
        stop = min(start + config.detection_nms_batch_size, len(times))
        batch_times = times[start:stop]
        batch_channels = channels[start:stop]
        sample_grid = batch_times[:, None] + offsets[None]
        in_bounds = (sample_grid >= 0) & (sample_grid < n_samples)
        samples = sample_grid.clamp(0, n_samples - 1)
        neighbors = safe_ids[batch_channels]
        neighbor_valid = valid_neighbors[batch_channels]
        values = absolute[samples[:, :, None], neighbors[:, None, :]]
        valid = in_bounds[:, :, None] & neighbor_valid[:, None, :]
        values = values.masked_fill(~valid, float("-inf"))
        own = absolute[batch_times, batch_channels]
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
    nms_proposals = len(times)
    cap_enforced = False
    if (
        config.max_proposals_per_pass is not None
        and len(times) > config.max_proposals_per_pass
    ):
        cap_enforced = True
        _, selected = torch.topk(
            absolute[times, channels],
            config.max_proposals_per_pass,
            largest=True,
            sorted=False,
        )
        times = times[selected]
        channels = channels[selected]
    order = torch.argsort(times, stable=True)
    times = times[order]
    channels = channels[order]
    scores = signed[times, channels]
    counts = {
        "matched_filter_crossings": crossings,
        "spatiotemporal_nms_proposals": nms_proposals,
        "proposals": len(times),
        "positive_proposals": int((scores > 0).sum().item()),
        "negative_proposals": int((scores < 0).sum().item()),
        "proposal_cap_enforced": cap_enforced,
        "score_merge": config.score_merge,
    }
    return times, channels, scores, counts


def convolving_detect_events(
    residual,
    noise,
    omega,
    safe_detection_ids,
    valid_neighbors,
    positions,
    config,
    fs,
    valid_start,
    valid_stop,
):
    n_samples, n_channels = residual.shape
    n_before = int(round(config.ms_before * fs / 1000))
    merged = merge_scores(
        matched_filter_scores(residual, noise, omega),
        torch.as_tensor(positions, dtype=torch.float32, device=residual.device),
        config,
    )
    absolute = torch.full(
        (n_samples, n_channels),
        float("-inf"),
        dtype=torch.float32,
        device=residual.device,
    )
    signed = torch.zeros(
        (n_samples, n_channels), dtype=torch.float32, device=residual.device
    )
    n_windows = merged.shape[1]
    if n_windows > 0:
        absolute[n_before:n_before + n_windows] = merged.abs().T
        signed[n_before:n_before + n_windows] = merged.T
    times, channels, scores, counts = select_proposals(
        absolute,
        signed,
        safe_detection_ids,
        valid_neighbors,
        config,
        valid_start,
        valid_stop,
    )
    unavailable = torch.full_like(times, -1)
    return times, channels, scores, unavailable, unavailable, counts


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
    channel_positions=None,
):
    if config.detector == "convolving":
        valid_neighbors = footprints.abs().sum(dim=1) > 0
        return convolving_detect_events(
            residual,
            noise,
            omega,
            safe_detection_ids,
            valid_neighbors,
            channel_positions,
            config,
            fs,
            valid_start,
            valid_stop,
        )
    return _0019_detect_events(
        residual,
        noise,
        omega,
        footprints,
        safe_detection_ids,
        config,
        fs,
        valid_start,
        valid_stop,
    )


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
    noise_np = BASE.robust_channel_noise(data)
    noise = torch.as_tensor(noise_np, device=config.device)
    residual = data_tensor
    if prior_prediction is not None:
        residual = (data_tensor - prior_prediction).contiguous()
    omega_t = PIPELINE.orient_omega(torch.as_tensor(omega, device=config.device))
    omega_similarity = (omega_t @ omega_t.T).abs() >= config.duplicate_temporal_correlation
    safe_fit_ids, fit_mask = BASE.gpu_neighborhood(fit_ids, config.device)
    fit_offsets_t = torch.as_tensor(fit_offsets, device=config.device)
    detection_bank, safe_detection_ids = BASE.detection_footprints(
        fit_offsets, fit_ids, sigmas.detach().cpu().numpy(), noise_np, config.device
    )
    adjacency = PIPELINE.merge_adjacency(merge_ids, config.device)
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
    channel_fraction = _pass_all_channel_fraction(config, pass_index)
    proposal_rejection_bit = (
        CONVOLVING_REJECTION_BIT if config.detector == "convolving" else 0
    )
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
            channel_positions,
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
            fit = PIPELINE.fit_grouped(
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
            all_ok, min_channel_fraction = _all_channel_acceptance(
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
            reasons = torch.zeros(
                (len(batch_times),),
                dtype=torch.int32,
                device=config.device,
            )
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
            duplicate = PIPELINE.duplicate_mask(
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
                    "all_channel_ok": BASE.tensor_numpy(
                        batch["all_ok"], in_core
                    ).astype(np.int8),
                    "all_channel_fraction": np.full(
                        count, channel_fraction, dtype=np.float32
                    ),
                    "min_channel_captured_fraction": BASE.tensor_numpy(
                        batch["min_channel_fraction"], in_core
                    ).astype(np.float32),
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
                    "recording_pass": np.full(count, pass_index, dtype=np.int16),
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
                logged_reasons = reasons | proposal_rejection_bit
                if not bool(logged.any()):
                    continue
                sel = torch.nonzero(logged, as_tuple=False).squeeze(1)
                count = len(sel)
                rejected_parts.append(
                    {
                        "rejected_spike_times": (
                            read_start + BASE.tensor_numpy(batch["times"], sel)
                        ).astype(np.int64),
                        "rejected_spike_channels": BASE.tensor_numpy(
                            batch["channels"], sel
                        ).astype(np.int32),
                        "rejected_detection_score": BASE.tensor_numpy(
                            batch["detection_score"], sel
                        ).astype(np.float32),
                        "rejected_sigma_index": BASE.tensor_numpy(
                            fit["sigma_index"], sel
                        ).astype(np.int16),
                        "rejected_temporal_index": BASE.tensor_numpy(
                            fit["temporal_index"], sel
                        ).astype(np.int16),
                        "rejected_alpha": BASE.tensor_numpy(fit["alpha"], sel).astype(np.float32),
                        "rejected_captured_fraction": BASE.tensor_numpy(
                            fit["captured_fraction"], sel
                        ).astype(np.float32),
                        "rejected_projection_score": BASE.tensor_numpy(
                            fit["fitted_projection_score"], sel
                        ).astype(np.float32),
                        "rejected_max_channel_rmse": BASE.tensor_numpy(
                            fit["maximum_channel_normalized_rmse"], sel
                        ).astype(np.float32),
                        "rejected_min_channel_fraction": BASE.tensor_numpy(
                            batch["min_channel_fraction"], sel
                        ).astype(np.float32),
                        "rejected_all_ok": BASE.tensor_numpy(
                            batch["all_ok"], sel
                        ).astype(np.int8),
                        "rejected_reason": BASE.tensor_numpy(logged_reasons, sel).astype(np.int32),
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
                    BASE.tensor_numpy(logged_reasons, sel), return_counts=True
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
            "proposal_score_quantiles": _quantiles(detection_score),
            "fitted_score_quantiles": _quantiles(fitted_scores),
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
    result = _concatenate_parts(parts, fit_ids.shape[1], waveform_length, config.save_waveforms)
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


def validate_config(config):
    ALLCHANNEL.validate_config(config)
    if config.detector not in ("bipolar", "convolving"):
        raise ValueError("detector must be bipolar or convolving")
    if config.detector == "bipolar":
        for name, default in _CONVOLVING_ONLY_DEFAULTS.items():
            if getattr(config, name) != default:
                raise ValueError(
                    f"--{name.replace('_', '-')} applies only to --detector convolving"
                )
        return
    if config.score_merge not in ("perchannel", "gaussian", "growsum"):
        raise ValueError("score merge must be perchannel, gaussian, or growsum")
    if config.proposal_lockout < 0:
        raise ValueError("proposal lockout must be nonnegative")
    if config.proposal_threshold <= 0:
        raise ValueError("proposal threshold must be positive")
    if config.gaussian_sigma_um <= 0:
        raise ValueError("gaussian sigma must be positive")
    if config.growsum_channels < 1:
        raise ValueError("growsum channel count must be positive")
    if config.max_proposals_per_pass < 1:
        raise ValueError("max proposals per pass must be positive")


def self_test(device):
    config = Config(device=device, detector="convolving")
    length = 2000
    waveform_length = 90
    omega = torch.zeros(1, waveform_length, device=device)
    omega[0] = 1.0 / waveform_length**0.5
    residual = torch.zeros(length, 2, device=device)
    residual[500:500 + waveform_length, 0] = 8.0 * omega[0]
    noise = torch.ones(2, device=device)
    positions_np = np.array([[0.0, 0.0], [32.0, 0.0]], dtype=np.float32)
    ids, _, _ = BASE.build_neighborhoods(positions_np, 48.0)
    safe_ids, valid_neighbors = BASE.gpu_neighborhood(ids, torch.device(device))
    footprints = torch.ones(2, 1, 2, device=device)
    if not float(residual.abs().max()) < config.proposal_threshold:
        raise AssertionError("planted spike must stay below the single-sample threshold")
    times, channels, scores, initial_sigma, initial_temporal, counts = detect_events(
        residual,
        noise,
        omega,
        footprints,
        safe_ids,
        config,
        30000.0,
        45,
        length - 45,
        positions_np,
    )
    if times.tolist() != [545] or channels.tolist() != [0]:
        raise AssertionError("matched-filter nomination of the weak spike is incorrect")
    if not torch.allclose(scores, torch.tensor([8.0], device=device), atol=1e-4):
        raise AssertionError("matched-filter signed score is incorrect")
    if initial_sigma.tolist() != [-1] or initial_temporal.tolist() != [-1]:
        raise AssertionError("convolving proposals must carry -1 sigma/temporal indices")
    if counts["proposals"] != 1 or counts["matched_filter_crossings"] < 1:
        raise AssertionError("convolving proposal counts are incorrect")

    residual[1500:1500 + waveform_length, 0] = 6.5 * omega[0]
    capped_config = replace(config, max_proposals_per_pass=1)
    times, channels, scores, _, _, counts = detect_events(
        residual,
        noise,
        omega,
        footprints,
        safe_ids,
        capped_config,
        30000.0,
        45,
        length - 45,
        positions_np,
    )
    if times.tolist() != [545] or not counts["proposal_cap_enforced"]:
        raise AssertionError("per-pass proposal cap is not enforced")

    weak = torch.zeros(length, 2, device=device)
    weak[500:500 + waveform_length, 0] = 5.5 * omega[0]
    times, _, _, _, _, counts = detect_events(
        weak,
        noise,
        omega,
        footprints,
        safe_ids,
        config,
        30000.0,
        45,
        length - 45,
        positions_np,
    )
    if len(times) or counts["proposals"] != 0:
        raise AssertionError("below-threshold matched-filter scores must not be proposed")

    nms_positions = np.array(
        [[0.0, 0.0], [30.0, 0.0], [200.0, 200.0]], dtype=np.float32
    )
    nms_ids, _, _ = BASE.build_neighborhoods(nms_positions, 48.0)
    nms_safe_ids, nms_valid = BASE.gpu_neighborhood(nms_ids, torch.device(device))
    nms_config = replace(config, proposal_lockout=60)

    def grid(values):
        absolute = torch.full((300, 3), float("-inf"), device=device)
        signed = torch.zeros(300, 3, device=device)
        for time, channel, score in values:
            absolute[time, channel] = abs(score)
            signed[time, channel] = score
        return absolute, signed

    absolute, signed = grid([(100, 0, 10.0), (100, 1, -8.0)])
    times, channels, scores, _ = select_proposals(
        absolute, signed, nms_safe_ids, nms_valid, nms_config, 0, 300
    )
    if times.tolist() != [100] or channels.tolist() != [0] or scores.tolist() != [10.0]:
        raise AssertionError("cross-channel NMS did not keep the larger merged score")
    absolute, signed = grid([(100, 0, 8.0), (100, 1, 10.0)])
    times, channels, scores, _ = select_proposals(
        absolute, signed, nms_safe_ids, nms_valid, nms_config, 0, 300
    )
    if times.tolist() != [100] or channels.tolist() != [1]:
        raise AssertionError("cross-channel NMS must keep the larger channel's proposal")
    absolute, signed = grid([(100, 0, 10.0), (105, 1, 8.0)])
    times, _, _, _ = select_proposals(
        absolute, signed, nms_safe_ids, nms_valid, nms_config, 0, 300
    )
    if times.tolist() != [100]:
        raise AssertionError("time-offset cross-channel duplicate did not merge")
    absolute, signed = grid([(100, 0, 9.0), (130, 0, 7.0)])
    times, _, _, _ = select_proposals(
        absolute, signed, nms_safe_ids, nms_valid, nms_config, 0, 300
    )
    if times.tolist() != [100]:
        raise AssertionError("peaks within the lockout window must merge")
    absolute, signed = grid([(100, 0, 9.0), (161, 0, 7.0)])
    times, _, _, _ = select_proposals(
        absolute, signed, nms_safe_ids, nms_valid, nms_config, 0, 300
    )
    if times.tolist() != [100, 161]:
        raise AssertionError("peaks just outside the lockout window must both survive")
    absolute, signed = grid([(100, 0, 9.0), (130, 0, 9.0)])
    times, channels, scores, _ = select_proposals(
        absolute, signed, nms_safe_ids, nms_valid, nms_config, 0, 300
    )
    if times.tolist() != [100] or channels.tolist() != [0] or scores.tolist() != [9.0]:
        raise AssertionError("tie must keep the earliest maximum")
    absolute, signed = grid([(100, 0, 9.0), (100, 1, 9.0)])
    times, _, _, _ = select_proposals(
        absolute, signed, nms_safe_ids, nms_valid, nms_config, 0, 300
    )
    if times.tolist() != [100, 100]:
        raise AssertionError("equal-time cross-channel ties must survive")

    rng = np.random.default_rng(11)
    scores_np = rng.standard_normal((6, 40)).astype(np.float32)
    merge_positions = np.array(
        [[0.0, 0.0], [30.0, 0.0], [65.0, 0.0], [0.0, 95.0], [30.0, 95.0], [200.0, 200.0]],
        dtype=np.float32,
    )
    scores_t = torch.from_numpy(scores_np).to(device)
    merge_positions_t = torch.from_numpy(merge_positions).to(device)
    merged = merge_scores(scores_t, merge_positions_t, replace(config, score_merge="perchannel"))
    if not torch.equal(merged, scores_t):
        raise AssertionError("perchannel merge must return the score unchanged")
    d2 = ((merge_positions[:, None, :] - merge_positions[None, :, :]) ** 2).sum(axis=2)
    weights = np.exp(-d2 / (2.0 * 30.0**2)) * (d2 <= 48.0**2)
    reference = (weights @ scores_np) / np.sqrt((weights**2).sum(axis=1))[:, None]
    merged = merge_scores(
        scores_t,
        merge_positions_t,
        replace(config, score_merge="gaussian", gaussian_sigma_um=30.0),
    )
    if not torch.allclose(
        merged.cpu(), torch.from_numpy(reference), atol=1e-4, rtol=1e-4
    ):
        raise AssertionError("gaussian merge disagrees with the brute-force reference")
    count = min(7, len(merge_positions))
    order = np.argsort(d2, axis=1, kind="stable")[:, :count]
    best_abs = np.full(scores_np.shape, -np.inf, dtype=np.float64)
    best_signed = np.zeros(scores_np.shape, dtype=np.float64)
    for size in range(1, count + 1):
        prefix = scores_np[order[:, :size]].sum(axis=1) / np.sqrt(size)
        update = np.abs(prefix) > best_abs
        best_abs = np.where(update, np.abs(prefix), best_abs)
        best_signed = np.where(update, prefix, best_signed)
    merged = merge_scores(
        scores_t,
        merge_positions_t,
        replace(config, score_merge="growsum", growsum_channels=7),
    )
    if not torch.allclose(
        merged.cpu(), torch.from_numpy(best_signed.astype(np.float32)), atol=1e-4, rtol=1e-4
    ):
        raise AssertionError("growsum merge disagrees with the brute-force reference")

    for name, value in (
        ("score_merge", "gaussian"),
        ("proposal_lockout", 5),
        ("proposal_threshold", 7.0),
        ("gaussian_sigma_um", 20.0),
        ("growsum_channels", 5),
        ("max_proposals_per_pass", 1000),
    ):
        try:
            validate_config(replace(Config(device="cuda"), **{name: value}))
        except ValueError:
            continue
        raise AssertionError(f"bipolar mode must reject the convolving-only flag {name}")
    try:
        validate_config(replace(Config(device="cuda"), detector="template"))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown detector must be rejected")
    validate_config(Config(device="cuda"))
    validate_config(
        Config(device="cuda", detector="convolving", score_merge="gaussian")
    )

    bipolar_config = Config(device=device)
    bipolar_residual = torch.zeros(128, 2, device=device)
    bipolar_residual[40, 0] = 6
    bipolar_residual[70, 1] = -7
    bipolar_residual[90, 0] = 6
    bipolar_residual[91, 1] = -8
    routing_ids = torch.tensor([[0, 1], [0, 1]], device=device)
    routing_valid = torch.ones(2, 2, dtype=torch.bool, device=device)
    routing_bank = torch.ones(2, 1, 2, device=device)
    routing_positions = np.array([[0.0, 0.0], [32.0, 0.0]], dtype=np.float32)
    mine = detect_events(
        bipolar_residual,
        torch.ones(2, device=device),
        torch.empty(1, 10, device=device),
        routing_bank,
        routing_ids,
        bipolar_config,
        1000.0,
        0,
        128,
        routing_positions,
    )
    reference = _0019_detect_events(
        bipolar_residual,
        torch.ones(2, device=device),
        torch.empty(1, 10, device=device),
        routing_bank,
        routing_ids,
        bipolar_config,
        1000.0,
        0,
        128,
    )
    if mine[0].tolist() != [40, 70, 91] or mine[1].tolist() != [0, 1, 1]:
        raise AssertionError("bipolar routing changed 0019 detection behavior")
    if mine[2].tolist() != [6.0, -7.0, -8.0]:
        raise AssertionError("bipolar routing changed the signed scores")
    if mine[3].tolist() != [-1, -1, -1] or mine[4].tolist() != [-1, -1, -1]:
        raise AssertionError("bipolar routing changed the proposal sentinels")
    if mine[5] != reference[5]:
        raise AssertionError("bipolar routing changed the detector counts")

    bipolar_metadata = output_metadata(
        Config(device="cuda"),
        Path("rec.bin"),
        30000.0,
        2,
        0,
        1000,
    )
    allchannel_metadata = ALLCHANNEL.output_metadata(
        ALLCHANNEL.Config(device="cuda"),
        Path("rec.bin"),
        30000.0,
        2,
        0,
        1000,
    )
    if bipolar_metadata != allchannel_metadata:
        raise AssertionError("bipolar metadata must match 0019 exactly")
    convolving_metadata = output_metadata(
        Config(device="cuda", detector="convolving", score_merge="growsum"),
        Path("rec.bin"),
        30000.0,
        2,
        0,
        1000,
    )
    if convolving_metadata["proposals"]["rejection_audit_bit"] != CONVOLVING_REJECTION_BIT:
        raise AssertionError("convolving metadata must record the rejection bit")

    print("024 self-test passed", flush=True)


def main():
    PIPELINE.Config = Config
    PIPELINE.output_metadata = output_metadata
    PIPELINE.detect_events = detect_events
    PIPELINE.process_chunk = process_chunk
    PIPELINE.validate_config = validate_config
    PIPELINE.self_test = self_test
    PIPELINE.alternating_fit = ALLCHANNEL.alternating_fit
    PIPELINE.orient_omega = ALLCHANNEL.preserve_omega_polarity
    PIPELINE.pursue = ALLCHANNEL.pursue
    PIPELINE.empty_chunk = ALLCHANNEL.empty_chunk
    PIPELINE.concatenate_parts = ALLCHANNEL.concatenate_parts
    OLD.calibration_detect = ALLCHANNEL.calibration_detect
    ALLCHANNEL.process_chunk = process_chunk
    ALLCHANNEL.detect_events = detect_events
    PIPELINE.main()


if __name__ == "__main__":
    main()
