"""All-channel peeling: 0018 plus a per-channel reconstruction bar and recording passes.

Same model as 0018: alpha>=0 * monopole(x,y,z,sigma) * Omega[q], q one-hot, with the
two-prototype cone codebook. Two things change:

    1. The fit objective is TOTAL channel error (mean-channel noise-normalized SSE)
       instead of the worst channel, so a narrow template that touches only the peak
       channel can no longer win the position search.
    2. Acceptance requires EVERY valid channel in the fit mask to capture at least
       `all_channel_min_fraction` of its OWN noise-normalized energy. The bar tightens
       by `pass_fraction_step` per recording pass. Detection stays at the same
       threshold on every pass; only the reconstruction bar escalates.

Structure: the recording is walked `recording_passes` times. Each chunk visit runs
`peeling_rounds` (default 1) detect-fit-subtract rounds. Passes 2+ rebuild each chunk's
starting residual on the GPU by replaying every saved event from earlier passes onto
the raw chunk, so the loop continues exactly where the last pass ended. A chunk visit
that accepts no events marks the chunk exhausted, and later passes skip it: its interior
residual is unchanged and the acceptance bar only escalates, so re-detection could only
reproduce the same rejected proposals. Every detected
candidate that is not accepted is logged with a reason bitmask and the rejected/rollback
audit tables are consolidated per pass and at the run root; nothing is silently discarded.
"""

import importlib.util
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "onehot_0016_for_0019", HERE / "0016_onehot_lattice_peeling.py"
)
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)
OLD = PIPELINE.OLD
BASE = PIPELINE.BASE
EPS = PIPELINE.EPS
_output_metadata_0016 = PIPELINE.output_metadata
_validate_config_0016 = PIPELINE.validate_config
_process_chunk_0016 = PIPELINE.process_chunk


@dataclass(frozen=True)
class Config(PIPELINE.Config):
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
    log_rejections: bool = True


def output_metadata(config, recording_path, fs, n_channels, first, stop):
    metadata = _output_metadata_0016(
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
            "every valid channel must capture at least the pass fraction of its own "
            "noise-normalized input energy" if config.all_channel_improvement else
            f"at least {config.min_improved_channels} channels improve"
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


def empty_chunk(width, waveform_length, save_waveforms):
    result = BASE.empty_chunk(width, waveform_length, save_waveforms)
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
    output = Path(output)
    summaries_path = output / "pass_summaries.json"
    summaries = (
        json.loads(summaries_path.read_text()) if summaries_path.exists() else []
    )
    stopped_after = next(
        (entry["recording_pass"] for entry in summaries if entry.get("stopped")), None
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
        if stopped_after is not None and pass_index > stopped_after:
            stopping_reason = f"stopped_after_pass_{stopped_after}"
            break
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
            else:
                core_stop = min(core_start + chunk_samples, stop)
            read_start = max(0, core_start - margin)
            read_stop = min(reader.ns, core_stop + margin)
            data = BASE.preprocess_voltage(
                reader[read_start:read_stop, :n_channels], sos
            )
            null_shifts = None
            if config.empirical_null:
                data, null_shifts = PIPELINE.shifted_channel_null(
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
            OLD.atomic_npz(path, result)
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
        OLD.atomic_json(pass_dir / "consolidation.json", pass_summary)
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
        OLD.atomic_json(summaries_path, summaries)
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
    OLD.atomic_json(output / "summary.json", summary)
    OLD.atomic_json(output / "pursuit_footprint_cache.json", cache.diagnostics())





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
            channel_floor = channel_input * channel_fraction
            all_ok = (
                (fit["channel_improvement"] >= channel_floor - 1e-6) | ~mask
            ).all(dim=1)
            min_channel_fraction = torch.where(
                channel_input > 1e-8,
                fit["channel_improvement"] / channel_input.clamp_min(1e-8),
                torch.full_like(channel_input, float("inf")),
            ).amin(dim=1)
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
                        "rejected_reason": BASE.tensor_numpy(reasons, sel).astype(np.int32),
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
                    BASE.tensor_numpy(reasons, sel), return_counts=True
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
    for _, waveforms_np, _, _, mask_np in OLD.iter_calibration_batches(
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
    root, shard_dir = OLD.calibration_paths(output)
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
    safe_ids, valid_neighbors = BASE.gpu_neighborhood(
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
        data = BASE.preprocess_voltage(
            reader[read_start:read_stop, :n_channels], sos
        )
        noise = BASE.robust_channel_noise(data)
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
        keep = BASE.isolated_events(
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
        OLD.atomic_npz(
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
    OLD.atomic_json(
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
    root, shards = OLD.calibration_paths(output)
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
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=config.device)
    cache = OLD.FootprintCache(sites, sigmas, config.device)
    history = []
    assignment_root = root / "assignments"

    OLD.atomic_npy(root / "initial_omega.npy", omega.cpu().numpy().astype(np.float32))
    OLD.atomic_npy(
        root / "initial_prototypes.npy",
        prototypes.cpu().numpy().astype(np.float32),
    )
    OLD.atomic_json(
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
            OLD.iter_calibration_batches(reader, shards, fs, fit_ids, sos, config)
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
                fit = PIPELINE.fit_grouped(
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
                OLD.atomic_npz(
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
        OLD.atomic_json(history_path, history)
        OLD.atomic_npy(omega_path, omega.cpu().numpy().astype(np.float32))
        OLD.atomic_npy(
            prototypes_path, prototypes.cpu().numpy().astype(np.float32)
        )
        OLD.atomic_npy(
            assignment_path, assignment.cpu().numpy().astype(np.int16)
        )
        print(json.dumps(history[-1]), flush=True)
        if not accepted or change < config.alternating_tolerance:
            break

    OLD.atomic_json(root / "footprint_cache.json", cache.diagnostics())
    OLD.atomic_json(
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
    OLD.atomic_npy(output / "omega.npy", omega)
    OLD.atomic_npy(output / "prototypes.npy", prototypes)
    OLD.atomic_npy(output / "atom_prototype.npy", assignment)
    OLD.atomic_json(
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
    _validate_config_0016(config)
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
            OLD.atomic_npz(
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

    print("0019 self-test passed", flush=True)


def main():
    PIPELINE.Config = Config
    PIPELINE.output_metadata = output_metadata
    PIPELINE.detect_events = detect_events
    PIPELINE.process_chunk = process_chunk
    PIPELINE.validate_config = validate_config
    PIPELINE.self_test = self_test
    PIPELINE.alternating_fit = alternating_fit
    PIPELINE.orient_omega = preserve_omega_polarity
    PIPELINE.pursue = pursue
    PIPELINE.empty_chunk = empty_chunk
    PIPELINE.concatenate_parts = concatenate_parts
    OLD.calibration_detect = calibration_detect
    PIPELINE.main()


if __name__ == "__main__":
    main()
