"""Two-prototype temporal-cone calibration with 0017 residual pursuit."""

import importlib.util
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "onehot_0016_for_0018", HERE / "0016_onehot_lattice_peeling.py"
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


def process_chunk(*args, **kwargs):
    result = _process_chunk_0016(*args, **kwargs)
    signed_score = result["detection_score"]
    result["absolute_detection_score"] = np.abs(signed_score).astype(np.float32)
    result["detection_polarity"] = np.where(
        signed_score > 0, 1, -1
    ).astype(np.int8)
    result["detection_amplitude"] = (
        signed_score * result["noise"][result["spike_channels"]]
    ).astype(np.float32)
    result["temporal_prototype_index"] = (
        result["temporal_index"] % 2
    ).astype(np.int8)
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
        raise ValueError("0018 fixes the temporal prototype count at two")
    if config.q < config.prototype_count:
        raise ValueError(
            "the temporal codebook must contain at least one atom per prototype"
        )
    if not 0 < config.prototype_cone_deg < 90:
        raise ValueError("prototype cone angle must be strictly between 0 and 90 degrees")
    if config.prototype_kmeans_iterations < 1:
        raise ValueError("prototype spherical k-means iterations must be positive")
    if config.omega_prior:
        raise ValueError("0018 learns its constrained codebook from calibration events")
    if not config.positive_gain:
        raise ValueError("0018 requires nonnegative gains so prototype polarity is identifiable")


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
    print("0018 self-test passed", flush=True)


def main():
    PIPELINE.Config = Config
    PIPELINE.output_metadata = output_metadata
    PIPELINE.detect_events = detect_events
    PIPELINE.process_chunk = process_chunk
    PIPELINE.validate_config = validate_config
    PIPELINE.self_test = self_test
    PIPELINE.alternating_fit = alternating_fit
    PIPELINE.orient_omega = preserve_omega_polarity
    OLD.calibration_detect = calibration_detect
    PIPELINE.main()


if __name__ == "__main__":
    main()
