"""Resumable XYZ-sigma, one-waveform residual pursuit.

This is deliberately a new entry point.  It retains session 0012's tested
SpikeGLX detection and subtraction semantics, but learns and uses a frozen
one-row temporal codebook with grouped discrete ``(x, y, z, sigma, q)`` fits.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("residuals_0012", HERE / "residuals_0012.py")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)
EPS = BASE.EPS


@dataclass(frozen=True)
class Config:
    q: int = 8
    threshold: float = 8.0
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
    min_fitted_projection: float = 8.0
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
        return BASE.Config(
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
    BASE.atomic_json(path, value)


def atomic_npy(path, value):
    BASE.atomic_npy(path, value)


def atomic_npz(path, values):
    BASE.atomic_npz(path, values)


def output_metadata(config, recording_path, fs, n_channels, first, stop):
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
        "sigma_values_um": BASE.sigma_bank(config.base()).tolist(),
        "lattice_bounds_um": [list(BASE.XYZ_LO), list(BASE.XYZ_HI)],
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
            value = BASE.monopole_footprint(off, self.sites, self.sigmas, valid)[0]
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


def fit_grouped(waveforms, offsets, mask, local_noise, omega, sites, axes, sigmas, config, cache):
    """Cached coarse scoring plus session-0012's ordered integer refinement."""
    omega = F.normalize(omega, dim=1)
    site_index, _, _, _, projected, channel_energy = grouped_coarse_assignment(
        waveforms, offsets, mask, local_noise, omega, sites, sigmas, config, cache)
    source, coarse, sigma_index, temporal_index, alpha, objective, levels = BASE.refine_sites(
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


def calibration_detect(reader, output, first, stop, offsets, fit_ids, merge_ids, sos, config, resume):
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
        data = BASE.preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
        noise = BASE.robust_channel_noise(data)
        times, channels, _ = BASE.raw_negative_peaks(data, noise, merge_ids, config.threshold,
                                                      peak_radius, config.device)
        times, channels = times.cpu().numpy(), channels.cpu().numpy()
        keep = ((times >= core_start - read_start) & (times < core_stop - read_start) &
                (times >= before) & (times + after <= len(data)))
        keep &= BASE.isolated_events(times, channels, merge_ids, isolation)
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
        data = BASE.preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
        noise = BASE.robust_channel_noise(data)
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


def alternating_fit(reader, output, fs, fit_ids, offsets, sos, config, resume):
    root, shards = calibration_paths(output)
    omega_path, history_path = root / "omega.npy", root / "alternating_history.json"
    if resume and omega_path.exists() and history_path.exists():
        return np.load(omega_path).astype(np.float32)
    omega = initial_omega(reader, shards, fs, fit_ids, sos, config)
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=config.device)
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
                fit = fit_grouped(waveforms, local_offsets, mask, local_noise, omega,
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


def pursue(reader, output, first, stop, positions, fit_ids, offsets, counts, merge_ids, sos,
           omega, config, resume):
    """Run one-second fresh residual chunks, replacing only the localizer with the cache."""
    fs, n_channels = float(reader.fs), len(positions)
    chunk_dir = Path(output) / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=config.device)
    cache = FootprintCache(sites, sigmas, config.device)
    original = BASE.fit_spatial_batch
    def cached_fit(waveforms, local_offsets, mask, local_noise, omega_t, _sites, _axes, _sigmas, base_config):
        return fit_grouped(waveforms, local_offsets, mask, local_noise, omega_t, sites, axes,
                           sigmas, base_config, cache)
    BASE.fit_spatial_batch = cached_fit
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
            data = BASE.preprocess_voltage(reader[read_start:read_stop, :n_channels], sos)
            with torch.inference_mode():
                result = BASE.process_chunk(data, read_start, core_start, core_stop, positions, fit_ids,
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
        BASE.fit_spatial_batch = original
    summary = consolidate_chunks(chunk_dir, Path(output))
    atomic_json(Path(output) / "summary.json", summary)
    atomic_json(Path(output) / "pursuit_footprint_cache.json", cache.diagnostics())


def validate_config(config):
    if config.calibration_max_events < config.q or config.calibration_chunks < 1:
        raise ValueError("calibration limits must provide at least Q events and one chunk")
    BASE.validate_config(config.base())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibration-detect", "alternating-fit", "pursue", "all"))
    parser.add_argument("recording_path", type=Path)
    parser.add_argument("output_path", type=Path)
    for field in Config.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=field.default)
        else:
            parser.add_argument(name, type=type(field.default), default=field.default)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    validate_config(config)
    import spikeglx
    output = args.output_path
    if output.exists() and not args.resume and args.stage != "calibration-detect":
        raise FileExistsError(f"refusing to overwrite {output}; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    reader = spikeglx.Reader(args.recording_path)
    try:
        fs = float(reader.fs)
        positions = np.column_stack((reader.geometry["x"], reader.geometry["y"])).astype(np.float32)
        fit_ids, offsets, _ = BASE.build_neighborhoods(positions, config.radius_um)
        merge_ids, _, _ = BASE.build_neighborhoods(positions, config.merge_radius_um)
        sos = BASE.make_filter(fs, config.base())
        first = max(0, int(round(args.start_seconds * fs)))
        stop = reader.ns if args.duration_seconds is None else min(reader.ns, first + int(round(args.duration_seconds * fs)))
        metadata = output_metadata(config, args.recording_path, fs, len(positions), first, stop)
        atomic_json(output / "config.json", metadata)
        atomic_json(output / "metadata.json", metadata)
        atomic_npy(output / "channel_positions.npy", positions)
        atomic_npy(output / "fit_neighborhood_ids.npy", fit_ids)
        atomic_npy(output / "fit_neighborhood_offsets.npy", offsets)
        atomic_npy(output / "merge_neighborhood_ids.npy", merge_ids)
        if args.stage in ("calibration-detect", "all"):
            calibration_detect(reader, output, first, stop, offsets, fit_ids, merge_ids, sos, config, args.resume)
        if args.stage in ("alternating-fit", "all"):
            omega = alternating_fit(reader, output, fs, fit_ids, offsets, sos, config, args.resume)
            atomic_npy(output / "omega.npy", omega)
        else:
            omega_path = calibration_paths(output)[0] / "omega.npy"
            if args.stage == "pursue" and not omega_path.exists():
                raise FileNotFoundError(f"{omega_path} is required for pursue")
            omega = np.load(omega_path).astype(np.float32) if omega_path.exists() else None
        if args.stage in ("pursue", "all"):
            pursue(reader, output, first, stop, positions, fit_ids, offsets, fit_ids.shape[1] - (fit_ids < 0).sum(1),
                   merge_ids, sos, omega, config, args.resume)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
