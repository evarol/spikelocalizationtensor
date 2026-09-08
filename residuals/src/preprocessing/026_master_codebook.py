"""Master temporal codebook (0026): one Omega for all recordings, learned from every IBL mouse.

Every run in the 0018/0019/0024/0025 lineage calibrates its own temporal codebook
from a 100k-event pool drawn from the single recording being peeled. This module
learns ONE Q32 two-prototype-cone codebook from the pooled event waveforms of all
812 public IBL Brain-Wide-Map probe recordings (first ~100 s AP clips, mtscomp
SpikeGLX triplets), so downstream runs can consume a shared, recording-independent
Omega.

The shape machinery is 0019's exactly, loaded via importlib (the 0019 import pulls
0016 and 0012 with it): two polarity prototypes, atoms seeded by spherical k-means
over real peak-aligned waveforms and projected into 35-degree cones, then the
alternating fixed-assignment fit with cone-projected proposals and backtracked
updates. Q = 32 atoms alternate between the two polarity cones. The Config
inherits 0019's production defaults (threshold 5, 48 um neighborhoods, the 0014
lattice and sigma bank) so the fit path behaves identically to a production
calibration.

What differs from a per-recording calibration is only the data source:

    harvest  one pass per clip: read_cbin_ibl -> preprocess -> locally-exclusive
             peaks (threshold 5, both polarities, 1 ms sweep, 1 ms isolation) ->
             48 um neighborhood waveforms -> one atomic npz shard per clip, with
             each event's own anchor-relative offsets and mask. Shards are
             self-contained, so the alternating fit never re-decompresses raw.
    init     pool every shard's peak-aligned waveforms, unit-normalize, polarity
             label, then 0019's initialize_codebook (Q atoms alternate between the
             two polarity cones).
    fit      0019's alternating loop over the pooled shards: fixed-Omega
             fit_grouped per batch, per-atom numerator/denominator/gain
             accumulation, prototype_cone_proposal, backtracked_update with
             objective guard, atomic checkpoint per iteration.

The waveform window is FIXED in samples (2 x round(1.5 ms x 30 kHz) = 90) rather
than per-clip milliseconds, so every shard and the learned Omega share one T even
though clip rates vary 29.9-30.0 kHz. Geometry differs per probe (NP1.0/NP2.0)
but the fit only ever sees anchor-relative offsets, and each event carries its
own, so probes pool without any rescaling.

Usage (through singularity, like everything in this pipeline):
    python residuals/src/preprocessing/026_master_codebook.py all \
        --data-root /scratch/ap7151/_RAW_DATA/ibl-data \
        --output residuals/runs/ibl_bwm/master_codebook_q32 --resume
Stages can run separately (harvest, init, fit) and each is resumable: harvest
skips clips whose shard exists, init and fit check their checkpoint files.
--clip-start/--clip-stop/--limit-clips partition the harvest for parallel jobs.
"""

import argparse
import concurrent.futures
import importlib.util
from dataclasses import dataclass, fields
import json
import multiprocessing
from pathlib import Path
import sys
from time import perf_counter
import zlib

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "allchannel_0019_for_0026", HERE / "0019_allchannel_peeling.py"
)
N19 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = N19
SPEC.loader.exec_module(N19)
PIPELINE = N19.PIPELINE
BASE = N19.BASE
EPS = N19.EPS
# 0016 exposes fit_grouped; FootprintCache lives one import deeper (0014, 0016's OLD)
FOOTPRINT_CACHE = PIPELINE.OLD.FootprintCache


@dataclass(frozen=True)
class Config(N19.Config):
    q: int = 32
    chunk_seconds: float = 10.0
    reference_fs: float = 30000.0
    harvest_events_per_clip: int = 4000
    harvest_workers: int = 8
    fit_pool_events_per_clip: int = 250


def atomic_json(path, value):
    BASE.atomic_json(path, value)


def atomic_npy(path, value):
    BASE.atomic_npy(path, value)


def atomic_npz(path, values):
    BASE.atomic_npz(path, values)


def enumerate_clips(root):
    clips = []
    for probe_dir in sorted(Path(root).glob("*/*/*/*")):
        if not probe_dir.is_dir():
            continue
        cbins = list(probe_dir.glob("*.ap.cbin"))
        if len(cbins) != 1:
            continue
        lab, subject, session, probe = probe_dir.parts[-4:]
        clips.append({
            "name": f"{lab}--{subject}--{session}--{probe}",
            "lab": lab, "subject": subject, "session": session, "probe": probe,
            "dir": probe_dir, "cbin": cbins[0],
        })
    if not clips:
        raise FileNotFoundError(f"no *.ap.cbin clips under {root}")
    return clips


def shard_path(shard_dir, clip):
    return shard_dir / f"clip_{clip['name']}.npz"


def clip_seed(clip, config):
    return config.seed + zlib.crc32(clip["name"].encode())


def harvest_clip(clip, path, config):
    from spikeinterface.extractors import read_cbin_ibl

    device = config.device
    rec = read_cbin_ibl(clip["dir"])
    fs = float(rec.get_sampling_frequency())
    ns = int(rec.get_num_frames())
    positions = np.asarray(rec.get_property("location"), dtype=np.float32)
    fit_ids, offsets, _ = BASE.build_neighborhoods(positions, config.radius_um)
    merge_ids, _, _ = BASE.build_neighborhoods(positions, config.merge_radius_um)
    sos = BASE.make_filter(fs, config.base())
    before_ref = int(round(config.ms_before * config.reference_fs / 1000))
    after_ref = int(round(config.ms_after * config.reference_fs / 1000))
    sample_offsets = np.arange(-before_ref, after_ref, dtype=np.int64)
    temporal_radius = max(1, int(config.exclude_sweep_ms * fs / 1000))
    isolation = int(round(config.calibration_isolation_ms * fs / 1000))
    chunk_samples = max(1, int(round(config.chunk_seconds * fs)))
    margin = max(
        int(round(config.read_margin_ms * fs / 1000)),
        before_ref + after_ref,
        temporal_radius + 1,
        128,
    )
    merge_on_device = BASE.gpu_neighborhood(merge_ids, device)
    rng = np.random.default_rng(clip_seed(clip, config))
    remaining = config.harvest_events_per_clip
    parts = []
    counts = {"positive": 0, "negative": 0}
    for core_start in range(0, ns, chunk_samples):
        if remaining <= 0:
            break
        core_stop = min(core_start + chunk_samples, ns)
        read_start = max(0, core_start - margin)
        read_stop = min(ns, core_stop + margin)
        raw = rec.get_traces(start_frame=read_start, end_frame=read_stop,
                             return_in_uV=False)
        data = BASE.preprocess_voltage(np.asarray(raw), sos)
        del raw
        noise = BASE.robust_channel_noise(data)
        residual = torch.as_tensor(data, dtype=torch.float32, device=device)
        noise_t = torch.as_tensor(noise, device=device)
        valid_start = max(before_ref, core_start - read_start)
        valid_stop = min(len(data) - after_ref + 1, core_stop - read_start)
        times_t, channels_t, scores_t, _ = N19.locally_exclusive_peaks(
            residual, noise_t, merge_on_device[0], merge_on_device[1],
            config.threshold, temporal_radius, valid_start, valid_stop,
            None, config.detection_nms_batch_size,
        )
        del residual
        times = times_t.cpu().numpy()
        channels = channels_t.cpu().numpy()
        scores = scores_t.cpu().numpy()
        keep = BASE.isolated_events(times, channels, merge_ids, isolation)
        times, channels, scores = times[keep], channels[keep], scores[keep]
        take = min(remaining, config.calibration_events_per_chunk, len(times))
        if take:
            pick = np.sort(rng.choice(len(times), take, replace=False))
            times, channels, scores = times[pick], channels[pick], scores[pick]
        else:
            times, channels, scores = times[:0], channels[:0], scores[:0]
        if len(times):
            safe = np.maximum(fit_ids[channels], 0)
            mask = fit_ids[channels] >= 0
            waveforms = data[
                times[:, None, None] - read_start + sample_offsets[None, None, :],
                safe[:, :, None],
            ]
            waveforms = (waveforms * mask[:, :, None]).astype(np.float32)
            parts.append({
                "spike_times": (times + read_start).astype(np.int64),
                "spike_channels": channels.astype(np.int32),
                "waveforms": waveforms,
                "mask": mask,
                "local_noise": noise[safe].astype(np.float32),
                "local_offsets": offsets[channels].astype(np.float32),
                "peak_score": scores.astype(np.float32),
                "peak_polarity": np.where(scores > 0, 1, -1).astype(np.int8),
            })
            counts["positive"] += int((scores > 0).sum())
            counts["negative"] += int((scores < 0).sum())
            remaining -= len(times)
    if parts:
        values = {key: np.concatenate([part[key] for part in parts])
                  for key in parts[0]}
    else:
        width = fit_ids.shape[1]
        values = {
            "spike_times": np.empty(0, np.int64),
            "spike_channels": np.empty(0, np.int32),
            "waveforms": np.empty((0, width, before_ref + after_ref), np.float32),
            "mask": np.empty((0, width), bool),
            "local_noise": np.empty((0, width), np.float32),
            "local_offsets": np.empty((0, width, 2), np.float32),
            "peak_score": np.empty(0, np.float32),
            "peak_polarity": np.empty(0, np.int8),
        }
    values["fs"] = np.asarray(fs, dtype=np.float64)
    values["positions"] = positions
    values["clip"] = np.asarray(clip["name"])
    atomic_npz(path, values)
    return {
        "clip": clip["name"], "events": int(len(values["spike_times"])),
        "positive": counts["positive"], "negative": counts["negative"],
        "fs": fs, "samples": ns,
    }


def _harvest_worker(clip, path_str, config_dict):
    config = Config(**config_dict)
    try:
        return harvest_clip(clip, Path(path_str), config)
    except Exception as error:
        return {"clip": clip["name"], "skipped": True, "error": f"{type(error).__name__}: {error}"}


def output_root_skipped(summary_path):
    return summary_path.parent / "skipped_clips.json"


def _shard_summary(path, clip_name):
    counts = {"events": 0, "positive": 0, "negative": 0}
    try:
        with np.load(path) as shard:
            polarity = shard["peak_polarity"]
            counts["events"] = int(len(polarity))
            counts["positive"] = int((polarity > 0).sum())
            counts["negative"] = int((polarity < 0).sum())
            fs = float(shard["fs"])
    except (OSError, KeyError):
        return None
    return {"clip": clip_name, **counts, "fs": fs}


def run_harvest(config, clips, shard_dir, resume, summary_path):
    done = {}
    if summary_path.exists():
        done = {item["clip"]: item for item in json.loads(summary_path.read_text())}
    for clip in clips:
        path = shard_path(shard_dir, clip)
        if clip["name"] in done or not path.exists():
            continue
        recovered = _shard_summary(path, clip["name"])
        if recovered is not None:
            done[clip["name"]] = recovered
    if resume and summary_path.exists() and len(done) != len(json.loads(summary_path.read_text())):
        atomic_json(summary_path, sorted(done.values(), key=lambda item: item["clip"]))
        print(f"harvest: recovered {len(done)} shard summaries from disk", flush=True)
    total = sum(item["events"] for item in done.values())
    pending = [
        clip for clip in clips
        if not (resume and clip["name"] in done
                and shard_path(shard_dir, clip).exists())
    ]
    print(f"harvest: {len(pending)} clips to harvest, "
          f"{len(clips) - len(pending)} already done", flush=True)
    if not pending:
        return
    config_dict = {field.name: getattr(config, field.name) for field in fields(Config)}
    workers = min(config.harvest_workers, len(pending))
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                                                mp_context=context) as pool:
        futures = {
            pool.submit(_harvest_worker, clip, str(shard_path(shard_dir, clip)),
                        config_dict): clip
            for clip in pending
        }
        for future in concurrent.futures.as_completed(futures):
            summary = future.result()
            done[summary["clip"]] = summary
            if summary.get("skipped"):
                print(f"harvest SKIPPED {summary['clip']}: {summary['error']}",
                      flush=True)
                continue
            total += summary["events"]
            atomic_json(summary_path,
                        sorted(done.values(), key=lambda item: item["clip"]))
            print(f"harvest {len(done)}/{len(clips)} {summary['clip']} "
                  f"events={summary['events']:,} "
                  f"(+{summary['positive']:,}/-{summary['negative']:,}) "
                  f"total={total:,}", flush=True)
    skipped = [item for item in done.values() if item.get("skipped")]
    if skipped:
        atomic_json(output_root_skipped(summary_path), skipped)
        print(f"harvest complete: {len(done) - len(skipped)} clips, "
              f"{len(skipped)} skipped (see {output_root_skipped(summary_path).name})",
              flush=True)


def load_shard_events(path, per_clip_cap, rng):
    keys = ("spike_times", "spike_channels", "waveforms", "mask",
            "local_noise", "local_offsets", "peak_polarity")
    with np.load(path) as shard:
        count = len(shard["spike_times"])
        if per_clip_cap is not None and count > per_clip_cap:
            pick = np.sort(rng.choice(count, per_clip_cap, replace=False))
            return {key: shard[key][pick] for key in keys}
        return {key: shard[key] for key in keys}


def iter_pool_batches(config, shard_dir, per_clip_cap):
    rng = np.random.default_rng(config.seed + 1)
    for path in sorted(shard_dir.glob("clip_*.npz")):
        values = load_shard_events(path, per_clip_cap, rng)
        count = len(values["spike_times"])
        for start in range(0, count, config.fit_batch_size):
            stop = min(start + config.fit_batch_size, count)
            yield {key: value[start:stop] for key, value in values.items()}


def build_pool(shard_dir, device):
    aligned_parts = []
    polarity_parts = []
    per_clip = []
    for path in sorted(shard_dir.glob("clip_*.npz")):
        with np.load(path) as shard:
            waveforms = torch.as_tensor(shard["waveforms"])
            mask = torch.as_tensor(shard["mask"], dtype=torch.bool)
            clip_name = str(shard["clip"])
        aligned, polarity = N19.peak_aligned_waveforms(waveforms, mask)
        norms = torch.linalg.vector_norm(aligned, dim=1)
        valid = torch.isfinite(norms) & (norms > EPS)
        aligned = aligned[valid] / norms[valid, None]
        polarity = polarity[valid]
        aligned_parts.append(aligned)
        polarity_parts.append(polarity)
        per_clip.append({"clip": clip_name, "events": int(valid.sum()),
                         "positive": int((polarity == 0).sum()),
                         "negative": int((polarity == 1).sum())})
    if not aligned_parts:
        raise RuntimeError("no shards to pool; run the harvest stage first")
    aligned = torch.cat(aligned_parts)
    polarity = torch.cat(polarity_parts)
    return aligned.to(device), polarity.to(device), per_clip


def run_init(config, shard_dir, root, resume):
    omega_path = root / "initial_omega.npy"
    complete = root / "prototype_initialization.json"
    if resume and omega_path.exists() and complete.exists():
        print("init: checkpoint found, skipping", flush=True)
        return
    aligned, polarity, per_clip = build_pool(shard_dir, config.device)
    cosine_limit = float(np.cos(np.radians(config.prototype_cone_deg)))
    omega, prototypes, assignment, polarity_counts = N19.initialize_codebook(
        aligned, polarity, config.q, cosine_limit, config.seed,
        config.prototype_kmeans_iterations,
    )
    atomic_npy(omega_path, omega.cpu().numpy().astype(np.float32))
    atomic_npy(root / "initial_prototypes.npy",
               prototypes.cpu().numpy().astype(np.float32))
    atomic_npy(root / "initial_atom_prototype.npy",
               assignment.cpu().numpy().astype(np.int16))
    atomic_json(complete, {
        "pool_events": int(len(aligned)),
        "polarity_group_counts": polarity_counts,
        "cone_half_angle_degrees": config.prototype_cone_deg,
        "atom_assignment": assignment.cpu().tolist(),
        "per_clip": per_clip,
    })
    print(f"init: pooled {len(aligned):,} waveforms -> Q{config.q}", flush=True)


def run_fit(config, shard_dir, root, resume):
    omega_path = root / "omega.npy"
    prototypes_path = root / "prototypes.npy"
    assignment_path = root / "atom_prototype.npy"
    history_path = root / "alternating_history.json"
    complete_path = root / "prototype_fit_complete.json"
    if resume and all(path.exists() for path in
                      (omega_path, prototypes_path, assignment_path, history_path,
                       complete_path)):
        print("fit: checkpoint found, skipping", flush=True)
        return
    initial_path = root / "initial_omega.npy"
    if not initial_path.exists():
        raise FileNotFoundError(f"{initial_path} missing; run the init stage first")
    omega = torch.as_tensor(np.load(initial_path).astype(np.float32),
                            device=config.device)
    prototypes = torch.as_tensor(np.load(root / "initial_prototypes.npy").astype(np.float32),
                                 device=config.device)
    assignment = torch.as_tensor(np.load(root / "initial_atom_prototype.npy").astype(np.int16),
                                 device=config.device)
    cosine_limit = float(np.cos(np.radians(config.prototype_cone_deg)))
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=config.device)
    cache = FOOTPRINT_CACHE(sites, sigmas, config.device)

    pool_rng = np.random.default_rng(config.seed + 1)
    pool = [load_shard_events(path, config.fit_pool_events_per_clip, pool_rng)
            for path in sorted(shard_dir.glob("clip_*.npz"))]
    pool = [values for values in pool if len(values["spike_times"])]
    if not pool:
        raise RuntimeError("fit pool is empty; run the harvest stage first")
    pool_events = sum(len(values["spike_times"]) for values in pool)
    atomic_json(root / "fit_pool.json", {
        "events": pool_events, "shards": len(pool),
        "per_clip_cap": config.fit_pool_events_per_clip,
    })
    print(f"fit: pool {pool_events:,} events from {len(pool)} clips", flush=True)

    history = []
    for iteration in range(1, config.alternating_iterations + 1):
        started = perf_counter()
        numerator = torch.zeros_like(omega)
        denominator = torch.zeros(config.q, device=config.device)
        atom_weight = torch.zeros(config.q, device=config.device)
        counts = torch.zeros(config.q, dtype=torch.long, device=config.device)
        input_energy = torch.zeros((), device=config.device)
        for values in pool:
            for start in range(0, len(values["spike_times"]), config.fit_batch_size):
                stop = min(start + config.fit_batch_size, len(values["spike_times"]))
                waveforms = torch.as_tensor(values["waveforms"][start:stop],
                                            device=config.device)
                local_offsets = torch.as_tensor(values["local_offsets"][start:stop],
                                                device=config.device)
                mask = torch.as_tensor(values["mask"][start:stop], device=config.device)
                local_noise = torch.as_tensor(values["local_noise"][start:stop],
                                              device=config.device)
                fit = PIPELINE.fit_grouped(
                    waveforms, local_offsets, mask, local_noise, omega,
                    sites, axes, sigmas, config, cache,
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
        proposed_omega, proposed_prototypes = N19.prototype_cone_proposal(
            omega, prototypes, assignment, numerator, atom_weight, cosine_limit
        )
        updated, updated_prototypes, before, after, step_size = N19.backtracked_update(
            omega, prototypes, proposed_omega, proposed_prototypes, assignment,
            numerator, denominator, input_energy, cosine_limit,
        )
        change = float(torch.linalg.vector_norm(updated - omega, dim=1).amax().item())
        accepted = step_size > 0
        omega, prototypes = updated, updated_prototypes
        history.append({
            "iteration": iteration,
            "fixed_assignment_objective": float(before.item()),
            "objective_after_basis": float(after.item()),
            "basis_accepted": accepted,
            "step_size": step_size,
            "maximum_row_change": change,
            "row_counts": counts.cpu().tolist(),
            "footprint_cache": cache.diagnostics(),
            "wall_s": perf_counter() - started,
        })
        atomic_json(history_path, history)
        atomic_npy(omega_path, omega.cpu().numpy().astype(np.float32))
        atomic_npy(prototypes_path, prototypes.cpu().numpy().astype(np.float32))
        atomic_npy(assignment_path, assignment.cpu().numpy().astype(np.int16))
        print(json.dumps(history[-1]), flush=True)
        if not accepted or change < config.alternating_tolerance:
            break

    atomic_npy(root / "omega.npy", omega.cpu().numpy().astype(np.float32))
    atomic_npy(root / "prototypes.npy", prototypes.cpu().numpy().astype(np.float32))
    atomic_npy(root / "atom_prototype.npy", assignment.cpu().numpy().astype(np.int16))
    atomic_json(root / "omega_source.json", {
        "kind": "master_ibl_bwm_clips",
        "clips": len(pool),
        "pool_events": pool_events,
        "prototype_count": config.prototype_count,
        "cone_half_angle_degrees": config.prototype_cone_deg,
        "orientation": "prototype_polarity_preserved",
        "frozen_during_pursuit": True,
    })
    atomic_json(complete_path, {
        "iterations": len(history),
        "basis_accepted": history[-1]["basis_accepted"],
    })


def self_test(config):
    device = "cpu"
    config = Config(q=6, device=device)
    root = HERE.parent.parent / "runs" / "_0026_selftest"
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(7)
    t = 90
    time_axis = torch.linspace(-1.5, 1.5, t)
    shapes = torch.stack([
        torch.exp(-((time_axis - 0.2) / 0.25).pow(2))
        - 0.6 * torch.exp(-((time_axis - 0.8) / 0.3).pow(2)),
        -torch.exp(-((time_axis + 0.3) / 0.3).pow(2))
        + 0.5 * torch.exp(-((time_axis - 0.3) / 0.25).pow(2)),
        torch.exp(-((time_axis + 0.1) / 0.4).pow(2)),
        -torch.exp(-((time_axis - 0.05) / 0.5).pow(2)),
    ])
    shapes = shapes / shapes.norm(dim=1, keepdim=True)
    grid = np.stack(np.meshgrid(np.arange(0, 80, 20.0),
                                np.arange(0, 80, 20.0), indexing="ij"),
                    axis=-1).reshape(-1, 2).astype(np.float32)
    for shard in range(3):
        count, width = 128, grid.shape[0]
        offsets = np.stack([grid - grid[event % len(grid)]
                            for event in range(count)]).astype(np.float32)
        mask = np.ones((count, width), bool)
        noise = np.full((count, width), 0.1, np.float32)
        waveforms = 0.05 * torch.randn(count, width, t, generator=generator)
        for event in range(count):
            atom = shapes[event % len(shapes)]
            polarity = 1.0 if (event % 7) else -1.0
            for channel in range(width):
                weight = float(np.exp(-np.square(offsets[event, channel]).sum() / 800.0))
                waveforms[event, channel] += polarity * weight * atom
        atomic_npz(shard_dir / f"clip_synthetic_{shard}.npz", {
            "spike_times": np.arange(count) * 50,
            "spike_channels": np.zeros(count, np.int32),
            "waveforms": waveforms.numpy().astype(np.float32),
            "mask": mask, "local_noise": noise, "local_offsets": offsets,
            "peak_polarity": np.ones(count, np.int8),
            "clip": np.asarray(f"synthetic_{shard}"),
        })
    aligned, polarity, per_clip = build_pool(shard_dir, device)
    assert len(aligned) == 3 * 128, len(aligned)
    cosine_limit = float(np.cos(np.radians(config.prototype_cone_deg)))
    omega, prototypes, assignment, polarity_counts = N19.initialize_codebook(
        aligned, polarity, config.q, cosine_limit, config.seed,
        config.prototype_kmeans_iterations,
    )
    assert omega.shape == (config.q, t) and len(prototypes) == 2
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=device)
    axes = [torch.as_tensor(axis, device=device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=device)
    cache = FOOTPRINT_CACHE(sites, sigmas, device)
    numerator = torch.zeros_like(omega)
    denominator = torch.zeros(config.q, device=device)
    atom_weight = torch.zeros(config.q, device=device)
    input_energy = torch.zeros((), device=device)
    for start in range(0, len(aligned), 64):
        stop = min(start + 64, len(aligned))
        waveforms = aligned[start:stop, None, :].expand(-1, 4, -1).contiguous()
        offsets = torch.zeros(len(waveforms), 4, 2)
        mask = torch.ones(len(waveforms), 4, dtype=torch.bool)
        local_noise = torch.full((len(waveforms), 4), 0.1)
        fit = PIPELINE.fit_grouped(waveforms, offsets, mask, local_noise, omega,
                                   sites, axes, sigmas, config, cache)
        labels = fit["temporal_index"]
        distance_xy = (offsets - fit["sources"][:, None, :2]).square().sum(dim=2)
        footprint = fit["sigma"][:, None] / torch.sqrt(
            distance_xy + fit["sources"][:, 2, None].square()
            + fit["sigma"][:, None].square()).clamp_min(EPS)
        spatial = fit["alpha"][:, None] * footprint * mask
        numerator.index_add_(0, labels,
                             torch.einsum("bct,bc->bt", waveforms, spatial))
        denominator.index_add_(0, labels, spatial.square().sum(dim=1))
        atom_weight.index_add_(0, labels, fit["alpha"])
        input_energy += waveforms.square().sum()
    proposed, proposed_prototypes = N19.prototype_cone_proposal(
        omega, prototypes, assignment, numerator, atom_weight, cosine_limit)
    updated, updated_prototypes, before, after, step_size = N19.backtracked_update(
        omega, prototypes, proposed, proposed_prototypes, assignment,
        numerator, denominator, input_energy, cosine_limit)
    assert float(after) <= float(before) + 1e-5 * max(1.0, abs(float(before)))
    assert step_size > 0, "self-test proposal should improve the objective"
    print(f"0026 self-test passed: objective {float(before):.4f} -> {float(after):.4f} "
          f"(step {step_size})", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("harvest", "init", "fit", "all"),
                        nargs="?", default="all")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/scratch/ap7151/_RAW_DATA/ibl-data"))
    parser.add_argument("--output", type=Path,
                        default=Path("residuals/runs/ibl_bwm/master_codebook_q32"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--clip-stop", type=int)
    parser.add_argument("--limit-clips", type=int)
    parser.add_argument("--self-test", action="store_true")
    for field in Config.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction,
                                default=field.default)
        elif field.default is not None:
            parser.add_argument(name, type=type(field.default), default=field.default)
        else:
            parser.add_argument(name, type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(**{name: getattr(args, name)
                       for name in Config.__dataclass_fields__})
    if args.self_test:
        self_test(config)
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output / "harvest_summary.json"
    clips = enumerate_clips(args.data_root)
    stop = args.clip_stop if args.clip_stop is not None else len(clips)
    clips = clips[args.clip_start:stop]
    if args.limit_clips:
        clips = clips[:args.limit_clips]
    print(f"0026 master codebook: {len(clips)} clips, Q={config.q}, "
          f"output={output}", flush=True)
    if args.stage in ("harvest", "all"):
        run_harvest(config, clips, shard_dir, args.resume, summary_path)
    if args.stage in ("init", "all"):
        run_init(config, shard_dir, output, args.resume)
    if args.stage in ("fit", "all"):
        run_fit(config, shard_dir, output, args.resume)


if __name__ == "__main__":
    main()
