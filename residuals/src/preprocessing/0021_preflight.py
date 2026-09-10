"""Per-recording 0021 preflight: pool a recording, fit both priors and both lag settings, compare.

Before committing to a four-prior shift-invariant codebook for a recording, this
script measures what each choice buys on that recording's own calibration pool:

    pool     extract peak-anchored neighborhood waveforms with the production
             recipe (locally exclusive threshold 5, 1 ms isolation, 48 um
             neighborhoods, the 0012 filter), capped at --pool-events over the
             first --pool-seconds. The pool npz is cached, so requeues skip the
             raw read.
    variants fit a Q-atom codebook for each requested combination of
             priors (2 polarity cones vs 4 = polarity x droop-speed cones, the
             speed split a per-polarity 2-means on trough-to-peak width) and
             shift learning (fixed onset vs per-event lag chosen against the
             renormalized shift bank, lag undone before accumulation), then
             report each variant's alternating-fit history, final fixed-
             assignment SSE, dead atoms, lag histogram, speed-split quality,
             and per-atom trough-to-peak widths.
    compare  final objectives are comparable across variants (same pool, same
             events; the shift model minimizes over a superset of the zero-lag
             assignments, and lag learning can only tie or beat fixed-onset
             learning in expectation). comparison.json ranks variants and
             records what each dial changed, so the 2-vs-4-priors decision is
             per recording and data-driven.

Sources: SpikeGLX .bin/.cbin/.meta via spikeglx.Reader, primate .nwb via
0025's NwbReader (duck-typed: fs, ns, geometry, reader[start:stop, :n]).
The fitting machinery is 0021_shift_invariant_codebook's exactly.

Usage (through singularity):
    python residuals/src/preprocessing/0021_preflight.py \
        /scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin \
        --output residuals/runs/0021_preflight/dataset1_p1 --resume
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


N19 = load_module("allchannel_0019_for_0021p", HERE / "0019_allchannel_peeling.py")
M21 = load_module("shift_codebook_0021_for_preflight", HERE / "0021_shift_invariant_codebook.py")
M26 = M21.M26
BASE = N19.BASE
PIPELINE = N19.PIPELINE
EPS = N19.EPS
FOOTPRINT_CACHE = PIPELINE.OLD.FootprintCache


from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PreflightConfig(M26.Config):
    pool_events: int = 60000
    pool_seconds: float = 600.0
    preflight_iterations: int = 5
    max_shift: int = 10
    shift_learning: bool = True
    width_kmeans_iterations: int = 50


def open_reader(path):
    path = Path(path)
    if path.suffix == ".nwb":
        primate = load_module(
            "primate_0025_for_0021p", HERE / "0025_primate_peeling.py"
        )
        return primate.NwbReader(path)
    import spikeglx

    return spikeglx.Reader(path)


def reader_geometry(reader):
    return np.column_stack(
        (reader.geometry["x"], reader.geometry["y"])
    ).astype(np.float32)


def extract_pool(reader, config, pool_path, summary_path, resume):
    if resume and pool_path.exists() and summary_path.exists():
        print("pool: cached pool found, skipping extraction", flush=True)
        return json.loads(summary_path.read_text())
    fs = float(reader.fs)
    ns = int(reader.ns)
    positions = reader_geometry(reader)
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
    merge_on_device = BASE.gpu_neighborhood(merge_ids, config.device)
    pool_stop = min(ns, int(round(config.pool_seconds * fs)))
    rng = np.random.default_rng(config.seed)
    remaining = config.pool_events
    parts = []
    counts = {"positive": 0, "negative": 0}
    for core_start in range(0, pool_stop, chunk_samples):
        if remaining <= 0:
            break
        core_stop = min(core_start + chunk_samples, pool_stop)
        read_start = max(0, core_start - margin)
        read_stop = min(ns, core_stop + margin)
        data = BASE.preprocess_voltage(
            np.asarray(reader[read_start:read_stop, :positions.shape[0]]), sos
        )
        noise = BASE.robust_channel_noise(data)
        residual = torch.as_tensor(data, dtype=torch.float32, device=config.device)
        noise_t = torch.as_tensor(noise, device=config.device)
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
        print(f"pool: chunk {core_start // chunk_samples + 1} "
              f"({core_stop / fs:.0f}s) remaining={remaining:,}", flush=True)
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
    M26.atomic_npz(pool_path, values)
    summary = {
        "events": int(len(values["spike_times"])),
        "positive": counts["positive"],
        "negative": counts["negative"],
        "fs": fs,
        "pool_seconds": config.pool_seconds,
        "recording_samples": ns,
    }
    M26.atomic_json(summary_path, summary)
    return summary


def build_pool_arrays(pool, device):
    waveforms = torch.as_tensor(pool["waveforms"], device=device)
    mask = torch.as_tensor(pool["mask"], dtype=torch.bool, device=device)
    aligned, polarity = N19.peak_aligned_waveforms(waveforms, mask)
    norms = torch.linalg.vector_norm(aligned, dim=1)
    valid = torch.isfinite(norms) & (norms > EPS)
    return aligned[valid] / norms[valid, None], polarity[valid]


def fit_variant(pool, aligned, polarity, priors, shift_learning, config, root):
    config = replace(config, shift_learning=shift_learning)
    started = perf_counter()
    cosine_limit = float(np.cos(np.radians(config.prototype_cone_deg)))
    q = config.q
    if priors == 4:
        speed, widths, _, thresholds = M21.speed_split(aligned, polarity, config)
    else:
        speed = torch.zeros(len(aligned), dtype=torch.long, device=aligned.device)
        widths = torch.zeros(len(aligned), device=aligned.device)
        thresholds = {}
    omega, prototypes, assignment, group_counts = M21.initialize_codebook_priors(
        aligned, polarity, speed, q, cosine_limit, config.seed,
        config.prototype_kmeans_iterations, priors,
    )
    sites_np, axes_np = BASE.coarse_lattice(config.base())
    sites = torch.as_tensor(sites_np, device=config.device)
    axes = [torch.as_tensor(axis, device=config.device) for axis in axes_np]
    sigmas = torch.as_tensor(BASE.sigma_bank(config.base()), device=config.device)
    cache = FOOTPRINT_CACHE(sites, sigmas, config.device)

    waveforms_all = torch.as_tensor(pool["waveforms"], device=config.device)
    offsets_all = torch.as_tensor(pool["local_offsets"], device=config.device)
    mask_all = torch.as_tensor(pool["mask"], device=config.device)
    noise_all = torch.as_tensor(pool["local_noise"], device=config.device)

    history = []
    for iteration in range(1, config.preflight_iterations + 1):
        numerator = torch.zeros_like(omega)
        denominator = torch.zeros(q, device=config.device)
        atom_weight = torch.zeros(q, device=config.device)
        counts = torch.zeros(q, dtype=torch.long, device=config.device)
        input_energy = torch.zeros((), device=config.device)
        lag_histogram = torch.zeros(2 * config.max_shift + 1,
                                    dtype=torch.long, device=config.device)
        bank = bank_raw = None
        if shift_learning:
            bank, bank_raw = M21.build_shift_bank(omega, config.max_shift)
        for start in range(0, len(waveforms_all), config.fit_batch_size):
            stop = min(start + config.fit_batch_size, len(waveforms_all))
            fit = PIPELINE.fit_grouped(
                waveforms_all[start:stop], offsets_all[start:stop],
                mask_all[start:stop], noise_all[start:stop], omega,
                sites, axes, sigmas, config, cache,
            )
            spatial = M21.spatial_weights(fit, offsets_all[start:stop],
                                          mask_all[start:stop])
            labels, batch_numerator, _ = M21.accumulate_statistics(
                waveforms_all[start:stop], spatial, fit, omega.shape, config,
                bank, bank_raw, lag_histogram,
            )
            numerator += batch_numerator
            denominator.index_add_(0, labels, spatial.square().sum(dim=1))
            atom_weight.index_add_(0, labels, fit["alpha"])
            counts += torch.bincount(labels, minlength=q)
            input_energy += waveforms_all[start:stop].square().sum()
        proposed_omega, proposed_prototypes = N19.prototype_cone_proposal(
            omega, prototypes, assignment, numerator, atom_weight, cosine_limit
        )
        updated, updated_prototypes, before, after, step_size = N19.backtracked_update(
            omega, prototypes, proposed_omega, proposed_prototypes, assignment,
            numerator, denominator, input_energy, cosine_limit,
        )
        omega, prototypes = updated, updated_prototypes
        history.append({
            "iteration": iteration,
            "fixed_assignment_objective": float(before.item()),
            "objective_after_basis": float(after.item()),
            "basis_accepted": step_size > 0,
            "step_size": step_size,
            "row_counts": counts.cpu().tolist(),
            "lag_histogram": lag_histogram.cpu().tolist(),
        })
        print(f"  variant priors={priors} shift={shift_learning} "
              f"iter {iteration}/{config.preflight_iterations}: "
              f"SSE {before.item():.1f} -> {after.item():.1f}", flush=True)

    final_numerator_norm = float(torch.linalg.vector_norm(omega, dim=1).amin())
    omega_np = omega.cpu().numpy().astype(np.float32)
    atom_widths, _, _, _ = M21.waveform_shape_features(
        torch.as_tensor(omega_np, device=config.device)
    )
    lag_axis = np.arange(-config.max_shift, config.max_shift + 1)
    lag_hist = history[-1]["lag_histogram"]
    diagnostics = {
        "priors": priors,
        "shift_learning": shift_learning,
        "max_shift": config.max_shift,
        "iterations": len(history),
        "final_objective": history[-1]["fixed_assignment_objective"],
        "history": history,
        "dead_atoms": int((np.asarray(history[-1]["row_counts"]) == 0).sum()),
        "minimum_row_norm": final_numerator_norm,
        "atom_widths_samples": atom_widths.cpu().tolist(),
        "speed_split": thresholds,
        "group_counts": group_counts,
        "lag_mean_abs": float((np.asarray(lag_hist) * np.abs(lag_axis)).sum()
                              / max(int(np.asarray(lag_hist).sum()), 1)),
        "lag_nonzero_fraction": float(
            1.0 - np.asarray(lag_hist)[config.max_shift]
            / max(int(np.asarray(lag_hist).sum()), 1)
        ),
        "wall_s": perf_counter() - started,
    }
    root.mkdir(parents=True, exist_ok=True)
    M26.atomic_npy(root / "omega.npy", omega_np)
    M26.atomic_npy(root / "prototypes.npy",
                   prototypes.cpu().numpy().astype(np.float32))
    M26.atomic_npy(root / "atom_prototype.npy",
                   assignment.cpu().numpy().astype(np.int16))
    M26.atomic_json(root / "diagnostics.json", diagnostics)
    return diagnostics


def run_preflight(config, pool, root, resume, variants):
    aligned, polarity = build_pool_arrays(pool, config.device)
    results = {}
    for priors, shift_learning in variants:
        name = f"priors{priors}_shift{int(shift_learning)}"
        variant_root = root / name
        diagnostics_path = variant_root / "diagnostics.json"
        if resume and diagnostics_path.exists():
            results[name] = json.loads(diagnostics_path.read_text())
            print(f"variant {name}: cached, skipping", flush=True)
            continue
        print(f"variant {name}: fitting", flush=True)
        results[name] = fit_variant(
            pool, aligned, polarity, priors, shift_learning, config, variant_root
        )
    ranked = sorted(results.items(), key=lambda item: item[1]["final_objective"])
    comparison = {
        "ranking": [
            {"variant": name, "final_objective": result["final_objective"]}
            for name, result in ranked
        ],
        "objective_by_variant": {
            name: result["final_objective"] for name, result in results.items()
        },
        "note": (
            "lower fixed_assignment_objective is better; variants share one pool, "
            "so objectives are directly comparable; shift variants minimize over a "
            "superset of the fixed-onset assignments"
        ),
        "lag_diagnostics": {
            name: {
                "lag_mean_abs": result["lag_mean_abs"],
                "lag_nonzero_fraction": result["lag_nonzero_fraction"],
            }
            for name, result in results.items() if result["shift_learning"]
        },
        "speed_split": next(
            (result["speed_split"] for result in results.values()
             if result["priors"] == 4), {}
        ),
        "dead_atoms": {
            name: result["dead_atoms"] for name, result in results.items()
        },
    }
    M26.atomic_json(root / "comparison.json", comparison)
    print(json.dumps(comparison["ranking"], indent=2), flush=True)
    return comparison


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, nargs="?", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--priors", type=str, default="2,4",
                        help="comma list from {2,4}")
    parser.add_argument("--shift", type=str, default="off,on",
                        help="comma list from {off,on}")
    parser.add_argument("--self-test", action="store_true")
    for field in PreflightConfig.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction,
                                default=field.default)
        elif field.default is not None:
            parser.add_argument(name, type=type(field.default), default=field.default)
        else:
            parser.add_argument(name, type=str, default=None)
    return parser.parse_args()


def synthetic_pool(path, device="cpu"):
    config = PreflightConfig(q=8, device=device)
    root = HERE.parent.parent / "runs" / "_0021_preflight_selftest"
    shard_dir, _ = M21.synthetic_shards(root, config)
    values = M26.load_shard_events(
        sorted(shard_dir.glob("clip_*.npz"))[0], None, np.random.default_rng(0)
    )
    pool = {
        "spike_times": values["spike_times"],
        "spike_channels": values["spike_channels"],
        "waveforms": values["waveforms"],
        "mask": values["mask"],
        "local_noise": values["local_noise"],
        "local_offsets": values["local_offsets"],
        "peak_score": np.ones(len(values["spike_times"]), np.float32),
        "peak_polarity": np.ones(len(values["spike_times"]), np.int8),
        "fs": np.asarray(30000.0),
    }
    np.savez(path, **pool)
    return pool


def main():
    args = parse_args()
    config = PreflightConfig(**{
        name: getattr(args, name) for name in PreflightConfig.__dataclass_fields__
    })
    if args.self_test:
        root = HERE.parent.parent / "runs" / "_0021_preflight_selftest"
        pool_path = root / "pool.npz"
        pool = synthetic_pool(pool_path)
        config = PreflightConfig(
            q=8, device="cpu", preflight_iterations=2, max_shift=4,
            pool_events=768,
        )
        variants = [(2, False), (4, False), (4, True)]
        comparison = run_preflight(config, pool, root, False, variants)
        objectives = comparison["objective_by_variant"]
        assert objectives["priors4_shift1"] <= objectives["priors4_shift0"] + 1e-3 * max(
            1.0, abs(objectives["priors4_shift0"])
        ), "shift learning must not be worse on lagged synthetic data"
        print("0021 preflight self-test passed", flush=True)
        return
    if args.recording is None:
        raise SystemExit("a recording path is required unless --self-test")
    if args.output is None:
        stem = args.recording.stem.replace(".imec0.ap", "").replace(".ap", "")
        args.output = Path(f"residuals/runs/0021_preflight/{stem}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    priors = [int(value) for value in args.priors.split(",") if value]
    shifts = [value.strip() == "on" for value in args.shift.split(",") if value.strip()]
    for value in priors:
        if value not in (2, 4):
            raise ValueError(f"priors must be 2 or 4, got {value}")
    variants = [(priors_index, shift)
                for priors_index in priors for shift in shifts]
    print(f"0021 preflight: {args.recording} -> {output} "
          f"variants={variants} Q={config.q}", flush=True)
    reader = open_reader(args.recording)
    try:
        summary = extract_pool(
            reader, config, output / "pool.npz", output / "pool_summary.json",
            args.resume,
        )
    finally:
        reader.close()
    print(f"pool: {summary['events']:,} events "
          f"(+{summary['positive']:,}/-{summary['negative']:,})", flush=True)
    pool = dict(np.load(output / "pool.npz"))
    run_preflight(config, pool, output, args.resume, variants)


if __name__ == "__main__":
    main()
