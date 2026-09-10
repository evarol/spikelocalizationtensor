"""Shift-invariant, four-prior temporal codebook (0021): lag-aligned learning over the 0026 IBL pool.

0026 learned one Q32 two-cone Omega from every IBL Brain-Wide-Map clip and proved
that one pooled codebook calibrates all recordings. 0021 changes two things about
HOW the atoms are learned and one thing about their organization:

    priors   the two polarity cones become up to four: polarity (positive /
             negative extremum) x droop speed (fast/narrow vs slow/wide decay),
             with the speed split derived from the data (per-polarity 1-D
             2-means on trough-to-peak width), never a hand-picked cutoff.
             Group order is prototype_index = polarity + 2*speed, which keeps
             0019's "even index positive" polarity convention intact.
             --freeze-prototypes keeps the prototypes at their init-split means
             for the whole fit: without it the prototypes are re-derived from
             their own atoms' SVD each iteration, and since a 35-degree cone
             admits both speeds, the atoms (and their prototypes with them)
             drift across the speed classes — the learned "slow" cones ended
             up holding narrow atoms. Freezing plus a tighter
             --prototype-cone-deg is what pins the classes.
    lags     every event is peak-anchored, but the waveform around the anchor
             jitters by a few samples; the fixed-onset Omega forces that jitter
             into the gain, a wrong atom, or a biased sigma. Learning therefore
             picks an integer lag per event (correlation against the renormalized
             shift bank at the fit_grouped-winning source), undoes it with a
             zero-padded alignment before adding to the per-atom numerator, and
             rescales by 1/||S_tau Omega|| so the accumulated statistics live in
             the renormalized-bank frame the pursuit will use.
    guard    input energy stays the RAW waveform energy while the numerator uses
             aligned waveforms: <y, S_tau Omega / ||S_tau Omega||> =
             <undo_tau(y), Omega> / ||S_tau Omega|| exactly for zero-padded
             shifts, so fixed_assignment_objective remains the true SSE of the
             shifted model with the renormalized bank rows.

Shift conventions follow spiketensor/unified.py (the reference implementation
this lineage ports): bank rows S_tau Omega with tau >= 0 delayed and zero-pad
renormalized to unit norm; lag undo drops the head and zero-pads the tail
(their basis_proposal lines 389-399); atom proposal by cone-projecting the
accumulated lag-undone profiles, prototypes by amplitude-weighted SVD with a
re-projection afterwards (their lines 406-423). The one deliberate difference:
their acceptance guard re-runs full inference on the candidate basis, so their
statistics need no 1/||S_tau Omega|| rescale; this module keeps 0019's
closed-form fixed_assignment_objective guard, where the rescale is what makes
the accumulated statistics exact.

Data source is 0026's harvested shard directory (default:
residuals/runs/ibl_bwm/master_codebook_q32/shards, 808 clips with stored
channel-level waveforms); there is no harvest stage. --priors {2,4} and
--shift-learning/--no-shift-learning make every variant runnable from the same
pool; --priors 2 --no-shift-learning reproduces the 026 fit (same init code
path, same accumulation convention).

Usage (through singularity):
    python residuals/src/preprocessing/0021_shift_invariant_codebook.py all \
        --shards residuals/runs/ibl_bwm/master_codebook_q32/shards \
        --output residuals/runs/ibl_bwm/master_codebook_0021_q32_p4_shift \
        --priors 4 --shift-learning --resume
"""

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


N19 = load_module("allchannel_0019_for_0021", HERE / "0019_allchannel_peeling.py")
M26 = load_module("master_codebook_0026_for_0021", HERE / "026_master_codebook.py")
PIPELINE = N19.PIPELINE
BASE = N19.BASE
EPS = N19.EPS
FOOTPRINT_CACHE = PIPELINE.OLD.FootprintCache


@dataclass(frozen=True)
class Config21(M26.Config):
    prototype_priors: int = 4
    max_shift: int = 10
    shift_learning: bool = True
    width_kmeans_iterations: int = 50
    freeze_prototypes: bool = False


def atomic_json(path, value):
    BASE.atomic_json(path, value)


def atomic_npy(path, value):
    BASE.atomic_npy(path, value)


def waveform_shape_features(aligned):
    aligned = aligned.float()
    count, width = aligned.shape
    rows = torch.arange(count, device=aligned.device)
    columns = torch.arange(width, device=aligned.device)
    extremum = aligned.abs().argmax(dim=1)
    sign = torch.sign(aligned[rows, extremum])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    flipped = aligned * sign[:, None]
    after = columns[None, :] > extremum[:, None]
    opposite = torch.where(after, -flipped, torch.full_like(flipped, float("-inf")))
    opposite_index = opposite.argmax(dim=1)
    has_opposite = after.any(dim=1)
    trough_to_peak = (opposite_index - extremum).float().clamp_min(0)
    peak_value = flipped[rows, extremum]
    below = after & (flipped < 0.5 * peak_value[:, None])
    half_index = torch.where(
        below, columns[None, :].expand(count, width),
        torch.full_like(flipped, float(width)),
    ).argmin(dim=1)
    has_half = below.any(dim=1)
    half_width = (half_index - extremum).float().clamp_min(0)
    width_ttp = torch.where(has_opposite, trough_to_peak, half_width)
    width_half = torch.where(has_half, half_width, trough_to_peak)
    return width_ttp, width_half, has_opposite, has_half


def one_d_two_means(values, seed, iterations):
    values = values.float()
    generator = torch.Generator(device=values.device).manual_seed(seed)
    finite = torch.isfinite(values)
    observed = values[finite]
    if len(observed) < 2:
        center = float(observed.mean()) if len(observed) else 0.0
        centers = torch.tensor([center, center], device=values.device)
    else:
        centers = torch.stack([
            torch.quantile(observed, 0.25).float(),
            torch.quantile(observed, 0.75).float(),
        ])
        for _ in range(iterations):
            distance = (observed[:, None] - centers[None, :]).abs()
            labels = distance.argmin(dim=1)
            for index in range(2):
                member = observed[labels == index]
                if len(member):
                    centers[index] = member.mean()
            centers = torch.stack(sorted(centers, key=float))
    distance = (values[:, None] - centers[None, :]).abs()
    distance = torch.where(
        torch.isfinite(distance), distance, torch.full_like(distance, float("inf"))
    )
    labels = distance.argmin(dim=1)
    return labels, centers


def speed_split(aligned, polarity, config):
    widths, half_widths, _, _ = waveform_shape_features(aligned)
    speed = torch.zeros(len(aligned), dtype=torch.long, device=aligned.device)
    thresholds = {}
    for polarity_index in (0, 1):
        member = polarity == polarity_index
        labels, centers = one_d_two_means(
            widths[member], config.seed + 101 + polarity_index,
            config.width_kmeans_iterations,
        )
        speed[member] = labels
        thresholds[f"polarity_{polarity_index}"] = {
            "width_split_samples": [float(center) for center in centers],
            "narrow_fast_events": int((labels == 0).sum()),
            "wide_slow_events": int((labels == 1).sum()),
        }
    return speed, widths, half_widths, thresholds


def initialize_codebook_priors(aligned, polarity, speed, q, cosine_limit, seed,
                               iterations, priors):
    if priors == 2:
        return N19.initialize_codebook(aligned, polarity, q, cosine_limit, seed,
                                       iterations)
    if q % 4:
        raise ValueError(f"four priors need q divisible by 4, got q={q}")
    group_polarity_speed = ((0, 0), (1, 0), (0, 1), (1, 1))
    groups = [
        aligned[(polarity == p) & (speed == s)] for p, s in group_polarity_speed
    ]
    counts = [len(group) for group in groups]
    if min(counts) == 0:
        raise RuntimeError(
            "four-prior calibration requires all polarity x speed groups; "
            f"observed group counts {counts}"
        )
    prototypes = N19.fix_polarity(
        torch.stack([F.normalize(group.mean(dim=0), dim=0) for group in groups])
    )
    per_cone = q // 4
    assignment = torch.arange(q, device=aligned.device) // per_cone
    atoms = torch.zeros(q, aligned.shape[1], device=aligned.device)
    for prototype_index, group in enumerate(groups):
        rows = torch.nonzero(assignment == prototype_index, as_tuple=False).squeeze(1)
        centers = N19.spherical_kmeans(
            group, len(rows), seed + prototype_index, iterations
        )
        for local_index, atom_index in enumerate(rows.tolist()):
            atoms[atom_index] = N19.project_cone(
                centers[local_index % len(centers)],
                prototypes[prototype_index],
                cosine_limit,
            )
    return atoms, prototypes, assignment, counts


def build_shift_bank(omega, max_shift):
    count, width = omega.shape
    omega = F.normalize(omega, dim=1)
    bank = torch.zeros(count, 2 * max_shift + 1, width, device=omega.device)
    raw_norm = torch.zeros(count, 2 * max_shift + 1, device=omega.device)
    for shift_index, lag in enumerate(range(-max_shift, max_shift + 1)):
        shifted = torch.zeros_like(omega)
        if lag >= 0:
            shifted[:, lag:] = omega[:, : width - lag]
        else:
            shifted[:, : width + lag] = omega[:, -lag:]
        norm = torch.linalg.vector_norm(shifted, dim=1)
        raw_norm[:, shift_index] = norm
        bank[:, shift_index] = shifted / norm.clamp_min(EPS)[:, None]
    return bank, raw_norm


def undo_lag_last_dim(values, lag):
    width = values.shape[-1]
    flat = values.reshape(-1, width)
    rows = flat.shape[0]
    columns = torch.arange(width, device=values.device)
    source = columns + lag
    valid = (source >= 0) & (source < width)
    gathered = flat.gather(
        1, source.clamp(0, width - 1)[None, :].expand(rows, width)
    )
    return (gathered * valid[None, :]).reshape(values.shape)


def accumulate_statistics(waveforms, spatial, fit, omega_shape, config,
                          bank, bank_raw, lag_histogram):
    labels = fit["temporal_index"].long()
    lag_index = torch.full_like(labels, config.max_shift)
    if config.shift_learning:
        combined = torch.einsum("bct,bc->bt", waveforms, spatial)
        normalized = combined / torch.linalg.vector_norm(
            combined, dim=1).clamp_min(EPS)[:, None]
        scores = torch.einsum("bt,mst->bms", normalized, bank)
        flat = scores.reshape(len(normalized), -1).argmax(dim=1)
        labels = flat // (2 * config.max_shift + 1)
        lag_index = flat % (2 * config.max_shift + 1)
    lags = lag_index - config.max_shift
    lag_histogram += torch.bincount(lag_index, minlength=2 * config.max_shift + 1)
    numerator = torch.zeros(omega_shape, device=waveforms.device)
    scale = torch.ones(len(waveforms), device=waveforms.device)
    if config.shift_learning:
        scale = 1.0 / bank_raw[labels, lag_index].clamp_min(EPS)
    for lag_value in lags.unique().tolist():
        member = lags == lag_value
        aligned = undo_lag_last_dim(waveforms[member], int(lag_value))
        aligned = aligned * scale[member][:, None, None]
        numerator.index_add_(
            0, labels[member],
            torch.einsum("bct,bc->bt", aligned, spatial[member]),
        )
    return labels, numerator, lag_index


def spatial_weights(fit, local_offsets, mask):
    selected_sigma = fit["sigma"]
    distance_xy = (local_offsets - fit["sources"][:, None, :2]).square().sum(dim=2)
    footprint = selected_sigma[:, None] / torch.sqrt(
        distance_xy
        + fit["sources"][:, 2, None].square()
        + selected_sigma[:, None].square()
    ).clamp_min(EPS)
    return fit["alpha"][:, None] * footprint * mask


def frozen_cone_proposal(omega, prototypes, assignment, numerator, cosine_limit):
    atoms = omega.clone()
    for atom_index in range(len(atoms)):
        if float(numerator[atom_index].norm()) > EPS:
            atoms[atom_index] = N19.project_cone(
                numerator[atom_index],
                prototypes[assignment[atom_index]],
                cosine_limit,
            )
    return atoms, prototypes


def run_init(config, shard_dir, root, resume):
    omega_path = root / "initial_omega.npy"
    complete = root / "prototype_initialization.json"
    if resume and omega_path.exists() and complete.exists():
        print("init: checkpoint found, skipping", flush=True)
        return
    aligned, polarity, per_clip = M26.build_pool(shard_dir, config.device)
    cosine_limit = float(np.cos(np.radians(config.prototype_cone_deg)))
    if config.prototype_priors == 4:
        speed, widths, _, thresholds = speed_split(aligned, polarity, config)
        omega, prototypes, assignment, group_counts = initialize_codebook_priors(
            aligned, polarity, speed, config.q, cosine_limit, config.seed,
            config.prototype_kmeans_iterations, config.prototype_priors,
        )
    else:
        speed = torch.zeros(len(aligned), dtype=torch.long, device=aligned.device)
        widths = torch.zeros(len(aligned), device=aligned.device)
        thresholds = {}
        omega, prototypes, assignment, group_counts = N19.initialize_codebook(
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
        "priors": config.prototype_priors,
        "group_counts": group_counts,
        "cone_half_angle_degrees": config.prototype_cone_deg,
        "atom_assignment": assignment.cpu().tolist(),
        "speed_split": thresholds,
        "width_percentiles_samples": {
            str(p): float(torch.quantile(widths, p / 100))
            for p in (5, 25, 50, 75, 95)
        },
        "per_clip": per_clip,
    })
    print(f"init: pooled {len(aligned):,} waveforms -> Q{config.q} "
          f"priors={config.prototype_priors}", flush=True)


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
    pool = [M26.load_shard_events(path, config.fit_pool_events_per_clip, pool_rng)
            for path in sorted(shard_dir.glob("clip_*.npz"))]
    pool = [values for values in pool if len(values["spike_times"])]
    if not pool:
        raise RuntimeError("fit pool is empty; 0026's harvest stage must run first")
    pool_events = sum(len(values["spike_times"]) for values in pool)
    atomic_json(root / "fit_pool.json", {
        "events": pool_events, "shards": len(pool),
        "per_clip_cap": config.fit_pool_events_per_clip,
        "priors": config.prototype_priors,
        "shift_learning": config.shift_learning,
        "max_shift": config.max_shift,
    })
    print(f"fit: pool {pool_events:,} events from {len(pool)} clips "
          f"(priors={config.prototype_priors}, shift={config.shift_learning})",
          flush=True)

    history = []
    for iteration in range(1, config.alternating_iterations + 1):
        started = perf_counter()
        numerator = torch.zeros_like(omega)
        denominator = torch.zeros(config.q, device=config.device)
        atom_weight = torch.zeros(config.q, device=config.device)
        counts = torch.zeros(config.q, dtype=torch.long, device=config.device)
        input_energy = torch.zeros((), device=config.device)
        lag_histogram = torch.zeros(2 * config.max_shift + 1,
                                    dtype=torch.long, device=config.device)
        bank = bank_raw = None
        if config.shift_learning:
            bank, bank_raw = build_shift_bank(omega, config.max_shift)
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
                spatial = spatial_weights(fit, local_offsets, mask)
                labels, batch_numerator, _ = accumulate_statistics(
                    waveforms, spatial, fit, omega.shape, config,
                    bank, bank_raw, lag_histogram,
                )
                numerator += batch_numerator
                denominator.index_add_(0, labels, spatial.square().sum(dim=1))
                atom_weight.index_add_(0, labels, fit["alpha"])
                counts += torch.bincount(labels, minlength=config.q)
                input_energy += waveforms.square().sum()
        if config.freeze_prototypes:
            proposed_omega, proposed_prototypes = frozen_cone_proposal(
                omega, prototypes, assignment, numerator, cosine_limit
            )
        else:
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
        lag_axis = torch.abs(torch.arange(
            -config.max_shift, config.max_shift + 1, device=config.device
        ))
        mean_abs_lag = float((lag_histogram * lag_axis).sum()) / max(
            int(lag_histogram.sum()), 1
        )
        history.append({
            "iteration": iteration,
            "fixed_assignment_objective": float(before.item()),
            "objective_after_basis": float(after.item()),
            "basis_accepted": accepted,
            "step_size": step_size,
            "maximum_row_change": change,
            "row_counts": counts.cpu().tolist(),
            "lag_histogram": lag_histogram.cpu().tolist(),
            "mean_abs_lag": mean_abs_lag,
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
        "kind": "0021_shift_invariant_master_ibl_bwm",
        "clips": len(pool),
        "pool_events": pool_events,
        "priors": config.prototype_priors,
        "shift_learning": config.shift_learning,
        "max_shift": config.max_shift,
        "prototype_count": config.prototype_count,
        "cone_half_angle_degrees": config.prototype_cone_deg,
        "orientation": "prototype_polarity_preserved",
        "frozen_during_pursuit": True,
        "guard_convention": "raw input energy, renormalized-bank cross term",
    })
    atomic_json(complete_path, {
        "iterations": len(history),
        "basis_accepted": history[-1]["basis_accepted"],
    })


def synthetic_shards(root, config):
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(11)
    t = 90
    time_axis = torch.linspace(-1.5, 1.5, t)

    def biphasic(main_sigma, main_center, under_sigma, under_center, under_amplitude):
        return (
            torch.exp(-((time_axis - main_center) / main_sigma).pow(2))
            - under_amplitude * torch.exp(-((time_axis - under_center) / under_sigma).pow(2))
        )

    fast = biphasic(0.15, 0.1, 0.20, 0.50, 0.45)
    slow = biphasic(0.45, 0.2, 0.40, 1.05, 0.45)
    shapes = torch.stack([fast, -fast, slow, -slow])
    shapes = shapes / shapes.norm(dim=1, keepdim=True)
    grid = np.stack(np.meshgrid(np.arange(0, 80, 20.0),
                                np.arange(0, 80, 20.0), indexing="ij"),
                    axis=-1).reshape(-1, 2).astype(np.float32)
    sigma = 32.0
    true_lags = []
    for shard in range(3):
        count, width = 256, grid.shape[0]
        rng = np.random.default_rng(shard)
        sites = grid[rng.integers(0, len(grid), count)]
        depths = rng.uniform(10, 40, count).astype(np.float32)
        offsets = np.stack([grid - sites[event]
                            for event in range(count)]).astype(np.float32)
        distance_sq = (offsets ** 2).sum(axis=2)
        weights = (sigma / np.sqrt(
            distance_sq + depths[:, None] ** 2 + sigma ** 2
        )).astype(np.float32)
        mask = np.ones((count, width), bool)
        noise = np.full((count, width), 0.05, np.float32)
        waveforms = 0.01 * torch.randn(count, width, t, generator=generator)
        lags = torch.randint(-config.max_shift, config.max_shift + 1, (count,),
                             generator=torch.Generator().manual_seed(shard + 50)
                             ).tolist()
        for event in range(count):
            lag = lags[event]
            true_lags.append(lag)
            shifted = torch.zeros(t)
            source = torch.arange(t) - lag
            valid = (source >= 0) & (source < t)
            shifted[valid] = shapes[event % len(shapes)][source[valid]]
            waveforms[event] += torch.as_tensor(weights[event])[:, None] * shifted[None, :]
        M26.atomic_npz(shard_dir / f"clip_synthetic_{shard}.npz", {
            "spike_times": np.arange(count) * 50,
            "spike_channels": np.zeros(count, np.int32),
            "waveforms": waveforms.numpy().astype(np.float32),
            "mask": mask, "local_noise": noise, "local_offsets": offsets,
            "peak_polarity": np.ones(count, np.int8),
            "clip": np.asarray(f"synthetic_{shard}"),
        })
    return shard_dir, np.asarray(true_lags)


def self_test():
    device = "cpu"
    config = Config21(
        q=8, prototype_priors=4, max_shift=4, shift_learning=True,
        alternating_iterations=1, device=device,
    )
    root = HERE.parent.parent / "runs" / "_0021_selftest"
    shard_dir, true_lags = synthetic_shards(root, config)
    aligned, polarity, per_clip = M26.build_pool(shard_dir, device)
    assert len(aligned) == 3 * 256, len(aligned)
    speed, widths, _, thresholds = speed_split(aligned, polarity, config)
    assert len(speed) == len(aligned)
    assert set(speed.tolist()) <= {0, 1}
    assert thresholds["polarity_0"]["narrow_fast_events"] > 0
    assert thresholds["polarity_0"]["wide_slow_events"] > 0
    cosine_limit = float(np.cos(np.radians(config.prototype_cone_deg)))
    omega, prototypes, assignment, group_counts = initialize_codebook_priors(
        aligned, polarity, speed, config.q, cosine_limit, config.seed,
        config.prototype_kmeans_iterations, config.prototype_priors,
    )
    assert omega.shape == (config.q, t := 90), omega.shape
    assert len(prototypes) == 4
    assert min(group_counts) > 0, group_counts
    for index in range(4):
        extremum = prototypes[index, prototypes[index].abs().argmax()]
        assert (extremum > 0) == (index % 2 == 0), (index, float(extremum))
    two_omega, two_prototypes, _, _ = initialize_codebook_priors(
        aligned, polarity, speed, config.q, cosine_limit, config.seed,
        config.prototype_kmeans_iterations, 2,
    )
    assert len(two_prototypes) == 2 and two_omega.shape == (config.q, t)

    shard_values = []
    for path in sorted(shard_dir.glob("clip_*.npz")):
        shard_values.append(M26.load_shard_events(path, None, np.random.default_rng(0)))

    def fit_once(shift_learning):
        run_config = Config21(
            q=config.q, prototype_priors=config.prototype_priors,
            max_shift=config.max_shift, shift_learning=shift_learning,
            alternating_iterations=1, device=device,
        )
        sites_np, axes_np = BASE.coarse_lattice(run_config.base())
        sites = torch.as_tensor(sites_np, device=device)
        axes = [torch.as_tensor(axis, device=device) for axis in axes_np]
        sigmas = torch.as_tensor(BASE.sigma_bank(run_config.base()), device=device)
        cache = FOOTPRINT_CACHE(sites, sigmas, device)
        numerator = torch.zeros_like(omega)
        denominator = torch.zeros(run_config.q, device=device)
        atom_weight = torch.zeros(run_config.q, device=device)
        input_energy = torch.zeros((), device=device)
        lag_histogram = torch.zeros(2 * run_config.max_shift + 1,
                                    dtype=torch.long, device=device)
        bank = bank_raw = None
        if shift_learning:
            bank, bank_raw = build_shift_bank(omega, run_config.max_shift)
        recovered = []
        for values in shard_values:
            waveforms_all = torch.as_tensor(values["waveforms"])
            offsets_all = torch.as_tensor(values["local_offsets"])
            mask_all = torch.as_tensor(values["mask"])
            noise_all = torch.as_tensor(values["local_noise"])
            for start in range(0, len(waveforms_all), 128):
                stop = min(start + 128, len(waveforms_all))
                waveforms = waveforms_all[start:stop]
                offsets = offsets_all[start:stop]
                mask = mask_all[start:stop]
                local_noise = noise_all[start:stop]
                fit = PIPELINE.fit_grouped(waveforms, offsets, mask, local_noise,
                                           omega, sites, axes, sigmas, run_config,
                                           cache)
                spatial = spatial_weights(fit, offsets, mask)
                labels, batch_numerator, lag_index = accumulate_statistics(
                    waveforms, spatial, fit, omega.shape, run_config,
                    bank, bank_raw, lag_histogram,
                )
                numerator += batch_numerator
                denominator.index_add_(0, labels, spatial.square().sum(dim=1))
                atom_weight.index_add_(0, labels, fit["alpha"])
                input_energy += waveforms.square().sum()
                recovered.append(
                    (lag_index - run_config.max_shift).numpy()
                )
        proposed, proposed_prototypes = N19.prototype_cone_proposal(
            omega, prototypes, assignment, numerator, atom_weight, cosine_limit)
        updated, _, before, after, step_size = N19.backtracked_update(
            omega, prototypes, proposed, proposed_prototypes, assignment,
            numerator, denominator, input_energy, cosine_limit)
        assert float(after) <= float(before) + 1e-5 * max(1.0, abs(float(before)))
        return float(before), float(after), step_size, recovered

    before_fixed, after_fixed, _, _ = fit_once(False)
    before_shift, after_shift, step_shift, recovered_batches = fit_once(True)
    recovered = np.concatenate(recovered_batches)
    error = np.abs(recovered - true_lags)
    print(f"0021 self-test: fixed objective {before_fixed:.4f} -> {after_fixed:.4f}; "
          f"shift objective {before_shift:.4f} -> {after_shift:.4f} "
          f"(step {step_shift}); lag error mean {error.mean():.2f} "
          f"median {np.median(error):.0f}", flush=True)
    assert after_shift <= after_fixed + 1e-3 * max(1.0, abs(after_fixed)), (
        "lag-aligned learning must not be worse than fixed-onset learning"
    )
    assert np.median(error) <= 2.0, "recovered lags should track the applied lags"
    random_numerator = torch.randn_like(omega)
    frozen_atoms, frozen_prototypes = frozen_cone_proposal(
        omega, prototypes, assignment, random_numerator, cosine_limit
    )
    assert torch.equal(frozen_prototypes, prototypes)
    for atom_index in range(len(omega)):
        cosine = float(F.normalize(frozen_atoms[atom_index:atom_index + 1], dim=1)
                       @ prototypes[assignment[atom_index]])
        assert cosine >= cosine_limit - 1e-5, (atom_index, cosine)
    print("0021 self-test passed", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "fit", "all"),
                        nargs="?", default="all")
    parser.add_argument("--shards", type=Path,
                        default=Path("residuals/runs/ibl_bwm/master_codebook_q32/shards"))
    parser.add_argument("--output", type=Path,
                        default=Path("residuals/runs/ibl_bwm/master_codebook_0021_q32_p4_shift"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--priors", dest="prototype_priors", type=int, default=None,
                        help="alias for --prototype-priors")
    for field in Config21.__dataclass_fields__.values():
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
    overrides = {
        name: getattr(args, name) for name in Config21.__dataclass_fields__
        if getattr(args, name) is not None
    }
    config = Config21(**overrides)
    if args.self_test:
        self_test()
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if not args.shards.exists():
        raise FileNotFoundError(
            f"{args.shards} missing; run 026_master_codebook.py harvest first"
        )
    print(f"0021 shift-invariant codebook: shards={args.shards} Q={config.q} "
          f"priors={config.prototype_priors} shift={config.shift_learning} "
          f"max_shift={config.max_shift} output={output}", flush=True)
    if args.stage in ("init", "all"):
        run_init(config, args.shards, output, args.resume)
    if args.stage in ("fit", "all"):
        run_fit(config, args.shards, output, args.resume)


if __name__ == "__main__":
    main()
