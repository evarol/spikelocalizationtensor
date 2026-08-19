"""Continuously refine a frozen masked monopole fit inside its final voxels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from continuous_refine import (
    curvature_width,
    monopole_profile,
    refine_batch,
    score,
    voxel_cell_bounds,
)


DTYPE = torch.float64
EPS = 1e-30


def quantiles(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {}
    levels = np.quantile(finite, (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1))
    names = ("min", "p25", "median", "p75", "p90", "p99", "max")
    result = {name: float(value) for name, value in zip(names, levels)}
    result["mean"] = float(finite.mean())
    return result


def load_frozen_fit(path, max_spikes=None):
    required = {
        "sources",
        "sigma",
        "omega",
        "temporal_idx",
        "alpha",
        "profile_idx",
        "voxel_bounds_um",
        "voxel_size_um",
        "nmse",
    }
    with np.load(path, allow_pickle=True) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"fit is missing fields: {sorted(missing)}")
        n_spikes = len(archive["sources"])
        if max_spikes is not None:
            n_spikes = min(n_spikes, max_spikes)
        fit = {
            "sources": np.asarray(archive["sources"][:n_spikes], dtype=np.float64),
            "sigma": np.asarray(archive["sigma"][:n_spikes], dtype=np.float64),
            "omega": np.asarray(archive["omega"], dtype=np.float64),
            "temporal_idx": np.asarray(
                archive["temporal_idx"][:n_spikes], dtype=np.int64),
            "alpha": np.asarray(archive["alpha"][:n_spikes], dtype=np.float64),
            "profile_idx": np.asarray(
                archive["profile_idx"][:n_spikes], dtype=np.int64),
            "voxel_bounds_um": np.asarray(
                archive["voxel_bounds_um"], dtype=np.float64),
            "voxel_size_um": float(archive["voxel_size_um"]),
            "nmse": float(archive["nmse"]),
        }
    return fit


def validate_grid(fit):
    source = fit["sources"]
    bounds = fit["voxel_bounds_um"]
    spacing = fit["voxel_size_um"]
    grid_coordinate = (source - bounds[0]) / spacing
    error = np.max(np.abs(grid_coordinate - np.rint(grid_coordinate)))
    if error > 1e-6:
        raise ValueError(
            f"frozen sources are not on the saved voxel grid; max error={error:g}")
    return voxel_cell_bounds(source, bounds, spacing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_path", type=Path)
    parser.add_argument("fit_path", type=Path)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--kernel", choices=("monopole",), default="monopole")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--spike-chunk", type=int, default=131_072)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--backtracks", type=int, default=30)
    parser.add_argument("--drop", type=float, default=0.01)
    parser.add_argument("--max-spikes", type=int)
    args = parser.parse_args()

    summary_path = args.result_path.with_suffix(".json")
    partial_result = args.result_path.with_name(
        f".{args.result_path.stem}.partial.npz")
    partial_summary = summary_path.with_name(
        f".{summary_path.stem}.partial.json")
    existing = [
        path for path in (
            args.result_path, summary_path, partial_result, partial_summary)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    fit = load_frozen_fit(args.fit_path, args.max_spikes)
    source_grid = fit["sources"]
    lower, upper = validate_grid(fit)
    n_spikes = len(source_grid)

    waveforms = np.load(
        args.session_path / "neighborhood_waveforms.npy", mmap_mode="r")
    offsets_all = np.load(
        args.session_path / "local_coords.npy", mmap_mode="r")
    neighbor_ids = np.load(
        args.session_path / "neighbor_ids.npy", mmap_mode="r")
    centroids = np.load(args.session_path / "centroids.npy", mmap_mode="r")
    lengths = (len(waveforms), len(offsets_all), len(neighbor_ids), len(centroids))
    if min(lengths) < n_spikes:
        raise ValueError(
            f"session arrays are shorter than the fit: {lengths} versus {n_spikes}")

    omega = torch.as_tensor(fit["omega"], dtype=DTYPE, device=device)
    omega_energy_all = np.square(fit["omega"]).sum(axis=1)
    frozen_energy = np.square(fit["alpha"]) * omega_energy_all[
        fit["temporal_idx"]]

    source_continuous = np.empty_like(source_grid)
    energy_grid = np.empty(n_spikes, dtype=np.float64)
    energy_continuous = np.empty(n_spikes, dtype=np.float64)
    waveform_energy = np.empty(n_spikes, dtype=np.float64)
    alpha_continuous = np.empty(n_spikes, dtype=np.float64)
    eigenvalues = np.empty((n_spikes, 3), dtype=np.float64)
    widths = np.empty((n_spikes, 3), dtype=np.float64)
    centered_energy = 0.0
    started = time.perf_counter()

    print(
        f"continuous masked monopole refinement: {n_spikes:,} spikes; "
        f"{fit['voxel_size_um']:g} um voxels; {device}",
        flush=True,
    )
    for start in range(0, n_spikes, args.spike_chunk):
        stop = min(start + args.spike_chunk, n_spikes)
        mask_np = np.asarray(neighbor_ids[start:stop] >= 0)
        raw = torch.as_tensor(
            np.asarray(waveforms[start:stop], dtype=np.float64),
            dtype=DTYPE,
            device=device,
        )
        mask_bool = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        mask = mask_bool.to(DTYPE)
        centered = raw - raw.mean(dim=2, keepdim=True)
        centered_energy += float((centered * centered).sum().item())
        measured = raw.masked_fill(~mask_bool[:, :, None], 0)
        waveform_energy[start:stop] = (
            measured * measured).sum(dim=(1, 2)).cpu().numpy()

        offsets_np = np.asarray(
            offsets_all[start:stop], dtype=np.float64).copy()
        offsets_np[~mask_np] = 0
        offsets = torch.as_tensor(offsets_np, dtype=DTYPE, device=device)
        temporal_idx = torch.as_tensor(
            fit["temporal_idx"][start:stop], dtype=torch.long, device=device)
        selected_omega = omega[temporal_idx]
        omega_energy = (selected_omega * selected_omega).sum(dim=1)
        projected = torch.einsum("nct,nt->nc", measured, selected_omega)
        form = (
            projected[:, :, None] * projected[:, None, :]
            / omega_energy[:, None, None]
        )

        sigma = torch.as_tensor(
            fit["sigma"][start:stop], dtype=DTYPE, device=device)
        mu_grid = torch.as_tensor(
            source_grid[start:stop], dtype=DTYPE, device=device)
        cell_lower = torch.as_tensor(
            lower[start:stop], dtype=DTYPE, device=device)
        cell_upper = torch.as_tensor(
            upper[start:stop], dtype=DTYPE, device=device)
        energy_grid[start:stop] = score(
            form, offsets, mu_grid, sigma, mask).cpu().numpy()
        mu, value, _, hessian = refine_batch(
            form,
            offsets,
            mu_grid,
            sigma,
            cell_lower,
            cell_upper,
            mask=mask,
            max_iterations=args.max_iterations,
            backtracks=args.backtracks,
        )
        values, width = curvature_width(hessian, value, args.drop)
        footprint, _, _ = monopole_profile(offsets, mu, sigma, mask)
        footprint = footprint / footprint.norm(dim=1, keepdim=True)
        response = (footprint * projected).sum(dim=1)

        source_continuous[start:stop] = mu.cpu().numpy()
        energy_continuous[start:stop] = value.cpu().numpy()
        alpha_continuous[start:stop] = (
            response / omega_energy).cpu().numpy()
        eigenvalues[start:stop] = values.cpu().numpy()
        widths[start:stop] = width.cpu().numpy()

        fraction = stop / n_spikes
        elapsed = time.perf_counter() - started
        print(
            f"  {stop:,}/{n_spikes:,} ({fraction:.1%})  {elapsed:.1f}s  "
            f"eta {elapsed / fraction - elapsed:.1f}s",
            flush=True,
        )

    delta = source_continuous - source_grid
    distance = np.linalg.norm(delta, axis=1)
    cell = upper - lower
    on_bound = (
        (source_continuous - lower < 1e-9 * cell)
        | (upper - source_continuous < 1e-9 * cell)
    )
    gain = energy_continuous - energy_grid
    relative_gain = gain / np.maximum(np.abs(energy_grid), EPS)
    fraction_grid = energy_grid / np.maximum(waveform_energy, EPS)
    fraction_continuous = energy_continuous / np.maximum(
        waveform_energy, EPS)
    variance = centered_energy / (
        n_spikes * waveforms.shape[1] * waveforms.shape[2])
    nmse_grid = (
        np.mean(waveform_energy - energy_grid)
        / (waveforms.shape[1] * waveforms.shape[2])
        / variance
    )
    nmse_continuous = (
        np.mean(waveform_energy - energy_continuous)
        / (waveforms.shape[1] * waveforms.shape[2])
        / variance
    )

    localization_grid = np.column_stack(
        (
            np.asarray(centroids[:n_spikes, 0]) + source_grid[:, 0],
            source_grid[:, 2],
            np.asarray(centroids[:n_spikes, 1]) + source_grid[:, 1],
        )
    )
    localization_continuous = np.column_stack(
        (
            np.asarray(centroids[:n_spikes, 0]) + source_continuous[:, 0],
            source_continuous[:, 2],
            np.asarray(centroids[:n_spikes, 1]) + source_continuous[:, 1],
        )
    )

    interior = ~on_bound.any(axis=1)
    flat = widths.max(axis=1) > np.linalg.norm(cell, axis=1)
    summary = {
        "model": "masked monopole; frozen temporal and profile choices",
        "objective": "continuous gain-eliminated captured energy inside final voxel",
        "kernel": args.kernel,
        "refit_of_omega": False,
        "sigma_refined": False,
        "temporal_assignment_refined": False,
        "n_spikes": n_spikes,
        "voxel_size_um": fit["voxel_size_um"],
        "wall_s": time.perf_counter() - started,
        "monotone_violations": int(
            (energy_continuous < energy_grid - 1e-12).sum()),
        "outside_cell": int(
            ((source_continuous < lower - 1e-9)
             | (source_continuous > upper + 1e-9)).any(axis=1).sum()),
        "displacement_um": quantiles(distance),
        "displacement_x_um": quantiles(np.abs(delta[:, 0])),
        "displacement_y_um": quantiles(np.abs(delta[:, 1])),
        "displacement_z_um": quantiles(np.abs(delta[:, 2])),
        "delta_energy_relative": quantiles(relative_gain),
        "captured_fraction_grid": quantiles(fraction_grid),
        "captured_fraction_continuous": quantiles(fraction_continuous),
        "nmse_frozen_archive_full_dataset": fit["nmse"],
        "nmse_recomputed_grid_for_processed_spikes": float(nmse_grid),
        "nmse_continuous_for_processed_spikes": float(nmse_continuous),
        "localization_columns": [
            "probe_x_um", "distance_from_probe_um", "probe_depth_um"],
        "interior_optimum_fraction": float(interior.mean()),
        "on_bound_fraction_x": float(on_bound[:, 0].mean()),
        "on_bound_fraction_y": float(on_bound[:, 1].mean()),
        "on_bound_fraction_z": float(on_bound[:, 2].mean()),
        "flat_cell_fraction": float(flat.mean()),
        "frozen_energy_parity_max_rel": float(np.max(
            np.abs(energy_grid - frozen_energy)
            / np.maximum(np.abs(frozen_energy), EPS))),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }

    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        partial_result,
        sources_grid=source_grid.astype(np.float32),
        sources_continuous=source_continuous.astype(np.float32),
        delta_source=delta.astype(np.float32),
        delta_source_norm=distance.astype(np.float32),
        localizations_grid=localization_grid.astype(np.float32),
        localizations_continuous=localization_continuous.astype(np.float32),
        sigma=fit["sigma"].astype(np.float32),
        profile_idx=fit["profile_idx"].astype(np.int64),
        temporal_idx=fit["temporal_idx"].astype(np.int64),
        alpha_grid=fit["alpha"].astype(np.float32),
        alpha_continuous=alpha_continuous.astype(np.float32),
        energy_grid=energy_grid.astype(np.float32),
        energy_continuous=energy_continuous.astype(np.float32),
        delta_energy=gain.astype(np.float32),
        delta_energy_relative=relative_gain.astype(np.float32),
        waveform_energy=waveform_energy.astype(np.float32),
        cell_lower=lower.astype(np.float32),
        cell_upper=upper.astype(np.float32),
        on_bound=on_bound,
        hessian_eigenvalues=eigenvalues.astype(np.float32),
        curvature_width=widths.astype(np.float32),
        spike_index=np.arange(n_spikes, dtype=np.int64),
    )
    partial_summary.write_text(json.dumps(summary, indent=2) + "\n")
    partial_result.replace(args.result_path)
    partial_summary.replace(summary_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2))
    print(f"saved {args.result_path}", flush=True)


if __name__ == "__main__":
    main()
