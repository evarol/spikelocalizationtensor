"""Fit the analytic spike model to one extracted session."""

import argparse
from pathlib import Path

import numpy as np

from maths import fit_spike_model, load_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_path", type=Path)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kernel", default="monopole")
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--n-scales", type=int, default=10)
    parser.add_argument("--n-sites", type=int, default=16)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--refine-levels", type=int, default=6)
    parser.add_argument("--refine-stop-um", type=float, default=3.0)
    args = parser.parse_args()

    Y, off, _, _ = load_session(args.session_path)
    print(f"loaded Y={Y.shape}, off={off.shape}", flush=True)
    fit = fit_spike_model(
        off,
        Y,
        Q=args.q,
        kernels=tuple(part.strip() for part in args.kernel.split(",")),
        n_scales=args.n_scales,
        n_sites=args.n_sites,
        n_iters=args.n_iters,
        tol=1e-5,
        refine_levels=args.refine_levels,
        refine_stop_um=args.refine_stop_um,
        device=args.device,
    )
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.result_path,
        pick=fit["pick"],
        site_idx=fit["site_idx"],
        profile_idx=fit["profile_idx"],
        sources=fit["sources"],
        voxel_coordinates=fit["voxel_coordinates"],
        voxel_idx=fit["voxel_idx"],
        voxel_bounds_um=fit["voxel_bounds_um"],
        voxel_size_um=np.array(fit["voxel_size_um"]),
        coarse_sources=fit["coarse_sources"],
        refinement_levels=fit["refinement_levels"],
        refinement_displacement_um=fit["refinement_displacement_um"],
        sigma=fit["sigma"],
        a=fit["a"],
        omega=fit["omega"],
        pi=fit["pi"],
        v=fit["v"],
        temporal_idx=fit["temporal_idx"],
        temporal_one_hot=fit["temporal_one_hot"],
        nmse=np.array(fit["nmse"]),
        nmse_coarse=np.array(fit["nmse_coarse"]),
        history=np.array(fit["history"], dtype=object),
        lattice=fit["lattice"],
    )
    print(f"saved {args.result_path}", flush=True)
    print(f"final nMSE={fit['nmse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
