"""Whitened dense two-stage pursuit with per-channel RMSE acceptance.

All fitting and subtraction are performed in the selected normalized coordinate
system.  The output also contains the forward and pseudoinverse transforms for
displaying reconstructions in preprocessed-voltage coordinates.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from raw_residual import ResidualConfig, run_recording


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", type=Path)
    parser.add_argument("codebook_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument(
        "--whitening", choices=("none", "diagonal", "local-zca", "zca"),
        default="local-zca",
    )
    parser.add_argument("--whitening-range-um", type=float, default=48.0)
    parser.add_argument("--whitening-regularization", type=float, default=1e-3)
    parser.add_argument("--whitening-max-samples", type=int, default=300000)
    parser.add_argument("--whitening-seed", type=int, default=42)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--radius-um", type=float, default=48.0)
    parser.add_argument("--temporal-radius-ms", type=float, default=0.5)
    parser.add_argument("--pursuit-rounds", type=int, default=60)
    parser.add_argument("--pursuit-lockout-ms", type=float)
    parser.add_argument(
        "--pursuit-min-round-energy-drop-fraction", type=float, default=0.0
    )
    parser.add_argument("--max-peaks-per-round", type=int, default=10000)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument("--localization-config-batch-size", type=int, default=32)
    parser.add_argument("--template-time-batch", type=int, default=4096)
    parser.add_argument("--n-scales", type=int, default=9)
    parser.add_argument("--max-channel-normalized-rmse", type=float, default=3.0)
    parser.add_argument("--learn-omega", action="store_true")
    parser.add_argument(
        "--learn-omega-chunks", type=int, default=1,
        help="number of selected-interval chunks used to refine the input omega before extraction",
    )
    parser.add_argument("--omega-momentum", type=float, default=0.9)
    parser.add_argument("--omega-min-events-per-row", type=int, default=32)
    parser.add_argument("--omega-learning-seed", type=int, default=42)
    parser.add_argument("--save-waveforms", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.learn_omega_chunks < 1:
        parser.error("--learn-omega-chunks must be positive")

    config = ResidualConfig(
        threshold=args.threshold,
        radius_um=args.radius_um,
        temporal_radius_ms=args.temporal_radius_ms,
        chunk_seconds=args.chunk_seconds,
        pursuit_rounds=args.pursuit_rounds,
        pursuit_lockout_ms=args.pursuit_lockout_ms,
        pursuit_min_round_energy_drop_fraction=(
            args.pursuit_min_round_energy_drop_fraction
        ),
        max_peaks_per_round=args.max_peaks_per_round,
        fit_batch_size=args.fit_batch_size,
        localization_config_batch_size=args.localization_config_batch_size,
        template_time_batch=args.template_time_batch,
        kernel="monopole",
        n_scales=args.n_scales,
        codebook_learning_chunks=(
            args.learn_omega_chunks if args.learn_omega else 0
        ),
        codebook_momentum=args.omega_momentum,
        codebook_min_events_per_row=args.omega_min_events_per_row,
        codebook_learning_seed=args.omega_learning_seed,
        whitening=args.whitening,
        whitening_range_um=args.whitening_range_um,
        whitening_regularization=args.whitening_regularization,
        whitening_max_samples=args.whitening_max_samples,
        whitening_seed=args.whitening_seed,
        max_channel_normalized_rmse=args.max_channel_normalized_rmse,
        use_0010_math=True,
        # χ² is deliberately not an acceptance gate in this workflow.
        min_delta_chi2=0.0,
        device=args.device,
        save_waveforms=args.save_waveforms,
        profile_stages=args.profile_stages,
    )
    started = perf_counter()
    run_recording(
        args.recording_path, args.codebook_path, args.output_path, config,
        start_seconds=args.start_seconds, duration_seconds=args.duration_seconds,
        resume=args.resume,
    )
    runtime = {"elapsed_seconds": perf_counter() - started, "config": asdict(config)}
    (args.output_path / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
