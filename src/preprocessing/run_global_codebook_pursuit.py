"""Run frozen global-codebook pursuit with fixed or matched detection threshold."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from raw_residual import ResidualConfig, load_omega, run_recording


def resolve_threshold(codebook_path, q, mode, fixed_threshold, calibration_path):
    if mode == "fixed":
        return float(fixed_threshold), None
    if calibration_path is None:
        raise ValueError("matched threshold mode requires --threshold-calibration")
    calibration = json.loads(Path(calibration_path).read_text())
    result = calibration["results"].get(str(q))
    if result is None:
        raise KeyError(f"threshold calibration has no entry for Q={q}")
    calibrated_path = Path(result["codebook_path"]).resolve()
    if calibrated_path != Path(codebook_path).resolve():
        raise ValueError(
            f"calibration Q={q} used {calibrated_path}, not {Path(codebook_path).resolve()}"
        )
    return float(result["matched_threshold"]), calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_path", type=Path)
    parser.add_argument("codebook_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--threshold-mode", choices=("fixed", "matched"), default="matched")
    parser.add_argument("--threshold-calibration", type=Path)
    parser.add_argument("--fixed-threshold", type=float, default=6.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=8.748)
    parser.add_argument("--chunk-seconds", type=float, default=2.187)
    parser.add_argument("--pursuit-rounds", type=int, default=60)
    parser.add_argument("--pursuit-min-round-energy-drop-fraction", type=float, default=0.0)
    parser.add_argument("--max-peaks-per-round", type=int, default=10000)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument("--localization-config-batch-size", type=int, default=32)
    parser.add_argument("--template-time-batch", type=int, default=4096)
    parser.add_argument("--continuous-max-iterations", type=int, default=80)
    parser.add_argument("--continuous-backtracks", type=int, default=30)
    parser.add_argument("--save-waveforms", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    omega = load_omega(args.codebook_path)
    q = len(omega)
    threshold, calibration = resolve_threshold(
        args.codebook_path,
        q,
        args.threshold_mode,
        args.fixed_threshold,
        args.threshold_calibration,
    )
    config = ResidualConfig(
        threshold=threshold,
        min_captured_fraction=0.05,
        chunk_seconds=args.chunk_seconds,
        max_peaks_per_round=args.max_peaks_per_round,
        fit_batch_size=args.fit_batch_size,
        localization_config_batch_size=args.localization_config_batch_size,
        template_time_batch=args.template_time_batch,
        pursuit_rounds=args.pursuit_rounds,
        pursuit_min_round_energy_drop_fraction=(
            args.pursuit_min_round_energy_drop_fraction
        ),
        codebook_learning_chunks=0,
        kernel="monopole",
        n_scales=10,
        n_sites=16,
        refine_levels=6,
        continuous_refine=True,
        continuous_max_iterations=args.continuous_max_iterations,
        continuous_backtracks=args.continuous_backtracks,
        device=args.device,
        save_waveforms=args.save_waveforms,
        profile_stages=args.profile_stages,
    )
    started = perf_counter()
    run_recording(
        args.recording_path,
        args.codebook_path,
        args.output_path,
        config,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
    )
    runtime = {
        "q": q,
        "threshold_mode": args.threshold_mode,
        "threshold": threshold,
        "elapsed_seconds": perf_counter() - started,
        "codebook_path": str(args.codebook_path.resolve()),
        "threshold_calibration": (
            str(args.threshold_calibration.resolve())
            if args.threshold_calibration is not None
            else None
        ),
        "config": asdict(config),
    }
    (args.output_path / "runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
