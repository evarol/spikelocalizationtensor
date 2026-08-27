"""Controlled session-0010 diagonal-whitening localization smoke run."""

import os
from pathlib import Path

from run_whitened_dense_pursuit import main


ROOT = Path(__file__).resolve().parents[3]
RECORDING = Path("/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin")
CODEBOOK = ROOT / "residuals/runs/dataset1_p1/raw_peak_channel_codebooks_16358267/global_codebook_q8.npz"
OUTPUT_ROOT = ROOT / "residuals/runs/dataset1_p1"

PARAMS = {
    "start_seconds": 0.0,
    "duration_seconds": 1.0,
    "threshold": 6.0,
    "whitening": "diagonal",
    "chunk_seconds": 1.0,
    "radius_um": 48.0,
    "temporal_radius_ms": 0.5,
    "pursuit_rounds": 4,
    "pursuit_lockout_ms": None,
    "pursuit_min_round_energy_drop_fraction": 0.01,
    "max_peaks_per_round": 10000,
    "fit_batch_size": 1024,
    "localization_config_batch_size": 32,
    "template_time_batch": 4096,
    "n_scales": 9,
    "max_channel_normalized_rmse": 3.0,
    "learn_omega": False,
    "save_waveforms": True,
    "profile_stages": True,
    "device": "cuda",
}


def cli_args(output_path):
    args = [str(RECORDING), str(CODEBOOK), str(output_path)]
    for name, value in PARAMS.items():
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend((flag, str(value)))
    return args


if __name__ == "__main__":
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    main(cli_args(OUTPUT_ROOT / f"diagonal_continuous_0010_{job_id}"))
