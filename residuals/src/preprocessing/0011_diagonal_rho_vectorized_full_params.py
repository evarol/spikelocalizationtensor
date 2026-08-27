"""Full-recording identifiable-rho pursuit with vectorized monopole refinement."""

from pathlib import Path

from run_whitened_dense_pursuit import main


ROOT = Path(__file__).resolve().parents[3]
RECORDING = Path("/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin")
CODEBOOK = ROOT / "residuals/runs/dataset1_p1/raw_peak_channel_codebooks_16358267/global_codebook_q8.npz"
OUTPUT = ROOT / "residuals/runs/dataset1_p1/unwhitened_rho_0011_vectorized_full"
PARAMS = {
    "duration_seconds": 1957.1908, "threshold": 6.0, "whitening": "none",
    "chunk_seconds": 1.0, "radius_um": 48.0, "temporal_radius_ms": 0.5,
    "pursuit_rounds": 4, "pursuit_min_round_energy_drop_fraction": 0.01,
    "max_peaks_per_round": 10000, "fit_batch_size": 2048,
    "localization_config_batch_size": 32, "template_time_batch": 4096,
    "n_scales": 9, "max_channel_normalized_rmse": 3.0, "identifiable_rho": True,
    "save_waveforms": True, "profile_stages": True, "resume": True, "device": "cuda",
}


def cli_args():
    args = [str(RECORDING), str(CODEBOOK), str(OUTPUT)]
    for name, value in PARAMS.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend((flag, str(value)))
    return args


if __name__ == "__main__":
    main(cli_args())
