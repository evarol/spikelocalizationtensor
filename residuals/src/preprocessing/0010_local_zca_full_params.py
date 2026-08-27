"""Full-recording reproduction of the original session-0010 local-ZCA pursuit."""

from pathlib import Path

from run_whitened_dense_pursuit import main


ROOT = Path(__file__).resolve().parents[3]
RECORDING = Path("/scratch/ap7151/_RAW_DATA/extra-motion/dataset1_p1/p1_g0_t0.imec0.ap.bin")
CODEBOOK = ROOT / "residuals/runs/dataset1_p1/raw_peak_channel_codebooks_16358267/global_codebook_q8.npz"
OUTPUT = ROOT / "residuals/runs/dataset1_p1/unwhitened_local_0010_full"
PARAMS = {
    "duration_seconds": 1957.1908, "threshold": 6.0, "whitening": "none",
    "whitening_range_um": 48.0, "chunk_seconds": 2.187, "radius_um": 48.0,
    "temporal_radius_ms": 0.5, "pursuit_rounds": 60,
    "pursuit_min_round_energy_drop_fraction": 0.0, "max_peaks_per_round": 10000,
    "fit_batch_size": 1024, "localization_config_batch_size": 32,
    "template_time_batch": 4096, "n_scales": 9, "max_channel_normalized_rmse": 3.0,
    "learn_omega": True, "learn_omega_chunks": 4, "save_waveforms": True,
    "profile_stages": True, "resume": True, "device": "cuda",
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
