import argparse
import json
from pathlib import Path

import iblsorter
from iblutil.util import setup_logger
from iblsorter.ibl import ibl_pykilosort_params, run_spike_sorting_ibl


def main():
    parser = argparse.ArgumentParser(
        description="Run IBL pykilosort (iblsorter) spike sorting on a SpikeGLX AP binary"
    )
    parser.add_argument("bin_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--stop-after",
        default=None,
        choices=[
            "whitening_matrix",
            "preprocess",
            "drift_correction",
            "learn",
            "merge",
            "split_1",
            "cutoff",
        ],
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    scratch_dir = output_dir / "scratch"
    ks_output_dir = output_dir / "iblsorter"
    alf_path = None if args.stop_after else output_dir / "alf"

    setup_logger(name="iblsorter", level="INFO")
    params = ibl_pykilosort_params(args.bin_file)
    print(
        f"ibl-sorter {iblsorter.__version__} "
        f"probe NP{params.probe.neuropixel_version} "
        f"nchan {len(params.probe.chanMap)} "
        f"bin {args.bin_file}",
        flush=True,
    )

    run_spike_sorting_ibl(
        args.bin_file,
        scratch_dir=scratch_dir,
        ks_output_dir=ks_output_dir,
        alf_path=alf_path,
        stop_after=args.stop_after,
        params=params,
    )

    if alf_path is None:
        print(f"stopped after {args.stop_after}; outputs in {ks_output_dir}", flush=True)
        return

    import one.alf.io as alfio

    spikes = alfio.load_object(alf_path, "spikes")
    clusters = alfio.load_object(alf_path, "clusters")
    times = spikes["times"]
    duration_s = float(times.max()) if times.size else 0.0
    summary = {
        "bin_file": str(args.bin_file),
        "iblsorter_version": str(iblsorter.__version__),
        "stop_after": args.stop_after,
        "n_spikes": int(times.size),
        "n_clusters": int(clusters["channels"].size),
        "duration_s": duration_s,
        "mean_firing_rate_hz": float(times.size / duration_s) if duration_s else 0.0,
        "ks_output_dir": str(ks_output_dir),
        "alf_path": str(alf_path),
    }
    (output_dir / "kilosort_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
