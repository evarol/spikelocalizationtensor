import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

PEAKS_DTYPE = np.dtype(
    [
        ("sample_index", "<i8"),
        ("channel_index", "<i4"),
        ("amplitude", "<f4"),
        ("segment_index", "<i4"),
    ]
)
LOCATIONS_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])


def read_recording(path, series):
    from spikeinterface.extractors import read_nwb_recording

    return read_nwb_recording(path, electrical_series_path=series, use_pynwb=False)


def log_geometry(rec, tag):
    chans = rec.get_channel_locations()
    fs = rec.get_sampling_frequency()
    ns = rec.get_num_samples()
    print(
        f"{tag}: fs={fs} ns={ns} ({ns / fs / 60.0:.1f} min) "
        f"n_channels={rec.get_num_channels()} "
        f"x=[{chans[:, 0].min():.0f}, {chans[:, 0].max():.0f}] "
        f"y=[{chans[:, 1].min():.0f}, {chans[:, 1].max():.0f}]",
        flush=True,
    )


def run_prepare(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "_peaks_done"
    peaks_path = out / "peaks.npy"
    locations_path = out / "peak_locations.npy"
    if marker.exists() and peaks_path.exists() and locations_path.exists():
        print("peaks stage already complete; skipping", flush=True)
        return

    run = Path(args.residual_run)
    spike_times = np.load(run / "spike_times.npy")
    global_sources = np.load(run / "global_sources.npy")
    alpha = np.load(run / "alpha.npy")
    spike_channels = np.load(run / "spike_channels.npy")
    assert spike_times.shape[0] == global_sources.shape[0] == alpha.shape[0] == spike_channels.shape[0]
    n = spike_times.shape[0]
    print(f"residual run {run}: {n} accepted events", flush=True)

    peaks = np.zeros(n, dtype=PEAKS_DTYPE)
    peaks["sample_index"] = spike_times
    peaks["channel_index"] = spike_channels
    peaks["amplitude"] = alpha * 1e6
    peaks["segment_index"] = 0

    locations = np.zeros(n, dtype=LOCATIONS_DTYPE)
    locations["x"] = global_sources[:, 0]
    locations["y"] = global_sources[:, 1]
    locations["z"] = global_sources[:, 2]

    depths = locations["y"]
    qs = np.percentile(depths, [1, 5, 25, 50, 75, 95, 99])
    print("depth percentiles (1/5/25/50/75/95/99 um):", np.round(qs, 1), flush=True)
    print(f"amplitude uV: median={np.median(peaks['amplitude']):.1f}", flush=True)

    tmp_peaks = out / "peaks.tmp.npy"
    np.save(tmp_peaks, peaks)
    os.replace(tmp_peaks, peaks_path)
    tmp_locs = out / "peak_locations.tmp.npy"
    np.save(tmp_locs, locations)
    os.replace(tmp_locs, locations_path)

    meta = dict(
        residual_run=str(run),
        n_peaks=int(n),
        amplitude_field="alpha_volts_to_uV",
        position_field="global_sources",
        source="0025 residual accepted events",
    )
    tmp_meta = out / "peaks_meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, out / "peaks_meta.json")
    marker.write_text("ok\n")
    print("peaks stage complete", flush=True)


def run_motion(args):
    import torch
    from spikeinterface.sortingcomponents.motion import estimate_motion

    out = Path(args.output_dir)
    marker = out / "_motion_done"
    motion_path = out / "motion.npz"
    if marker.exists() and motion_path.exists():
        print("motion stage already complete; skipping", flush=True)
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")

    peaks = np.load(out / "peaks.npy")
    peak_locations = np.load(out / "peak_locations.npy")
    print(f"loaded {peaks.size} peaks", flush=True)

    rec = read_recording(args.recording_path, args.electrical_series_path)
    log_geometry(rec, "nwb-ap")

    t0 = time.time()
    motion, extra = estimate_motion(
        rec,
        peaks,
        peak_locations,
        method="dredge_ap",
        direction="y",
        rigid=args.rigid,
        win_shape="gaussian",
        win_step_um=args.win_step_um,
        win_scale_um=args.win_scale_um,
        extra_outputs=True,
        progress_bar=True,
        bin_s=args.bin_s,
        bin_um=args.bin_um,
        max_disp_um=args.max_disp_um,
        device=args.device,
    )
    print(f"dredge_ap finished in {time.time() - t0:.0f}s", flush=True)

    displacement = motion.displacement[0]
    time_bin_centers = motion.temporal_bins_s[0]
    window_centers = motion.spatial_bins_um
    print(f"displacement shape={displacement.shape}", flush=True)
    ad = np.abs(displacement)
    print(
        f"displacement um: median={np.median(ad):.1f} p95={np.percentile(ad, 95):.1f} "
        f"max={ad.max():.1f} | window spread median={np.median(np.ptp(displacement, axis=1)):.1f}",
        flush=True,
    )

    tmp = out / "motion.tmp.npz"
    np.savez(
        tmp,
        displacement=displacement,
        time_bin_centers=time_bin_centers,
        window_centers=window_centers,
        D=extra["D"],
        C=extra["C"],
        weights_orig=extra["weights_orig"],
        mincorr=extra.get("mincorr"),
        max_disp_um=extra.get("max_disp_um"),
    )
    os.replace(tmp, motion_path)

    meta = dict(
        method="dredge_ap",
        rigid=args.rigid,
        bin_s=args.bin_s,
        bin_um=args.bin_um,
        max_disp_um=args.max_disp_um,
        win_step_um=args.win_step_um,
        win_scale_um=args.win_scale_um,
        device=args.device,
        n_peaks=int(peaks.size),
        displacement_shape=list(displacement.shape),
    )
    tmp_meta = out / "motion_meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, out / "motion_meta.json")
    marker.write_text("ok\n")
    print("motion stage complete", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "motion"])
    p.add_argument("--recording-path")
    p.add_argument("--electrical-series-path", default="acquisition/ElectricalSeriesAP")
    p.add_argument("--residual-run")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bin-s", type=float, default=1.0)
    p.add_argument("--bin-um", type=float, default=1.0)
    p.add_argument("--max-disp-um", type=float, default=150.0)
    p.add_argument("--win-step-um", type=float, default=200.0)
    p.add_argument("--win-scale-um", type=float, default=300.0)
    p.add_argument("--rigid", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.stage == "prepare":
        run_prepare(args)
    else:
        run_motion(args)


if __name__ == "__main__":
    main()
