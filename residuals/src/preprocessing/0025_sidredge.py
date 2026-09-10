import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from spikeinterface.core.node_pipeline import run_node_pipeline, ExtractDenseWaveforms
from spikeinterface.sortingcomponents.peak_localization.monopolar import LocalizeMonopolarTriangulation
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCADenoising
from sklearn.decomposition import PCA


def read_recording(path, series):
    return se.read_nwb_recording(path, electrical_series_path=series, use_pynwb=False)


def preprocess_recording(rec, freq_min=300.0, freq_max=6000.0):
    rec = sp.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max, dtype="float32")
    rec = sp.common_reference(rec, reference="global", operator="median")
    return rec


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


def atomic_save(path, arr):
    tmp = str(path) + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def fit_tpca_model(rec, peaks, n_components, fit_seconds, n_train, ms_before, ms_after, seed=42):
    fs = rec.get_sampling_frequency()
    fit_end = min(int(fit_seconds * fs), rec.get_num_samples())
    pool = np.flatnonzero(peaks["sample_index"] < fit_end)
    if pool.size == 0:
        raise ValueError(f"no spikes in the first {fit_seconds}s for TPCA")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(pool, size=min(n_train, pool.size), replace=False))
    selected_peaks = peaks[selected]
    locations = rec.get_channel_locations()
    n_neighbors = min(10, len(locations))
    distances = np.linalg.norm(locations[:, None] - locations[None, :], axis=-1)
    neighbor_table = np.argsort(distances, axis=1)[:, :n_neighbors]
    neighbor_ids = neighbor_table[selected_peaks["channel_index"]]
    n_before = int(ms_before * fs / 1000)
    n_after = int(ms_after * fs / 1000)
    n_samples = n_before + n_after
    waveforms = np.empty((len(selected), n_neighbors, n_samples), dtype=np.float32)
    times = selected_peaks["sample_index"]
    chunk_samples = int(10 * fs)
    for start_frame in range(0, fit_end, chunk_samples):
        end_frame = min(start_frame + chunk_samples, fit_end)
        in_chunk = np.flatnonzero((times >= start_frame) & (times < end_frame))
        if not len(in_chunk):
            continue
        load_start = max(0, start_frame - n_before)
        load_end = min(rec.get_num_samples(), end_frame + n_after)
        traces = rec.get_traces(start_frame=load_start, end_frame=load_end)
        for index in in_chunk:
            w0 = int(times[index]) - n_before - load_start
            waveforms[index] = traces[w0 : w0 + n_samples, neighbor_ids[index]].T
    rows = waveforms.reshape(-1, n_samples)
    print(f"[tpca] fitting {n_components} comps on {len(rows)} traces from {len(selected)} spikes", flush=True)
    model = PCA(n_components=n_components, svd_solver="auto")
    model.fit(rows)
    print(f"[tpca] explained_variance={model.explained_variance_ratio_.sum():.4f}", flush=True)
    return model


def run_peaks(args):
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.core import get_noise_levels
    from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import LocallyExclusivePeakDetector

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "_peaks_done"
    if marker.exists() and (out / "peaks.npy").exists() and (out / "peak_locations.npy").exists():
        print("peaks stage already complete; skipping", flush=True)
        return

    rec = read_recording(args.recording_path, args.electrical_series_path)
    log_geometry(rec, "nwb-ap")
    rec = sp.bandpass_filter(rec, freq_min=300.0, freq_max=6000.0, dtype="float32")
    rec = sp.common_reference(rec, reference="global", operator="median")
    log_geometry(rec, "preprocessed-lazy")
    fs = rec.get_sampling_frequency()

    job_kwargs = dict(n_jobs=args.n_jobs, chunk_duration="1s", progress_bar=True, mp_context="spawn")
    detect_kwargs = dict(
        peak_sign="neg",
        detect_threshold=8.0,
        exclude_sweep_ms=0.8,
        radius_um=80.0,
    )

    fit_end = min(int(300.0 * fs), rec.get_num_samples())
    sub = rec.frame_slice(0, fit_end)
    t0 = time.time()
    fit_peaks = detect_peaks(
        sub,
        method="locally_exclusive",
        method_kwargs=dict(detect_kwargs),
        job_kwargs=job_kwargs,
    )
    print(f"[tpca-fit] detected {fit_peaks.size} peaks in first 300s in {time.time() - t0:.0f}s", flush=True)

    tpca_model = fit_tpca_model(rec, fit_peaks, 7, 300.0, 10000, 5.0, 5.0)

    noise_levels = get_noise_levels(rec, return_in_uV=False, method="mad")
    detector = LocallyExclusivePeakDetector(
        rec,
        peak_sign="neg",
        detect_threshold=8.0,
        exclude_sweep_ms=0.8,
        radius_um=80.0,
        noise_levels=noise_levels,
        return_output=True,
    )
    dense_wf = ExtractDenseWaveforms(rec, ms_before=5.0, ms_after=5.0, parents=[detector], return_output=False)
    tpca_node = TemporalPCADenoising(rec, parents=[detector, dense_wf], pca_model=tpca_model, return_output=False)
    localize = LocalizeMonopolarTriangulation(
        rec,
        parents=[detector, tpca_node],
        return_output=True,
        radius_um=75.0,
        max_distance_um=150.0,
        feature="ptp",
    )

    print("[pipeline] detect+localize single pass...", flush=True)
    t0 = time.time()
    outputs = run_node_pipeline(
        rec,
        nodes=[detector, dense_wf, tpca_node, localize],
        job_kwargs=job_kwargs,
    )
    peaks, peak_locations = outputs[0], outputs[1]
    print(f"detected+localized {peaks.size} peaks in {time.time() - t0:.0f}s", flush=True)
    if peaks.size == 0:
        raise RuntimeError("no peaks detected")

    depths = peak_locations["y"]
    qs = np.percentile(depths, [1, 5, 25, 50, 75, 95, 99])
    print("depth percentiles (1/5/25/50/75/95/99 um):", np.round(qs, 1), flush=True)
    print(f"amplitude uV: median={np.median(np.abs(peaks['amplitude'])):.1f}", flush=True)

    atomic_save(out / "peaks.npy", peaks)
    atomic_save(out / "peak_locations.npy", peak_locations)
    meta = dict(
        recording_path=str(args.recording_path),
        electrical_series_path=args.electrical_series_path,
        fs=float(fs),
        n_peaks=int(peaks.size),
        detection="LocallyExclusivePeakDetector thr8 sweep0.8 radius80",
        localization="run_node_pipeline Detector->DenseWaveforms5ms->TPCA7->monopolar_triangulation r75 d150 ptp",
        preprocessing="bandpass 300-6000 float32 + global median CMR (lazy, no cache)",
    )
    tmp = out / "peaks_meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out / "peaks_meta.json")
    marker.write_text("ok\n")
    print("peaks stage complete", flush=True)


def run_motion(args):
    import torch
    from spikeinterface.sortingcomponents.motion import estimate_motion

    out = Path(args.output_dir)
    marker = out / "_motion_done"
    if marker.exists() and (out / "motion.npz").exists():
        print("motion stage already complete; skipping", flush=True)
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")

    rec = read_recording(args.recording_path, args.electrical_series_path)
    rec = sp.bandpass_filter(rec, freq_min=300.0, freq_max=6000.0, dtype="float32")
    rec = sp.common_reference(rec, reference="global", operator="median")
    log_geometry(rec, "preprocessed-lazy")

    peaks = np.load(out / "peaks.npy")
    peak_locations = np.load(out / "peak_locations.npy")
    print(f"loaded {peaks.size} peaks", flush=True)

    t0 = time.time()
    motion, extra = estimate_motion(
        rec,
        peaks,
        peak_locations,
        method="dredge_ap",
        direction="y",
        rigid=False,
        win_shape="gaussian",
        win_step_um=400.0,
        win_scale_um=400.0,
        win_margin_um=None,
        extra_outputs=True,
        progress_bar=True,
        bin_s=1.0,
        bin_um=1.0,
        max_disp_um=args.max_disp_um,
        device=args.device,
    )
    print(f"dredge_ap finished in {time.time() - t0:.0f}s", flush=True)

    displacement = motion.displacement[0]
    ad = np.abs(displacement)
    print(f"displacement shape={displacement.shape}", flush=True)
    print(
        f"displacement um: median={np.median(ad):.1f} p95={np.percentile(ad, 95):.1f} "
        f"max={ad.max():.1f} | window spread median={np.median(np.ptp(displacement, axis=1)):.1f}",
        flush=True,
    )

    tmp = out / "motion.tmp.npz"
    np.savez(
        tmp,
        displacement=displacement,
        time_bin_centers=motion.temporal_bins_s[0],
        window_centers=motion.spatial_bins_um,
        D=extra["D"],
        C=extra["C"],
        weights_orig=extra["weights_orig"],
        mincorr=extra.get("mincorr"),
        max_disp_um=extra.get("max_disp_um"),
    )
    os.replace(tmp, out / "motion.npz")
    meta = dict(
        method="dredge_ap",
        preset="dredge (SI)",
        max_disp_um=args.max_disp_um,
        bin_s=1.0,
        bin_um=1.0,
        win_step_um=400.0,
        win_scale_um=400.0,
        device=args.device,
        n_peaks=int(peaks.size),
        displacement_shape=list(displacement.shape),
    )
    tmp = out / "motion_meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out / "motion_meta.json")
    marker.write_text("ok\n")
    print("motion stage complete", flush=True)


def run_raster(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks

    out = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(out / "motion.npz")
    motion = Motion(z["displacement"], z["time_bin_centers"], z["window_centers"], direction="y")
    peaks = np.load(out / "peaks.npy")
    locs = np.load(out / "peak_locations.npy")

    fs = float(json.loads((out / "peaks_meta.json").read_text())["fs"])
    rec = read_recording(args.recording_path, args.electrical_series_path)

    corrected = correct_motion_on_peaks(peaks, locs, motion, rec)
    atomic_save(out / "peak_locations_corrected.npy", corrected)
    print("corrected locations saved", flush=True)

    t_min = peaks["sample_index"] / fs / 60.0

    lo = min(0.0, float(np.floor(corrected["y"].min() / 500.0) * 500.0))
    hi = max(3820.0, float(np.ceil(corrected["y"].max() / 500.0) * 500.0))
    print(f"depth axis: [{lo}, {hi}] um", flush=True)

    t_edges = np.linspace(0, t_min.max() + 1e-6, 5200)
    d_edges = np.arange(lo, hi + 2.0, 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 12), sharey=True)
    for ax, y, title in [
        (axes[0], locs["y"], "Unregistered"),
        (axes[1], corrected["y"], "DREDge (AP)\nregistered"),
    ]:
        H, _, _ = np.histogram2d(t_min, np.asarray(y), bins=(t_edges, d_edges))
        vmax = np.percentile(H[H > 0], 99.9)
        ax.pcolormesh(
            t_edges,
            d_edges,
            np.ma.masked_where(H == 0, H).T,
            cmap="gray_r",
            norm=PowerNorm(0.4, vmin=1, vmax=vmax),
            shading="auto",
        )
        ax.set_ylim(hi, lo)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (min)")
    axes[0].set_ylabel("depth (um)")
    fig.suptitle("macaque3 SI dredge preset", fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_dir / "spike_raster_4b_style.png", dpi=800)

    fig, ax = plt.subplots(figsize=(6, 5))
    wc = z["window_centers"]
    im = ax.imshow(
        z["displacement"].T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-args.max_disp_um,
        vmax=args.max_disp_um,
        extent=[z["time_bin_centers"][0] / 60.0, z["time_bin_centers"][-1] / 60.0, wc[-1], wc[0]],
    )
    fig.colorbar(im, ax=ax, label="displacement (um)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("window center depth (um)")
    ax.set_title("dredge_ap nonrigid displacement")
    fig.tight_layout()
    fig.savefig(plot_dir / "motion_heatmap.png", dpi=800)
    print("rasters saved", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["peaks", "motion", "raster"])
    p.add_argument("--recording-path")
    p.add_argument("--electrical-series-path", default="acquisition/ElectricalSeriesAP")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--plot-dir")
    p.add_argument("--n-jobs", type=int, default=12)
    p.add_argument("--max-disp-um", type=float, default=2500.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.stage == "peaks":
        run_peaks(args)
    elif args.stage == "motion":
        run_motion(args)
    else:
        run_raster(args)


if __name__ == "__main__":
    main()
