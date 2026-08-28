"""Run CANONICAL DREDge (spikeinterface's dredge_ap) on a fit's localizations.

The project's internal pipeline is a DREDge-style estimator with deliberate departures
(3-D amplitude-weighted volumes, 8-s-subsampled bins, 4 um integer shifts, a Laplacian-
smoothed rigid solve). Those keep 50+ fits mutually comparable, but the absolute motion
estimate is not what DREDge itself would produce. This script feeds a fit's per-spike
(x, y) localizations, amplitudes and times into spikeinterface's dredge_ap exactly as it
is meant to be run: every spike, every 1-s bin, 1 um depth bins, a depth x log-amplitude
histogram, mincorr thresholding, and the Thomas centralization solve -- rigid and
nonrigid.

The recording object exists only to carry the sampling rate and probe geometry;
dredge_ap never reads traces for AP-based registration, so a lazy noise recording of the
right duration and probe is sufficient.

Outputs under figs/<tag>/:
    dredge_real.npz    p_rigid (T,), p_nonrigid (T, n_win), t, win_centers, gt scores
    dredge_real.png    both traces vs the imposed motion + corrected depth rasters
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                      # noqa: E402
from spiketensor.gtscore import score            # noqa: E402
from spiketensor.drift import flatness               # noqa: E402


def load_positions(runs: Path, tag: str, rec):
    """(pos_xyz, amp) for a fit tag or the 'monopole' reference."""
    ptp = None
    if tag in ("monopole", "BASELINE_monopole", "MONOPOLE_matched",
               "CONTROL_anchor_only_ptp", "CONTROL_anchor_only_flat"):
        ptp = np.ptp(np.asarray(rec.waveforms), axis=2).max(1).astype(np.float32)
    if tag in ("monopole", "BASELINE_monopole", "MONOPOLE_matched"):
        return rec.mp_xyz.astype(np.float32), ptp
    if tag.startswith("CONTROL_anchor_only"):
        pos = np.empty((rec.n_spikes, 3), np.float32)
        pos[:, :2] = rec.anchors[rec.spike_channels][:, :2]
        pos[:, 2] = 20.0
        return pos, (ptp if tag.endswith("_ptp")
                     else np.ones(rec.n_spikes, np.float32))
    z = np.load(runs / f"pi_{tag}.npz")
    amp = np.linalg.norm(z["v"], axis=1).astype(np.float32)
    # dredge_ap log1p-bins amplitudes and thresholds window weights at 0.2; model
    # amplitudes (||v|| median ~0.2) sit in the linear regime and the histogram
    # effectively vanishes -- rigid r collapsed from +0.93 to +0.25 before this rescale.
    # Scale-invariant fix: put the median at 100, comparable to measured ptp.
    med = float(np.median(amp))
    if not np.isfinite(med) or med <= 0:
        # a degenerate fit would otherwise be rescaled by 1/eps into an all-zero
        # histogram and produce a confident-looking but meaningless motion trace
        raise ValueError(f"{tag}: model amplitude median is {med}; cannot rescale for "
                         f"dredge_ap. This fit has no usable amplitudes.")
    amp *= 100.0 / med
    if "pos" in z.files:
        return z["pos"].astype(np.float32), amp
    k = z["k"].astype(np.int64)
    if "mu_site" in z.files:
        mu = z["mu_site"].astype(np.float32)[k // int(z["S"])]
    else:                       # legacy layout: mu is the expanded (KS, 3) candidate table
        mu = z["mu"].astype(np.float32)[k]
    pos = np.empty((rec.n_spikes, 3), np.float32)
    pos[:, :2] = rec.anchors[rec.spike_channels][:, :2] + mu[:, :2]
    pos[:, 2] = mu[:, 2]
    return pos, amp


def run_dredge(rec, pos, amp, rigid, bin_s=1.0, bin_um=1.0, smooth_um=1.0):
    from spikeinterface.core import generate  # lazy noise recording, traces never read
    from spikeinterface.sortingcomponents.motion.dredge import dredge_ap
    import probeinterface

    dur = float(rec.spike_times.max() / rec.fs) + 1.0
    rec_si = generate.generate_recording(num_channels=len(rec.channel_locations),
                                         sampling_frequency=float(rec.fs),
                                         durations=[dur])
    probe = probeinterface.Probe(ndim=2)
    probe.set_contacts(positions=rec.channel_locations[:, :2],
                       shapes="square", shape_params={"width": 12})
    probe.set_device_channel_indices(np.arange(len(rec.channel_locations)))
    rec_si = rec_si.set_probe(probe)

    peaks = np.zeros(rec.n_spikes,
                     dtype=[("sample_index", "int64"), ("channel_index", "int64"),
                            ("amplitude", "float64"), ("segment_index", "int64")])
    peaks["sample_index"] = rec.spike_times
    peaks["channel_index"] = rec.spike_channels
    # dredge_ap bins log1p(|amplitude|) as the second histogram axis
    peaks["amplitude"] = -np.abs(amp)
    locs = np.zeros(rec.n_spikes, dtype=[("x", "float64"), ("y", "float64")])
    locs["x"], locs["y"] = pos[:, 0], pos[:, 1]

    t0 = time.perf_counter()
    motion, extra = dredge_ap(rec_si, peaks, locs, rigid=rigid,
                              bin_s=bin_s, bin_um=bin_um,
                              histogram_depth_smooth_um=smooth_um,
                              progress_bar=False, extra_outputs=True)
    secs = time.perf_counter() - t0
    # spikeinterface's Motion stores per-segment lists in some versions and bare
    # arrays in others; normalise both
    disp = motion.displacement
    if isinstance(disp, (list, tuple)):
        disp = disp[0]
    tb = motion.temporal_bins_s
    if isinstance(tb, (list, tuple)):
        tb = tb[0]
    tb = np.atleast_1d(np.asarray(tb, float))
    disp = np.asarray(disp, float)
    if rigid:
        disp = disp.reshape(len(tb))          # (T, 1) -> (T,)
    wc = np.asarray(motion.spatial_bins_um, float)
    return np.asarray(disp, float), np.asarray(tb, float), wc, secs


def nonrigid_at(y, disp_nr, win_c):
    """Interpolate the nonrigid field across depth windows at spike depths."""
    if disp_nr.ndim == 1 or disp_nr.shape[1] == 1:
        return np.broadcast_to(disp_nr.reshape(-1), (len(y),))
    return None  # handled per-spike by the caller with np.interp over windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="monopole")
    ap.add_argument("--runs", type=Path, default=REPO / "zncc/runs/lattice")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/lattice")
    ap.add_argument("--bin_s", type=float, default=1.0)
    ap.add_argument("--bin_um", type=float, default=1.0)
    ap.add_argument("--amp", choices=["model", "ptp"], default="model",
                    help="amplitude fed to dredge_ap: the model's ||v|| or measured ptp")
    ap.add_argument("--smooth_um", type=float, default=1.0,
                    help="depth-histogram gaussian smoothing. Hard-assignment models put "
                         "every spike on a finite set of sites, so at 1 µm bins the "
                         "histogram is a comb of fixed delta spikes and rigid xcorr locks "
                         "to zero shift; smoothing over the site spacing restores a "
                         "continuous image. Continuous localizers (monopole) need none.")
    a = ap.parse_args()

    rec = D.load("np1")
    pos, amp = load_positions(a.runs, a.tag, rec)
    if a.amp == "ptp":
        amp = np.ptp(np.asarray(rec.waveforms), axis=2).max(1).astype(np.float32)
    out = a.figs / ("BASELINE_monopole" if a.tag == "monopole" else a.tag)
    if (out / "dredge_real.npz").exists():
        print(f"{a.tag}: dredge_real.npz exists, skipping"); return
    out.mkdir(parents=True, exist_ok=True)
    ts = rec.spike_times / rec.fs

    print(f"{a.tag}: canonical dredge_ap on {rec.n_spikes:,} spikes, "
          f"bin_s={a.bin_s}, bin_um={a.bin_um}", flush=True)
    p_r, t_r, _, s_r = run_dredge(rec, pos, amp, rigid=True,
                                  bin_s=a.bin_s, bin_um=a.bin_um,
                                  smooth_um=a.smooth_um)
    print(f"  rigid: {len(t_r)} bins in {s_r:.0f}s", flush=True)
    p_n, t_n, wc, s_n = run_dredge(rec, pos, amp, rigid=False,
                                   bin_s=a.bin_s, bin_um=a.bin_um,
                                   smooth_um=a.smooth_um)
    print(f"  nonrigid: {p_n.shape} over windows at {np.round(wc, 0)} in {s_n:.0f}s",
          flush=True)

    sc_r = score(p_r, t_r)
    # score the nonrigid field at the window nearest the imposed manipulator
    sc_w = [score(p_n[:, w], t_n) for w in range(p_n.shape[1])] if p_n.ndim == 2 \
        else [score(p_n, t_n)]
    best_w = int(np.argmax([s["r"] for s in sc_w]))
    print(f"  GT r: rigid {sc_r['r']:+.3f} (gain {sc_r['gain']:.3f}) · nonrigid best "
          f"window {best_w} {sc_w[best_w]['r']:+.3f} (gain {sc_w[best_w]['gain']:.3f})",
          flush=True)

    # ---- corrected depths, both ways
    y = pos[:, 1]
    y_r = y - np.interp(ts, t_r, p_r)
    if p_n.ndim == 2 and p_n.shape[1] > 1:
        # bilinear in (time, depth), fully chunked and float32 -- the all-spike float64
        # stack was ~180 MB x copies and died under memory pressure alongside the movie
        # render; per-chunk interp keeps the peak footprint a few MB
        corr = np.empty(len(y), np.float32)
        for c0 in range(0, len(y), 200_000):
            sl = slice(c0, min(c0 + 200_000, len(y)))
            yy = np.clip(y[sl], wc[0], wc[-1]).astype(np.float32)
            wi = np.clip(np.searchsorted(wc, yy) - 1, 0, len(wc) - 2)
            f = ((yy - wc[wi]) / np.maximum(wc[wi + 1] - wc[wi], 1e-9)).astype(np.float32)
            lo = np.empty(len(yy), np.float32); hi = np.empty(len(yy), np.float32)
            for w in np.unique(np.concatenate([wi, wi + 1])):
                m = wi == w
                if m.any():
                    lo[m] = np.interp(ts[sl][m], t_n, p_n[:, w]).astype(np.float32)
                m = (wi + 1) == w
                if m.any():
                    hi[m] = np.interp(ts[sl][m], t_n, p_n[:, w]).astype(np.float32)
            corr[sl] = (1 - f) * lo + f * hi
        y_n = y - corr
    else:
        y_n = y - np.interp(ts, t_n, p_n.reshape(len(t_n), -1)[:, 0])

    ylim = (400., 900.)
    fl = {k: flatness(ts, v, ylim) for k, v in
          (("raw", y), ("rigid", y_r), ("nonrigid", y_n))}
    print(f"  flatness (CoM wander, µm): raw {fl['raw']:.2f} · "
          f"rigid {fl['rigid']:.2f} · nonrigid {fl['nonrigid']:.2f}", flush=True)

    # ---- figure: traces + three rasters
    gt = D.gt_motion_trace(t_r)
    fig = plt.figure(figsize=(16.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.1, 1.1, 1.1])
    A = fig.add_subplot(gs[0])
    A.plot(t_r, gt - gt.mean(), color="#2f6df6", lw=1.2, label="imposed motion")
    A.plot(t_r, p_r - p_r.mean(), color="#e03131", lw=1.0,
           label=f"dredge_ap rigid  (r {sc_r['r']:+.3f}, gain {sc_r['gain']:.3f})")
    if p_n.ndim == 2:
        A.plot(t_n, p_n[:, best_w] - p_n[:, best_w].mean(), color="#2f9e44", lw=1.0,
               label=f"dredge_ap nonrigid, window y≈{wc[best_w]:.0f} µm  "
                     f"(r {sc_w[best_w]['r']:+.3f}, gain {sc_w[best_w]['gain']:.3f})")
    A.set_ylabel("depth (µm)"); A.legend(fontsize=8); A.grid(alpha=.3)
    A.set_title(f"canonical DREDge on {a.tag} localizations — all "
                f"{rec.n_spikes:,} spikes, every {a.bin_s:g} s bin, "
                f"{a.bin_um:g} µm depth bins", fontsize=11)
    for ax_i, (yy, lab) in zip(range(1, 4),
                               ((y, f"raw   (CoM wander {fl['raw']:.1f} µm)"),
                                (y_r, f"rigid-corrected   ({fl['rigid']:.1f} µm)"),
                                (y_n, f"nonrigid-corrected   ({fl['nonrigid']:.1f} µm)"))):
        B = fig.add_subplot(gs[ax_i])
        m = (yy >= ylim[0]) & (yy <= ylim[1])
        H, xe, ye = np.histogram2d(ts[m], yy[m], bins=(780, 420),
                                   range=[[0, ts.max()], list(ylim)], weights=amp[m])
        from matplotlib.colors import PowerNorm
        nz = H[H > 0]
        # A collapsed or out-of-window localizer can leave the canonical display
        # interval genuinely empty.  Preserve that empty panel rather than failing
        # after DREDge has already completed; this does not alter estimation/scores.
        vmax = float(np.percentile(nz, 99.4)) if len(nz) else 1.0
        B.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
                 extent=[0, ts.max(), *ylim],
                 norm=PowerNorm(0.45, vmin=0, vmax=vmax))
        B.set_ylabel("depth (µm)", fontsize=9)
        B.text(.005, .96, lab, transform=B.transAxes, color="w", fontsize=10,
               va="top", weight="bold")
    B.set_xlabel("recording time (s)")
    fig.savefig(out / "dredge_real.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    np.savez_compressed(out / "dredge_real.npz", p_rigid=p_r, t=t_r,
                        p_nonrigid=p_n, t_nonrigid=t_n, win_centers=wc,
                        gt_r_rigid=sc_r["r"], gt_gain_rigid=sc_r["gain"],
                        gt_r_nonrigid_best=sc_w[best_w]["r"],
                        flat_raw=fl["raw"], flat_rigid=fl["rigid"],
                        flat_nonrigid=fl["nonrigid"])
    print(f"wrote {out}/dredge_real.png, dredge_real.npz")


if __name__ == "__main__":
    main()
