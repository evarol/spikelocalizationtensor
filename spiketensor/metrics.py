"""Internal D/C and canonical DREDge metrics for multipole readouts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                                      # noqa: E402
from spiketensor.dredge import DredgeConfig, solve_rigid               # noqa: E402
from spiketensor.volume import GridSpec, VolumeSmoother                # noqa: E402
from spiketensor.zncc import ShiftMaxZNCC, ZNCCConfig                  # noqa: E402
from spiketensor.gtscore import score                            # noqa: E402
from spiketensor.dc_movie import build_volume, render_dc             # noqa: E402
from spiketensor.drift import flatness                               # noqa: E402
from spiketensor.source_figures import load_run                       # noqa: E402


def enforce_pairwise_contract(D: np.ndarray, C: np.ndarray,
                              *, hard: bool = False
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Make independently evaluated pairwise scores exactly reciprocal.

    ShiftMaxZNCC evaluates both directions separately.  Floating-point noise is
    normally negligible, but an exact hard-lag tie can occasionally select
    adjacent 4-um bins in opposite directions.  Correlations are averaged.  Soft
    shifts are projected onto the antisymmetric subspace.  For hard shifts, the
    smaller-magnitude tied lag is retained so the result stays on the native lag
    grid and the tie resolution is conservative and deterministic.
    """
    D = np.asarray(D).copy()
    C = np.asarray(C).copy()
    C = (C + C.T) / 2
    if hard:
        upper_i, upper_j = np.triu_indices_from(D, k=1)
        forward = D[upper_i, upper_j]
        reverse = -D[upper_j, upper_i]
        chosen = np.where(np.abs(forward) <= np.abs(reverse), forward, reverse)
        D[upper_i, upper_j] = chosen
        D[upper_j, upper_i] = -chosen
    else:
        D = (D - D.T) / 2
    np.fill_diagonal(D, 0)
    return D, C


def event_readout(state: dict, rec, readout: str) -> dict[str, np.ndarray]:
    spike_row = np.arange(len(state["spike_index"]))
    total_amp = state["source_amp"].sum(1)
    if readout == "all":
        active = state["source_index"] >= 0
        parent, slot = np.nonzero(active)
        pos = state["source_pos"][parent, slot]
        amp = state["source_amp"][parent, slot]
        spike_row = parent
    elif readout == "dominant":
        pos, amp = state["pos_dominant"], total_amp
    elif readout == "barycenter":
        pos, amp = state["pos_barycenter"], total_amp
    else:
        raise ValueError("readout must be all, dominant, or barycenter")
    spike = state["spike_index"][spike_row]
    return {
        "pos": np.asarray(pos, np.float32),
        "amp": np.asarray(amp, np.float32),
        "spike": spike,
        "time": rec.spike_times[spike].astype(np.float64) / rec.fs,
        "sample": rec.spike_times[spike].astype(np.int64),
        "channel": rec.spike_channels[spike].astype(np.int64),
        "total_amp": total_amp,
    }


def internal_dc(events: dict, rec, tag: str, out: Path, device: str,
                stride: int = 8, tau: float = .2, smooth_um: float = 4.0,
                margin: int = 4, clamp_positions: bool = False) -> dict:
    grid = GridSpec(**torch.load(REPO / "zncc/runs/pretrain_np1/model.pt",
                                 map_location="cpu", weights_only=False)["grid"])
    smoother = VolumeSmoother(smooth_um, grid, device=device).to(device)
    zncc = ShiftMaxZNCC(ZNCCConfig(max_shift_y_vox=int(80.0 / grid.y_bin), tau=tau))
    sec = np.floor(events["time"]).astype(np.int64)
    order = np.argsort(sec, kind="stable")
    count = np.bincount(sec, minlength=int(sec.max()) + 1)
    offsets = np.concatenate([[0], np.cumsum(count)])
    bins = np.flatnonzero(count > 0)[::stride]
    t_sec = bins.astype(float) + .5
    pos = np.asarray(events["pos"], np.float32)
    if clamp_positions:
        # Match the established lattice safeguard: clip every coordinate just
        # inside the reconstruction volume.  This is a sensitivity analysis,
        # never a replacement for the raw-position primary result.
        pos = pos.copy()
        axes = ((grid.x_lo, grid.nx, grid.x_bin),
                (grid.y_lo, grid.ny, grid.y_bin),
                (grid.z_lo, grid.nz, grid.z_bin))
        for axis, (lo, n, width) in enumerate(axes):
            pos[:, axis] = np.clip(
                pos[:, axis], lo + 1e-3, lo + n * width - 1e-3)
    volumes = torch.empty(len(bins), *grid.shape, dtype=torch.float16)
    dropped = 0
    n_binned_events = 0
    for i, b in enumerate(bins):
        rows = order[offsets[b]:offsets[b + 1]]
        v, dr = build_volume(grid, smoother, pos[rows, 0],
                             pos[rows, 1], pos[rows, 2],
                             events["amp"][rows], device)
        volumes[i] = v.to(torch.float16).cpu()
        dropped += dr
        n_binned_events += len(rows)
    T = len(bins)
    Ds = np.zeros((T, T), np.float32); Cs = np.zeros((T, T), np.float32)
    Dh = np.zeros((T, T), np.float32); Ch = np.zeros((T, T), np.float32)
    chunk = 100
    for i in range(0, T, chunk):
        ni = min(chunk, T - i)
        for j in range(0, T, chunk):
            nj = min(chunk, T - j)
            sub = torch.cat([volumes[i:i + ni].to(device).float(),
                             volumes[j:j + nj].to(device).float()], 0)
            with torch.no_grad():
                result = zncc(sub, return_hard=True)
            Ds[i:i + ni, j:j + nj] = result[
                "d_y_soft"][:ni, ni:ni + nj].cpu().numpy()
            Cs[i:i + ni, j:j + nj] = result[
                "rho_soft"][:ni, ni:ni + nj].cpu().numpy()
            Dh[i:i + ni, j:j + nj] = result[
                "d_y"][:ni, ni:ni + nj].cpu().numpy()
            Ch[i:i + ni, j:j + nj] = result[
                "rho_hard"][:ni, ni:ni + nj].cpu().numpy()
    Ds, Cs = enforce_pairwise_contract(Ds, Cs)
    Dh, Ch = enforce_pairwise_contract(Dh, Ch, hard=True)
    Ds = np.nan_to_num(Ds).astype(float) * grid.y_bin
    Dh = np.nan_to_num(Dh).astype(float) * grid.y_bin
    Cs = np.nan_to_num(Cs).astype(float); Ch = np.nan_to_num(Ch).astype(float)
    inner = slice(margin, T - margin)
    cfg = DredgeConfig(c_thresh=0.0, lam_smooth=1.0, irls_iters=3)
    sol = solve_rigid(Ds[inner, inner], np.clip(Cs[inner, inner], 0, None), cfg)
    solh = solve_rigid(Dh[inner, inner], np.clip(Ch[inner, inner], 0, None), cfg)
    gt = D.gt_motion_trace(t_sec)
    sc = score(sol["p"], t_sec[inner]); sch = score(solh["p"], t_sec[inner])
    out.mkdir(parents=True, exist_ok=True)
    render_dc(out, tag, Ds, Cs, Dh, Ch, sol, solh, sc, sch, t_sec, gt,
              inner, tau, stride,
              note=("positions clipped just inside all grid bounds; sensitivity only"
                    if clamp_positions else
                    "raw positions; multipole events conserve parent model amplitude"))
    dc_path = out / "dc.npz"
    with np.load(dc_path) as saved:
        dc_arrays = {key: saved[key] for key in saved.files}
    dc_config = {
        "one_second_images": int(T), "temporal_stride_s": int(stride),
        "spatial_blur_um": float(smooth_um), "max_displacement_um": 80.0,
        "softmax_temperature": float(tau), "solver_margin": int(margin),
        "grid": {"x_lo": grid.x_lo, "y_lo": grid.y_lo, "z_lo": grid.z_lo,
                 "nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
                 "x_bin": grid.x_bin, "y_bin": grid.y_bin, "z_bin": grid.z_bin},
        "position_policy": "clipped_sensitivity" if clamp_positions else "raw_primary",
        "pairwise_reciprocity": (
            "C averaged; soft D antisymmetrized; hard D uses the smaller-absolute "
            "native-grid lag for exact directional ties"
        ),
    }
    np.savez_compressed(dc_path, **dc_arrays,
                        config_json=np.asarray(json.dumps(dc_config, sort_keys=True)))
    offdiag = ~np.eye(T, dtype=bool)
    gt_pair = gt[:, None] - gt[None]
    return {
        "T": T,
        "C_soft": float(Cs[offdiag].mean()),
        "C_hard": float(Ch[offdiag].mean()),
        "gt_r_soft": float(sc["r"]),
        "gt_gain_soft": float(sc["gain"]),
        "gt_r_hard": float(sch["r"]),
        "gt_gain_hard": float(sch["gain"]),
        "r_D_gt": float(np.corrcoef(Dh[offdiag], gt_pair[offdiag])[0, 1]),
        "outside_fraction": float(dropped / max(1, n_binned_events)),
        "n_events": int(len(events["pos"])),
        "n_binned_events": int(n_binned_events),
        "position_policy": "clipped_sensitivity" if clamp_positions else "raw_primary",
        "config": dc_config,
    }


def clamped_internal_sensitivity(events: dict, rec, tag: str, panel: Path,
                                 device: str, stride: int) -> dict:
    """Run the established all-axis clipping safeguard without replacing raw panels."""
    scratch = panel / "_dc_clamped_sensitivity"
    result = internal_dc(events, rec, f"{tag} / clipped sensitivity", scratch,
                         device, stride=stride, clamp_positions=True)
    panel.mkdir(parents=True, exist_ok=True)
    for name in ("dc.png", "dc.npz"):
        src = scratch / name
        if src.exists():
            os.replace(src, panel / name.replace("dc.", "dc_clamped."))
    try:
        scratch.rmdir()
    except OSError:
        pass
    return result


def run_dredge_events(rec, events: dict, rigid: bool, bin_s: float = 1.0,
                      bin_um: float = 1.0, smooth_um: float = 1.0):
    from spikeinterface.core import generate
    from spikeinterface.sortingcomponents.motion.dredge import dredge_ap
    import probeinterface

    duration = float(rec.spike_times.max() / rec.fs) + 1.0
    recording = generate.generate_recording(
        num_channels=len(rec.channel_locations), sampling_frequency=float(rec.fs),
        durations=[duration])
    probe = probeinterface.Probe(ndim=2)
    probe.set_contacts(positions=rec.channel_locations[:, :2], shapes="square",
                       shape_params={"width": 12})
    probe.set_device_channel_indices(np.arange(len(rec.channel_locations)))
    recording = recording.set_probe(probe)
    peaks = np.zeros(len(events["pos"]),
                    dtype=[("sample_index", "int64"), ("channel_index", "int64"),
                           ("amplitude", "float64"), ("segment_index", "int64")])
    peaks["sample_index"] = events["sample"]
    peaks["channel_index"] = events["channel"]
    # Apply the established scale safeguard to the parent total, then split it
    # over source events.  This preserves each parent's amplitude exactly.
    scale = 100.0 / max(float(np.median(events["total_amp"])), 1e-9)
    peaks["amplitude"] = -np.abs(events["amp"] * scale)
    locations = np.zeros(len(events["pos"]), dtype=[("x", "float64"), ("y", "float64")])
    locations["x"] = events["pos"][:, 0]; locations["y"] = events["pos"][:, 1]
    t0 = time.perf_counter()
    motion, _ = dredge_ap(recording, peaks, locations, rigid=rigid, bin_s=bin_s,
                          bin_um=bin_um, histogram_depth_smooth_um=smooth_um,
                          progress_bar=False, extra_outputs=True)
    displacement = motion.displacement
    if isinstance(displacement, (list, tuple)):
        displacement = displacement[0]
    temporal = motion.temporal_bins_s
    if isinstance(temporal, (list, tuple)):
        temporal = temporal[0]
    temporal = np.atleast_1d(np.asarray(temporal, float))
    displacement = np.asarray(displacement, float)
    if rigid:
        displacement = displacement.reshape(len(temporal))
    return (displacement, temporal, np.asarray(motion.spatial_bins_um, float),
            time.perf_counter() - t0, scale)


def _nonrigid_correction(time_s: np.ndarray, depth: np.ndarray,
                         temporal: np.ndarray, displacement: np.ndarray,
                         windows: np.ndarray) -> np.ndarray:
    if displacement.ndim == 1 or displacement.shape[1] == 1:
        return np.interp(time_s, temporal, displacement.reshape(len(temporal), -1)[:, 0])
    out = np.empty(len(depth), np.float32)
    for lo in range(0, len(depth), 200000):
        sl = slice(lo, min(lo + 200000, len(depth)))
        yy = np.clip(depth[sl], windows[0], windows[-1])
        wi = np.clip(np.searchsorted(windows, yy) - 1, 0, len(windows) - 2)
        f = (yy - windows[wi]) / np.maximum(windows[wi + 1] - windows[wi], 1e-9)
        a = np.empty(len(yy)); b = np.empty(len(yy))
        for w in np.unique(np.concatenate([wi, wi + 1])):
            use = wi == w
            if use.any():
                a[use] = np.interp(time_s[sl][use], temporal, displacement[:, w])
            use = wi + 1 == w
            if use.any():
                b[use] = np.interp(time_s[sl][use], temporal, displacement[:, w])
        out[sl] = (1 - f) * a + f * b
    return out


def canonical_dredge(events: dict, rec, tag: str, out: Path,
                     smooth_um: float = 1.0) -> dict:
    rigid, tr, _, sr, scale = run_dredge_events(rec, events, True, smooth_um=smooth_um)
    print(f"  canonical rigid complete: {tag} ({sr:.1f}s)", flush=True)
    nonrigid, tn, windows, sn, _ = run_dredge_events(
        rec, events, False, smooth_um=smooth_um)
    print(f"  canonical nonrigid complete: {tag} ({sn:.1f}s)", flush=True)
    score_r = score(rigid, tr)
    score_w = ([score(nonrigid[:, w], tn) for w in range(nonrigid.shape[1])]
               if nonrigid.ndim == 2 else [score(nonrigid, tn)])
    best = int(np.argmax([q["r"] for q in score_w]))
    y = events["pos"][:, 1]
    y_r = y - np.interp(events["time"], tr, rigid)
    y_n = y - _nonrigid_correction(events["time"], y, tn, nonrigid, windows)
    zoom = (400.0, 900.0)
    fl = {name: flatness(events["time"], yy, zoom)
          for name, yy in (("raw", y), ("rigid", y_r), ("nonrigid", y_n))}
    fig = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.1, 1.1, 1.1])
    ax = fig.add_subplot(gs[0])
    gt = D.gt_motion_trace(tr)
    ax.plot(tr, gt - gt.mean(), lw=1.2, label="imposed motion")
    ax.plot(tr, rigid - rigid.mean(), lw=1.0,
            label=f"rigid r={score_r['r']:+.3f}, gain={score_r['gain']:.3f}")
    if nonrigid.ndim == 2:
        ax.plot(tn, nonrigid[:, best] - nonrigid[:, best].mean(), lw=1.0,
                label=f"nonrigid best r={score_w[best]['r']:+.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_ylabel("depth shift (µm)")
    ax.set_title(f"canonical DREDge — {tag} — {len(y):,} source events")
    for rr, (yy, label) in enumerate(((y, "raw"), (y_r, "rigid-corrected"),
                                      (y_n, "nonrigid-corrected")), 1):
        a = fig.add_subplot(gs[rr])
        use = (yy >= zoom[0]) & (yy <= zoom[1])
        hist, _, _ = np.histogram2d(events["time"][use], yy[use], bins=(780, 420),
                                    range=[[0, rec.duration_s], list(zoom)],
                                    weights=events["amp"][use])
        vmax = np.quantile(hist[hist > 0], .994) if (hist > 0).any() else 1.0
        from matplotlib.colors import PowerNorm
        a.imshow(hist.T, origin="lower", aspect="auto", cmap="magma",
                 extent=[0, rec.duration_s, *zoom],
                 norm=PowerNorm(.45, vmin=0, vmax=vmax))
        a.text(.005, .96, f"{label} · CoM wander {fl[label.split('-')[0]]:.1f} µm",
               transform=a.transAxes, color="w", va="top", weight="bold")
        a.set_ylabel("depth (µm)")
    a.set_xlabel("recording time (s)")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "dredge_real.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    import spikeinterface
    import probeinterface
    dredge_config = {
        "method": "spikeinterface.sortingcomponents.motion.dredge.dredge_ap",
        "bin_s": 1.0, "bin_um": 1.0,
        "histogram_depth_smooth_um": float(smooth_um),
        "rigid_and_nonrigid": True,
        "amplitude_policy": "parent total median scaled to 100; conserved over sources",
    }
    dredge_software = {
        "spikeinterface": getattr(spikeinterface, "__version__", "unknown"),
        "probeinterface": getattr(probeinterface, "__version__", "unknown"),
        "numpy": np.__version__, "python": sys.version,
    }
    np.savez_compressed(out / "dredge_real.npz", p_rigid=rigid, t=tr,
                        p_nonrigid=nonrigid, t_nonrigid=tn, win_centers=windows,
                        amplitude_scale=scale, gt_r_rigid=score_r["r"],
                        gt_gain_rigid=score_r["gain"],
                        gt_r_nonrigid_best=score_w[best]["r"],
                        flat_raw=fl["raw"], flat_rigid=fl["rigid"],
                        flat_nonrigid=fl["nonrigid"],
                        config_json=np.asarray(json.dumps(dredge_config, sort_keys=True)),
                        software_json=np.asarray(json.dumps(dredge_software, sort_keys=True)))
    return {
        "n_events": int(len(y)), "amplitude_scale": scale,
        "gt_r_rigid": float(score_r["r"]), "gt_gain_rigid": float(score_r["gain"]),
        "gt_r_nonrigid_best": float(score_w[best]["r"]),
        "gt_gain_nonrigid_best": float(score_w[best]["gain"]),
        "flat_raw": float(fl["raw"]), "flat_rigid": float(fl["rigid"]),
        "flat_nonrigid": float(fl["nonrigid"]),
        "rigid_wall_s": sr, "nonrigid_wall_s": sn,
        "config": dredge_config, "software": dredge_software,
    }


def persist_records(path: Path, records: list[dict]) -> None:
    """Atomically merge completed metric rows so long runs are resumable by tag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = json.loads(path.read_text()) if path.exists() else []
        by_key = {(r["tag"], r["readout"]): r for r in existing}
        for record in records:
            key = (record["tag"], record["readout"])
            by_key[key] = {**by_key.get(key, {}), **record}
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(list(by_key.values()), indent=2))
        os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=REPO / "zncc/runs/multipole")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/multipole")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--readouts", default="all")
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--clamp-sensitivity-threshold", type=float, default=.05,
                    help="also score all-axis clipped positions when the raw outside-grid "
                         "fraction exceeds this value; negative disables the sensitivity")
    ap.add_argument("--skip-internal", action="store_true")
    ap.add_argument("--skip-canonical", action="store_true")
    a = ap.parse_args()
    device = ("mps" if a.device == "auto" and torch.backends.mps.is_available()
              else "cpu" if a.device == "auto" else a.device)
    rec = D.load("np1")
    readouts = [x for x in a.readouts.split(",") if x]
    records = []
    path = a.runs / "validation" / "localization_metrics.json"
    for tag in a.tags:
        state, summary, _ = load_run(a.runs, tag)
        if len(state["spike_index"]) != rec.n_spikes:
            raise SystemExit(f"{tag} is not a full-data result")
        for readout in readouts:
            print(f"metrics: {tag} / {readout}", flush=True)
            events = event_readout(state, rec, readout)
            panel = a.figs / tag / ("" if readout == "all" else readout)
            record = {"tag": tag, "model": summary["model"],
                      "kind": summary["dictionary"]["kind"], "readout": readout}
            if not a.skip_internal:
                record["internal"] = internal_dc(
                    events, rec, f"{tag} / {readout}", panel, device, stride=a.stride)
                if (a.clamp_sensitivity_threshold >= 0 and
                        record["internal"]["outside_fraction"] >
                        a.clamp_sensitivity_threshold):
                    record["internal_clamped_sensitivity"] = clamped_internal_sensitivity(
                        events, rec, f"{tag} / {readout}", panel, device, a.stride)
            if not a.skip_canonical:
                record["canonical"] = canonical_dredge(
                    events, rec, f"{tag} / {readout}", panel)
            if readout != "all":
                for name in ("dc.png", "dc.npz", "dc_clamped.png", "dc_clamped.npz",
                             "dredge_real.png", "dredge_real.npz"):
                    src = panel / name
                    if src.exists():
                        dst = a.figs / tag / name.replace(".", f"_{readout}.", 1)
                        os.replace(src, dst)
                try:
                    panel.rmdir()
                except OSError:
                    pass
            records.append(record)
            persist_records(path, [record])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
