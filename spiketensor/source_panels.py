"""DREDge-corrected standard panels for the MULTI-SOURCE model rows.

The multipole / analytic-sparse fits place up to R sources per spike, so the standard
generators -- which assume one position per spike keyed by `rec` -- cannot render their
corrected variants. This script expands each fit to per-SOURCE events (one event per
active source: its position, its amplitude `source_amp`, its dominant temporal component
`argmax|b_n|`, its parent spike's time) and renders the same corrected panel family the
single-source rows carry, reusing the identical plotting primitives from
viz_centroid_basis and the identical volume/projection machinery from dc_movie, so the
output is visually indistinguishable from the standard set.

Corrections come from each row's OWN solves (dc.npz for soft/hard, dredge_real.npz for
real-rigid/real-nonrigid) via drift.correction, applied to every event's depth using its
parent spike's time. The aggregate panels here are two-panel (events at 4 µm blur +
monopole reference); the model-kernel soft stamp of the single-source aggregate needs a
per-event kernel and is already shown, uncorrected, in each row's own aggregate panel.

Outputs, per row root and per mode X in {_drr, _drn}:
    centroid_basis_{zoom,full}X.png       depth_time_{density,basis}_{zoom,full}X.png
    aggregate_1s{_zoom,}X.png             It_{zoom,full}X.mp4
    centroid_basis_movie_{zoom,full}X.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
import torch                                 # noqa: E402
from matplotlib.colors import PowerNorm          # noqa: E402
from scipy.ndimage import gaussian_filter        # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                                    # noqa: E402
from spiketensor.volume import GridSpec, VolumeSmoother              # noqa: E402
from spiketensor.dc_movie import build_volume                      # noqa: E402
from spiketensor.drift import SUFFIX, correction                   # noqa: E402
from spiketensor.spike_error import error_metric, spike_energy    # noqa: E402
from spiketensor.viz_centroid_basis import (FULL_Y, ZOOM_Y,        # noqa: E402
                                            pca_rgb_local, save_basis_raster,
                                            save_aggregate, save_basis_error,
                                            save_error_raster,
                                            save_centroid, save_density_raster,
                                            save_movies)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

ROWS = {  # browser tag -> (state npz, figure root)
    "MULTIPOLE_learned_m1_x2":
        ("zncc/runs/multipole/multipole_learned_m1_x2.npz",
         "zncc/figures/multipole/learned_m1_x2"),
    "MULTIPOLE_learned_m2_r2":
        ("zncc/runs/multipole/multipole_learned_m2_r2.npz",
         "zncc/figures/multipole/learned_m2_r2"),
    "analytic_l0_mono_N512_M8_R4":
        ("zncc/runs/analytic_sparse/multipole_analytic_l0_mono_N512_M8_R4.npz",
         "zncc/figures/analytic_sparse/analytic_l0_mono_N512_M8_R4"),
    "analytic_l1_mono_N512_M8_group9975_R4":
        ("zncc/runs/analytic_sparse/multipole_analytic_l1_mono_N512_M8_group9975_R4.npz",
         "zncc/figures/analytic_sparse/analytic_l1_mono_N512_M8_group9975_R4"),
}


def load_events(state: Path, rec):
    """Expand a multipole npz to per-source events, in parent-time order."""
    z = np.load(state)
    idx = z["spike_index"].astype(np.int64)          # parent spike per row
    si = z["source_index"]                           # (n, R), -1 or amp==0 = inactive
    amp = z["source_amp"].astype(np.float32)
    pos = z["source_pos"].astype(np.float32)         # (n, R, 3) absolute
    if "shape_feature" in z.files:
        co = z["shape_feature"].astype(np.float32)  # C5 explicit nonseparable feature
    else:
        co = z["source_coeff"].astype(np.float32)   # ordinary separable source row
    act = (si >= 0) & (amp > 0)
    n, R = si.shape
    parent = np.repeat(idx, R)[act.ravel()]
    ev_pos = pos.reshape(-1, 3)[act.ravel()]
    ev_amp = amp.ravel()[act.ravel()]
    ev_co = co.reshape(-1, co.shape[2])[act.ravel()]
    o = np.argsort(parent, kind="stable")
    # colour = 3-PC PCA of each SOURCE's own coefficient vector, so sub-sources of one
    # spike can carry different colours when their waveforms differ; the PCA is fitted
    # inside overlapping depth blocks (see pca_rgb_local) so LOCAL shape variance gets
    # the colour cube rather than the recording-wide axes
    colors, evr = pca_rgb_local(ev_co[o], ev_pos[o][:, 1])
    return parent[o], ev_pos[o], ev_amp[o], colors, evr, co.shape[2]


def _it_index(sec_ev):
    nb = int(sec_ev.max()) + 1
    order = np.argsort(sec_ev, kind="stable")
    cnt = np.bincount(sec_ev, minlength=nb)
    off = np.concatenate([[0], np.cumsum(cnt)])
    return order, cnt, off, np.flatnonzero(cnt > 0)


def _it_projections(g, sm, pos, amp, order, cnt, off, bins, batch=32,
                    device="cpu"):
    """Yield standard blurred projections while smoothing several seconds at once."""
    nvox = g.nx * g.ny * g.nz
    with torch.no_grad():
        for lo in range(0, len(bins), batch):
            labels = bins[lo:lo + batch]
            indices = np.concatenate([order[off[b]:off[b + 1]] for b in labels])
            local = np.repeat(np.arange(len(labels)), cnt[labels])
            px = torch.as_tensor(pos[indices, 0], device=device)
            py = torch.as_tensor(pos[indices, 1], device=device)
            pz = torch.as_tensor(pos[indices, 2], device=device)
            ix = torch.floor((px - g.x_lo) / g.x_bin).long()
            iy = torch.floor((py - g.y_lo) / g.y_bin).long()
            iz = torch.floor((pz - g.z_lo) / g.z_bin).long()
            local_t = torch.as_tensor(local, device=device, dtype=torch.long)
            ok = ((ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny)
                  & (iz >= 0) & (iz < g.nz))
            flat = local_t[ok] * nvox + (ix[ok] * g.ny + iy[ok]) * g.nz + iz[ok]
            volume = torch.zeros(len(labels) * nvox, device=device)
            weight = torch.as_tensor(amp[indices], device=device,
                                     dtype=torch.float32)
            volume.index_add_(0, flat, weight[ok])
            volume = sm(volume.reshape(len(labels), g.nx, g.ny, g.nz))
            xy = volume.sum(3).transpose(1, 2).cpu().numpy()
            zy = volume.sum(1).cpu().numpy()
            for row, b in enumerate(labels):
                yield int(b), int(cnt[b]), xy[row], zy[row]


def it_movies(requests, g, sm, pos, amp, sec_ev, rec, tag, note, fps=24):
    """Render zoom/full standard I_t movies from one shared projection stream."""
    from spiketensor.viz_common import extents, project
    ex_xy, ex_zy = extents(g)
    order, cnt, off, allb = _it_index(sec_ev)
    hi = [0.0, 0.0]
    for b in allb[::max(1, len(allb) // 40)]:
        s = order[off[b]:off[b + 1]]
        v, _ = build_volume(g, sm, pos[s, 0], pos[s, 1], pos[s, 2], amp[s], DEV)
        for k, img in enumerate(project(v.cpu().numpy())):
            sb = img[img > 0]
            if sb.size:
                hi[k] = max(hi[k], float(np.percentile(sb, 99.7)))
    canvases = []
    wx, wz = g.nx * g.x_bin, g.nz * g.z_bin
    projection_sm = VolumeSmoother(sm.sigma_um, g, device="cpu").to("cpu")
    processes = []
    try:
        for path, ylim in requests:
            span = ylim[1] - ylim[0]
            h = 7.0 if span < 1000 else 13.0
            fw = min(15.0, h * (wx + wz) / span + 1.4)
            fig, ax = plt.subplots(
                1, 2, figsize=(fw, h + .8), facecolor="#0f1115",
                constrained_layout=True, gridspec_kw={"width_ratios": [wx, wz]})
            fig.set_dpi(90)
            sup = fig.suptitle("", color="w", fontsize=9)
            channel_mask = ((ylim[0] <= rec.channel_locations[:, 1])
                            & (rec.channel_locations[:, 1] <= ylim[1]))
            images = []
            for k, (shape, ex, xl) in enumerate(
                    (((g.ny, g.nx), ex_xy, "x (µm)"),
                     ((g.ny, g.nz), ex_zy, "z (µm)"))):
                images.append(ax[k].imshow(
                    np.zeros(shape, dtype=np.float32), origin="lower",
                    aspect="equal", extent=ex, cmap="magma",
                    norm=PowerNorm(.5, vmin=0, vmax=max(hi[k], 1e-9))))
                ax[k].set_ylim(*ylim)
                ax[k].set_xlabel(xl, color="w", fontsize=8)
                ax[k].tick_params(colors="w", labelsize=7)
            channels = ax[0].scatter(
                rec.channel_locations[channel_mask, 0],
                rec.channel_locations[channel_mask, 1], s=5,
                marker="s", c="none", edgecolors="w",
                linewidths=.3, alpha=.55)
            ax[0].set_ylabel("depth y (µm)", color="w", fontsize=8)
            fig.canvas.draw()
            fig.set_layout_engine(None)
            background = fig.canvas.copy_from_bbox(fig.bbox)
            width, height = fig.canvas.get_width_height()
            command = [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                "-vcodec", "rawvideo", "-s", f"{width}x{height}",
                "-pix_fmt", "rgba", "-framerate", str(fps), "-i", "-",
                "-an", "-vcodec", "libx264", "-threads", "2",
                "-b", "4000k", "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                str(path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            processes.append((path, process))
            canvases.append((path, ylim, fig, ax, sup, process, images,
                             channels, background))
        for frame, (b, count, xy, zy) in enumerate(
                _it_projections(g, projection_sm, pos, amp, order, cnt, off,
                                allb, device="cpu")):
            for path, ylim, fig, ax, sup, process, images, channels, background in canvases:
                fig.canvas.restore_region(background)
                images[0].set_data(xy)
                images[1].set_data(zy)
                ax[0].draw_artist(images[0])
                ax[1].draw_artist(images[1])
                ax[0].draw_artist(channels)
                sup.set_text(f"{tag} — I_t at t = {b}–{b + 1} s · "
                             f"{count:,} source events · {note}")
                fig.draw_artist(sup)
                process.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
            if frame and frame % 400 == 0:
                names = ", ".join(path.name for path, *_ in canvases)
                print(f"    {names}: {frame}/{len(allb)}", flush=True)
    finally:
        for _, process in processes:
            if process.stdin:
                process.stdin.close()
        failures = [(path, process.wait()) for path, process in processes]
        for _, _, fig, *_ in canvases:
            plt.close(fig)
    failed = [(path, code) for path, code in failures if code]
    if failed:
        raise RuntimeError(f"ffmpeg failed for {failed}")


def it_movie(path, g, sm, pos, amp, sec_ev, rec, ylim, tag, note, fps=24):
    """Compatibility wrapper for one standard I_t movie."""
    it_movies([(path, ylim)], g, sm, pos, amp, sec_ev, rec, tag, note, fps)


def aggregate(path, pos_xy, amp, sec_ev, rec, ylim, tag, note, t0=1200, res=2.0):
    """Delegates to the ONE shared aggregate panel so every model matches.

    The monopole reference panel that used to sit beside this one is gone: it is its own
    browser row, and repeating it inside every model's figure spent half the width on an
    image identical across all of them.
    """
    save_aggregate(path, pos_xy[:, 0], pos_xy[:, 1], amp, sec_ev, rec, ylim, tag,
                   note, t0=int(t0), res=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="",
                    help="browser tag for one generic source-event state")
    ap.add_argument("--state", type=Path,
                    help="generic source-event NPZ (used with --tag and --root)")
    ap.add_argument("--root", type=Path,
                    help="figure directory for one generic source-event state")
    ap.add_argument("--modes", default="none,real-rigid,real-nonrigid")
    ap.add_argument("--error-metric", dest="error_metric",
                    choices=["relative", "absolute"], default="relative")
    ap.add_argument("--force", action="store_true",
                    help="re-render files that already exist (e.g. after a recolour)")
    ap.add_argument("--skip_movies", action="store_true")
    ap.add_argument("--movies_only", action="store_true")
    ap.add_argument("--movie_family", choices=("all", "it", "centroid"),
                    default="all")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--it_fps", type=int,
                    help="override I_t movie frame rate")
    ap.add_argument("--centroid_fps", type=int,
                    help="override centroid movie frame rate")
    a = ap.parse_args()

    rows = ROWS
    if any(value is not None and value != "" for value in (a.state, a.root, a.tag)):
        if not (a.state and a.root and a.tag):
            raise SystemExit("--state, --root, and --tag are required together")
        rows = {a.tag: (str(a.state), str(a.root))}

    rec = D.load("np1")
    g = GridSpec(**torch.load(REPO / "zncc/runs/pretrain_np1/model.pt",
                              map_location="cpu", weights_only=False)["grid"])
    sm = VolumeSmoother(4.0, g, device=DEV).to(DEV)
    rng = np.random.default_rng(7)

    for tag, (state, root) in rows.items():
        if a.only and a.only not in tag:
            continue
        root = Path(root)
        if not root.is_absolute():
            root = REPO / root
        state = Path(state)
        if not state.is_absolute():
            state = REPO / state
        parent, pos, amp, colors, evr, M = load_events(state, rec)
        # per-EVENT reconstruction error: the parent spike's error, carried by each of
        # its sources, so the error panels match the other per-event rasters
        with np.load(state, mmap_mode="r") as _z:
            _sse = np.asarray(_z["sse"], np.float64)
        err = error_metric(_sse, spike_energy(rec)[np.arange(len(_sse))],
                           a.error_metric)[parent]
        t_smp = rec.spike_times[parent]
        ts = t_smp / rec.fs
        sec = np.floor(ts).astype(np.int64)
        palette = colors
        dom = np.arange(len(pos), dtype=np.int64)
        jx = rng.uniform(-1.5, 1.5, len(pos)).astype(np.float32)
        jy = rng.uniform(-1.5, 1.5, len(pos)).astype(np.float32)
        jt = rng.uniform(-.18, .18, len(pos)).astype(np.float32)
        print(f"{tag}: {len(pos):,} source events from {len(np.unique(parent)):,} "
              f"spikes, M={M} · PCA colours evr={np.round(evr, 3)}", flush=True)
        for mode in [m for m in a.modes.split(",") if m]:
            sfx = SUFFIX[mode]
            dy = correction(root.parent, root.name, mode, t_smp, rec.fs,
                            y=pos[:, 1])
            yc = pos[:, 1] - dy
            x = pos[:, 0] + jx
            note = "uncorrected" if mode == "none" else f"{mode} DREDge-corrected"
            t0 = time.perf_counter()
            if not a.movies_only:
                for ylim, z in ((ZOOM_Y, "zoom"), (FULL_Y, "full")):
                    p = root / f"centroid_basis_{z}{sfx}.png"
                    (p.exists() and not a.force) or save_centroid(p, x, yc + jy, dom, palette, rec, ylim,
                                                tag, False)
                    p = root / f"depth_time_density_{z}{sfx}.png"
                    (p.exists() and not a.force) or save_density_raster(p, ts + jt, yc, amp, ylim, tag)
                    p = root / f"depth_time_basis_{z}{sfx}.png"
                    (p.exists() and not a.force) or save_basis_raster(p, ts + jt, yc + jy, dom, palette,
                                                    ylim, tag, False)
                for ylim, z in ((ZOOM_Y, "zoom"), (FULL_Y, "full")):
                    p = root / f"depth_time_mse_{z}{sfx}.png"
                    (p.exists() and not a.force) or save_error_raster(
                        p, ts + jt, yc, err, ylim, tag, note,
                        metric=a.error_metric, point_weight=amp)
                if not sfx:
                    p = root / "basis_error.png"
                    (p.exists() and not a.force) or save_basis_error(
                        p, pos[:, 0], pos[:, 1], err, tag, metric=a.error_metric)
                for ylim, nm in ((ZOOM_Y, f"aggregate_1s_zoom{sfx}.png"),
                                 (FULL_Y, f"aggregate_1s{sfx}.png")):
                    p = root / nm
                    if not p.exists() or a.force:
                        pxy = np.stack([pos[:, 0], yc], 1)
                        aggregate(p, pxy, amp, sec, rec, ylim, tag, note)
            if not a.skip_movies:
                pc = pos.copy(); pc[:, 1] = yc
                if a.movie_family in ("all", "it"):
                    requests = []
                    for ylim, z in ((ZOOM_Y, "zoom"), (FULL_Y, "full")):
                        p = root / f"It_{z}{sfx}.mp4"
                        if not p.exists() or a.force:
                            requests.append((p, ylim))
                    if requests:
                        it_movies(requests, g, sm, pc, amp, sec, rec, tag, note,
                                  a.it_fps if a.it_fps is not None else a.fps)
                if a.movie_family in ("all", "centroid"):
                    requests = []
                    for ylim, z in ((ZOOM_Y, "zoom"), (FULL_Y, "full")):
                        p = root / f"centroid_basis_movie_{z}{sfx}.mp4"
                        if not p.exists() or a.force:
                            requests.append((p, ylim))
                    if requests:
                        save_movies(
                            requests, x, yc + jy, dom, palette, sec, rec, tag,
                            a.centroid_fps if a.centroid_fps is not None else a.fps,
                            False)
            print(f"  {mode}: done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
