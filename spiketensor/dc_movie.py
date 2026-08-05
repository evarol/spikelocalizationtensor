"""Drift pipeline for a fitted hard-lattice model: I_t, D/C, DREDge, and a movie.

Takes the per-spike localizations a hard-assignment fit produces, builds per-second
images, and runs them through EXACTLY the machinery the ZNCC project uses -- the same
GridSpec, the same ShiftMaxZNCC with a +-80 um y search, the same rigid DREDge solve and
the same detrended ground-truth score. That is deliberate: it makes the output directly
comparable to the monopole reference (D 17 distinct values, lattice-fraction 0.000,
GT r +0.887) and to the 140 trained ZNCC runs, rather than being a new scale nobody can
place.

Each spike contributes a point at (anchor + mu_k) weighted by ||v||, since the model's
readout is a single lattice site, not a heat map. z comes from the site's depth, so I_t
is a genuine 3-D image and the existing pipeline applies unchanged.

Outputs, all under figures/<tag>/:
    dc.png      D and C, soft and hard, with the post-hoc motion solved from each
    dc.npz      the raw matrices
    It_zoom.mp4 / It_full.mp4   the images the ZNCC actually sees, 1 s per frame
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402
from matplotlib.animation import FFMpegWriter          # noqa: E402
from matplotlib.colors import PowerNorm                # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                                    # noqa: E402
from spiketensor.dredge import DredgeConfig, solve_rigid             # noqa: E402
from spiketensor.volume import GridSpec, VolumeSmoother              # noqa: E402
from spiketensor.zncc import ShiftMaxZNCC, ZNCCConfig                # noqa: E402
from spiketensor.gtscore import score                          # noqa: E402
from spiketensor.baselines import reference_positions              # noqa: E402


def load_any(runs, tag):
    """(mu indexed by candidate, choice, v, meta) for either model family.

    The wide-lattice fits keep mu in the npz -- 64^3 x 10 scales is 2.6M rows -- and their
    codebook_*.pt is a plain dict, not a module state dict, so viz_hard.load cannot read
    them."""
    ck = torch.load(runs / f"codebook_{tag}.pt", map_location="cpu", weights_only=False)
    z = np.load(runs / f"pi_{tag}.npz")
    ck.setdefault("dataset", "np1")
    # the candidate index is site*S + profile, so the position is a floor-divide away and
    # the 2.6M-row expanded table never has to be materialized
    return (z["mu_site"].astype(np.float32),
            z["k"].astype(np.int64) // int(z["S"]), z["v"], ck)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def build_volume(g, smoother, px, py, pz, w, dev):
    """Weighted point scatter into the 3-D voxel grid, then the standard blur."""
    ix = torch.floor((torch.as_tensor(px, device=dev) - g.x_lo) / g.x_bin).long()
    iy = torch.floor((torch.as_tensor(py, device=dev) - g.y_lo) / g.y_bin).long()
    iz = torch.floor((torch.as_tensor(pz, device=dev) - g.z_lo) / g.z_bin).long()
    ok = ((ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny)
          & (iz >= 0) & (iz < g.nz))
    V = torch.zeros(g.nx * g.ny * g.nz, device=dev)
    flat = (ix[ok] * g.ny + iy[ok]) * g.nz + iz[ok]
    V.index_add_(0, flat, torch.as_tensor(w, device=dev, dtype=torch.float32)[ok])
    return smoother(V.view(1, g.nx, g.ny, g.nz))[0], int((~ok).sum())


def render_dc(o, tag, Ds, Cs, Dh, Ch, sol, solh, sc, sch, t_sec, gt, inn, tau, stride,
              note=""):
    """The six-panel D/C figure plus dc.npz. Shared with dc_batch.py so every run's panel
    is produced by one implementation and the numbers on it cannot drift from the table."""
    T = len(t_sec)
    off_d = ~np.eye(T, dtype=bool)
    t_in = t_sec[inn]
    fig = plt.figure(figsize=(15.6, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
    ext = [t_sec[0], t_sec[-1], t_sec[0], t_sec[-1]]
    for r, (Dm, Cm, s_, lab) in enumerate((
            (Ds, Cs, sc, f"soft — softmax over lags, τ={tau}"),
            (Dh, Ch, sch, "hard — argmax over lags (what DREDge reads)"))):
        dv = float(max(np.percentile(np.abs(Dm[off_d]), 99.5), 1.0))
        A = fig.add_subplot(gs[r, 0])
        im = A.imshow(Dm, origin="lower", cmap="RdBu_r", vmin=-dv, vmax=dv, extent=ext,
                      interpolation="nearest", aspect="equal")
        cb_ = fig.colorbar(im, ax=A, fraction=0.046, pad=0.02,
                           extend="both" if np.abs(Dm[off_d]).max() > dv else "neither")
        cb_.set_label("D = depth shift (µm)", fontsize=7.5); cb_.ax.tick_params(labelsize=6)
        gp = gt[:, None] - gt[None, :]
        rg = np.corrcoef(Dm[off_d], gp[off_d])[0, 1] if Dm[off_d].std() > 0 else 0.0
        A.set_title(f"D — {lab}\nr_gt {rg:+.2f} · amplitude {Dm[off_d].std():.1f} µm · "
                    f"zero {100*(Dm[off_d]==0).mean():.0f}%", fontsize=8)
        B = fig.add_subplot(gs[r, 1])
        im = B.imshow(Cm, origin="lower", cmap="magma", vmin=min(0, Cm[off_d].min()),
                      vmax=1, extent=ext, interpolation="nearest", aspect="equal")
        cb_ = fig.colorbar(im, ax=B, fraction=0.046, pad=0.02)
        cb_.set_label("C = ZNCC", fontsize=7.5); cb_.ax.tick_params(labelsize=6)
        B.set_title(f"C — {lab}\nmean {Cm[off_d].mean():.3f} · "
                    f"range [{Cm[off_d].min():.3f}, {Cm[off_d].max():.3f}]", fontsize=8)
        E = fig.add_subplot(gs[r, 2])
        p = (sol if r == 0 else solh)["p"]
        gi = gt[inn] - gt[inn].mean()
        E.plot(t_in, gi, color="#2f6df6", lw=1.1, label="imposed motion")
        E.plot(t_in, p, color="#e03131", lw=1.0, label="DREDge from D")
        E.set_title(f"DREDge — r {s_['r']:+.3f} · gain {s_['gain']:.3f} · "
                    f"ptp {np.ptp(p):.1f} µm vs {np.ptp(gi):.1f}", fontsize=8)
        E.set_xlabel("time (s)"); E.set_ylabel("depth (µm)", fontsize=8)
        E.legend(fontsize=7); E.grid(alpha=0.3)
        for X in (A, B):
            X.set_xlabel("t' (s)", fontsize=8); X.set_ylabel("t (s)", fontsize=8)
            X.tick_params(labelsize=7)
    fig.suptitle(f"pairwise ZNCC of the model's own localizations — {tag}\n"
                 f"T={T} one-second images sampled every {stride} s · ±80 µm y search · "
                 f"same grid, ZNCC and DREDge as the monopole reference "
                 f"(C_hard 0.504, GT r +0.865)" + (f"\n{note}" if note else ""),
                 fontsize=11)
    fig.savefig(o / "dc.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    np.savez_compressed(o / "dc.npz", D_soft=Ds.astype(np.float32),
                        C_soft=Cs.astype(np.float32), D_hard=Dh.astype(np.float32),
                        C_hard=Ch.astype(np.float32), p_soft=sol["p"],
                        p_hard=solh["p"], t=t_sec, gt=gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--control", default="",
                    help="render a reference localizer instead of a fit: "
                         "monopole | anchor_ptp | anchor_flat")
    ap.add_argument("--monopole", action="store_true",
                    help="use the raw monopole localizations instead of a fitted model, "
                         "through this identical pipeline, as a matched baseline")
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=REPO / "figures")
    ap.add_argument("--tau", type=float, default=0.2, help="softmax temperature over lags")
    ap.add_argument("--stride", type=int, default=2, help="bins for the D/C matrices")
    ap.add_argument("--smooth_um", type=float, default=4.0)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--movie_only", action="store_true",
                    help="skip the D/C recomputation and just render the movies")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    o = a.out / a.tag; o.mkdir(parents=True, exist_ok=True)

    if a.control:
        a.monopole = True                       # same waveform-free position path
        rec = D.load("np1"); cb = KS = V = None; ck = {"dataset": "np1"}
    elif a.monopole:
        rec = D.load("np1"); cb = KS = V = None; ck = {"dataset": "np1"}
    else:
        mu, KS, V, ck = load_any(a.runs, a.tag)
        rec = D.load(ck["dataset"])
    # the SAME grid the ZNCC project uses, so every number lands on a known scale
    g = GridSpec(**torch.load(REPO / "runs/gridspec.pt",
                              map_location="cpu", weights_only=False)["grid"])
    sm = VolumeSmoother(a.smooth_um, g, device=DEV).to(DEV)

    if a.monopole:
        # the monopole's own (x, y, z), weighted by measured max ptp. The fitted models
        # weight by ||v||, their internal amplitude; both are "how loud is this spike"
        # in their own units and the two are highly correlated, so the images are
        # comparable on WHERE the mass sits, which is the point of the comparison.
        pos, amp = reference_positions(rec, a.control or "monopole")
    else:
        anc = rec.anchors[rec.spike_channels][:, :2]
        pos = np.empty((rec.n_spikes, 3), np.float32)
        pos[:, :2] = anc + mu[KS][:, :2]
        pos[:, 2] = mu[KS][:, 2]
        amp = np.linalg.norm(V, axis=1).astype(np.float32)
    sec = np.floor(rec.spike_times / rec.fs).astype(np.int64)
    nb = int(sec.max()) + 1
    order = np.argsort(sec, kind="stable")
    cnt = np.bincount(sec, minlength=nb)
    off = np.concatenate([[0], np.cumsum(cnt)])
    print(f"{a.tag}: {rec.n_spikes:,} spikes, {nb} one-second bins, "
          f"grid {g.shape}, device {DEV}", flush=True)

    if a.movie_only:
        a.movie = True
    bins = (np.flatnonzero(cnt > 0)[:2] if a.movie_only
            else np.flatnonzero(cnt > 0)[::a.stride])
    T = len(bins); t_sec = bins.astype(float) + 0.5
    gt = D.gt_motion_trace(t_sec)
    t0 = time.perf_counter(); drop = 0
    vols = torch.empty(T, *g.shape, dtype=torch.float16)
    for i, b in enumerate(bins):
        s = order[off[b]:off[b + 1]]
        v, dr = build_volume(g, sm, pos[s, 0], pos[s, 1], pos[s, 2], amp[s], DEV)
        vols[i] = v.to(torch.float16).cpu(); drop += dr
    print(f"  built {T} volumes in {time.perf_counter()-t0:.0f}s "
          f"({100*drop/max(1,cnt[bins].sum()):.2f}% of spikes fell outside the grid)",
          flush=True)

    if a.movie_only:
        Ds = Cs = Dh = Ch = None
    zc = None if a.movie_only else ShiftMaxZNCC(ZNCCConfig(max_shift_y_vox=int(80.0 / g.y_bin), tau=a.tau))
    Ds = np.zeros((T, T), np.float32); Cs = np.zeros((T, T), np.float32)
    Dh = np.zeros((T, T), np.float32); Ch = np.zeros((T, T), np.float32)
    ch = 100
    for i in ([] if a.movie_only else range(0, T, ch)):
        ni = min(ch, T - i)
        for j in range(0, T, ch):
            nj = min(ch, T - j)
            sub = torch.cat([vols[i:i+ni].to(DEV).float(),
                             vols[j:j+nj].to(DEV).float()], 0)
            with torch.no_grad():
                r = zc(sub, return_hard=True)
            Ds[i:i+ni, j:j+nj] = r["d_y_soft"][:ni, ni:ni+nj].cpu().numpy()
            Cs[i:i+ni, j:j+nj] = r["rho_soft"][:ni, ni:ni+nj].cpu().numpy()
            Dh[i:i+ni, j:j+nj] = r["d_y"][:ni, ni:ni+nj].cpu().numpy()
            Ch[i:i+ni, j:j+nj] = r["rho_hard"][:ni, ni:ni+nj].cpu().numpy()
            del sub, r
    if not a.movie_only:
        Ds = np.nan_to_num(Ds).astype(float) * g.y_bin
        Dh = np.nan_to_num(Dh).astype(float) * g.y_bin
        Cs = np.nan_to_num(Cs).astype(float); Ch = np.nan_to_num(Ch).astype(float)

        inn = slice(a.margin, T - a.margin); t_in = t_sec[inn]
        cfg = DredgeConfig(c_thresh=0.0, lam_smooth=1.0, irls_iters=3)
        sol = solve_rigid(Ds[inn, inn], np.clip(Cs[inn, inn], 0, None), cfg)
        solh = solve_rigid(Dh[inn, inn], np.clip(Ch[inn, inn], 0, None), cfg)
        sc, sch = score(sol["p"], t_in), score(solh["p"], t_in)

        off_d = ~np.eye(T, dtype=bool)
        q = np.abs(Dh[off_d]) / 40.0
        lat = float((np.abs(q - np.round(q)) * 40.0 <= 1.0)[Dh[off_d] != 0].mean()) \
            if (Dh[off_d] != 0).any() else 0.0
        print(f"  D_hard: {len(np.unique(np.round(Dh[off_d],3)))} distinct values, "
              f"{100*(Dh[off_d]!=0).mean():.0f}% non-zero, lattice-frac {lat:.3f}", flush=True)
        print(f"  GT r  soft {sc['r']:+.3f} (gain {sc['gain']:.3f})   "
              f"hard {sch['r']:+.3f} (gain {sch['gain']:.3f})", flush=True)

        render_dc(o, a.tag, Ds, Cs, Dh, Ch, sol, solh, sc, sch, t_sec, gt, inn,
                  a.tau, a.stride)
        print(f"  wrote {o}/dc.png", flush=True)

    if not a.movie:
        return
    from spiketensor.viz_common import extents, project
    ex_xy, ex_zy = extents(g)
    allb = np.flatnonzero(cnt > 0)
    tm = allb.astype(float) + 0.5
    gm = D.gt_motion_trace(tm); gm0 = float(np.median(gm))
    probe = allb[::max(1, len(allb) // 40)]
    hi = [0.0, 0.0]
    for b in probe:
        s = order[off[b]:off[b + 1]]
        v, _ = build_volume(g, sm, pos[s, 0], pos[s, 1], pos[s, 2], amp[s], DEV)
        for k, img in enumerate(project(v.cpu().numpy())):
            sb = img[img > 0]
            if sb.size:
                hi[k] = max(hi[k], float(np.percentile(sb, 99.7)))
    for name, ylim, h in (("It_zoom", (400., 900.), 7.0),
                          ("It_full", (g.y_lo, g.y_lo + g.ny * g.y_bin), 13.0)):
        span = ylim[1] - ylim[0]
        wx, wz = g.nx * g.x_bin, g.nz * g.z_bin
        fw = min(15.0, h * (wx + wz) / span + 1.4)
        fig, ax = plt.subplots(1, 2, figsize=(fw, h + .8), facecolor="#0f1115",
                               constrained_layout=True,
                               gridspec_kw={"width_ratios": [wx, wz]})
        sup = fig.suptitle("", color="w", fontsize=9)
        sel = (ylim[0] <= rec.channel_locations[:, 1]) & \
              (rec.channel_locations[:, 1] <= ylim[1])
        w = FFMpegWriter(fps=a.fps, codec="libx264", bitrate=4000,
                         extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                     "-pix_fmt", "yuv420p"])
        t1 = time.perf_counter()
        with w.saving(fig, str(o / f"{name}.mp4"), dpi=90):
            for i, b in enumerate(allb):
                s = order[off[b]:off[b + 1]]
                v, _ = build_volume(g, sm, pos[s, 0], pos[s, 1], pos[s, 2], amp[s], DEV)
                xy, zy = project(v.cpu().numpy())
                for k, (img, ex, xl) in enumerate(((xy, ex_xy, "x (µm)"),
                                                   (zy, ex_zy, "z (µm)"))):
                    ax[k].clear()
                    ax[k].imshow(img, origin="lower", aspect="equal", extent=ex,
                                 cmap="magma",
                                 norm=PowerNorm(0.45, vmin=0, vmax=hi[k] or 1.0),
                                 interpolation="nearest")
                    ax[k].scatter(rec.channel_locations[sel, 0] if k == 0
                                  else np.zeros(sel.sum()),
                                  rec.channel_locations[sel, 1], s=8, marker="s",
                                  c="none", edgecolors="w", linewidths=.5, alpha=.7)
                    ax[k].axhline(np.clip(ylim[0] + span / 2 + (gm[i] - gm0),
                                          ylim[0], ylim[1]), color="#4c8dff", lw=1.2)
                    ax[k].set_xlim(ex[0], ex[1]); ax[k].set_ylim(*ylim)
                    ax[k].set_xlabel(xl, color="w", fontsize=8)
                    ax[k].tick_params(colors="w", labelsize=7)
                    for sp in ax[k].spines.values():
                        sp.set_color("#444")
                ax[0].set_ylabel("depth y (µm)", color="w", fontsize=9)
                sup.set_text(f"I_t from {a.tag}\nt = {tm[i]:7.1f} s   ·   imposed motion "
                             f"{gm[i]-gm0:+6.1f} µm (blue)   ·   fixed colour scale")
                w.grab_frame(facecolor="#0f1115")
        plt.close(fig)
        print(f"  wrote {o}/{name}.mp4  ({len(allb)} frames, "
              f"{(time.perf_counter()-t1)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
