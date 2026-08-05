"""Panels for the wide-lattice single-source fits.

Mirrors viz_hard.py's panel set, but the model is different enough to need its own
implementation: the codebook is a (site x scale) product too large to materialize, the
time basis is orthonormal and stored directly rather than as a module state dict, and
each spike's footprint is reconstructed from its own chosen candidate rather than sliced
out of a precomputed (B, C, K) tensor.

Writes, per fit, into figures/<tag>/:
    components.png   the Q time courses, and where in the volume the fits actually land
    basis.png        the time basis one panel per component, sorted by usage, with the
                     aggregate and lattice scatters coloured by each spike's dominant one
    usage.png        how the K sites and 10 scales are used, and how concentrated that is
    spikes.png       example spikes: waveform vs model, per-channel ptp bars, footprint, v
    localize.png     implied (x, y, z) against the monopole
    aggregate_1s*.png  one second of localizations, full probe and zoomed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
import torch                                 # noqa: E402
from matplotlib.colors import PowerNorm      # noqa: E402
from scipy.ndimage import gaussian_filter    # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                                     # noqa: E402
from spiketensor.volume import GridSpec, quantized_anchor_xyz         # noqa: E402
from spiketensor.waveforms import load_batch                              # noqa: E402
from spiketensor.fit_lattice import Candidates, footprint, phi      # noqa: E402


def load(runs: Path, tag: str):
    """(meta, choice, v, mu-per-candidate, sigma-per-candidate).

    Dictionary fits store the site lattice plus the profile count rather than the expanded
    candidate array; expand lazily here only for the columns the panels index by candidate,
    and never materialize a 15.7M x 3 position array."""
    ck = torch.load(runs / f"codebook_{tag}.pt", map_location="cpu", weights_only=False)
    z = np.load(runs / f"pi_{tag}.npz")
    k = z["k"].astype(np.int64)
    if "mu_site" in z.files:
        S = int(z["S"])
        site, prof = k // S, k % S
        return (ck, k, z["v"], z["mu_site"].astype(np.float32)[site],
                z["prof_sigma"][prof], site, prof,
                z["mu_site"].astype(np.float32))
    # legacy layout: mu is the EXPANDED (KS, 3) candidate table, so the site lattice is
    # every S-th row of it. Deriving it from the per-spike view instead gave an array of
    # the wrong length and broke fig_components.
    S = ck["S"]
    return (ck, k, z["v"], z["mu"][k], z["sigma"][k], k // S, k % S,
            z["mu"][::S].astype(np.float32))


def kern_img(kernel, XX, YY, m, sig):
    """The chosen profile on an x-y slice at the site's depth, for the spike panels."""
    import torch as _t
    from spiketensor.fit_lattice import KERNELS
    dxy2 = _t.as_tensor((XX - m[0]) ** 2 + (YY - m[1]) ** 2, dtype=_t.float32)
    dz2 = _t.full_like(dxy2, float(m[2]) ** 2)
    f = KERNELS.get(kernel, KERNELS["monopole"])
    p = sig if isinstance(sig, (tuple, list)) else (float(sig), 2.0)
    return f(dxy2, dz2, p).numpy()


def fig_components(ck, site, prof, musite, out: Path, tag):
    a = ck["a"].numpy()
    cnt = np.bincount(site, minlength=ck["K"]).astype(float)
    su = musite
    fig, ax = plt.subplots(1, 4, figsize=(18.5, 4.3), constrained_layout=True)
    for q in range(a.shape[0]):
        ax[0].plot(np.arange(a.shape[1]) / 30.0, a[q], lw=1.1,
                   label=f"q={q}" if q < 8 else None)
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("amplitude")
    ax[0].set_title(f"the shared time basis a  (Q={a.shape[0]}, orthonormal)", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=.3)

    hb = ax[1].hexbin(su[:, 0], su[:, 1], C=cnt, reduce_C_function=np.sum,
                      gridsize=45, cmap="magma", bins="log")
    plt.colorbar(hb, ax=ax[1]).set_label("spikes assigned", fontsize=8)
    ax[1].set_xlabel("site x (µm)"); ax[1].set_ylabel("site y (µm)")
    ax[1].set_title("where in the volume the sites are used", fontsize=9)

    zs = np.unique(su[:, 2])
    zc = np.array([cnt[np.isclose(su[:, 2], z)].sum() for z in zs])
    ax[2].semilogx(zs, zc / max(1, zc.sum()), "-o", ms=3, color="#4c8dff")
    ax[2].set_xlabel("site depth z (µm, geometric lattice)")
    ax[2].set_ylabel("fraction of spikes")
    ax[2].set_title("depth usage", fontsize=9); ax[2].grid(alpha=.3)

    sh = np.bincount(prof, minlength=ck["S"])
    prf = ck.get("profiles") or [(ck.get("kernel", "?"), (s_,)) for s_ in ck["sigmas"]]
    col = {k: c for k, c in zip(sorted({p[0] for p in prf}),
                                ["#4c8dff", "#e8590c", "#2f9e44", "#845ef7",
                                 "#f59f00", "#c92a2a", "#12b886"])}
    ax[3].bar(np.arange(ck["S"]), sh / max(1, sh.sum()),
              color=[col[p[0]] for p in prf])
    ax[3].set_xlabel("profile index (colour = kernel family)")
    ax[3].set_ylabel("fraction of spikes")
    lg = "  ".join(f"{k} {100*sh[[i for i,p in enumerate(prf) if p[0]==k]].sum()/max(1,sh.sum()):.0f}%"
                   for k in col)
    ax[3].set_title(f"profile usage · {lg}", fontsize=8.5); ax[3].grid(alpha=.3, axis="y")
    for k, c in col.items():
        ax[3].plot([], [], "s", color=c, label=k)
    ax[3].legend(fontsize=6, ncol=2)
    fig.suptitle(f"codebook — {tag}   ·   {ck['K']:,} sites × {ck['S']} profiles = "
                 f"{ck['KS']:,} candidates, one chosen per spike", fontsize=11)
    fig.savefig(out / "components.png", dpi=125, bbox_inches="tight"); plt.close(fig)


def fig_basis(ck, rec, V, site, musite, out: Path, tag, t0=1200.0, n_scat=120000,
              zoom=(400., 900.), seed=5):
    """The time basis one-per-panel, sorted by usage, with space coloured by which
    component each spike leans on.

    v_s is a free coefficient vector, so no spike "uses" a single component outright. The
    hard label here is argmax_q |v_{s,q}| -- the component carrying the most weight for
    that spike. Usage is how many spikes take each as their dominant one; energy is the
    share of sum_s v^2, which is the softer version of the same question. They are reported
    together because they can disagree: a component can dominate few spikes but carry a lot
    of amplitude in many."""
    a = ck["a"].numpy()
    Q, T = a.shape
    dom = np.argmax(np.abs(V), axis=1)
    use = np.bincount(dom, minlength=Q).astype(float)
    eng = (V.astype(np.float64) ** 2).sum(0)
    order = np.argsort(-use)                      # most-used first
    rank = np.empty(Q, int); rank[order] = np.arange(Q)
    cmap = plt.get_cmap("turbo")
    col = cmap(np.linspace(0.04, 0.96, Q))        # colour BY RANK, so hue = popularity
    qcol = col[rank]                              # colour of original component q

    ncol = 8 if Q > 16 else (4 if Q > 4 else Q)
    nrow = int(np.ceil(Q / ncol))
    fig = plt.figure(figsize=(2.05 * ncol + 1.0, 1.55 * nrow + 8.6),
                     constrained_layout=True)
    gs = fig.add_gridspec(nrow + 2, ncol,
                          height_ratios=[1.0] * nrow + [0.85, 5.4])
    ymax = float(np.abs(a).max()) * 1.08
    for r, q in enumerate(order):
        A = fig.add_subplot(gs[r // ncol, r % ncol])
        A.plot(np.arange(T) / 30.0, a[q], color=qcol[q], lw=1.5)
        A.axhline(0, color="0.6", lw=.5)
        A.set_ylim(-ymax, ymax)
        A.set_title(f"q={q}  ·  {100*use[q]/use.sum():.1f}%", fontsize=7.5,
                    color=qcol[q])
        A.tick_params(labelsize=5.5)
        if r % ncol:
            A.set_yticklabels([])
        if r // ncol != nrow - 1:
            A.set_xticklabels([])
        else:
            A.set_xlabel("ms", fontsize=7)
    B = fig.add_subplot(gs[nrow, :])
    xx = np.arange(Q)
    B.bar(xx - .2, use[order] / use.sum(), .4, color=col, label="spikes dominated")
    B.bar(xx + .2, eng[order] / eng.sum(), .4, color=col, alpha=.45, hatch="///",
          label="share of Σv²")
    B.set_xticks(xx); B.set_xticklabels([f"q{q}" for q in order], fontsize=6)
    B.set_ylabel("fraction", fontsize=8); B.legend(fontsize=7)
    B.set_title("component usage, sorted — solid = spikes whose |v| peaks there, "
                "hatched = energy share", fontsize=8.5)
    B.grid(alpha=.3, axis="y")

    sec = np.floor(rec.spike_times / rec.fs)
    one = np.flatnonzero((sec >= t0) & (sec < t0 + 1))
    anc = rec.anchors[rec.spike_channels]
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(rec.n_spikes, min(n_scat, rec.n_spikes), replace=False))
    P1 = anc[one][:, :2] + musite[site[one]][:, :2]
    for j, (idxs, ylim, ttl) in enumerate((
            (one, (0., 3840.), f"aggregate, t={t0:.0f}–{t0+1:.0f} s — full probe"),
            (one, zoom, f"aggregate, same second — {zoom[0]:.0f}–{zoom[1]:.0f} µm zoom"),
            (sub, None, f"lattice sites, {len(sub):,} spikes (anchor-relative)"))):
        A = fig.add_subplot(gs[nrow + 1, j * ncol // 3:(j + 1) * ncol // 3])
        # draw RAREST component last so it is not buried under the dominant one --
        # q0 alone takes half the spikes and would otherwise paint over everything
        dsub = dom[idxs]
        draw = order[::-1]
        if ylim is None:
            pos = musite[site[idxs]]
            for q in draw:
                m = dsub == q
                if m.any():
                    A.scatter(pos[m, 0], pos[m, 1], s=1.2, color=qcol[q], alpha=.35,
                              linewidths=0, rasterized=True)
            A.set_xlabel("site x rel. anchor (µm)", fontsize=8)
            A.set_ylabel("site y rel. anchor (µm)", fontsize=8)
        else:
            pos = anc[idxs][:, :2] + musite[site[idxs]][:, :2]
            for q in draw:
                m = dsub == q
                if m.any():
                    A.scatter(pos[m, 0], pos[m, 1], s=5.0, color=qcol[q], alpha=.75,
                              linewidths=0, rasterized=True)
            sel = ((ylim[0] <= rec.channel_locations[:, 1])
                   & (rec.channel_locations[:, 1] <= ylim[1]))
            A.scatter(rec.channel_locations[sel, 0], rec.channel_locations[sel, 1],
                      s=7, marker="s", c="none", edgecolors="0.45", linewidths=.4)
            A.set_ylim(*ylim); A.set_xlim(-160, 210)
            A.set_xlabel("x (µm)", fontsize=8); A.set_ylabel("depth y (µm)", fontsize=8)
        A.set_title(ttl, fontsize=8.5); A.tick_params(labelsize=7)
    fig.suptitle(f"time basis and where it is used — {tag}\n"
                 f"Q={Q} orthonormal components, sorted by how many spikes they dominate; "
                 f"every dot is one spike, coloured by its argmax_q |v_q|", fontsize=11)
    fig.savefig(out / "basis.png", dpi=125, bbox_inches="tight"); plt.close(fig)


def fig_usage(ck, K, V, site, out: Path, tag):
    cs = np.bincount(site, minlength=ck["K"])
    cc = np.bincount(K, minlength=ck["KS"])
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.1), constrained_layout=True)
    for A, c, lab, tot in ((ax[0], cs, "sites", ck["K"]),
                           (ax[1], cc, "candidates", ck["KS"])):
        s = np.sort(c)[::-1]
        A.plot(np.arange(1, len(s) + 1), np.cumsum(s) / s.sum(), color="#4c8dff", lw=1.5)
        used = int((c > 0).sum())
        h = np.searchsorted(np.cumsum(s) / s.sum(), 0.5) + 1
        A.axvline(h, color="#c92a2a", ls="--", lw=1.2)
        A.set_xscale("log"); A.set_xlabel(f"{lab}, ranked by usage")
        A.set_ylabel("cumulative fraction of spikes")
        A.set_title(f"{used:,}/{tot:,} {lab} used · half the spikes in the top {h:,}",
                    fontsize=9)
        A.grid(alpha=.3)
    amp = np.linalg.norm(V, axis=1)
    ax[2].hist(np.log10(np.maximum(amp, 1e-6)), bins=90, color="#845ef7")
    ax[2].set_xlabel("log10 ‖v‖ (model amplitude)"); ax[2].set_ylabel("spikes")
    ax[2].set_title(f"amplitude · median {np.median(amp):.3f}", fontsize=9)
    ax[2].grid(alpha=.3)
    fig.suptitle(f"codebook usage — {tag}", fontsize=11)
    fig.savefig(out / "usage.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_spikes(ck, rec, off_all, K, V, mu, sig, musite, idx, out: Path, tag, lim=160.0):
    Y, off = load_batch(rec, idx, off_all, "cpu")
    a = ck["a"]
    prf = ck.get("profiles") or [(ck.get("kernel", "monopole"), (s_,))
                                 for s_ in ck["sigmas"]]
    cand = Candidates(musite, [(p[0], tuple(p[1])) for p in prf], "cpu")
    g = footprint(cand, off, torch.as_tensor(K[idx]))
    Yh = g[:, :, None] * (torch.as_tensor(V[idx]) @ a)[:, None, :]
    Y, Yh, offs = Y.numpy(), Yh.numpy(), off.numpy()
    n = len(idx); amp = 16.0 / max(1e-6, np.abs(Y).max())
    gx = np.linspace(-lim, lim, 141)
    XX, YY = np.meshgrid(gx, gx, indexing="ij")
    fig, ax = plt.subplots(4, n, figsize=(3.0 * n, 12.0), constrained_layout=True,
                          squeeze=False,
                          gridspec_kw={"height_ratios": [1.25, .62, 1.2, .55]})
    for i in range(n):
        A = ax[0][i]; t = np.arange(Y.shape[2]) * .32
        for c in range(Y.shape[1]):
            A.plot(offs[i, c, 0] + t, offs[i, c, 1] + Y[i, c] * amp, "#e03131", lw=.9)
            A.plot(offs[i, c, 0] + t, offs[i, c, 1] + Yh[i, c] * amp, "#2f9e44",
                   lw=1.05, ls="--")
            A.plot(offs[i, c, 0], offs[i, c, 1], "s", ms=2.6, c="#999")
        e = ((Y[i] - Yh[i]) ** 2).mean() / max(1e-12, (Y[i] ** 2).mean())
        A.set_title(f"spike {idx[i]}  rel.err {e:.3f}", fontsize=8)
        A.tick_params(labelsize=6)
        # measured vs modelled peak-to-peak, per channel
        B = ax[1][i]; w = 0.38; xc = np.arange(Y.shape[1])
        pm, pf = np.ptp(Y[i], axis=1), np.ptp(Yh[i], axis=1)
        B.bar(xc - w / 2, pm, w, color="#e03131", label="measured" if i == 0 else None)
        B.bar(xc + w / 2, pf, w, color="#2f9e44", label="model" if i == 0 else None)
        B.set_title(f"ptp · rel.err {np.abs(pf-pm).sum()/max(1e-9,pm.sum()):.3f}",
                    fontsize=7.5)
        B.tick_params(labelsize=6); B.set_xlabel("channel", fontsize=7)
        if i == 0:
            B.legend(fontsize=6)
        m, s_ = mu[idx[i]], sig[idx[i]]
        C = ax[2][i]
        C.imshow(kern_img(ck["kernel"], XX, YY, m, s_).T, origin="lower",
                 extent=[-lim, lim, -lim, lim], cmap="magma", aspect="equal",
                 vmin=0, vmax=1)
        C.scatter(offs[i, :, 0], offs[i, :, 1], s=12, marker="s", c="none",
                  edgecolors="w", linewidths=.6)
        C.plot(0, 0, "c+", ms=11, mew=1.7)
        C.plot(m[0], m[1], "o", mfc="none", mec="#4c8dff", ms=13, mew=2.0)
        C.set_title(f"({m[0]:+.0f},{m[1]:+.0f}) z={m[2]:.0f} σ={s_:.0f}", fontsize=7.5)
        C.tick_params(labelsize=6)
        E = ax[3][i]
        E.bar(np.arange(len(V[idx[i]])), V[idx[i]], color="#4c8dff")
        E.axhline(0, color="0.5", lw=.5); E.set_xlabel("shape q", fontsize=7)
        E.tick_params(labelsize=6)
    ax[0][0].set_ylabel("measured (red) vs model (green)", fontsize=8)
    ax[3][0].set_ylabel("v", fontsize=8)
    fig.suptitle(f"example spikes — {tag}   ·   ONE (site, scale) per spike (blue ○); "
                 f"cyan + anchor, white ▫ contacts", fontsize=11)
    fig.savefig(out / "spikes.png", dpi=125, bbox_inches="tight"); plt.close(fig)


def fig_localize(ck, rec, K, mu, out: Path, tag, n=60000, seed=3):
    g = GridSpec(**torch.load(REPO / "runs/gridspec.pt",
                              map_location="cpu", weights_only=False)["grid"])
    qa = quantized_anchor_xyz(rec.anchors, g)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(rec.n_spikes, n, replace=False))
    mp = rec.mp_xyz[idx] - qa[rec.spike_channels[idx]]
    mp[:, 2] = rec.mp_xyz[idx, 2]
    pos = mu[idx]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    ax[0].hexbin(pos[:, 0], pos[:, 1], gridsize=60, cmap="magma", bins="log")
    ax[0].set_title("chosen site (x, y rel. anchor)", fontsize=9)
    ax[0].set_xlabel("x (µm)"); ax[0].set_ylabel("y (µm)")
    for j, lab in enumerate("xyz"):
        A = ax[1 + j]
        A.hexbin(mp[:, j], pos[:, j], gridsize=60, cmap="magma", bins="log")
        lo = min(mp[:, j].min(), pos[:, j].min())
        hi = max(mp[:, j].max(), pos[:, j].max())
        A.plot([lo, hi], [lo, hi], "--", color="#4c8dff", lw=1.3)
        r = np.corrcoef(mp[:, j], pos[:, j])[0, 1]
        sl = np.polyfit(mp[:, j], pos[:, j], 1)[0]
        A.set_xlabel(f"monopole {lab} (µm)"); A.set_ylabel(f"lattice {lab} (µm)")
        A.set_title(f"{lab}: r {r:+.3f}  slope {sl:.3f}  "
                    f"spread {pos[:, j].std():.1f} vs {mp[:, j].std():.1f}", fontsize=9)
    fig.suptitle(f"lattice readout vs monopole — {tag}", fontsize=11)
    fig.savefig(out / "localize.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_aggregate(ck, rec, K, V, mu, sig, out: Path, tag, t0=1200.0, res=2.0,
                  ktrunc=60.0, zoom=(400., 900.), gamma=.45):
    sec = np.floor(rec.spike_times / rec.fs)
    idx = np.flatnonzero((sec >= t0) & (sec < t0 + 1))
    anc = rec.anchors[rec.spike_channels[idx]][:, :2]
    pos = anc + mu[idx][:, :2]
    amp = np.linalg.norm(V[idx], axis=1)
    x_lo, x_hi, y_lo, y_hi = -70., 130., 0., 3840.
    nx, ny = int((x_hi - x_lo) / res), int((y_hi - y_lo) / res)

    def scatter(px, py, w):
        A = np.zeros((ny, nx))
        ix = np.floor((px - x_lo) / res).astype(int)
        iy = np.floor((py - y_lo) / res).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.add.at(A, (iy[ok], ix[ok]), w[ok]); return A

    nk = int(ktrunc / res); gg = np.arange(-nk, nk + 1) * res
    KX, KY = np.meshgrid(gg, gg, indexing="xy")
    soft = np.zeros((ny, nx))
    for i in range(len(idx)):
        m, s_ = mu[idx[i]], sig[idx[i]]
        d2 = KX ** 2 + KY ** 2 + m[2] ** 2
        ker = ((s_ / np.sqrt(d2 + s_ ** 2)) if ck["kernel"] == "monopole"
               else np.exp(-d2 / (2 * s_ ** 2))) * (np.sqrt(KX ** 2 + KY ** 2) <= ktrunc)
        cx = int((pos[i, 0] - x_lo) / res); cy = int((pos[i, 1] - y_lo) / res)
        x0, x1 = max(0, cx - nk), min(nx, cx + nk + 1)
        y0, y1 = max(0, cy - nk), min(ny, cy + nk + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        soft[y0:y1, x0:x1] += amp[i] * ker[y0 - cy + nk:y1 - cy + nk,
                                           x0 - cx + nk:x1 - cx + nk]
    hard = gaussian_filter(scatter(pos[:, 0], pos[:, 1], amp), 4.0 / res)
    mono = gaussian_filter(scatter(rec.mp_xyz[idx, 0], rec.mp_xyz[idx, 1],
                                   np.ptp(np.asarray(rec.waveforms[idx]),
                                          axis=2).max(1)), 4.0 / res)
    panels = [("lattice SOFT  Σ_s ‖v‖ g_{k(s)}", soft),
              ("lattice HARD (4 µm blur)", hard),
              ("MONOPOLE reference (4 µm blur)", mono)]
    for name, ylim, h in (("aggregate_1s", (y_lo, y_hi), 13.),
                          ("aggregate_1s_zoom", zoom, 7.)):
        fig, ax = plt.subplots(1, 3, figsize=(3.1 * 3 + 1.2, h + 1.),
                              constrained_layout=True)
        for j, (ttl, img) in enumerate(panels):
            s = img[img > 0]
            v = np.percentile(s, 99.7) if s.size else 1.
            ax[j].imshow(img, origin="lower", extent=[x_lo, x_hi, y_lo, y_hi],
                         cmap="magma", aspect="equal",
                         norm=PowerNorm(gamma, vmin=0, vmax=v), interpolation="nearest")
            sel = ((ylim[0] <= rec.channel_locations[:, 1])
                   & (rec.channel_locations[:, 1] <= ylim[1]))
            ax[j].scatter(rec.channel_locations[sel, 0], rec.channel_locations[sel, 1],
                          s=6, marker="s", c="none", edgecolors="w", linewidths=.4,
                          alpha=.6)
            ax[j].set_ylim(*ylim); ax[j].set_xlim(x_lo, x_hi)
            ax[j].set_title(ttl, fontsize=9); ax[j].set_xlabel("x (µm)", fontsize=8)
            ax[j].tick_params(labelsize=7)
        ax[0].set_ylabel("depth y (µm)", fontsize=9)
        fig.suptitle(f"aggregate, t = {t0:.0f}–{t0+1:.0f} s — {tag}\n{len(idx)} spikes · "
                     f"true aspect · kernels truncated at {ktrunc:.0f} µm", fontsize=10)
        fig.savefig(out / f"{name}.png", dpi=125, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--figs", type=Path, default=REPO / "figures")
    ap.add_argument("--only", default="")
    ap.add_argument("--n_spikes", type=int, default=6)
    a = ap.parse_args()

    rec = D.load("np1")
    off_all = rec.channel_offsets().astype(np.float32)
    rng = np.random.default_rng(11)
    sidx = np.sort(rng.choice(rec.n_spikes, a.n_spikes, replace=False))
    for f in sorted(a.runs.glob("summary_*.json")):
        tag = f.stem[len("summary_"):]
        if a.only and a.only not in tag:
            continue
        out = a.figs / tag; out.mkdir(parents=True, exist_ok=True)
        ck, K, V, mu, sig, site, prof, musite = load(a.runs, tag)
        fig_components(ck, site, prof, musite, out, tag)
        fig_usage(ck, K, V, site, out, tag)
        fig_basis(ck, rec, V, site, musite, out, tag)
        fig_spikes(ck, rec, off_all, K, V, mu, sig, musite, sidx, out, tag)
        fig_localize(ck, rec, K, mu, out, tag)
        fig_aggregate(ck, rec, K, V, mu, sig, out, tag)
        print(f"  {tag}: panels written", flush=True)


if __name__ == "__main__":
    main()
