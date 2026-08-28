"""Candidate headline schematics for the README, drawn from a real fit.

Everything shown is measured, not illustrative: the waveforms are real spikes, the
footprint is the profile the model actually selected for that spike, the time course is the
fitted one, and the downstream panels are the real localization and coefficient spaces.

    python3 docs/make_schematic.py --runs <runs> --tag d_gauss_iso6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from matplotlib.colors import PowerNorm                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                     # noqa: E402
from spiketensor.fit_lattice import KERNELS, Candidates, footprint   # noqa: E402
from spiketensor.waveforms import load_batch          # noqa: E402

C_SPACE, C_TIME, C_IN, C_OUT = "#4c8dff", "#e8590c", "#c92a2a", "#2f9e44"
HEADLINE_TAG = "d_gauss_iso6"
HEADLINE_NOTE = "d_gauss_iso6 · canonical rigid r = +0.935 · gain = 0.550"


def compact_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} M"
    if value >= 1_000:
        return f"{value / 1_000:.0f} k"
    return str(value)


def load_fit(runs: Path, tag: str):
    ck = torch.load(runs / f"codebook_{tag}.pt", map_location="cpu", weights_only=False)
    z = np.load(runs / f"pi_{tag}.npz")
    k = z["k"].astype(np.int64)
    if "mu_site" in z.files:                    # compact layout: site lattice + profile count
        S = int(z["S"])
        return ck, k, z["v"], z["mu_site"].astype(np.float32), S, k // S, k % S
    # legacy layout: mu is the expanded (KS, 3) candidate table, one row per candidate
    S = int(ck["S"])
    return ck, k, z["v"], z["mu"][::S].astype(np.float32), S, k // S, k % S


def pick_spikes(rec, off_all, ck, k, V, musite, S, n=3, seed=4):
    """Spikes that reconstruct well AND lean on different basis components, so the figure
    shows the model working rather than its worst case."""
    dom = np.argmax(np.abs(V), 1)
    rng = np.random.default_rng(seed)
    cand = np.sort(rng.choice(rec.n_spikes, 40000, replace=False))
    Y, off = load_batch(rec, cand, off_all, "cpu")
    prf = ck.get("profiles") or [(ck["kernel"], (s_,)) for s_ in ck["sigmas"]]
    cb = Candidates(musite, [(p[0], tuple(p[1])) for p in prf], "cpu")
    g = footprint(cb, off, torch.as_tensor(k[cand]))
    Yh = g[:, :, None] * (torch.as_tensor(V[cand]) @ ck["a"])[:, None, :]
    err = (((Y - Yh) ** 2).mean((1, 2)) / (Y ** 2).mean((1, 2))).numpy()
    amp = np.abs(Y.numpy()).max((1, 2))
    ok = np.flatnonzero((err < np.percentile(err, 12)) & (amp > np.percentile(amp, 70)))
    out, seen = [], set()
    for i in ok[np.argsort(-amp[ok])]:
        q = dom[cand[i]]
        if q not in seen:
            seen.add(q); out.append(int(cand[i]))
        if len(out) == n:
            break
    while len(out) < n:
        out.append(int(cand[ok[len(out)]]))
    return np.sort(np.array(out))


def rendered(rec, off_all, ck, k, V, musite, S, idx):
    Y, off = load_batch(rec, idx, off_all, "cpu")
    prf = ck.get("profiles") or [(ck["kernel"], (s_,)) for s_ in ck["sigmas"]]
    cb = Candidates(musite, [(p[0], tuple(p[1])) for p in prf], "cpu")
    g = footprint(cb, off, torch.as_tensor(k[idx]))
    w = (torch.as_tensor(V[idx]) @ ck["a"]).numpy()          # (n, T) time course
    Yh = (g[:, :, None] * torch.as_tensor(w)[:, None, :]).numpy()
    return Y.numpy(), Yh, off.numpy(), g.numpy(), w


def draw_waves(A, Y, off, colour, lw=1.0, scale=None, ls="-"):
    """Waveforms laid out on the true contact geometry."""
    sc = scale or 15.0 / max(1e-9, np.abs(Y).max())
    t = np.arange(Y.shape[1]) * 0.33
    for c in range(Y.shape[0]):
        A.plot(off[c, 0] + t, off[c, 1] + Y[c] * sc, color=colour, lw=lw, ls=ls)
    A.set_xticks([]); A.set_yticks([])
    for s in A.spines.values():
        s.set_visible(False)
    return sc


def kernel_slice(ck, m, sig, lim=55.0, n=161):
    gx = np.linspace(-lim, lim, n)
    XX, YY = np.meshgrid(gx, gx, indexing="ij")
    dxy2 = torch.as_tensor((XX - m[0]) ** 2 + (YY - m[1]) ** 2, dtype=torch.float32)
    dz2 = torch.full_like(dxy2, float(m[2]) ** 2)
    nm = ck.get("profiles", [(ck.get("kernel", "monopole"), None)])[0][0]
    p = sig if isinstance(sig, (tuple, list)) else (float(sig), 2.0)
    return KERNELS[nm](dxy2, dz2, p).numpy().T, [-lim, lim, -lim, lim]


def between(ax0, ax1, y=None):
    """Midpoint of the gap between two axes, in figure coords, at their shared centre."""
    b0, b1 = ax0.get_position(), ax1.get_position()
    yy = y if y is not None else (b0.y0 + b0.y1) / 2
    return (b0.x1 + b1.x0) / 2, yy


def link(fig, ax0, ax1, glyph=None, colour="0.35", pad=0.012, fs=21):
    """An arrow across the gap between two axes, optionally with an operator glyph."""
    b0, b1 = ax0.get_position(), ax1.get_position()
    y = (max(b0.y0, b1.y0) + min(b0.y1, b1.y1)) / 2
    if glyph:
        xm = (b0.x1 + b1.x0) / 2
        fig.text(xm, y, glyph, ha="center", va="center", fontsize=fs, color="0.30",
                 zorder=6)
        return
    arrow(fig, (b0.x1 + pad, y), (b1.x0 - pad, y), colour=colour)


def arrow(fig, a, b, colour="0.35", lw=1.8, style="-|>", rad=0.0):
    fig.patches.append(FancyArrowPatch(
        a, b, transform=fig.transFigure, arrowstyle=style, mutation_scale=17,
        lw=lw, color=colour, connectionstyle=f"arc3,rad={rad}", zorder=5))


def band(fig, x0, y0, x1, y1, colour, label, alpha=0.055, fs=10):
    fig.patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                                 facecolor=colour, edgecolor=colour, alpha=alpha,
                                 lw=1.2, zorder=0))
    fig.text((x0 + x1) / 2, y1 - 0.012, label, transform=fig.transFigure, ha="center",
             va="top", fontsize=fs, color=colour, weight="bold", zorder=6)


# ===================================================================== candidate A
def schematic_A(ctx, out: Path):
    """One spike, followed all the way through: the equation made literal, with the two
    factors carrying on to their two downstream uses."""
    (rec, ck, k, V, musite, S, site, prof, idx, Y, Yh, off, g, w, dom, qcol) = ctx
    i = 0
    fig = plt.figure(figsize=(16.2, 8.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0],
                          left=.035, right=.985, top=.86, bottom=.07,
                          wspace=.42, hspace=.46)

    A = fig.add_subplot(gs[0, 0]); sc = draw_waves(A, Y[i], off[i], C_IN, 1.25)
    A.set_title("one spike\n10 contacts × 90 samples", fontsize=10.5, color=C_IN)

    B = fig.add_subplot(gs[0, 1])
    img, ext = kernel_slice(ck, musite[site[idx[i]]], ck["sigmas"][prof[idx[i]]]
                            if not ck.get("profiles") else ck["profiles"][prof[idx[i]]][1])
    B.imshow(img, origin="lower", extent=ext, cmap="magma",
             norm=PowerNorm(0.42, vmin=0, vmax=1))
    B.scatter(off[i, :, 0], off[i, :, 1], s=16, marker="s", c="none",
              edgecolors="w", linewidths=.7)
    m = musite[site[idx[i]]]
    B.plot(m[0], m[1], "o", mfc="none", mec=C_SPACE, ms=15, mew=2.4)
    B.set_xticks([]); B.set_yticks([])
    B.set_title(f"SPATIAL  $g_s$\none site chosen from "
                f"{compact_count(int(ck['KS']))}", fontsize=10.5, color=C_SPACE)

    Cx = fig.add_subplot(gs[0, 2])
    Cx.plot(np.arange(len(w[i])) / 30.0, w[i], color=C_TIME, lw=2.0)
    Cx.axhline(0, color="0.75", lw=.6); Cx.set_xlabel("ms", fontsize=9)
    Cx.set_yticks([]); Cx.tick_params(labelsize=8)
    for s_ in ("top", "right"):
        Cx.spines[s_].set_visible(False)
    Cx.set_title("TEMPORAL  $v_s^{\\top}a$\nfree weights on a shared basis",
                 fontsize=10.5, color=C_TIME)

    Dx = fig.add_subplot(gs[0, 3])
    draw_waves(Dx, Yh[i], off[i], C_OUT, 1.35, scale=sc)
    Dx.set_title("reconstruction\n$\\hat Y = g_s\\,(v_s^{\\top}a)$", fontsize=10.5,
                 color=C_OUT)

    E = fig.add_subplot(gs[0, 4:])
    draw_waves(E, Y[i], off[i], C_IN, 1.25, scale=sc)
    draw_waves(E, Yh[i], off[i], C_OUT, 1.5, scale=sc, ls="--")
    e = ((Y[i] - Yh[i]) ** 2).mean() / (Y[i] ** 2).mean()
    E.set_title(f"overlay — measured vs model   (rel. err {e:.2f})", fontsize=10.5)

    fig.canvas.draw()
    link(fig, A, B, glyph="\u2248")        # Y  ~  g
    link(fig, B, Cx, glyph="\u00d7")       # g  x  (v.a)
    link(fig, Cx, Dx)                       # ->  reconstruction

    # ---- downstream: the two factors are the two scientific readouts
    L = fig.add_subplot(gs[1, :3])
    sub = ctx_sub(rec, site, musite, dom, 220000)
    L.scatter(sub[0], sub[1], s=2.2, c=qcol[sub[2]], alpha=.55, linewidths=0,
              rasterized=True)
    L.set_xlim(-140, 200); L.set_ylim(380, 920)
    L.set_xlabel("x (µm)", fontsize=9); L.set_ylabel("depth y (µm)", fontsize=9)
    L.tick_params(labelsize=8)
    L.set_title("→ WHERE:  centroid = anchor + $\\mu_k$  ·  every spike localised",
                fontsize=10.5, color=C_SPACE)

    R = fig.add_subplot(gs[1, 3:])
    a = ck["a"].numpy()
    order = np.argsort(-np.bincount(dom, minlength=a.shape[0]))
    for r_, q in enumerate(order[:6]):
        R.plot(np.arange(a.shape[1]) / 30.0, a[q] - r_ * 0.34, color=qcol[q], lw=1.8)
        R.text(3.15, -r_ * 0.34, f"  $q_{{{q}}}$", color=qcol[q], fontsize=9,
               va="center", weight="bold")
    R.set_xlim(0, 3.5); R.set_yticks([]); R.set_xlabel("ms", fontsize=9)
    R.tick_params(labelsize=8)
    for s_ in ("top", "right", "left"):
        R.spines[s_].set_visible(False)
    R.set_title("→ WHAT:  the shared basis  ·  $v_s$ assigns each spike a waveform type",
                fontsize=10.5, color=C_TIME)

    bB, bC, bL, bR = (x.get_position() for x in (B, Cx, L, R))
    arrow(fig, ((bB.x0 + bB.x1) / 2, bB.y0 - .015),
          ((bL.x0 + bL.x1) / 2, bL.y1 + .055), colour=C_SPACE, rad=.16, lw=2.2)
    arrow(fig, ((bC.x0 + bC.x1) / 2, bC.y0 - .015),
          ((bR.x0 + bR.x1) / 2, bR.y1 + .055), colour=C_TIME, rad=-.16, lw=2.2)
    fig.suptitle("Single-source tensor factorization of spike waveforms\n"
                 "$Y_{s,c,t}\;\\approx\;g_s(c)\\,\\cdot\\,(v_s^{\\top}a)_t$   —   "
                 "a discrete choice of source position, times a shared time basis",
                 fontsize=14, y=.975)
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ctx_sub(rec, site, musite, dom, n, seed=3, ylim=(400., 900.)):
    """Centroids inside the displayed depth band. Subsampling AFTER the band cut keeps the
    panel dense enough to read -- sampling first left ~13% of the points in view."""
    p_all = rec.anchors[rec.spike_channels][:, :2] + musite[site][:, :2]
    keep = np.flatnonzero((p_all[:, 1] >= ylim[0]) & (p_all[:, 1] <= ylim[1]))
    rng = np.random.default_rng(seed)
    s = keep if len(keep) <= n else rng.choice(keep, n, replace=False)
    return p_all[s, 0], p_all[s, 1], dom[s]


# ===================================================================== candidate B
def schematic_B(ctx, out: Path):
    """Three spikes at once, stacked as a tensor: emphasises that the SITE is a discrete
    pick per spike while the BASIS is shared, which is the whole architecture."""
    (rec, ck, k, V, musite, S, site, prof, idx, Y, Yh, off, g, w, dom, qcol) = ctx
    n = len(idx); a = ck["a"].numpy(); Q = a.shape[0]
    fig = plt.figure(figsize=(16.6, 9.2)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(n, 8, left=.045, right=.985, top=.815, bottom=.09,
                          wspace=.55, hspace=.35)
    for x0, x1, col, lab in ((.028, .215, C_IN, "INPUT"),
                             (.232, .705, "0.40", "FACTORIZATION"),
                             (.722, .992, C_OUT, "OUTPUT")):
        fig.patches.append(Rectangle((x0, .075), x1 - x0, .80, transform=fig.transFigure,
                                     facecolor=col, edgecolor=col, alpha=.055, lw=1.1,
                                     zorder=0))
        fig.text((x0 + x1) / 2, .888, lab, transform=fig.transFigure, ha="center",
                 va="bottom", fontsize=11, color=col, weight="bold", zorder=6)

    for i in range(n):
        A = fig.add_subplot(gs[i, 0]); sc = draw_waves(A, Y[i], off[i], C_IN, 1.1)
        if i == 0:
            A.set_title("$Y_s$  (10 × 90)", fontsize=10, color=C_IN, pad=8)

        B = fig.add_subplot(gs[i, 1:3])
        pr = ck["profiles"][prof[idx[i]]][1] if ck.get("profiles") else \
            ck["sigmas"][prof[idx[i]]]
        img, ext = kernel_slice(ck, musite[site[idx[i]]], pr)
        B.imshow(img, origin="lower", extent=ext, cmap="magma",
                 norm=PowerNorm(0.42, vmin=0, vmax=1))
        B.scatter(off[i, :, 0], off[i, :, 1], s=13, marker="s", c="none",
                  edgecolors="w", linewidths=.6)
        m = musite[site[idx[i]]]
        B.plot(m[0], m[1], "o", mfc="none", mec=C_SPACE, ms=13, mew=2.2)
        B.set_xticks([]); B.set_yticks([])
        B.set_ylabel(f"spike {i+1}", fontsize=9, color="0.3")
        if i == 0:
            B.set_title("$g_s$ — ONE site + profile,\nchosen per spike",
                        fontsize=10, color=C_SPACE, pad=8)

        Cx = fig.add_subplot(gs[i, 3])
        Cx.bar(np.arange(Q), V[idx[i]], color=[qcol[q] for q in range(Q)])
        Cx.axhline(0, color="0.7", lw=.5); Cx.set_xticks([]); Cx.set_yticks([])
        for s_ in Cx.spines.values():
            s_.set_visible(False)
        if i == 0:
            Cx.set_title("$v_s$\nper spike", fontsize=10, color=C_TIME, pad=8)

        E = fig.add_subplot(gs[i, 6:])
        draw_waves(E, Y[i], off[i], C_IN, 1.0, scale=sc)
        draw_waves(E, Yh[i], off[i], C_OUT, 1.35, scale=sc, ls="--")
        if i == 0:
            E.set_title("$\\hat Y_s = g_s\\,(v_s^{\\top}a)$\nmeasured — model ---",
                        fontsize=10, color=C_OUT, pad=8)

    Bx = fig.add_subplot(gs[:, 4:6])
    order = np.argsort(-np.bincount(dom, minlength=Q))
    for r_, q in enumerate(order):
        Bx.plot(np.arange(a.shape[1]) / 30.0, a[q] - r_ * 0.30, color=qcol[q], lw=1.7)
        Bx.text(3.2, -r_ * 0.30, f" $q_{{{q}}}$", color=qcol[q], fontsize=8.5,
                va="center", weight="bold")
    Bx.set_xlim(0, 3.6); Bx.set_yticks([]); Bx.set_xlabel("ms", fontsize=9)
    Bx.tick_params(labelsize=8)
    for s_ in ("top", "right", "left"):
        Bx.spines[s_].set_visible(False)
    Bx.set_title("$a$ — the time basis,\nSHARED by all 2.5 M spikes", fontsize=10,
                 color=C_TIME, pad=8)
    fig.suptitle("$Y_{s,c,t}\;\\approx\;g_s(c)\;\\cdot\;(v_s^{\\top}a)_t$\n"
                 "position is a discrete choice per spike · shape is a shared basis with "
                 "free per-spike weights", fontsize=14.5, y=.965)
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ===================================================================== candidate C
def schematic_C(ctx, out: Path):
    """Factorization on top, then the two factors fan out to the two things they are FOR:
    where the neuron is, and what kind of waveform it has."""
    (rec, ck, k, V, musite, S, site, prof, idx, Y, Yh, off, g, w, dom, qcol) = ctx
    i = 0; a = ck["a"].numpy(); Q = a.shape[0]
    fig = plt.figure(figsize=(15.4, 10.4)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, .1, 1.25],
                          left=.05, right=.97, top=.87, bottom=.06,
                          wspace=.34, hspace=.10)

    A = fig.add_subplot(gs[0, 0]); sc = draw_waves(A, Y[i], off[i], C_IN, 1.3)
    A.set_title("measured spike  $Y_s$", fontsize=11, color=C_IN)
    B = fig.add_subplot(gs[0, 1])
    pr = ck["profiles"][prof[idx[i]]][1] if ck.get("profiles") else ck["sigmas"][prof[idx[i]]]
    img, ext = kernel_slice(ck, musite[site[idx[i]]], pr)
    B.imshow(img, origin="lower", extent=ext, cmap="magma",
             norm=PowerNorm(0.42, vmin=0, vmax=1))
    B.scatter(off[i, :, 0], off[i, :, 1], s=15, marker="s", c="none", edgecolors="w",
              linewidths=.7)
    m = musite[site[idx[i]]]
    B.plot(m[0], m[1], "o", mfc="none", mec=C_SPACE, ms=15, mew=2.4)
    B.set_xticks([]); B.set_yticks([])
    B.set_title("spatial factor  $g_s$", fontsize=11, color=C_SPACE)
    Cx = fig.add_subplot(gs[0, 2])
    Cx.plot(np.arange(len(w[i])) / 30.0, w[i], color=C_TIME, lw=2.2)
    Cx.axhline(0, color="0.75", lw=.6); Cx.set_yticks([]); Cx.set_xlabel("ms", fontsize=9)
    Cx.tick_params(labelsize=8)
    for s_ in ("top", "right"):
        Cx.spines[s_].set_visible(False)
    Cx.set_title("temporal factor  $v_s^{\\top}a$", fontsize=11, color=C_TIME)
    Dx = fig.add_subplot(gs[0, 3])
    draw_waves(Dx, Y[i], off[i], C_IN, 1.1, scale=sc)
    draw_waves(Dx, Yh[i], off[i], C_OUT, 1.45, scale=sc, ls="--")
    Dx.set_title("reconstruction  $\\hat Y_s$", fontsize=11, color=C_OUT)

    L = fig.add_subplot(gs[2, :2])
    sub = ctx_sub(rec, site, musite, dom, 260000)
    L.scatter(sub[0], sub[1], s=2.4, c=qcol[sub[2]], alpha=.55, linewidths=0,
              rasterized=True)
    L.set_xlim(-140, 200); L.set_ylim(380, 920)
    L.set_xlabel("x (µm)", fontsize=9.5); L.set_ylabel("depth y (µm)", fontsize=9.5)
    L.tick_params(labelsize=8)
    L.set_title("WHERE   ·   centroid = anchor + $\\mu_{k_s}$\n"
                "every spike gets a position — drift, depth, unit structure",
                fontsize=11.5, color=C_SPACE)

    R = fig.add_subplot(gs[2, 2:])
    order = np.argsort(-np.bincount(dom, minlength=Q))
    use = np.bincount(dom, minlength=Q) / len(dom)
    for r_, q in enumerate(order[:6]):
        R.plot(np.arange(a.shape[1]) / 30.0, a[q] - r_ * .32, color=qcol[q], lw=2.0)
        R.text(3.15, -r_ * .32, f"  $q_{{{q}}}$  {100*use[q]:.0f}%", color=qcol[q],
               fontsize=9.5, va="center", weight="bold")
    R.set_xlim(0, 3.8); R.set_yticks([]); R.set_xlabel("ms", fontsize=9.5)
    R.tick_params(labelsize=8)
    for s_ in ("top", "right", "left"):
        R.spines[s_].set_visible(False)
    R.set_title("WHAT   ·   $v_s$ over the shared basis\n"
                "the coefficient vector is a waveform-type signature",
                fontsize=11.5, color=C_TIME)

    fig.canvas.draw()
    link(fig, A, B, glyph="\u2248")
    link(fig, B, Cx, glyph="\u00d7")
    link(fig, Cx, Dx)
    bB, bC, bL, bR = (x.get_position() for x in (B, Cx, L, R))
    arrow(fig, ((bB.x0 + bB.x1) / 2, bB.y0 - .012),
          ((bL.x0 + bL.x1) / 2, bL.y1 + .055), colour=C_SPACE, rad=.18, lw=2.4)
    arrow(fig, ((bC.x0 + bC.x1) / 2, bC.y0 - .012),
          ((bR.x0 + bR.x1) / 2, bR.y1 + .055), colour=C_TIME, rad=-.18, lw=2.4)
    fig.suptitle("Single-source tensor factorization of spike waveforms\n"
                 "$Y_{s,c,t}\\approx g_s(c)\\,(v_s^{\\top}a)_t$   —   one factorization, "
                 "two readouts: position and waveform type", fontsize=14.5, y=.975)
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ===================================================================== candidate D
def schematic_D(ctx, out: Path):
    """Same flow as A, but WHERE is the depth-vs-time raster rather than an x-y scatter.

    The x-y view shows lattice quantization (64 levels over +-150 um, on top of 4 discrete
    anchor columns) which reads as noise. Depth against time is what localization is
    actually FOR: the imposed drift is visible directly, and it is the input the motion
    estimate is built from."""
    (rec, ck, k, V, musite, S, site, prof, idx, Y, Yh, off, g, w, dom, qcol) = ctx
    i = 0; a = ck["a"].numpy(); Q = a.shape[0]
    fig = plt.figure(figsize=(16.2, 8.6)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.1, 1.0],
                          left=.04, right=.985, top=.855, bottom=.075,
                          wspace=.45, hspace=.42)

    A = fig.add_subplot(gs[0, 0]); sc = draw_waves(A, Y[i], off[i], C_IN, 1.25)
    A.set_title("one spike\n10 contacts × 90 samples", fontsize=10.5, color=C_IN)
    B = fig.add_subplot(gs[0, 1])
    pr = ck["profiles"][prof[idx[i]]][1] if ck.get("profiles") else ck["sigmas"][prof[idx[i]]]
    img, ext = kernel_slice(ck, musite[site[idx[i]]], pr)
    B.imshow(img, origin="lower", extent=ext, cmap="magma",
             norm=PowerNorm(0.42, vmin=0, vmax=1))
    B.scatter(off[i, :, 0], off[i, :, 1], s=16, marker="s", c="none", edgecolors="w",
              linewidths=.7)
    m = musite[site[idx[i]]]
    B.plot(m[0], m[1], "o", mfc="none", mec=C_SPACE, ms=15, mew=2.4)
    B.set_xticks([]); B.set_yticks([])
    B.set_title(f"SPATIAL  $g_s$\none site chosen from "
                f"{compact_count(int(ck['KS']))}", fontsize=10.5, color=C_SPACE)
    Cx = fig.add_subplot(gs[0, 2])
    Cx.plot(np.arange(len(w[i])) / 30.0, w[i], color=C_TIME, lw=2.1)
    Cx.axhline(0, color="0.75", lw=.6); Cx.set_yticks([]); Cx.set_xlabel("ms", fontsize=9)
    Cx.tick_params(labelsize=8)
    for s_ in ("top", "right"):
        Cx.spines[s_].set_visible(False)
    Cx.set_title("TEMPORAL  $v_s^{\\top}a$\nfree weights on a shared basis",
                 fontsize=10.5, color=C_TIME)
    Dx = fig.add_subplot(gs[0, 3])
    draw_waves(Dx, Yh[i], off[i], C_OUT, 1.35, scale=sc)
    Dx.set_title("reconstruction\n$\\hat Y = g_s\\,(v_s^{\\top}a)$", fontsize=10.5,
                 color=C_OUT)
    E = fig.add_subplot(gs[0, 4:])
    draw_waves(E, Y[i], off[i], C_IN, 1.2, scale=sc)
    draw_waves(E, Yh[i], off[i], C_OUT, 1.5, scale=sc, ls="--")
    e = ((Y[i] - Yh[i]) ** 2).mean() / (Y[i] ** 2).mean()
    E.set_title(f"overlay — measured vs model   (rel. err {e:.2f})", fontsize=10.5)

    # WHERE: depth against recording time, amplitude-weighted
    L = fig.add_subplot(gs[1, :4])
    ylim = (400., 900.)
    pos_y = rec.anchors[rec.spike_channels][:, 1] + musite[site][:, 1]
    t = rec.spike_times / rec.fs
    amp = np.linalg.norm(V, axis=1)
    sel = (pos_y >= ylim[0]) & (pos_y <= ylim[1])
    H, xe, ye = np.histogram2d(t[sel], pos_y[sel], bins=(760, 420),
                               range=[[0, t.max()], list(ylim)], weights=amp[sel])
    L.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
             extent=[0, t.max(), ylim[0], ylim[1]],
             norm=PowerNorm(0.45, vmin=0, vmax=np.percentile(H[H > 0], 99.4)))
    L.set_xlabel("recording time (s)", fontsize=9.5)
    L.set_ylabel("centroid depth y (µm)", fontsize=9.5); L.tick_params(labelsize=8)
    L.set_title("→ WHERE:  every spike localised  ·  depth × time  —  the imposed drift "
                "is visible directly", fontsize=10.5, color=C_SPACE)

    R = fig.add_subplot(gs[1, 4:])
    order = np.argsort(-np.bincount(dom, minlength=Q)); use = np.bincount(dom, minlength=Q)
    for r_, q in enumerate(order[:6]):
        R.plot(np.arange(a.shape[1]) / 30.0, a[q] - r_ * .33, color=qcol[q], lw=1.9)
        R.text(3.1, -r_ * .33, f"  $q_{{{q}}}$ {100*use[q]/use.sum():.0f}%",
               color=qcol[q], fontsize=9, va="center", weight="bold")
    R.set_xlim(0, 3.9); R.set_yticks([]); R.set_xlabel("ms", fontsize=9.5)
    R.tick_params(labelsize=8)
    for s_ in ("top", "right", "left"):
        R.spines[s_].set_visible(False)
    R.set_title("→ WHAT:  $v_s$ over the shared basis\nis a waveform-type signature",
                fontsize=10.5, color=C_TIME)

    fig.canvas.draw()
    link(fig, A, B, glyph="\u2248"); link(fig, B, Cx, glyph="\u00d7"); link(fig, Cx, Dx)
    bB, bC, bL, bR = (x.get_position() for x in (B, Cx, L, R))
    arrow(fig, ((bB.x0 + bB.x1) / 2, bB.y0 - .015),
          (bL.x0 + .13, bL.y1 + .05), colour=C_SPACE, rad=.16, lw=2.2)
    arrow(fig, ((bC.x0 + bC.x1) / 2, bC.y0 - .015),
          ((bR.x0 + bR.x1) / 2, bR.y1 + .05), colour=C_TIME, rad=-.16, lw=2.2)
    fig.suptitle("Single-source tensor factorization of spike waveforms\n"
                 "$Y_{s,c,t}\\approx g_s(c)\\,(v_s^{\\top}a)_t$   —   one factorization, "
                 "two readouts: where the neuron is, and what its waveform is",
                 fontsize=14, y=.975)
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ================================================= candidate A, redrawn (V1 / V2)
def schematic_Av(ctx, out: Path, ylim=(400., 900.), xlim=(-100., 150.), n_pts=26000,
                 mode="scatter", tag_note=""):
    """Candidate A with the localization panel at TRUE aspect and v_s shown explicitly.

    The earlier version stretched a 500 x 250 um region into a wide box, so cluster shape
    was meaningless. `aspect="equal"` makes it portrait and the blobs read as they actually
    are. v_s -- the per-spike coefficient vector, which the time course is built FROM -- now
    gets its own panel instead of only appearing multiplied out."""
    (rec, ck, k, V, musite, S, site, prof, idx, Y, Yh, off, g, w, dom, qcol) = ctx
    i = 0; a = ck["a"].numpy(); Q = a.shape[0]
    use = np.bincount(dom, minlength=Q)
    order = np.argsort(-use)

    fig = plt.figure(figsize=(16.4, 11.2)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 12, height_ratios=[.92, 1.35],
                          left=.04, right=.98, top=.82 if tag_note else .885, bottom=.055,
                          wspace=1.05, hspace=.30)

    A = fig.add_subplot(gs[0, 0:2]); sc = draw_waves(A, Y[i], off[i], C_IN, 1.3)
    A.set_title("one spike\n10 contacts × 90 samples", fontsize=11, color=C_IN)
    B = fig.add_subplot(gs[0, 2:4])
    pr = ck["profiles"][prof[idx[i]]][1] if ck.get("profiles") else ck["sigmas"][prof[idx[i]]]
    img, ext = kernel_slice(ck, musite[site[idx[i]]], pr)
    B.imshow(img, origin="lower", extent=ext, cmap="magma",
             norm=PowerNorm(0.42, vmin=0, vmax=1))
    B.scatter(off[i, :, 0], off[i, :, 1], s=17, marker="s", c="none", edgecolors="w",
              linewidths=.7)
    m = musite[site[idx[i]]]
    B.plot(m[0], m[1], "o", mfc="none", mec=C_SPACE, ms=16, mew=2.5)
    B.set_xticks([]); B.set_yticks([])
    B.set_title(f"SPATIAL  $g_s$\none site from {compact_count(int(ck['KS']))} candidates",
                fontsize=11, color=C_SPACE)
    Cx = fig.add_subplot(gs[0, 4:6])
    Cx.plot(np.arange(len(w[i])) / 30.0, w[i], color=C_TIME, lw=2.2)
    Cx.axhline(0, color="0.75", lw=.6); Cx.set_yticks([]); Cx.set_xlabel("ms", fontsize=9)
    Cx.tick_params(labelsize=8)
    for s_ in ("top", "right"):
        Cx.spines[s_].set_visible(False)
    Cx.set_title("TEMPORAL  $v_s^{\\top}a$\nweights $v_s$ on a shared basis $a$",
                 fontsize=11, color=C_TIME)
    Dx = fig.add_subplot(gs[0, 6:8])
    draw_waves(Dx, Yh[i], off[i], C_OUT, 1.4, scale=sc)
    Dx.set_title("reconstruction\n$\\hat Y = g_s\\,(v_s^{\\top}a)$", fontsize=11,
                 color=C_OUT)
    E = fig.add_subplot(gs[0, 8:])
    draw_waves(E, Y[i], off[i], C_IN, 1.25, scale=sc)
    draw_waves(E, Yh[i], off[i], C_OUT, 1.55, scale=sc, ls="--")
    e = ((Y[i] - Yh[i]) ** 2).mean() / (Y[i] ** 2).mean()
    E.set_title(f"overlay — measured vs model\n(rel. err {e:.2f})", fontsize=11)

    # ---------- WHERE: true aspect, so cluster shape means something
    L = fig.add_subplot(gs[1, 0:4])
    if mode == "density":
        # amplitude-weighted density, the same convention as the aggregate panels. At 2.5 M
        # spikes a scatter saturates every lattice site; density is what shows unit shape.
        pxy = rec.anchors[rec.spike_channels][:, :2] + musite[site][:, :2]
        amp = np.linalg.norm(V, axis=1)
        m_ = ((pxy[:, 1] >= ylim[0]) & (pxy[:, 1] <= ylim[1])
              & (pxy[:, 0] >= xlim[0]) & (pxy[:, 0] <= xlim[1]))
        H, _, _ = np.histogram2d(pxy[m_, 0], pxy[m_, 1],
                                 bins=(int(xlim[1] - xlim[0]) // 2,
                                       int(ylim[1] - ylim[0]) // 2),
                                 range=[list(xlim), list(ylim)], weights=amp[m_])
        L.imshow(H.T, origin="lower", extent=[*xlim, *ylim], cmap="magma", aspect="equal",
                 norm=PowerNorm(.45, vmin=0, vmax=np.percentile(H[H > 0], 99.5)),
                 interpolation="nearest")
        note = f"{int(m_.sum()):,} spikes, amplitude-weighted density"
        ec = "w"
    else:
        sub = ctx_sub(rec, site, musite, dom, n_pts, ylim=ylim)
        keep = (sub[0] >= xlim[0]) & (sub[0] <= xlim[1])
        L.scatter(sub[0][keep], sub[1][keep], s=3.4, c=qcol[sub[2][keep]], alpha=.62,
                  linewidths=0, rasterized=True)
        note = f"{int(keep.sum()):,} spikes shown, coloured by dominant $q$"
        ec = "0.45"
    sel = ((ylim[0] <= rec.channel_locations[:, 1])
           & (rec.channel_locations[:, 1] <= ylim[1]))
    L.scatter(rec.channel_locations[sel, 0], rec.channel_locations[sel, 1], s=9,
              marker="s", c="none", edgecolors=ec, linewidths=.5)
    L.set_xlim(*xlim); L.set_ylim(*ylim); L.set_aspect("equal")
    L.set_xlabel("x (µm)", fontsize=10); L.set_ylabel("depth y (µm)", fontsize=10)
    L.tick_params(labelsize=8.5)
    L.set_title(f"→ WHERE   centroid = anchor + $\\mu_{{k_s}}$\n"
                f"{ylim[0]:.0f}–{ylim[1]:.0f} µm, true aspect · {note}",
                fontsize=10.5, color=C_SPACE)

    # ---------- v_s itself, for the example spike
    Vx = fig.add_subplot(gs[1, 4:6])
    Vx.barh(np.arange(Q), V[idx[i]][order][::-1],
            color=[qcol[q] for q in order[::-1]], height=.72)
    Vx.axvline(0, color="0.6", lw=.7)
    Vx.set_yticks(np.arange(Q))
    Vx.set_yticklabels([f"$q_{{{q}}}$" for q in order[::-1]],
                       fontsize=6.5 if Q > 16 else 8.5)
    Vx.set_xlabel("coefficient", fontsize=9.5); Vx.tick_params(labelsize=8)
    for s_ in ("top", "right"):
        Vx.spines[s_].set_visible(False)
    Vx.set_title("$v_s$ for this spike\nQ numbers per spike", fontsize=11, color=C_TIME)

    # ---------- the shared basis, with how often each component leads
    R = fig.add_subplot(gs[1, 6:])
    basis_show = order[:min(Q, 10)]
    for r_, q in enumerate(basis_show):
        R.plot(np.arange(a.shape[1]) / 30.0, a[q] - r_ * .30, color=qcol[q], lw=1.9)
        R.text(3.1, -r_ * .30, f"  $q_{{{q}}}$   {100*use[q]/use.sum():.0f}%",
               color=qcol[q], fontsize=9.5, va="center", weight="bold")
    R.set_xlim(0, 3.9); R.set_yticks([]); R.set_xlabel("ms", fontsize=9.5)
    R.tick_params(labelsize=8.5)
    for s_ in ("top", "right", "left"):
        R.spines[s_].set_visible(False)
    shown = f"top {len(basis_show)} of Q={Q}" if len(basis_show) < Q else f"Q={Q}"
    R.set_title(f"→ WHAT   the shared basis $a$, {shown}, ranked by spike usage\n"
                "$v_s$ over the full basis is the waveform-type signature",
                fontsize=11, color=C_TIME)

    fig.canvas.draw()
    link(fig, A, B, glyph="\u2248"); link(fig, B, Cx, glyph="\u00d7"); link(fig, Cx, Dx)
    bB, bC, bL, bV = (x.get_position() for x in (B, Cx, L, Vx))
    arrow(fig, ((bB.x0 + bB.x1) / 2, bB.y0 - .012),
          ((bL.x0 + bL.x1) / 2, bL.y1 + .055), colour=C_SPACE, rad=.16, lw=2.3)
    arrow(fig, ((bC.x0 + bC.x1) / 2, bC.y0 - .012),
          ((bV.x0 + bV.x1) / 2, bV.y1 + .055), colour=C_TIME, rad=-.10, lw=2.3)
    fig.suptitle("Single-source tensor factorization of spike waveforms\n"
                 "$Y_{s,c,t}\\approx g_s(c)\\,(v_s^{\\top}a)_t$   —   one factorization, "
                 "two readouts: where the neuron is, and what its waveform is",
                 fontsize=14.5, y=.985)
    if tag_note:
        fig.text(.5, .875, tag_note, ha="center", va="center", fontsize=11.5,
                 color="0.25", weight="bold")
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--tag", default=HEADLINE_TAG)
    ap.add_argument("--tag-note", default=None,
                    help="optional note appended to the headline; the default headline "
                         "fit gets its canonical rigid score automatically")
    ap.add_argument("--out", type=Path, default=REPO / "docs/panels")
    ap.add_argument("--only", default="", help="comma-separated schematic names")
    a_ = ap.parse_args()
    rec = D.load("np1")
    off_all = rec.channel_offsets().astype(np.float32)
    ck, k, V, musite, S, site, prof = load_fit(a_.runs, a_.tag)
    dom = np.argmax(np.abs(V), 1); Q = V.shape[1]
    order = np.argsort(-np.bincount(dom, minlength=Q))
    rank = np.empty(Q, int); rank[order] = np.arange(Q)
    qcol = plt.get_cmap("turbo")(np.linspace(.06, .94, Q))[rank]
    idx = pick_spikes(rec, off_all, ck, k, V, musite, S)
    Y, Yh, off, g, w = rendered(rec, off_all, ck, k, V, musite, S, idx)
    ctx = (rec, ck, k, V, musite, S, site, prof, idx, Y, Yh, off, g, w, dom, qcol)
    # the README headline; the alternates below were the other candidates considered
    tag_note = (a_.tag_note if a_.tag_note is not None else
                HEADLINE_NOTE if a_.tag == HEADLINE_TAG else "")
    jobs = [("", lambda c, o: schematic_Av(c, o, (400., 900.), (-100., 150.),
                                           n_pts=26000, tag_note=tag_note)),
            ("_flow_density", lambda c, o: schematic_Av(c, o, (400., 900.),
                                                        (-100., 150.), mode="density")),
            ("_flow_tight", lambda c, o: schematic_Av(c, o, (500., 700.), (-80., 130.),
                                                      n_pts=14000)),
            ("_threespike", schematic_B), ("_branch", schematic_C),
            ("_drift", schematic_D), ("_xy", schematic_A)]
    only = set(a_.only.split(",")) if a_.only else {""}   # default: just the headline
    for nm, fn in jobs:
        if only and nm not in only:
            continue
        p = a_.out / f"schematic{nm}.png"
        fn(ctx, p); print(f"  wrote {p}", flush=True)


if __name__ == "__main__":
    main()
