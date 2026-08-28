#!/usr/bin/env python3
"""Centroid/time-basis views for every method in the lattice browser.

For learned codebooks, a spike's global centroid is

    anchor_xy[s] + mu_site[chosen_site[s], :2]

and its displayed time-basis label is ``argmax_q |v[s,q]|``.  Colours are a
deterministic shuffled palette indexed by the original q (not by popularity),
and small deterministic jitter is applied only to rendering coordinates.

Reference localizers have no learned time basis.  They are still rendered so
that every browser method has the same panels, using one labelled neutral
reference colour.

Outputs per method under ``figs/<tag>/``:
  centroid_basis_{full,zoom}.png
  centroid_basis_movie_{full,zoom}.mp4
  depth_time_density_{full,zoom}.png
  depth_time_basis_{full,zoom}.png
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D
from spiketensor.drift import SUFFIX, correction  # noqa: E402


LEARNED_FILES = (
    "centroid_basis_full.png", "centroid_basis_zoom.png",
    "centroid_basis_movie_full.mp4", "centroid_basis_movie_zoom.mp4",
    "depth_time_density_full.png", "depth_time_density_zoom.png",
    "depth_time_basis_full.png", "depth_time_basis_zoom.png",
)
REFERENCE_TAGS = {
    "BASELINE_monopole": "monopole",
    # the same monopole localizer, measured at stride 2 earlier in the project; kept so
    # every browser row renders, though BASELINE_monopole supersedes it
    "MONOPOLE_matched": "monopole",
    "CONTROL_anchor_only_ptp": "anchor_ptp",
    "CONTROL_anchor_only_flat": "anchor_flat",
}
XY_LIM = (-70.0, 130.0)
FULL_Y = (0.0, 3840.0)
ZOOM_Y = (400.0, 900.0)
BG = np.array([15, 17, 21], dtype=np.float32) / 255.0


def stable_seed(tag: str, seed: int) -> int:
    raw = hashlib.sha1(f"{tag}:{seed}".encode()).digest()[:8]
    return int.from_bytes(raw, "little")


def shuffled_palette(q: int, reference: bool = False) -> np.ndarray:
    """High-contrast, deterministic q colours in deliberately shuffled hue order."""
    if reference:
        return np.array([[0.35, 0.78, 0.98]], np.float32)
    rng = np.random.default_rng(20260804 + q)
    hue = (0.07 + np.arange(q) * 0.618033988749895) % 1.0
    rng.shuffle(hue)
    out = []
    for i, h in enumerate(hue):
        sat = 0.68 + 0.22 * ((i * 7) % 3) / 2
        val = 0.82 + 0.16 * ((i * 5 + 1) % 3) / 2
        out.append(colorsys.hsv_to_rgb(float(h), float(sat), float(val)))
    return np.asarray(out, np.float32)


def load_learned(runs: Path, tag: str, rec) -> tuple[np.ndarray, ...]:
    ck = torch.load(runs / f"codebook_{tag}.pt", map_location="cpu",
                    weights_only=False)
    z = np.load(runs / f"pi_{tag}.npz")
    if "pos" in z.files:
        # centroid-anchored learned-basis fit: positions are explicit
        k = z["k"].astype(np.int64)
        V = z["v"]
        amp = np.linalg.norm(V, axis=1).astype(np.float32)
        pos = z["pos"].astype(np.float32)
        colors, evr = pca_rgb_local(V, pos[:, 1])
        return (pos,
                np.arange(len(V), dtype=np.int64), amp, colors,
                {"Q": V.shape[1], "pca_evr": evr})
    choice = z["k"].astype(np.int64)
    coeff = z["v"].astype(np.float32)
    amp = np.linalg.norm(coeff, axis=1).astype(np.float32)
    dom = np.arange(len(coeff), dtype=np.int64)
    if "mu_site" in z.files:
        site = choice // int(z["S"])
        local = z["mu_site"].astype(np.float32)[site]
    else:
        local = z["mu"].astype(np.float32)[choice]
    anc = rec.anchors[rec.spike_channels][:, :2].astype(np.float32)
    pos = anc + local[:, :2]
    # colour is keyed on the UNCORRECTED depth, so a spike keeps one colour in
    # every panel regardless of which drift correction the panel applies
    colors, evr = pca_rgb_local(coeff, pos[:, 1])
    del coeff
    return pos, dom, amp, colors, {"Q": int(ck["Q"]), "pca_evr": evr}


def measured_ptp(rec, chunk: int = 20000) -> np.ndarray:
    out = np.empty(rec.n_spikes, np.float32)
    for i in range(0, rec.n_spikes, chunk):
        w = np.asarray(rec.waveforms[i:i + chunk])
        out[i:i + len(w)] = np.ptp(w, axis=2).max(1)
    return out


def load_reference(tag: str, rec, ptp: np.ndarray | None) -> tuple[np.ndarray, ...]:
    kind = REFERENCE_TAGS[tag]
    if kind == "monopole":
        pos = rec.mp_xyz[:, :2].astype(np.float32).copy()
        amp = ptp
    else:
        pos = rec.anchors[rec.spike_channels][:, :2].astype(np.float32).copy()
        amp = ptp if kind == "anchor_ptp" else np.ones(rec.n_spikes, np.float32)
    dom = np.zeros(rec.n_spikes, np.int16)
    meta = {"Q": 0, "reference": kind}
    return pos, dom, amp.astype(np.float32), shuffled_palette(1, reference=True), meta


def pca_projection(V: np.ndarray, seed: int = 0):
    """Return the canonical deterministic 3-PC shape projection.

    The fitted transform is intentionally shared by RGB rasters/movies and the
    PCA embedding panel for one model.  Fitting on at most 300k rows keeps the
    full-recording path bounded while projecting every source event through the
    same axes.
    """
    Vn = np.asarray(V, np.float32)
    Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-9)
    n = len(Vn)
    rng = np.random.default_rng(seed)
    fit = Vn[rng.choice(n, min(300_000, n), replace=False)] if n > 300_000 else Vn
    mu = fit.mean(0)
    _, sv, vt = np.linalg.svd(fit - mu, full_matrices=False)
    P = (Vn - mu) @ vt[:3].T
    evr = (sv[:3] ** 2 / (sv ** 2).sum()).astype(float)
    return P, evr


def pca_rgb(V: np.ndarray, seed: int = 0):
    """Per-spike colours from a 3-component PCA of the shape coefficients.

    Each v_s is L2-normalised first (colour encodes SHAPE, not loudness -- ||v||
    spans ~240x and would swamp everything), then projected onto the top three
    principal components, each robust-scaled (1st..99th percentile) into an RGB
    channel. Nearby waveform shapes get nearby colours, and the three channels
    preserve the maximum shape variance a 3-D colour space can carry -- unlike the
    old argmax_q labelling, which collapsed 8+ dimensions onto discrete hues.
    Returns (colors (N,3) float32 in [0.06, 0.97], explained-variance fractions)."""
    P, evr = pca_projection(V, seed=seed)
    cols = np.empty((len(P), 3), np.float32)
    for c in range(3):
        lo, hi = np.percentile(P[:, c], [1, 99])
        cols[:, c] = np.clip((P[:, c] - lo) / max(hi - lo, 1e-9), 0, 1)
    return 0.06 + 0.91 * cols, evr


def pca_rgb_local(V, y, block_um=200.0, step_um=100.0, min_n=1500, seed=0,
                  rank=True):
    """Per-spike RGB from a PCA fitted inside overlapping depth blocks.

    The global pca_rgb spends most of the colour cube separating the recording's
    dominant shape axes, so two units 50 um apart that differ subtly can end up nearly
    the same colour. Fitting the PCA inside a ~200 um depth neighbourhood and rescaling
    there re-expands whatever variance is LOCAL, which is what the depth rasters are
    actually read for.

    Two things have to be handled or the result is artefact, not signal:

    * SIGN/ROTATION AMBIGUITY. Each block's PCA is defined only up to an orthogonal
      transform of its 3 components, so independent fits would flip colours at block
      boundaries. Every block basis is therefore aligned to the global basis by
      orthogonal Procrustes. Hue keeps a roughly consistent meaning across depth while
      the SCALING stays local -- that is the part that buys contrast.
    * SEAMS. Blocks overlap by (block_um - step_um) and each spike's colour is a
      triangular-weighted blend of the blocks covering it, so colour varies smoothly
      with depth rather than stepping at boundaries.

    `rank=True` maps each component to its within-block RANK rather than linearly
    rescaling it. PC projections are roughly Gaussian, so linear scaling parks most
    spikes near the middle of the colour cube and the raster washes out to grey; the
    rank (histogram-equalising) transform spreads them across the whole cube. Measured
    on lat64_monopole_Q32, median within-50 um colour distance: 0.437 global+linear,
    0.463 local+linear, 0.546 global+rank, 0.583 local+rank -- the scaling matters more
    than the locality, and the two compose.

    Blocks with fewer than `min_n` spikes fall back to the global projection. `y` should
    be the UNCORRECTED depth so a spike keeps one colour across every panel.
    Returns (colors (N,3) float32, mean per-block explained-variance of PC1..3).
    """
    Vn = np.asarray(V, np.float32)
    Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-9)
    y = np.asarray(y, np.float64)
    n = len(Vn)
    rng = np.random.default_rng(seed)

    def fit(X):
        mu = X.mean(0)
        _, sv, vt = np.linalg.svd(X - mu, full_matrices=False)
        k = min(3, vt.shape[0])
        W = np.zeros((3, X.shape[1]), np.float32); W[:k] = vt[:k]
        evr = np.zeros(3); tot = (sv ** 2).sum()
        if tot > 0:
            evr[:k] = (sv[:k] ** 2 / tot)
        return mu.astype(np.float32), W, evr

    sub = Vn[rng.choice(n, min(300_000, n), replace=False)] if n > 300_000 else Vn
    g_mu, g_W, _ = fit(sub)

    lo, hi = np.percentile(y, [0.05, 99.95])
    centers = np.arange(lo, hi + step_um, step_um) if hi > lo else np.array([lo])
    acc = np.zeros((n, 3), np.float32)
    wsum = np.zeros(n, np.float32)
    evrs = []
    half = block_um / 2.0
    for c in centers:
        m = np.flatnonzero(np.abs(y - c) <= half)
        if len(m) < min_n:
            continue
        b_mu, b_W, evr = fit(Vn[m])
        # orthogonal Procrustes: rotate the block basis onto the global one so hue does
        # not flip arbitrarily between neighbouring blocks
        U, _, Vt = np.linalg.svd(b_W @ g_W.T)
        b_W = (U @ Vt).T @ b_W
        P = (Vn[m] - b_mu) @ b_W.T
        col = np.empty_like(P)
        for k in range(3):
            if rank:
                order = np.argsort(np.argsort(P[:, k]))
                col[:, k] = order / max(len(order) - 1, 1)
            else:
                p_lo, p_hi = np.percentile(P[:, k], [1, 99])
                col[:, k] = np.clip((P[:, k] - p_lo) / max(p_hi - p_lo, 1e-9), 0, 1)
        w = np.maximum(0.0, 1.0 - np.abs(y[m] - c) / half).astype(np.float32) + 1e-3
        acc[m] += col.astype(np.float32) * w[:, None]
        wsum[m] += w
        evrs.append(evr)

    out = np.empty((n, 3), np.float32)
    ok = wsum > 0
    out[ok] = acc[ok] / wsum[ok, None]
    if (~ok).any():                      # sparse depths: fall back to the global fit
        P = (Vn[~ok] - g_mu) @ g_W.T
        for k in range(3):
            if rank and len(P) > 1:
                out[~ok, k] = np.argsort(np.argsort(P[:, k])) / (len(P) - 1)
            else:
                p_lo, p_hi = np.percentile(P[:, k], [1, 99]) if len(P) > 10 else (0.0, 1.)
                out[~ok, k] = np.clip((P[:, k] - p_lo) / max(p_hi - p_lo, 1e-9), 0, 1)
    evr = np.mean(evrs, 0) if evrs else np.zeros(3)
    return (0.06 + 0.91 * out).astype(np.float32), evr


def jittered(pos: np.ndarray, tag: str, seed: int, jitter_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(tag, seed))
    x = pos[:, 0] + rng.uniform(-jitter_um, jitter_um, len(pos)).astype(np.float32)
    y = pos[:, 1] + rng.uniform(-jitter_um, jitter_um, len(pos)).astype(np.float32)
    jt = rng.uniform(-0.18, 0.18, len(pos)).astype(np.float32)
    return x, y, jt


def colour_raster(x, y, label, palette, xlim, ylim, nx, ny,
                  sigma_px=0.55, point_weight=None) -> np.ndarray:
    """Rasterize points; optional weights preserve fractional source mass.

    Single-source callers omit ``point_weight`` and retain the historical unit
    count.  Multipole callers pass q, so two active sources contribute q1 and q2
    rather than two unit counts; q1 + q2 = 1 for every parent spike.
    """
    ix = np.floor((x - xlim[0]) / (xlim[1] - xlim[0]) * nx).astype(np.int64)
    iy = np.floor((y - ylim[0]) / (ylim[1] - ylim[0]) * ny).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]
    n = nx * ny
    weight = (np.ones(len(x), np.float32) if point_weight is None
              else np.asarray(point_weight, np.float32))
    if weight.shape != np.shape(x):
        raise ValueError("point_weight must match point coordinates")
    count = np.bincount(flat, weights=weight[ok], minlength=n).astype(np.float32)
    rgb = np.empty((ny, nx, 3), np.float32)
    for c in range(3):
        val = palette[label[ok], c] * weight[ok]
        rgb[..., c] = np.bincount(flat, weights=val, minlength=n).reshape(ny, nx)
    count2 = count.reshape(ny, nx)
    rgb /= np.maximum(count2[..., None], 1.0)
    intensity = 1.0 - np.exp(-count2 / 1.8)
    if sigma_px:
        intensity = gaussian_filter(intensity, sigma_px)
        for c in range(3):
            weighted = gaussian_filter(rgb[..., c] * count2, sigma_px)
            mass = gaussian_filter(count2, sigma_px)
            rgb[..., c] = weighted / np.maximum(mass, 1e-7)
    intensity = np.clip(intensity * 1.3, 0, 1)
    return BG[None, None, :] * (1 - intensity[..., None]) + rgb * intensity[..., None]


def add_contacts(ax, rec, ylim):
    m = ((rec.channel_locations[:, 1] >= ylim[0])
         & (rec.channel_locations[:, 1] <= ylim[1]))
    ax.scatter(rec.channel_locations[m, 0], rec.channel_locations[m, 1], s=8,
               marker="s", facecolors="none", edgecolors="0.72", linewidths=.45)


PCA_NOTE = ("colour = RGB(PC1, PC2, PC3) of a 3-component PCA of unit-normalised $v_s$, "
            "fitted in overlapping 200 µm depth blocks and rank-equalised")


def add_palette_legend(ax, palette, reference: bool, anchor_y: float = -0.07):
    q = len(palette)
    if q > 64 and not reference:      # per-spike PCA colours: a swatch legend is meaningless
        ax.text(.5, anchor_y + .035, PCA_NOTE, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.8, color="0.35")
        return
    if reference:
        ax.scatter([], [], s=18, color=palette[0], label="reference: no learned basis")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, anchor_y), fontsize=6.5,
                  frameon=False)
        return
    handles = [ax.scatter([], [], s=17, color=palette[i], label=f"q{i}") for i in range(q)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, anchor_y),
              ncol=min(8, q), fontsize=5.8, frameon=False, handletextpad=.25,
              columnspacing=.7)


def save_centroid(path, x, y, dom, palette, rec, ylim, tag, reference,
                  point_weight=None):
    # One micrometre pixels retain the requested small visual jitter before imshow.
    nx = int(XY_LIM[1] - XY_LIM[0])
    ny = int(ylim[1] - ylim[0])
    rgb = colour_raster(x, y, dom, palette, XY_LIM, ylim, nx, ny,
                        point_weight=point_weight)
    full = ylim == FULL_Y
    fig, ax = plt.subplots(figsize=(7.45, 11.45) if full else (9.1, 7.15),
                           constrained_layout=True)
    ax.imshow(rgb, origin="lower", extent=[*XY_LIM, *ylim], aspect="equal",
              interpolation="nearest")
    add_contacts(ax, rec, ylim)
    ax.set_xlim(*XY_LIM); ax.set_ylim(*ylim)
    ax.set_xlabel("lateral x (µm)"); ax.set_ylabel("probe depth y (µm)")
    weighted_note = "; intensity = fractional source mass" if point_weight is not None else ""
    colour_note = (PCA_NOTE if len(palette) > 64
                   else "colour = dominant time basis argmax_q |v_q|")
    ax.set_title(f"all spike localization centroids — {tag}\n"
                 f"visual jitter ±1.5 µm; {colour_note}{weighted_note}"
                 if not reference else
                 f"all spike localization centroids — {tag}\n"
                 f"visual jitter ±1.5 µm; reference has no learned time basis",
                 fontsize=10)
    add_palette_legend(ax, palette, reference)
    fig.savefig(path, dpi=154, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_density_raster(path, t, y, amp, ylim, tag):
    nt = 979                       # two seconds per displayed pixel
    ny = int((ylim[1] - ylim[0]) / 4.0)
    H, _, _ = np.histogram2d(y, t, bins=(ny, nt),
                             range=[ylim, (0.0, 1958.0)], weights=amp)
    H = gaussian_filter(H.astype(np.float32), (.65, .35))
    nz = H[H > 0]
    vmax = float(np.percentile(nz, 99.7)) if nz.size else 1.0
    fig, ax = plt.subplots(figsize=(12.8, 7.2 if ylim == FULL_Y else 5.1),
                           constrained_layout=True)
    im = ax.imshow(H, origin="lower", extent=[0, 1958, *ylim], aspect="auto",
                   cmap="magma", norm=PowerNorm(.45, vmin=0, vmax=vmax),
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=.012)
    cb.set_label("summed model amplitude")
    ax.set_xlabel("recording time (s)"); ax.set_ylabel("centroid depth y (µm)")
    ax.set_title(f"depth × time — {tag}\namplitude-weighted density; same magma / "
                 "power-law convention as aggregate views", fontsize=10)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def save_basis_raster(path, t, y, dom, palette, ylim, tag, reference,
                      point_weight=None):
    nt = 979
    ny = int((ylim[1] - ylim[0]) / 4.0)
    rgb = colour_raster(t, y, dom, palette, (0.0, 1958.0), ylim, nt, ny,
                        sigma_px=.42, point_weight=point_weight)
    fig, ax = plt.subplots(figsize=(12.8, 7.2 if ylim == FULL_Y else 5.1),
                           constrained_layout=True)
    ax.imshow(rgb, origin="lower", extent=[0, 1958, *ylim], aspect="auto",
              interpolation="nearest")
    ax.set_xlabel("recording time (s)"); ax.set_ylabel("centroid depth y (µm)")
    weighted_note = "; intensity = fractional source mass" if point_weight is not None else ""
    cnote = (PCA_NOTE if len(palette) > 64 else "colour = dominant time basis")
    ax.set_title(f"depth × time centroid scatter — {tag}\n"
                 + (f"visual jitter: ±0.18 s, ±1.5 µm; {cnote}" + weighted_note
                    if not reference else
                    "visual jitter: ±0.18 s, ±1.5 µm; reference has no learned basis"),
                 fontsize=10)
    add_palette_legend(ax, palette, reference, anchor_y=-0.13)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def _movie_layout(ylim):
    if ylim == FULL_Y:
        # Leave room below the physically scaled 200 x 3840 um probe panel for
        # the x-axis label.  The original 1240 px canvas clipped its last rows.
        return {"W": 420, "H": 1320, "left": 125, "top": 90,
                "pw": 64, "ph": 1152}
    return {"W": 640, "H": 720, "left": 180, "top": 65,
            "pw": 240, "ph": 600}


def _base_movie_frame(layout, rec, ylim, palette, reference):
    im = Image.new("RGB", (layout["W"], layout["H"]), tuple((BG * 255).astype(int)))
    d = ImageDraw.Draw(im)
    L, T, pw, ph = layout["left"], layout["top"], layout["pw"], layout["ph"]
    d.rectangle([L, T, L + pw - 1, T + ph - 1], outline=(70, 75, 84), width=1)
    m = ((rec.channel_locations[:, 1] >= ylim[0])
         & (rec.channel_locations[:, 1] <= ylim[1]))
    for x, y in rec.channel_locations[m, :2]:
        px = int(L + (x - XY_LIM[0]) / (XY_LIM[1] - XY_LIM[0]) * pw)
        py = int(T + ph - (y - ylim[0]) / (ylim[1] - ylim[0]) * ph)
        if L <= px < L + pw and T <= py < T + ph:
            d.rectangle([px - 1, py - 1, px + 1, py + 1], outline=(135, 140, 148))
    # Pillow's portable bitmap font does not contain the micro sign.
    d.text((L, T + ph + 15), "lateral x (um)", fill=(220, 222, 227))
    d.text((12, T + ph // 2), "probe depth y (um)", fill=(220, 222, 227))
    legend_x = L + pw + 28
    if len(palette) > 64 and not reference:
        d.text((legend_x, T), "colour =", fill=(220, 222, 227))
        d.text((legend_x, T + 16), "RGB(PC1..3)", fill=(220, 222, 227))
        d.text((legend_x, T + 32), "of v_s PCA", fill=(220, 222, 227))
        return np.asarray(im).copy()
    d.text((legend_x, T), "reference" if reference else "dominant basis", fill=(220, 222, 227))
    for q, c in enumerate(palette):
        yy = T + 18 + q * 16
        if yy + 12 >= layout["H"]:
            break
        cc = tuple(np.clip(c * 255, 0, 255).astype(np.uint8))
        d.rectangle([legend_x, yy, legend_x + 10, yy + 10], fill=cc)
        d.text((legend_x + 15, yy - 1), "none" if reference else f"q{q}",
               fill=(205, 208, 214))
    return np.asarray(im).copy()


def save_movies(requests, x, y, dom, palette, sec, rec, tag, fps, reference,
                point_weight=None):
    """Write multiple standard centroid canvases from one shared event-bin pass."""
    order = np.argsort(sec, kind="stable")
    counts = np.bincount(sec, minlength=int(sec.max()) + 1)
    offs = np.concatenate([[0], np.cumsum(counts)])
    bins = np.flatnonzero(counts > 0)
    outputs = []
    for path, ylim in requests:
        layout = _movie_layout(ylim)
        W, H = layout["W"], layout["H"]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
               "-crf", "22", "-threads", "2", "-pix_fmt", "yuv420p",
               str(path)]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        outputs.append((path, ylim, layout,
                        _base_movie_frame(layout, rec, ylim, palette, reference),
                        process))
    font = ImageFont.load_default()
    try:
        for ii, b in enumerate(bins):
            idx = order[offs[b]:offs[b + 1]]
            if point_weight is None:
                all_col = np.clip(palette[dom[idx]] * 255, 0, 255).astype(np.uint8)
            else:
                # Display intensity follows fractional source mass.  The square
                # root keeps small but real secondary sources visible in movies.
                strength = np.sqrt(np.clip(np.asarray(point_weight)[idx], 0, 1))[:, None]
                rgb = BG[None, :] + (palette[dom[idx]] - BG[None, :]) * strength
                all_col = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            for path, ylim, layout, base, process in outputs:
                W, H = layout["W"], layout["H"]
                L, T = layout["left"], layout["top"]
                pw, ph = layout["pw"], layout["ph"]
                frame = base.copy()
                ix = np.floor(L + (x[idx] - XY_LIM[0]) /
                              (XY_LIM[1] - XY_LIM[0]) * pw).astype(int)
                iy = np.floor(T + ph - (y[idx] - ylim[0]) /
                              (ylim[1] - ylim[0]) * ph).astype(int)
                ok = (ix >= L) & (ix < L + pw) & (iy >= T) & (iy < T + ph)
                ix, iy, col = ix[ok], iy[ok], all_col[ok]
                # Direct vectorized 3x3 dots. Later assignments only affect exact
                # overlaps; deterministic jitter exposes different source colours.
                for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                    xx = np.clip(ix + dx, L, L + pw - 1)
                    yy = np.clip(iy + dy, T, T + ph - 1)
                    frame[yy, xx] = col
                pim = Image.fromarray(frame)
                dr = ImageDraw.Draw(pim)
                dr.rectangle([0, 0, W, 52], fill=tuple((BG * 255).astype(int)))
                dr.text((12, 10), f"{tag}  |  t={b + 0.5:7.1f} s  |  "
                        + ("reference (no basis)" if reference else
                           ("colour = RGB(v_s PCA 1..3)" if len(palette) > 64
                            else "colour = dominant time basis") +
                           (" · intensity = source q"
                            if point_weight is not None else "")),
                        fill=(235, 237, 241), font=font)
                process.stdin.write(np.asarray(pim, dtype=np.uint8).tobytes())
            if ii and ii % 400 == 0:
                names = ", ".join(path.name for path, *_ in outputs)
                print(f"    {names}: {ii}/{len(bins)}", flush=True)
    finally:
        for _, _, _, _, process in outputs:
            if process.stdin:
                process.stdin.close()
        failures = [(path, process.wait())
                    for path, _, _, _, process in outputs]
    failed = [(path, code) for path, code in failures if code]
    if failed:
        raise RuntimeError(f"ffmpeg failed for {failed}")


def save_movie(path, x, y, dom, palette, sec, rec, ylim, tag, fps, reference,
               point_weight=None):
    save_movies([(path, ylim)], x, y, dom, palette, sec, rec, tag, fps,
                reference, point_weight=point_weight)


def complete_sfx(out: Path, sfx: str) -> bool:
    return all((out / f.replace(".", sfx + ".", 1)).exists() for f in LEARNED_FILES)


def complete(out: Path) -> bool:
    return all((out / f).exists() and (out / f).stat().st_size > 0 for f in LEARNED_FILES)


def render_one(tag, runs, figs, rec, ptp, args):
    out = figs / tag
    out.mkdir(parents=True, exist_ok=True)
    sfx = SUFFIX[getattr(args, "correct", "none") or "none"]
    if sfx and complete_sfx(out, sfx) and not args.force:
        print(f"skip complete {tag}{sfx}", flush=True)
        return {"tag": tag, "status": "existing"}
    if not sfx and complete(out) and not args.force:
        print(f"skip complete {tag}", flush=True)
        return {"tag": tag, "status": "existing"}
    reference = tag in REFERENCE_TAGS
    if reference:
        pos, dom, amp, palette, meta = load_reference(tag, rec, ptp)
    else:
        pos, dom, amp, palette, meta = load_learned(runs, tag, rec)
    x, y, jt = jittered(pos, tag, args.seed, args.jitter_um)
    if sfx:
        # subtract this fit's own DREDge trace: the corrected panels are the test of
        # whether the motion the model implies is the motion that is actually there
        y = y - correction(figs, tag, args.correct, rec.spike_times, rec.fs, y=pos[:, 1])
    t = rec.spike_times.astype(np.float64) / rec.fs + jt
    sec = np.floor(rec.spike_times / rec.fs).astype(np.int64)
    print(f"{tag}: {len(pos):,} spikes, Q={meta.get('Q', 0)}"
          + (f" · PCA colours (evr {np.round(meta['pca_evr'], 3)})"
             if 'pca_evr' in meta else ""), flush=True)

    save_centroid(out / f"centroid_basis_full{sfx}.png", x, y, dom, palette, rec,
                  FULL_Y, tag, reference)
    save_centroid(out / f"centroid_basis_zoom{sfx}.png", x, y, dom, palette, rec,
                  ZOOM_Y, tag, reference)
    save_density_raster(out / f"depth_time_density_full{sfx}.png", t, y, amp,
                        FULL_Y, tag)
    save_density_raster(out / f"depth_time_density_zoom{sfx}.png", t, y, amp,
                        ZOOM_Y, tag)
    save_basis_raster(out / f"depth_time_basis_full{sfx}.png", t, y, dom, palette,
                      FULL_Y, tag, reference)
    save_basis_raster(out / f"depth_time_basis_zoom{sfx}.png", t, y, dom, palette,
                      ZOOM_Y, tag, reference)
    if not args.skip_movies:
        save_movie(out / f"centroid_basis_movie_full{sfx}.mp4", x, y, dom, palette,
                   sec, rec, FULL_Y, tag, args.fps, reference)
        save_movie(out / f"centroid_basis_movie_zoom{sfx}.mp4", x, y, dom, palette,
                   sec, rec, ZOOM_Y, tag, args.fps, reference)
    return {"tag": tag, "status": "generated", "reference": reference,
            "Q": 0 if reference else int(meta["Q"]),
            "jitter_um": args.jitter_um,
            "palette": (palette.tolist() if len(palette) <= 64
                        else f"per-spike PCA RGB ({len(palette)} rows)"),
            "files": [f for f in LEARNED_FILES if (out / f).exists()]}


def write_manifest(path, records, tags, args):
    path.write_text(json.dumps({
        "complete": len(records) == len(tags)
                    and all(r["status"] in {"generated", "existing"} for r in records),
        "n_methods": len(tags), "n_finished": len(records),
        "definitions": {
            "centroid": "anchor_xy + learned chosen codebook site mu_xy",
            "time_basis": "argmax_q abs(v_s,q)",
            "colour": "deterministic shuffled categorical palette indexed by original q",
            "jitter": f"display only: +/-{args.jitter_um:g} um in x/y and +/-0.18 s",
            "reference_colour": "single neutral colour; reference methods have no learned time basis",
        },
        "records": records,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "zncc/runs/lattice")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/lattice")
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--correct", choices=["none", "soft", "hard", "real-rigid", "real-nonrigid"], default="none",
                    help="subtract this fit's own DREDge motion before rendering; "
                         "output names get a _drr / _drn suffix")
    ap.add_argument("--jitter-um", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--skip-movies", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    learned = sorted(p.stem[len("codebook_"):] for p in a.runs.glob("codebook_*.pt"))
    tags = a.tags or learned + list(REFERENCE_TAGS)
    missing = [t for t in tags if t not in REFERENCE_TAGS
               and not (a.runs / f"codebook_{t}.pt").exists()]
    if missing:
        raise SystemExit(f"missing codebooks: {missing}")
    a.figs.mkdir(parents=True, exist_ok=True)
    rec = D.load("np1")
    need_ptp = any(t in REFERENCE_TAGS and REFERENCE_TAGS[t] != "anchor_flat"
                   for t in tags)
    ptp = measured_ptp(rec) if need_ptp else None
    records = []
    manifest = a.figs / "centroid_basis_manifest.json"
    start = time.time()
    for i, tag in enumerate(tags, 1):
        print(f"[{i}/{len(tags)}] {tag}", flush=True)
        try:
            records.append(render_one(tag, a.runs, a.figs, rec, ptp, a))
        except FileNotFoundError as e:
            # a correction source that has not been computed yet (e.g. dredge_real.npz)
            # should skip the row, not kill the whole batch
            print(f"  skip {tag}: {e}", flush=True)
            continue
        write_manifest(manifest, records, tags, a)
    print(f"wrote {manifest}; {len(tags)} methods in {(time.time()-start)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()


AGG_X = (-70.0, 130.0)


def save_aggregate(path: Path, x, y, amp, sec, rec, ylim, tag: str, note: str,
                   t0: int = 1200, res: float = 2.0, blur_um: float = 4.0,
                   gamma: float = 0.45) -> None:
    """ONE second of localizations, amplitude-weighted, blurred -- for EVERY model.

    Every row in the browser renders this panel through this one function, so a contact
    sheet of `aggregate_1s.png` across models compares like with like. That is the whole
    point: it previously differed by family (the lattice path drew three panels -- a
    kernel-stamp SOFT view, this blurred view, and a monopole reference -- while the
    multi-source path drew two, this view plus a monopole reference), so the same
    filename meant a different picture depending on which model you were looking at.

    Deliberately ONE panel showing only the model's own localizations. The monopole
    reference is a separate row in the browser and repeating it inside every model's
    panel wasted half the figure on an image identical everywhere. The SOFT kernel-stamp
    view is gone because it has no counterpart for the multi-source families, so it could
    not be part of a matched set.

    `x`, `y`, `amp` are per-EVENT (a multi-source spike contributes one entry per active
    source, amplitude already split); `sec` is each event's integer second.
    """
    x = np.asarray(x); y = np.asarray(y); amp = np.asarray(amp)
    m = np.asarray(sec) == t0
    x_lo, x_hi = AGG_X
    nx = int((x_hi - x_lo) / res); ny = int((ylim[1] - ylim[0]) / res)
    img = np.zeros((ny, nx))
    ix = np.floor((x[m] - x_lo) / res).astype(int)
    iy = np.floor((y[m] - ylim[0]) / res).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(img, (iy[ok], ix[ok]), amp[m][ok])
    img = gaussian_filter(img, blur_um / res)

    h = 13.0 if ylim[1] - ylim[0] > 1000 else 7.0
    fig, ax = plt.subplots(figsize=(4.2, h + 1.0), constrained_layout=True)
    s = img[img > 0]
    v = np.percentile(s, 99.7) if s.size else 1.0
    ax.imshow(img, origin="lower", extent=[x_lo, x_hi, *ylim], cmap="magma",
              aspect="equal", norm=PowerNorm(gamma, vmin=0, vmax=v),
              interpolation="nearest")
    sel = ((ylim[0] <= rec.channel_locations[:, 1])
           & (rec.channel_locations[:, 1] <= ylim[1]))
    ax.scatter(rec.channel_locations[sel, 0], rec.channel_locations[sel, 1],
               s=6, marker="s", c="none", edgecolors="w", linewidths=.4, alpha=.6)
    ax.set_ylim(*ylim); ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("x (µm)", fontsize=8); ax.set_ylabel("depth y (µm)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.suptitle(f"aggregate, t = {t0}–{t0 + 1} s — {tag}\n"
                 f"{int(m.sum()):,} events · amplitude-weighted · {blur_um:.0f} µm blur"
                 + (f" · {note}" if note else ""), fontsize=9)
    # NOT bbox_inches="tight": the trimmed width would follow the title length, so the
    # same panel would come out a few pixels wider for a model with a longer tag and a
    # contact sheet would not align. Fixed figsize -> identical pixel dimensions.
    fig.savefig(path, dpi=125); plt.close(fig)


ERROR_MIN_COUNT = 4


def save_error_raster(path: Path, t, y, err, ylim, tag: str, note: str,
                      metric: str = "relative", point_weight=None) -> None:
    """depth x time, coloured by the model's per-spike RECONSTRUCTION ERROR.

    Same grid and figure size as `save_density_raster` / `save_basis_raster`, so the
    three are directly overlayable: density says where spikes are, basis says what shape
    they have, this says where the model fits badly.

    Each pixel is the spike-count-weighted MEAN error of the spikes in it, not a sum --
    a sum would simply reproduce the density map. Pixels holding fewer than
    ERROR_MIN_COUNT spikes are left blank rather than shown as noise.

    metric="relative" colours by ||Y - Yhat||^2 / ||Y||^2, the fraction of each spike's
    energy the model fails to explain. That is the quantity that answers "does the model
    fit worse in some places", because absolute MSE is dominated by amplitude: loud
    spikes carry more absolute error wherever they are, so an absolute map largely
    redraws the amplitude map. metric="absolute" gives the raw per-spike MSE for when
    that is what is wanted.
    """
    nt = 979
    ny = int((ylim[1] - ylim[0]) / 4.0)
    rng = [ylim, (0.0, 1958.0)]
    w = np.ones(len(t), np.float32) if point_weight is None else np.asarray(
        point_weight, np.float32)
    num, _, _ = np.histogram2d(y, t, bins=(ny, nt), range=rng,
                               weights=np.asarray(err, np.float64) * w)
    den, _, _ = np.histogram2d(y, t, bins=(ny, nt), range=rng, weights=w)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(den >= ERROR_MIN_COUNT, num / np.maximum(den, 1e-9), np.nan)
    finite = M[np.isfinite(M)]
    lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
    fig, ax = plt.subplots(figsize=(12.8, 7.2 if ylim == FULL_Y else 5.1),
                           constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(BG)
    im = ax.imshow(M, origin="lower", extent=[0, 1958, *ylim], aspect="auto",
                   cmap=cmap, vmin=lo, vmax=max(hi, lo + 1e-9),
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=.012)
    cb.set_label("mean unexplained energy fraction" if metric == "relative"
                 else "mean per-spike MSE")
    ax.set_xlabel("recording time (s)"); ax.set_ylabel("centroid depth y (µm)")
    label = ("relative error  $\\|Y-\\hat{Y}\\|^2 / \\|Y\\|^2$" if metric == "relative"
             else "absolute MSE  $\\|Y-\\hat{Y}\\|^2$")
    ax.set_title(f"depth × time reconstruction error — {tag}\n"
                 f"pixel = mean {label} over its spikes "
                 f"(blank below {ERROR_MIN_COUNT} spikes)"
                 + (f" · {note}" if note else ""), fontsize=10)
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def save_basis_error(path: Path, sx, sy, err, tag: str, metric: str = "relative",
                     gridsize: int = 45) -> None:
    """Spatial-basis locations coloured by the MEAN reconstruction error there.

    The companion to the usage hexbin in `components.png`: same axes, but the colour is
    how badly the model fits the spikes assigned to each site rather than how many it
    was assigned. A site that is used a lot but fits badly is invisible in the usage
    view and obvious here.

    `sx`, `sy`, `err` are PER-SPIKE (the spike's assigned site position and its own
    error), so a site's colour is the mean over the spikes that actually chose it.
    """
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
    hb = ax[0].hexbin(sx, sy, C=err, reduce_C_function=np.mean, gridsize=gridsize,
                      cmap="viridis", mincnt=ERROR_MIN_COUNT)
    lab = ("mean unexplained energy fraction" if metric == "relative"
           else "mean per-spike MSE")
    fig.colorbar(hb, ax=ax[0]).set_label(lab, fontsize=8)
    ax[0].set_xlabel("assigned site x (µm)"); ax[0].set_ylabel("assigned site y (µm)")
    ax[0].set_title("error by where the spike was localized", fontsize=9)

    nz = np.isfinite(err)
    ax[1].hexbin(sy[nz], np.asarray(err)[nz], gridsize=(60, 40), cmap="magma",
                 bins="log", mincnt=1)
    ax[1].set_xlabel("assigned site depth y (µm)"); ax[1].set_ylabel(lab, fontsize=8)
    ax[1].set_title("error vs depth (log density)", fontsize=9)
    for A in ax:
        A.tick_params(labelsize=7)
    fig.suptitle(f"spatial basis coloured by reconstruction error — {tag}\n"
                 f"is the model's error biased by WHERE the spike sits?", fontsize=10.5)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
