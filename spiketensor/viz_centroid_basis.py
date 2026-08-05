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

from spiketensor import data as D  # noqa: E402


LEARNED_FILES = (
    "centroid_basis_full.png", "centroid_basis_zoom.png",
    "centroid_basis_movie_full.mp4", "centroid_basis_movie_zoom.mp4",
    "depth_time_density_full.png", "depth_time_density_zoom.png",
    "depth_time_basis_full.png", "depth_time_basis_zoom.png",
)
REFERENCE_TAGS = {
    "BASELINE_monopole": "monopole",
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
    choice = z["k"].astype(np.int64)
    coeff = z["v"].astype(np.float32)
    dom = np.argmax(np.abs(coeff), axis=1).astype(np.int16)
    amp = np.linalg.norm(coeff, axis=1).astype(np.float32)
    del coeff
    if "mu_site" in z.files:
        site = choice // int(z["S"])
        local = z["mu_site"].astype(np.float32)[site]
    else:
        local = z["mu"].astype(np.float32)[choice]
    anc = rec.anchors[rec.spike_channels][:, :2].astype(np.float32)
    pos = anc + local[:, :2]
    return pos, dom, amp, shuffled_palette(int(ck["Q"])), ck


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


def jittered(pos: np.ndarray, tag: str, seed: int, jitter_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(tag, seed))
    x = pos[:, 0] + rng.uniform(-jitter_um, jitter_um, len(pos)).astype(np.float32)
    y = pos[:, 1] + rng.uniform(-jitter_um, jitter_um, len(pos)).astype(np.float32)
    jt = rng.uniform(-0.18, 0.18, len(pos)).astype(np.float32)
    return x, y, jt


def colour_raster(x, y, label, palette, xlim, ylim, nx, ny,
                  sigma_px=0.55) -> np.ndarray:
    """Rasterize every point; collisions average their categorical colours."""
    ix = np.floor((x - xlim[0]) / (xlim[1] - xlim[0]) * nx).astype(np.int64)
    iy = np.floor((y - ylim[0]) / (ylim[1] - ylim[0]) * ny).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]
    n = nx * ny
    count = np.bincount(flat, minlength=n).astype(np.float32)
    rgb = np.empty((ny, nx, 3), np.float32)
    for c in range(3):
        val = palette[label[ok], c]
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


def add_palette_legend(ax, palette, reference: bool, anchor_y: float = -0.07):
    q = len(palette)
    if reference:
        ax.scatter([], [], s=18, color=palette[0], label="reference: no learned basis")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, anchor_y), fontsize=6.5,
                  frameon=False)
        return
    handles = [ax.scatter([], [], s=17, color=palette[i], label=f"q{i}") for i in range(q)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, anchor_y),
              ncol=min(8, q), fontsize=5.8, frameon=False, handletextpad=.25,
              columnspacing=.7)


def save_centroid(path, x, y, dom, palette, rec, ylim, tag, reference):
    # One micrometre pixels retain the requested small visual jitter before imshow.
    nx = int(XY_LIM[1] - XY_LIM[0])
    ny = int(ylim[1] - ylim[0])
    rgb = colour_raster(x, y, dom, palette, XY_LIM, ylim, nx, ny)
    full = ylim == FULL_Y
    fig, ax = plt.subplots(figsize=(7.45, 11.45) if full else (9.1, 7.15),
                           constrained_layout=True)
    ax.imshow(rgb, origin="lower", extent=[*XY_LIM, *ylim], aspect="equal",
              interpolation="nearest")
    add_contacts(ax, rec, ylim)
    ax.set_xlim(*XY_LIM); ax.set_ylim(*ylim)
    ax.set_xlabel("lateral x (µm)"); ax.set_ylabel("probe depth y (µm)")
    ax.set_title(f"all spike localization centroids — {tag}\n"
                 f"visual jitter ±1.5 µm; colour = dominant time basis "
                 f"argmax_q |v_q|" if not reference else
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


def save_basis_raster(path, t, y, dom, palette, ylim, tag, reference):
    nt = 979
    ny = int((ylim[1] - ylim[0]) / 4.0)
    rgb = colour_raster(t, y, dom, palette, (0.0, 1958.0), ylim, nt, ny,
                        sigma_px=.42)
    fig, ax = plt.subplots(figsize=(12.8, 7.2 if ylim == FULL_Y else 5.1),
                           constrained_layout=True)
    ax.imshow(rgb, origin="lower", extent=[0, 1958, *ylim], aspect="auto",
              interpolation="nearest")
    ax.set_xlabel("recording time (s)"); ax.set_ylabel("centroid depth y (µm)")
    ax.set_title(f"depth × time centroid scatter — {tag}\n"
                 + ("visual jitter: ±0.18 s, ±1.5 µm; colour = dominant time basis"
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


def save_movie(path, x, y, dom, palette, sec, rec, ylim, tag, fps, reference):
    layout = _movie_layout(ylim)
    W, H = layout["W"], layout["H"]
    L, T, pw, ph = layout["left"], layout["top"], layout["pw"], layout["ph"]
    base = _base_movie_frame(layout, rec, ylim, palette, reference)
    order = np.argsort(sec, kind="stable")
    counts = np.bincount(sec, minlength=int(sec.max()) + 1)
    offs = np.concatenate([[0], np.cumsum(counts)])
    bins = np.flatnonzero(counts > 0)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
           "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    font = ImageFont.load_default()
    try:
        for ii, b in enumerate(bins):
            idx = order[offs[b]:offs[b + 1]]
            frame = base.copy()
            ix = np.floor(L + (x[idx] - XY_LIM[0]) / (XY_LIM[1] - XY_LIM[0]) * pw).astype(int)
            iy = np.floor(T + ph - (y[idx] - ylim[0]) / (ylim[1] - ylim[0]) * ph).astype(int)
            ok = (ix >= L) & (ix < L + pw) & (iy >= T) & (iy < T + ph)
            ix, iy, lab = ix[ok], iy[ok], dom[idx][ok]
            # Direct vectorized 3x3 dots. Later assignments only affect exact overlaps;
            # deterministic jitter makes different q values visible around shared sites.
            col = np.clip(palette[lab] * 255, 0, 255).astype(np.uint8)
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                xx = np.clip(ix + dx, L, L + pw - 1)
                yy = np.clip(iy + dy, T, T + ph - 1)
                frame[yy, xx] = col
            pim = Image.fromarray(frame)
            dr = ImageDraw.Draw(pim)
            dr.rectangle([0, 0, W, 52], fill=tuple((BG * 255).astype(int)))
            dr.text((12, 10), f"{tag}  |  t={b + 0.5:7.1f} s  |  "
                    + ("reference (no basis)" if reference else
                       "colour = dominant time basis"),
                    fill=(235, 237, 241), font=font)
            proc.stdin.write(np.asarray(pim, dtype=np.uint8).tobytes())
            if ii and ii % 400 == 0:
                print(f"    {path.name}: {ii}/{len(bins)}", flush=True)
    finally:
        if proc.stdin:
            proc.stdin.close()
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"ffmpeg failed ({rc}) for {path}")


def complete(out: Path) -> bool:
    return all((out / f).exists() and (out / f).stat().st_size > 0 for f in LEARNED_FILES)


def render_one(tag, runs, figs, rec, ptp, args):
    out = figs / tag
    out.mkdir(parents=True, exist_ok=True)
    if complete(out) and not args.force:
        print(f"skip complete {tag}", flush=True)
        return {"tag": tag, "status": "existing"}
    reference = tag in REFERENCE_TAGS
    if reference:
        pos, dom, amp, palette, meta = load_reference(tag, rec, ptp)
    else:
        pos, dom, amp, palette, meta = load_learned(runs, tag, rec)
    x, y, jt = jittered(pos, tag, args.seed, args.jitter_um)
    t = rec.spike_times.astype(np.float64) / rec.fs + jt
    sec = np.floor(rec.spike_times / rec.fs).astype(np.int64)
    print(f"{tag}: {len(pos):,} spikes, Q={len(palette) if not reference else 0}", flush=True)

    save_centroid(out / "centroid_basis_full.png", x, y, dom, palette, rec,
                  FULL_Y, tag, reference)
    save_centroid(out / "centroid_basis_zoom.png", x, y, dom, palette, rec,
                  ZOOM_Y, tag, reference)
    save_density_raster(out / "depth_time_density_full.png", t, y, amp,
                        FULL_Y, tag)
    save_density_raster(out / "depth_time_density_zoom.png", t, y, amp,
                        ZOOM_Y, tag)
    save_basis_raster(out / "depth_time_basis_full.png", t, y, dom, palette,
                      FULL_Y, tag, reference)
    save_basis_raster(out / "depth_time_basis_zoom.png", t, y, dom, palette,
                      ZOOM_Y, tag, reference)
    if not args.skip_movies:
        save_movie(out / "centroid_basis_movie_full.mp4", x, y, dom, palette,
                   sec, rec, FULL_Y, tag, args.fps, reference)
        save_movie(out / "centroid_basis_movie_zoom.mp4", x, y, dom, palette,
                   sec, rec, ZOOM_Y, tag, args.fps, reference)
    return {"tag": tag, "status": "generated", "reference": reference,
            "Q": 0 if reference else int(meta["Q"]),
            "jitter_um": args.jitter_um,
            "palette": palette.tolist(),
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
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--figs", type=Path, default=REPO / "figures")
    ap.add_argument("--tags", nargs="*")
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
        records.append(render_one(tag, a.runs, a.figs, rec, ptp, a))
        write_manifest(manifest, records, tags, a)
    print(f"wrote {manifest}; {len(tags)} methods in {(time.time()-start)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
