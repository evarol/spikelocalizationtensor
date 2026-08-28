"""Oblique 3-D scatter of (time, lateral x, depth y) for one temporal atom.

The per-atom rasters project depth x time and lose the lateral coordinate, so two units
at the same depth but different x collapse onto each other. This keeps all three axes and
draws one point per source instead of a filled histogram, so the cloud stays transparent
and structure behind a dense band is still visible.

BRIGHTNESS ENCODES q, the source's fractional contribution to its parent spike. A source
that carries the whole spike is drawn at full brightness; a weak fourth source is dim.
On the dark background that reads directly as confidence, and it keeps the near-zero
sources from painting a fog over everything.

Subsampled by default: 9.9 M sources over 64 atoms is more than a scatter can usefully
show, and the structure is legible far below that.
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

from spiketensor import data as D                            # noqa: E402
from spiketensor.drift import SUFFIX, correction           # noqa: E402
from spiketensor.viz_centroid_basis import (FULL_Y, ZOOM_Y,  # noqa: E402
                                            shuffled_palette)

# The lateral extent of the LEARNED dictionary, not the probe: candidate
# offsets reach +-176 um, so sources legitimately sit well outside the
# 0-48 um column span. Clipping to the probe width silently piled ~4% of
# sources onto the plot edges as two false rays.
XLIM = (-160.0, 200.0)
TLIM = (0.0, 1958.0)


def atom_scatter(path: Path, t, x, y, q, ylim, wave, proto, colour, title, note,
                 elev=22.0, azim=45.0, n_max=40000, seed=0, point_size=1.6,
                 floor=0.12) -> int:
    m = (y >= ylim[0]) & (y <= ylim[1])
    idx = np.flatnonzero(m)
    rng = np.random.default_rng(seed)
    if len(idx) > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    if len(idx) == 0:
        return 0
    qq = np.asarray(q)[idx].astype(np.float32)
    hi = float(np.percentile(qq, 99)) if len(qq) > 10 else 1.0
    b = np.clip(qq / max(hi, 1e-6), 0.0, 1.0)
    b = floor + (1.0 - floor) * b                 # keep faint points visible
    rgba = np.zeros((len(idx), 4), np.float32)
    rgba[:, :3] = np.asarray(colour, np.float32)[None, :3] * b[:, None]
    rgba[:, 3] = b

    fig = plt.figure(figsize=(13.5, 9.0))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0f1115")
    ax.set_facecolor("#0f1115")
    order = np.argsort(b)                          # bright points drawn last
    ax.scatter(np.asarray(t)[idx][order], np.asarray(x)[idx][order],
               np.asarray(y)[idx][order], s=point_size, c=rgba[order],
               linewidths=0, depthshade=False, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(*TLIM); ax.set_ylim(*XLIM); ax.set_zlim(*ylim)
    ax.set_xlabel("recording time (s)", color="0.85", labelpad=12)
    ax.set_ylabel("lateral x (µm)", color="0.85", labelpad=8)
    ax.set_zlabel("depth y (µm)", color="0.85", labelpad=8)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.09, 0.10, 0.12, 1.0))
        a._axinfo["grid"]["color"] = (0.30, 0.32, 0.36, 0.5)
    ax.tick_params(colors="0.75", labelsize=8)
    ax.set_title(title + ("" if not note else f" · {note}") +
                 f"\n{len(idx):,} of {int(m.sum()):,} sources shown · "
                 f"brightness ∝ q (contribution share) · view elev {elev:.0f}° "
                 f"azim {azim:.0f}°", color="0.92", fontsize=10.5)

    ins = fig.add_axes([0.035, 0.72, 0.13, 0.19])
    tt = np.arange(len(wave)) / 30.0
    if proto is not None:
        ins.plot(tt, proto, color="0.45", lw=1.0, ls="--")
    ins.plot(tt, wave, color=colour, lw=1.8)
    ins.axhline(0, color=".7", lw=.5)
    ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_edgecolor("0.25"); sp.set_linewidth(.8)
    ins.text(.5, .965, "atom — · prototype - -", transform=ins.transAxes,
             ha="center", va="top", fontsize=6.2, color="0.2")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return len(idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path,
                    default=REPO / "zncc/runs/onehot_prior/multipole_prior2_shift_M64_R4.npz")
    ap.add_argument("--tag", default="prior2_shift_M64_R4")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/onehot_prior")
    ap.add_argument("--out", type=Path, default=REPO / "zncc/figures/atom_scatter3d")
    ap.add_argument("--modes", default="none")
    ap.add_argument("--views", default="zoom")
    ap.add_argument("--atoms", default="", help="comma list; empty = all")
    ap.add_argument("--n-max", type=int, default=40000)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=45.0)
    a = ap.parse_args()

    rec = D.load("np1")
    root = a.out / a.tag
    root.mkdir(parents=True, exist_ok=True)
    with np.load(a.state, mmap_mode="r") as z:
        act = np.asarray(z["source_index"]) >= 0
        parent, slot = np.nonzero(act)
        pos = np.asarray(z["source_pos"])[parent, slot]
        atom = np.asarray(z["source_temporal_atom"])[parent, slot].astype(np.int64)
        qv = np.asarray(z["source_weight"])[parent, slot].astype(np.float32)
        spike = np.asarray(z["spike_index"])[parent]
        omega = np.asarray(z["omega"])
        protos = np.asarray(z["prototypes"]) if "prototypes" in z.files else None
        assign = (np.asarray(z["atom_prototype"]) if "atom_prototype" in z.files
                  else np.zeros(len(omega), int))
    M = len(omega)
    t_smp = rec.spike_times[spike]
    t = (t_smp / rec.fs).astype(np.float32)
    palette = shuffled_palette(M)
    counts = np.bincount(atom, minlength=M)
    which = ([int(v) for v in a.atoms.split(",") if v != ""] if a.atoms
             else list(range(M)))
    views = {"zoom": ZOOM_Y, "full": FULL_Y}
    t0 = time.perf_counter()
    for mode in [m for m in a.modes.split(",") if m]:
        sfx = SUFFIX[mode]
        yy = pos[:, 1].astype(np.float32)
        if mode != "none":
            yy = yy - correction(a.figs, a.tag, mode, t_smp, rec.fs, y=pos[:, 1])
        note = "uncorrected" if mode == "none" else f"{mode} DREDge corrected"
        for vname in [v for v in a.views.split(",") if v]:
            for q in which:
                sel = atom == q
                n = atom_scatter(
                    root / f"atom{q:02d}_{vname}{sfx}_3d.png", t[sel],
                    pos[sel, 0], yy[sel], qv[sel], views[vname], omega[q],
                    None if protos is None else protos[assign[q]], palette[q],
                    f"atom q{q} — {a.tag} · {counts[q]:,} sources "
                    f"({100 * counts[q] / counts.sum():.1f}%)", note,
                    elev=a.elev, azim=a.azim, n_max=a.n_max)
                print(f"  q{q} {vname}{sfx}: {n:,} pts "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"wrote {root}", flush=True)


if __name__ == "__main__":
    main()
