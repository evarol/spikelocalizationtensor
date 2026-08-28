"""Per-atom depth x time rasters: can one shape be tracked through the recording?

One raster per temporal atom, showing ONLY the sources that selected that atom, so a
persistent horizontal streak means that shape keeps being found at the same depth --
i.e. the atom is behaving like a unit's signature rather than a generic basis element.

Every panel carries the same faint grey background of ALL sources. Without it a sparse
atom looks like structure simply because the plot is empty; with it you can see whether
the atom's streaks sit on real bands of activity, and whether it claims a band that
other atoms also claim.

An inset shows the atom waveform itself (solid) against its learned prototype (dashed),
so the shape being tracked is visible in the same figure as its spatial behaviour.

Written outside the browser tree, to zncc/figures/atom_rasters/<tag>/.
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
from matplotlib.colors import PowerNorm  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                            # noqa: E402
from spiketensor.drift import SUFFIX, correction           # noqa: E402
from spiketensor.viz_centroid_basis import FULL_Y, ZOOM_Y, shuffled_palette  # noqa: E402

NT = 979


def _hist(t, y, w, ylim):
    ny = int((ylim[1] - ylim[0]) / 4.0)
    H, _, _ = np.histogram2d(y, t, bins=(ny, NT), range=[ylim, (0.0, 1958.0)],
                             weights=w)
    return gaussian_filter(H.astype(np.float32), (.65, .35))


def atom_raster(path: Path, t, y, amp, mask, ylim, atom_wave, proto_wave, colour,
                title: str, note: str, bg: np.ndarray) -> None:
    H = _hist(t[mask], y[mask], amp[mask], ylim)
    fig, ax = plt.subplots(figsize=(12.8, 7.2 if ylim == FULL_Y else 5.1),
                           constrained_layout=True)
    nzb = bg[bg > 0]
    if nzb.size:                       # context: every source, faint, behind the atom
        ax.imshow(bg, origin="lower", extent=[0, 1958, *ylim], aspect="auto",
                  cmap="Greys", alpha=.30, interpolation="nearest",
                  norm=PowerNorm(.45, vmin=0, vmax=float(np.percentile(nzb, 99.7))))
    nz = H[H > 0]
    vmax = float(np.percentile(nz, 99.7)) if nz.size else 1.0
    im = ax.imshow(H, origin="lower", extent=[0, 1958, *ylim], aspect="auto",
                   cmap="magma", norm=PowerNorm(.45, vmin=0, vmax=vmax),
                   interpolation="nearest", alpha=.95)
    fig.colorbar(im, ax=ax, pad=.012).set_label("summed source amplitude (this atom)")
    ax.set_xlabel("recording time (s)"); ax.set_ylabel("source depth y (µm)")
    ax.set_title(title + ("" if not note else f" · {note}"), fontsize=10)

    # the inset sits ON the raster, so it needs an opaque box and an interior label:
    # a normal axes title would be drawn in dark text over the dark image
    ins = ax.inset_axes([0.010, 0.665, 0.135, 0.30])
    tt = np.arange(len(atom_wave)) / 30.0
    if proto_wave is not None:
        ins.plot(tt, proto_wave, color="0.45", lw=1.0, ls="--")
    ins.plot(tt, atom_wave, color=colour, lw=1.8)
    ins.axhline(0, color=".7", lw=.5)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_facecolor("white"); ins.patch.set_alpha(1.0)
    for sp in ins.spines.values():
        sp.set_edgecolor("0.25"); sp.set_linewidth(.8)
    ins.text(.5, .965, "atom — · prototype - -", transform=ins.transAxes,
             ha="center", va="top", fontsize=6.2, color="0.2")
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path,
                    default=REPO / "zncc/runs/onehot_prior/multipole_prior2_shift_M64_R4.npz")
    ap.add_argument("--tag", default="prior2_shift_M64_R4")
    ap.add_argument("--figs", type=Path, default=REPO / "zncc/figures/onehot_prior",
                    help="row directory holding dredge_real.npz for the corrections")
    ap.add_argument("--out", type=Path, default=REPO / "zncc/figures/atom_rasters")
    ap.add_argument("--modes", default="none,real-rigid,real-nonrigid")
    ap.add_argument("--views", default="zoom,full")
    ap.add_argument("--only-atom", type=int, default=-1)
    a = ap.parse_args()

    rec = D.load("np1")
    root = a.out / a.tag
    root.mkdir(parents=True, exist_ok=True)
    with np.load(a.state, mmap_mode="r") as z:
        act = np.asarray(z["source_index"]) >= 0
        parent, slot = np.nonzero(act)
        pos = np.asarray(z["source_pos"])[parent, slot]
        atom = np.asarray(z["source_temporal_atom"])[parent, slot].astype(np.int64)
        amp = np.asarray(z["source_amp"])[parent, slot].astype(np.float32)
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
    amps = np.bincount(atom, weights=amp.astype(np.float64), minlength=M)
    print(f"{a.tag}: {len(t):,} sources, {M} atoms", flush=True)

    views = {"zoom": ZOOM_Y, "full": FULL_Y}
    t0 = time.perf_counter()
    for mode in [m for m in a.modes.split(",") if m]:
        sfx = SUFFIX[mode]
        yy = pos[:, 1].astype(np.float32)
        if mode != "none":
            yy = yy - correction(a.figs, a.tag, mode, t_smp, rec.fs, y=pos[:, 1])
        note = "uncorrected" if mode == "none" else f"{mode} DREDge corrected"
        for vname in [v for v in a.views.split(",") if v]:
            ylim = views[vname]
            bg = _hist(t, yy, amp, ylim)
            sheet = []
            for q in range(M):
                if a.only_atom >= 0 and q != a.only_atom:
                    continue
                m = atom == q
                p = root / f"atom{q:02d}_{vname}{sfx}.png"
                atom_raster(p, t, yy, amp, m, ylim, omega[q],
                            None if protos is None else protos[assign[q]],
                            palette[q],
                            f"atom q{q} — {a.tag} · {int(m.sum()):,} sources "
                            f"({100 * counts[q] / counts.sum():.1f}% of sources, "
                            f"{100 * amps[q] / amps.sum():.1f}% of amplitude)",
                            note, bg)
                sheet.append(p)
            print(f"  {mode}/{vname}: {len(sheet)} atom rasters "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"wrote {root}", flush=True)


if __name__ == "__main__":
    main()
