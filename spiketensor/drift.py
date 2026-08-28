"""Apply a fit's own DREDge motion estimate back to its spikes.

Every fit already carries the motion it implies: dc.npz holds p_soft and p_hard, the rigid
traces solved from the soft and hard shift matrices. Correcting means subtracting that
trace from each spike's depth, so a spike from a stationary neuron should land at a
constant depth regardless of when it fired:

    y_corrected(s) = y(s) - p(t_s)

The panels rendered from corrected positions are the actual test of the whole pipeline. An
uncorrected depth-time raster shows the imposed sawtooth; a correctly corrected one is
flat. Anything left over is motion the model could not see.

TWO SUBTLETIES, both of which silently corrupt the result if ignored:

  * p may be shorter than t. A rigid solve runs on an INTERIOR window (it uses
    margin=4 bins at each end, so 245 bins give 237 samples of p) because the first and
    last bins have no neighbours on one side and their shifts are unreliable. p therefore
    lines up with t[margin:-margin], not with t.
  * outside that window there is no estimate. np.interp holds the end values flat, which
    is the conservative choice -- it neither invents motion nor leaves a discontinuity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ONE motion estimate: canonical spikeinterface dredge_ap, rigid and nonrigid. The
# project's earlier internal soft/hard ZNCC solve is not distributed -- three estimates
# that disagreed made every panel ambiguous about which was being shown.
WHICH = ("real-rigid", "real-nonrigid")
SUFFIX = {"real-rigid": "_drr", "real-nonrigid": "_drn", "none": ""}


def correction(figs: Path, tag: str, which: str, spike_times, fs, margin: int = 4,
               y=None):
    """Per-spike depth offset in um. Subtract it from y to correct.

    soft/hard come from the project's internal solve (dc.npz). real-rigid and
    real-nonrigid come from canonical spikeinterface dredge_ap (dredge_real.npz); the
    nonrigid mode is depth-dependent and therefore needs the per-spike depths `y`."""
    if which in (None, "", "none"):
        return np.zeros(len(spike_times), np.float32)
    ts = np.asarray(spike_times, np.float64) / fs
    if which in ("real-rigid", "real-nonrigid"):
        f = figs / tag / "dredge_real.npz"
        if not f.exists():
            raise FileNotFoundError(f"no dredge_real.npz for {tag}; run dredge_real first")
        z = np.load(f)
        if which == "real-rigid":
            return np.interp(ts, z["t"], z["p_rigid"]).astype(np.float32)
        if y is None:
            raise ValueError("real-nonrigid correction needs per-spike depths y")
        pn, tn, wc = z["p_nonrigid"], z["t_nonrigid"], z["win_centers"]
        if pn.ndim == 1 or pn.shape[1] == 1:
            return np.interp(ts, tn, pn.reshape(len(tn))).astype(np.float32)
        out = np.empty(len(ts), np.float32)
        y = np.asarray(y, np.float64)
        for c0 in range(0, len(ts), 200_000):
            sl = slice(c0, min(c0 + 200_000, len(ts)))
            yy = np.clip(y[sl], wc[0], wc[-1])
            wi = np.clip(np.searchsorted(wc, yy) - 1, 0, len(wc) - 2)
            fr = (yy - wc[wi]) / np.maximum(wc[wi + 1] - wc[wi], 1e-9)
            lo = np.empty(len(yy)); hi = np.empty(len(yy))
            for w in np.unique(np.concatenate([wi, wi + 1])):
                m = wi == w
                if m.any():
                    lo[m] = np.interp(ts[sl][m], tn, pn[:, w])
                m = (wi + 1) == w
                if m.any():
                    hi[m] = np.interp(ts[sl][m], tn, pn[:, w])
            out[sl] = (1 - fr) * lo + fr * hi
        return out
    raise ValueError(f'unknown correction {which!r}; this package ships real-rigid and real-nonrigid')


def flatness(t, y, ylim, bins=(600, 300)):
    """How much a depth-time raster moves: the std of its per-time-bin centre of mass.

    Reported before and after correction as a scalar sanity check -- a correction with the
    sign inverted makes this number go UP, which is otherwise easy to miss by eye.
    """
    m = (y >= ylim[0]) & (y <= ylim[1])
    H, _, ye = np.histogram2d(t[m], y[m], bins=bins,
                              range=[[float(t.min()), float(t.max())], list(ylim)])
    yc = 0.5 * (ye[1:] + ye[:-1])
    tot = H.sum(1)
    ok = tot > 0
    com = (H[ok] @ yc) / tot[ok]
    return float(com.std())
