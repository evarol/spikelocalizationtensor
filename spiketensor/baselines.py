"""Reference localizers that fit nothing, for calibrating every learned model.

Two of these matter, and they bracket the useful range.

MONOPOLE is the standard analytic point-source fit that the field already uses. It is the
target to beat: any learned model that cannot match it is not earning its complexity.

ANCHOR-ONLY is a collapse control. It throws the waveform away entirely and puts every
spike at its own peak channel, so it carries no positional information beyond which
contact fired. It exists because the pairwise-ZNCC score C rewards temporal
self-similarity of the localization image, and an image that has collapsed onto the
channel lattice is maximally self-similar. On this recording the control scores
C_hard 0.795 -- higher than every fitted model measured -- while recovering the imposed
drift far worse than the monopole (GT r +0.465 vs +0.865). Read C as distance BELOW this
control, never as an absolute quality.

The `flat` variant drops amplitude weighting too, isolating the effect of weighting from
the effect of position.
"""
from __future__ import annotations

import numpy as np

ANCHOR_Z = 20.0          # nominal depth for the anchor-only control, in um

NAMES = {"monopole": "BASELINE_monopole",
         "anchor_ptp": "CONTROL_anchor_only_ptp",
         "anchor_flat": "CONTROL_anchor_only_flat"}


def measured_ptp(rec):
    """Peak-to-peak on the strongest channel, the natural amplitude for an unfitted spike."""
    return np.ptp(np.asarray(rec.waveforms), axis=2).max(1).astype(np.float32)


def reference_positions(rec, which, ptp=None):
    """(pos, amp) in the same convention a fitted model produces: absolute (x, y, z) in um
    and a per-spike weight. `which` is one of monopole / anchor_ptp / anchor_flat."""
    if which not in NAMES:
        raise ValueError(f"unknown reference localizer {which!r}; expected {sorted(NAMES)}")
    ptp = measured_ptp(rec) if ptp is None else ptp
    if which == "monopole":
        return rec.mp_xyz.astype(np.float32).copy(), ptp
    pos = np.empty((rec.n_spikes, 3), np.float32)
    pos[:, :2] = rec.anchors[rec.spike_channels][:, :2]
    pos[:, 2] = ANCHOR_Z
    return pos, (ptp if which == "anchor_ptp" else np.ones(rec.n_spikes, np.float32))
