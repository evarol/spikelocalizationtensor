"""Score a recovered motion trace against the imposed ground-truth drift.

Both traces are quadratically detrended first: the recordings carry slow
non-imposed drift that would otherwise inflate the correlation for any trace
with a similar trend, including a flat one.
"""
from __future__ import annotations

import numpy as np

from spiketensor import data as D


def score(p, t_sec, detrend=True):
    """Correlation and least-squares gain of a trace against the triangle GT."""
    gt = D.gt_motion_trace(t_sec)
    p = np.asarray(p, float).copy(); gt = np.asarray(gt, float).copy()
    if detrend:                      # remove slow non-imposed drift from both
        A = np.vstack([np.ones_like(t_sec), t_sec, t_sec ** 2]).T
        p = p - A @ np.linalg.lstsq(A, p, rcond=None)[0]
        gt = gt - A @ np.linalg.lstsq(A, gt, rcond=None)[0]
    a, b = p - p.mean(), gt - gt.mean()
    if a.std() == 0 or b.std() == 0:
        return {"r": 0.0, "gain": 0.0, "resid_um": float(b.std()), "ptp_um": 0.0}
    gain = float((a @ b) / (b @ b))
    return {"r": float(np.corrcoef(a, b)[0, 1]), "gain": gain,
            "resid_um": float(np.sqrt(((a - gain * b) ** 2).mean())),
            "ptp_um": float(a.max() - a.min())}
