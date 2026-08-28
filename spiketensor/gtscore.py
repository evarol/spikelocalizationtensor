"""Re-score every finished arm against the CORRECTED (triangle) ground truth.

The manipulator entries are waypoints and the stage ramps between them, so the
imposed motion is a zigzag triangle, not a square wave. Runs were selected during
training against the square-wave model, so their `gt_r` values in train_log.jsonl
are all understated; this rescores the saved best trace (and re-solves from the
saved D/C) against the correct target.

Also computes the two controls that make the numbers interpretable:

  anchor-null      every spike at its channel anchor -- ZERO localization
  monopole ladder  y = anchor + k*(y_MP - anchor) for k in {0, 0.25, 0.5, 1}

The headline quantity is GAIN (the least-squares amplitude of the recovery
relative to the 50 um command), because gain separates zero-localization from
honest localization while correlation alone does not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from spiketensor import data as D                                    # noqa: E402
from spiketensor.dredge import DredgeConfig, solve_rigid             # noqa: E402
from spiketensor.zncc import ShiftMaxZNCC, ZNCCConfig                # noqa: E402


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


# --------------------------------------------------------------------------- #
def control_ladder(ks=(0.0, 0.25, 0.5, 1.0), T=300, t0=580, spb=800,
                   sigma_um=4.0, max_shift_um=80.0, band=20, match_eval=False):
    """Shrink-ladder control on real spikes, same pipeline, no model involved."""
    rec = D.load("np1")
    bins = (8.0, 4.0, 10.0)
    ext = ((-120.0, 180.0), (-64.0, 3904.0), (0.0, 224.0))
    sec = np.floor(rec.spike_times / rec.fs).astype(np.int64)
    if match_eval:
        # EXACTLY the bin set and pair set the training-time evaluation uses, so
        # the controls are comparable with the arms rather than merely adjacent.
        from spiketensor.engine import make_eval_bins
        chosen = make_eval_bins(rec, 120); T = len(chosen); spb = 1500; band = 10**9
    else:
        chosen = np.arange(t0, t0 + T)
    lut = -np.ones(int(sec.max()) + 2, np.int64); lut[chosen] = np.arange(T)
    b_all = lut[sec]
    idx = np.flatnonzero(b_all >= 0)
    rng = np.random.default_rng(0)
    bb = b_all[idx]; o = np.argsort(bb, kind="stable"); idx, bb = idx[o], bb[o]
    off = np.concatenate([[0], np.cumsum(np.bincount(bb, minlength=T))])
    keep = np.concatenate([idx[off[i]:off[i + 1]] if off[i + 1] - off[i] <= spb else
                           idx[rng.choice(np.arange(off[i], off[i + 1]), spb, replace=False)]
                           for i in range(T)])
    bid = lut[sec[keep]]
    anch = rec.anchors[rec.spike_channels[keep]]
    t_sec = chosen.astype(float) + 0.5
    bandm = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :]) <= band
    (xl, xh), (yl, yh), (zl, zh) = ext
    bx, by, bz = bins
    nx, ny, nz = int((xh - xl) / bx), int((yh - yl) / by), int((zh - zl) / bz)
    zc = ShiftMaxZNCC(ZNCCConfig(max_shift_y_vox=int(max_shift_um / by), tau=0.01))

    out = {}
    for k in ks:
        x = anch[:, 0] + k * (rec.mp_xyz[keep, 0] - anch[:, 0])
        y = anch[:, 1] + k * (rec.mp_xyz[keep, 1] - anch[:, 1])
        z = k * rec.mp_xyz[keep, 2]
        xi = np.floor((x - xl) / bx).astype(np.int64)
        yi = np.floor((y - yl) / by).astype(np.int64)
        zi = np.floor((z - zl) / bz).astype(np.int64)
        ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny) & (zi >= 0) & (zi < nz)
        flat = bid[ok] * (nx * ny * nz) + xi[ok] * (ny * nz) + yi[ok] * nz + zi[ok]
        H = np.bincount(flat, minlength=T * nx * ny * nz).astype(np.float32)
        H = H.reshape(T, nx, ny, nz)
        for ax, bw in zip((1, 2, 3), bins):
            H = gaussian_filter1d(H, sigma_um / bw, axis=ax, mode="constant")
        with torch.no_grad():
            r = zc(torch.as_tensor(H), return_hard=True)
        Dm = r["d_y"].numpy().astype(float) * by
        Cm = np.where(bandm, r["rho_hard"].numpy().astype(float), 0.0)
        sol = solve_rigid(Dm, Cm, DredgeConfig(c_thresh=0.0, lam_smooth=1.0, irls_iters=3))
        sc = score(sol["p"], t_sec)
        sc["resid_std_um"] = float(np.std(y - anch[:, 1]))
        out[f"k={k}"] = sc
        print(f"  ladder k={k:<5} resid_std={sc['resid_std_um']:5.2f}um  "
              f"r={sc['r']:+.4f}  gain={sc['gain']:.3f}  resid={sc['resid_um']:.2f}um",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "zncc/runs")
    ap.add_argument("--skip_ladder", action="store_true")
    a = ap.parse_args()

    out = {}
    if not a.skip_ladder:
        print("CONTROLS: monopole shrink ladder (k=0 is the anchor null)", flush=True)
        out["controls"] = control_ladder()

    print("\nARMS re-scored against the triangle GT:", flush=True)
    rows = []
    for d in sorted(a.runs.iterdir()):
        f = d / "best_motion.npz"
        if not f.exists():
            continue
        z = np.load(f)
        p, t = z["p"], z["t"]
        s_direct = score(p, t)
        Dm, Cm = z["D"], z["C"]
        sol = solve_rigid(Dm, Cm, DredgeConfig(c_thresh=0.0, lam_smooth=1.0, irls_iters=5))
        s_resolve = score(sol["p"], t)
        try:
            sm = json.load(open(d / "summary.json"))
            arm, lam = sm["arm"], sm["lam"]
        except Exception:
            arm, lam = "?", float("nan")
        rows.append({"run": d.name, "arm": arm, "lam": lam,
                     "triangle": s_direct, "triangle_resolved": s_resolve})
        print(f"  {d.name:26s} arm={arm:8s} lam={lam:<5} "
              f"r={s_direct['r']:+.4f} gain={s_direct['gain']:.3f} "
              f"resid={s_direct['resid_um']:5.2f}um ptp={s_direct['ptp_um']:5.1f}um",
              flush=True)
    out["arms"] = rows
    (a.runs / "rescore_triangle.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.runs}/rescore_triangle.json")


if __name__ == "__main__":
    main()
