"""Mean pairwise ZNCC for every tensor fit, on one common scale.

These models never saw the ZNCC objective -- they were fit to minimise waveform MSE
alone. So mean C is a HELD-OUT measurement here, not a training score, which is exactly
what makes the MSE-vs-C plot worth having: in the 140 trained ZNCC runs the two were
optimised against each other, and C rose mostly by collapsing the image.

C(t,t') is the same shift-max ZNCC the project optimises: rho_soft = sum_d softmax(rho_d/tau)
* rho_d over 41 lags spanning +-80 um in y, and rho_hard = rho at the argmax lag. Same
GridSpec, same 4 um blur, same one-second bins as dc_movie.py, so every number here is
directly comparable to the monopole reference and to the trained runs.

Only the STRIDE differs from dc_movie.py (8 vs 2) so 244 fits are affordable. Note what
the stride does and does not do: bins are always ONE SECOND wide (`sec = floor(t/fs)`);
the stride subsamples WHICH one-second images get compared, spacing them 8 s apart. It
never widens them. So every C here is the ZNCC of 1 s images holding ~1240 spikes each,
and 12.3% of the recording's spikes enter any image. Subsampling leaves each individual
C(t,t') untouched -- verified against the two stride-2 runs, where C_soft[::4,::4].mean()
reproduces this to 6 decimals -- so fits stay mutually comparable; only the absolute level
of C reflects 1 s images rather than longer ones.

CAVEAT on the third coordinate. z means different things across families: source depth
(softplus(raw_z)) for monopole kernels, spatial WIDTH for gauss, and the sigma lattice for
hard fits. They are not one physical axis. The shared grid ends at z=160 um, so fits whose
scale coordinate exceeds that lose spikes -- 4 hard fits at sigma=192/256 lose ALL of them
and score a meaningless C=0. `outside_frac` is recorded per fit so those can be excluded.

Positions, by model form:
    hard  (pi_*.npz)      one-hot site      pos = anchor + mu[k],      w = ||v||
    grid  (pi_*.npy, 3d)  pi_{s,p,q}        pos = anchor + wbar @ mu,  w = sum pi
    CP    (pi_*.npy, 2d)  pi_{s,k}          pos = anchor + wbar @ mu,  w = sum pi
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                                  # noqa: E402
from spiketensor.dredge import DredgeConfig, solve_rigid           # noqa: E402
from spiketensor.volume import GridSpec, VolumeSmoother            # noqa: E402
from spiketensor.zncc import ShiftMaxZNCC, ZNCCConfig              # noqa: E402
from spiketensor.gtscore import score                        # noqa: E402
from spiketensor.baselines import NAMES, measured_ptp, reference_positions
from spiketensor.dc_movie import build_volume, render_dc        # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def positions(runs: Path, tag: str, rec, keep: np.ndarray):
    """(pos, amp, kind) for the spikes in `keep` (sorted). Anchor-relative mu, z absolute."""
    s = json.loads((runs / f"summary_{tag}.json").read_text())
    npz = runs / f"pi_{tag}.npz"
    if npz.exists():
        z = np.load(npz)
        # the wide-lattice fits carry mu in the npz: at 64^3 x 10 scales it is 2.6M rows,
        # which would make summary.json a 100 MB document
        if "mu_site" in z.files:
            # dictionary fits store the SITE lattice plus the profile count; the candidate
            # index is site*S + profile, so position is just a floor-divide away
            mu, k = z["mu_site"].astype(np.float32), z["k"][keep] // int(z["S"])
        else:
            mu = (z["mu"].astype(np.float32) if "mu" in z.files
                  else np.asarray(s["mu"], np.float32))
            k = z["k"][keep]
        off, amp, kind = mu[k], np.linalg.norm(z["v"][keep], axis=1), "hard"
    else:
        mu = np.asarray(s["mu"], np.float32)
        P = np.load(runs / f"pi_{tag}.npy", mmap_mode="r")
        # chunk the fancy-index: mmap gather on 10 GB in one shot thrashes
        acc = []
        for c in range(0, len(keep), 200_000):
            b = np.asarray(P[keep[c:c + 200_000]], np.float32)
            acc.append(b.sum(2) if b.ndim == 3 else b)
        W = np.concatenate(acc)                      # (n, P)
        amp = W.sum(1)
        if mu.shape[1] == 2:
            # soft codebooks keep only (x, y) in mu; the third coordinate lives in
            # raw_z -- source depth for the monopole kernel, spatial width for gauss.
            # That is the same quantity the hard fits store in mu[:, 2], so using it
            # keeps one z convention across all three model families.
            pass                      # noqa: F401
            ck = torch.load(runs / f"codebook_{tag}.pt", map_location="cpu",
                            weights_only=False)
            st = ck["state"]
            pre = "space." if "space.raw_z" in st else ""
            if ck["kernel"] == "monopole":
                zk = torch.nn.functional.softplus(st[pre + "raw_z"]).numpy()
            else:
                d = torch.nn.functional.softplus(st[pre + "raw_L"][:, [0, 2]])
                L = torch.zeros(len(d), 2, 2)
                L[:, 0, 0], L[:, 1, 0], L[:, 1, 1] = d[:, 0], st[pre + "raw_L"][:, 1], d[:, 1]
                zk = torch.linalg.inv(L @ L.transpose(1, 2)).diagonal(
                    dim1=1, dim2=2).sqrt().mean(1).numpy()
            mu = np.concatenate([mu, zk[:len(mu), None].astype(np.float32)], 1)
        off = (W / np.maximum(amp, 1e-12)[:, None]) @ mu
        kind = "grid" if P.ndim == 3 else "cp"
    pos = np.empty((len(keep), 3), np.float32)
    pos[:, :2] = rec.anchors[rec.spike_channels[keep]][:, :2] + off[:, :2]
    pos[:, 2] = off[:, 2]
    return pos, amp.astype(np.float32), kind


def dc_for(tag, runs, rec, g, sm, zc, order, off_b, cnt, bins, t_sec, gt, margin,
           clamp_z=False, panel_dir=None, tau=0.2, stride=8):
    keep = np.concatenate([order[off_b[b]:off_b[b + 1]] for b in bins])
    keep.sort()
    if isinstance(runs, tuple):                 # ('control', pos, amp, kind)
        _, P, A, kind = runs
        pos, amp = P[keep], A[keep]
    else:
        pos, amp, kind = positions(runs, tag, rec, keep)
    if clamp_z:
        # clip on ALL THREE axes. The wide-lattice models reach +-150 um in x, past the
        # grid's 128 um edge, so clipping z alone would leave most of the loss in place.
        pos = pos.copy()
        for c, (lo, n, b) in enumerate(((g.x_lo, g.nx, g.x_bin), (g.y_lo, g.ny, g.y_bin),
                                        (g.z_lo, g.nz, g.z_bin))):
            pos[:, c] = np.clip(pos[:, c], lo + 1e-3, lo + n * b - 1e-3)
    T = len(bins)
    vols = torch.empty(T, *g.shape, dtype=torch.float16)
    drop = 0
    for i, b in enumerate(bins):
        # keep is sorted and the bin's ids are a subset of it, so searchsorted is exact
        r = np.searchsorted(keep, order[off_b[b]:off_b[b + 1]])
        v, dr = build_volume(g, sm, pos[r, 0], pos[r, 1], pos[r, 2], amp[r], DEV)
        vols[i] = v.to(torch.float16).cpu(); drop += dr

    Ds = np.zeros((T, T), np.float32); Cs = np.zeros((T, T), np.float32)
    Dh = np.zeros((T, T), np.float32); Ch = np.zeros((T, T), np.float32)
    ch = 100
    for i in range(0, T, ch):
        ni = min(ch, T - i)
        for j in range(0, T, ch):
            nj = min(ch, T - j)
            sub = torch.cat([vols[i:i + ni].to(DEV).float(),
                             vols[j:j + nj].to(DEV).float()], 0)
            with torch.no_grad():
                r = zc(sub, return_hard=True)
            Ds[i:i + ni, j:j + nj] = r["d_y_soft"][:ni, ni:ni + nj].cpu().numpy()
            Cs[i:i + ni, j:j + nj] = r["rho_soft"][:ni, ni:ni + nj].cpu().numpy()
            Dh[i:i + ni, j:j + nj] = r["d_y"][:ni, ni:ni + nj].cpu().numpy()
            Ch[i:i + ni, j:j + nj] = r["rho_hard"][:ni, ni:ni + nj].cpu().numpy()
            del sub, r
    Ds = np.nan_to_num(Ds).astype(float) * g.y_bin
    Dh = np.nan_to_num(Dh).astype(float) * g.y_bin
    Cs = np.nan_to_num(Cs).astype(float); Ch = np.nan_to_num(Ch).astype(float)

    inn = slice(margin, T - margin); t_in = t_sec[inn]
    cfg = DredgeConfig(c_thresh=0.0, lam_smooth=1.0, irls_iters=3)
    sol = solve_rigid(Ds[inn, inn], np.clip(Cs[inn, inn], 0, None), cfg)
    solh = solve_rigid(Dh[inn, inn], np.clip(Ch[inn, inn], 0, None), cfg)
    sc, sch = score(sol["p"], t_in), score(solh["p"], t_in)
    od = ~np.eye(T, dtype=bool)
    q = np.abs(Dh[od]) / 40.0
    lat = (float((np.abs(q - np.round(q)) * 40.0 <= 1.0)[Dh[od] != 0].mean())
           if (Dh[od] != 0).any() else 0.0)
    if panel_dir is not None:
        panel_dir.mkdir(parents=True, exist_ok=True)
        render_dc(panel_dir, tag, Ds, Cs, Dh, Ch, sol, solh, sc, sch, t_sec, gt, inn,
                  tau, stride,
                  note=("z clipped into the grid — this fit's sources otherwise fall "
                        "outside it" if clamp_z else ""))
    gp = gt[:, None] - gt[None, :]
    return {
        "tag": tag, "kind": kind, "T": T,
        "C_soft": float(Cs[od].mean()), "C_hard": float(Ch[od].mean()),
        "C_soft_med": float(np.median(Cs[od])), "C_hard_med": float(np.median(Ch[od])),
        "D_amp_soft": float(Ds[od].std()), "D_amp_hard": float(Dh[od].std()),
        "D_zero_frac": float((Dh[od] == 0).mean()),
        "D_distinct": int(len(np.unique(np.round(Dh[od], 3)))),
        "lattice_frac": lat,
        "r_D_gt": float(np.corrcoef(Dh[od], gp[od])[0, 1]) if Dh[od].std() > 0 else 0.0,
        "gt_r_soft": float(sc["r"]), "gt_gain_soft": float(sc["gain"]),
        "gt_r_hard": float(sch["r"]), "gt_gain_hard": float(sch["gain"]),
        "outside_frac": float(drop / max(1, len(keep))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path,
                    default=REPO / "figures/dc_all.jsonl")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--smooth_um", type=float, default=4.0)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--only", default="", help="substring filter on tag")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--controls", action="store_true",
                    help="measure reference localizers instead of fits: the monopole, and "
                         "an anchor-only collapse control that discards the waveform")
    ap.add_argument("--panels", action="store_true",
                    help="also write per-run dc.png (D/C soft+hard with both DREDge "
                         "motion solves) and dc.npz (the raw matrices and traces)")
    ap.add_argument("--figs", type=Path, default=REPO / "figures")
    ap.add_argument("--clamp_z", action="store_true",
                    help="clip positions into the grid on all axes, to recover a number "
                         "for fits whose support extends past it")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if a.out.exists() and not a.redo:
        done = {json.loads(l)["tag"] for l in a.out.read_text().splitlines() if l.strip()}
    rec = D.load("np1")
    g = GridSpec(**torch.load(REPO / "runs/gridspec.pt",
                              map_location="cpu", weights_only=False)["grid"])
    sm = VolumeSmoother(a.smooth_um, g, device=DEV).to(DEV)
    zc = ShiftMaxZNCC(ZNCCConfig(max_shift_y_vox=int(80.0 / g.y_bin), tau=a.tau))
    sec = np.floor(rec.spike_times / rec.fs).astype(np.int64)
    nb = int(sec.max()) + 1
    order = np.argsort(sec, kind="stable")
    cnt = np.bincount(sec, minlength=nb)
    off_b = np.concatenate([[0], np.cumsum(cnt)])
    bins = np.flatnonzero(cnt > 0)[::a.stride]
    t_sec = bins.astype(float) + 0.5
    gt = D.gt_motion_trace(t_sec)

    src = {}
    if a.controls:
        # Two reference localizers, through the identical pipeline. ANCHOR-ONLY is the one
        # that matters: it throws the waveform away and places every spike at its peak
        # channel. Any localizer scoring below it is being beaten by a method that carries
        # zero positional information, which is what makes C a collapse-sensitive score
        # rather than a localization score.
        ptp = measured_ptp(rec)
        src = {NAMES[w]: ("control", *reference_positions(rec, w, ptp), w)
               for w in ("anchor_ptp", "anchor_flat", "monopole")}
    tags = ([t for t in src if t not in done] if a.controls else
            [f.stem[len("summary_"):] for f in sorted(a.runs.glob("summary_*.json"))
             if a.only in f.stem[len("summary_"):]
             and f.stem[len("summary_"):] not in done])
    print(f"{len(tags)} fits to do ({len(done)} already done) · T={len(bins)} bins of "
          f"{a.stride}s · grid {g.shape} · device {DEV}", flush=True)

    t0 = time.perf_counter()
    for n, tag in enumerate(tags, 1):
        try:
            r = dc_for(tag, src[tag] if a.controls else a.runs, rec, g, sm, zc, order,
                       off_b, cnt, bins, t_sec, gt, a.margin, clamp_z=a.clamp_z,
                       panel_dir=(a.figs / tag) if a.panels else None,
                       tau=a.tau, stride=a.stride)
        except Exception as e:                       # a bad fit should not stop the sweep
            print(f"  [{n}/{len(tags)}] {tag}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        with a.out.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        el = time.perf_counter() - t0
        print(f"  [{n}/{len(tags)}] {tag:52s} C_soft {r['C_soft']:.3f}  "
              f"C_hard {r['C_hard']:.3f}  gt_r {r['gt_r_hard']:+.3f}  "
              f"[{el/n:.0f}s/fit, eta {(len(tags)-n)*el/n/60:.0f} min]", flush=True)


if __name__ == "__main__":
    main()
