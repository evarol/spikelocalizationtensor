"""Fully learned two-basis factorization: N spatial elements x M shape components.

The four-part model, with BOTH bases learned:

    Yhat_s[c,t] = ( sum_n pi_{s,n} ghat_n(r_{s,c}) ) * ( v_s^T omega )_t

    pi_s  in {0,1}^N   one-hot spatial choice          (per spike)
    v_s   in R^M       free shape coefficients          (per spike)
    g     = {(mu_n, sigma_n)}_{n=1..N}   learned source locations AND scales (shared)
    omega in R^{M x T}, orthonormal      learned time basis (shared)

    ghat_n = unit-normalised phi(||r - mu_n||; sigma_n) on the spike's 10 offsets.

Offsets r are taken RELATIVE TO THE CENTROID of the spike's 10-channel neighbourhood
(not the peak channel, as every earlier fit did). The centroid is the natural origin for
a learned basis: it is the mean of the geometry the footprint is evaluated on. Because
that breaks the anchor+mu convention every downstream loader assumes, the npz carries an
explicit per-spike `pos` array and the loaders prefer it when present.

Unit-normalising ghat resolves the g/v scale ambiguity (amplitude lives in v_s);
orthonormal omega resolves the rotation ambiguity inside the shape block. The fixed
lattice is the special case mu frozen on a grid with a sigma dictionary; learn_mu freed
mu but kept the dictionary. Here sigma_n is free per element too, so N=512 learned
elements answer to a 512-site x 10-sigma fixed codebook at a tenth of the candidates.

ALGORITHM -- three exact blocks + one gradient block per outer iteration:
    1. pi:    argmax_n  ghat_n^T (M_s M_s^T) ghat_n,  M_s = Y_s omega^T     (exact)
    2. v:     v_s = omega ghat_s^T Y_s                                       (exact)
    3. omega: top-M eigenvectors of sum_s u_s u_s^T,  u_s = ghat_s^T Y_s     (exact)
    4. g:     Adam on (mu_n, log-sigma_n) through each element's assigned
              spikes; softplus keeps sigma positive                          (gradient)
Blocks 1-3 cannot increase the objective; block 4 is inexact and its effect is logged.
Elements that lose every spike get no gradient and die; used/N is reported per iteration.

Usage:
    python3 zncc/tensor/fit_learned.py --N 512 --M 8 --init kmeans
    python3 zncc/tensor/fit_learned.py --N 512 --M 8 --init lattice
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

from spiketensor import data as D                                   # noqa: E402
from spiketensor.fit import load_batch, references                # noqa: E402
from spiketensor.fit_lattice import (KERNELS, SPAN_XY, VAR,       # noqa: E402
                                     Z_HI, Z_LO, sym_index)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


class LearnedBasis(torch.nn.Module):
    """N spatial elements, each a learned (mu, sigma)."""

    def __init__(self, mu0, sig0, kernel="monopole"):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.as_tensor(mu0, dtype=torch.float32))
        s0 = torch.as_tensor(sig0, dtype=torch.float32)
        self.raw_sig = torch.nn.Parameter(torch.log(torch.expm1(s0)))
        self.kernel = kernel
        self.N = len(mu0)

    def sigma(self):
        return torch.nn.functional.softplus(self.raw_sig).clamp_min(0.5)

    def ghat_all(self, off):
        """(N, C) unit-norm footprints for ONE channel geometry off (C, 2)."""
        r = torch.cat([off, torch.zeros(len(off), 1, device=off.device)], 1)
        dxy2 = ((r[None, :, 0] - self.mu[:, None, 0]) ** 2
                + (r[None, :, 1] - self.mu[:, None, 1]) ** 2)
        dz2 = (self.mu[:, None, 2] ** 2).expand_as(dxy2)
        g = KERNELS[self.kernel](dxy2, dz2, (self.sigma()[:, None],))
        return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def ghat_sel(self, off, n):
        """(b, C) footprints for each spike's OWN element; differentiable in mu, sigma."""
        m = self.mu[n]
        dxy2 = ((off[:, :, 0] - m[:, None, 0]) ** 2
                + (off[:, :, 1] - m[:, None, 1]) ** 2)
        dz2 = (m[:, None, 2] ** 2).expand_as(dxy2)
        g = KERNELS[self.kernel](dxy2, dz2, (self.sigma()[n][:, None],))
        return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)


def centroid_offsets(rec):
    """Per-config offsets re-anchored at the 10-channel centroid, plus the shift."""
    off_all = rec.channel_offsets().astype(np.float32)          # (384, C, 2), peak-rel
    shift = off_all.mean(axis=1)                                # centroid - anchor, (384,2)
    return off_all - shift[:, None, :], shift


def init_mu(kind, N, rec, shift, seed=0):
    if kind == "lattice":
        n = max(2, round(N ** (1 / 3)))
        ax = np.linspace(-SPAN_XY, SPAN_XY, n)
        az = np.geomspace(Z_LO, Z_HI, n)
        MX, MY, MZ = np.meshgrid(ax, ax, az, indexing="ij")
        mu = np.stack([MX.ravel(), MY.ravel(), MZ.ravel()], 1)[:N]
        if len(mu) < N:                       # N not a perfect cube: pad with jitter
            extra = mu[np.random.default_rng(seed).integers(0, len(mu), N - len(mu))]
            mu = np.vstack([mu, extra + np.random.default_rng(seed).normal(0, 3, extra.shape)])
        return mu.astype(np.float32)
    # k-means over the monopole localizations, expressed in the centroid frame
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(rec.n_spikes, min(300_000, rec.n_spikes), replace=False))
    p = np.empty((len(idx), 3), np.float32)
    p[:, :2] = (rec.mp_xyz[idx, :2] - rec.anchors[rec.spike_channels[idx]][:, :2]
                - shift[rec.spike_channels[idx]])
    p[:, 2] = np.clip(rec.mp_xyz[idx, 2], Z_LO, Z_HI)
    p[:, 0] = np.clip(p[:, 0], -SPAN_XY, SPAN_XY)
    p[:, 1] = np.clip(p[:, 1], -SPAN_XY, SPAN_XY)
    km = MiniBatchKMeans(n_clusters=N, random_state=seed, n_init=3,
                         batch_size=4096).fit(p)
    return km.cluster_centers_.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--kernel", default="monopole")
    ap.add_argument("--init", choices=["kmeans", "lattice"], default="kmeans")
    ap.add_argument("--sigma0", type=float, default=20.0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--g_steps", type=int, default=500)
    ap.add_argument("--g_lr", type=float, default=0.5)
    ap.add_argument("--g_batch", type=int, default=16384)
    ap.add_argument("--n_fit", type=int, default=400000)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--dataset", default="np1")
    ap.add_argument("--out", type=Path, default=REPO / "zncc/runs/lattice")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    rec = D.load(a.dataset)
    off_cent, shift = centroid_offsets(rec)                  # (384, C, 2), (384, 2)
    cfg_u, cfg_id = np.unique(off_cent.reshape(len(off_cent), -1), axis=0,
                              return_inverse=True)
    off_cfg = torch.as_tensor(cfg_u.reshape(-1, off_cent.shape[1], 2), device=DEV)
    C, T = rec.waveforms.shape[1], rec.waveforms.shape[2]
    tag = a.tag or f"lrn{a.N}_{a.kernel}_M{a.M}_{a.init}"
    gb = LearnedBasis(init_mu(a.init, a.N, rec, shift, a.seed),
                      np.full(a.N, a.sigma0, np.float32), a.kernel).to(DEV)
    print(f"{tag}: N={a.N} learned elements ({a.init} init) x M={a.M} shape basis · "
          f"{len(off_cfg)} channel configs · centroid-anchored · {DEV}", flush=True)
    ref = references(rec, off_cent, VAR, K=a.M)
    print("  refs: " + "  ".join(f"{k} {v:.4f}" for k, v in ref.items() if k != "n_ref"),
          flush=True)

    rng = np.random.default_rng(a.seed)
    pool = np.sort(rng.choice(rec.n_spikes, min(a.n_fit, rec.n_spikes), replace=False))
    conf_pool = cfg_id[rec.spike_channels[pool]]
    Yc = torch.empty(len(pool), C, T, dtype=torch.float16)
    OFFp = torch.as_tensor(off_cent[rec.spike_channels[pool]])
    for s0 in range(0, len(pool), 32768):
        Y, _ = load_batch(rec, pool[s0:s0 + 32768], off_cent, "cpu")
        Yc[s0:s0 + len(Y)] = Y.half()

    # omega init: top-M PCA of raw waveforms
    F = Yc[rng.choice(len(pool), 20000, replace=False)].float().reshape(-1, T).numpy()
    _, _, vt = np.linalg.svd(F - F.mean(0), full_matrices=False)
    omega = torch.as_tensor(vt[:a.M].copy(), dtype=torch.float32, device=DEV)

    ri, ci, wt = (t.to(DEV) for t in sym_index(C))
    hist, t0, prev = [], time.perf_counter(), np.inf

    def assign(idx_conf, Ysrc, blk=32768):
        """pi-step over the pool (or all spikes): returns pick, best, y2."""
        n = len(idx_conf)
        P = torch.empty(n, len(ri), device=DEV)
        y2 = torch.empty(n, device=DEV)
        for s0 in range(0, n, blk):
            Y = Ysrc(s0, min(s0 + blk, n))
            y2[s0:s0 + len(Y)] = (Y * Y).sum((1, 2))
            M_ = Y @ omega.T
            Ps = M_ @ M_.transpose(1, 2)
            P[s0:s0 + len(Y)] = Ps[:, ri, ci]
        best = torch.full((n,), -np.inf, device=DEV)
        pick = torch.zeros(n, dtype=torch.long, device=DEV)
        with torch.no_grad():
            for ic in range(len(off_cfg)):
                rows = np.flatnonzero(idx_conf == ic)
                if not len(rows):
                    continue
                gh = gb.ghat_all(off_cfg[ic])                 # (N, C)
                Phi = gh[:, ri] * gh[:, ci] * wt              # (N, 55)
                rt = torch.as_tensor(rows, device=DEV)
                sc = Phi @ P[rt].T                            # (N, b)
                m, k = sc.max(0)
                best[rt], pick[rt] = m, k
        return pick, best, y2

    for it in range(1, a.iters + 1):
        pick, best, y2 = assign(conf_pool,
                                lambda s0, s1: Yc[s0:s1].to(DEV).float())
        nmse = float(((y2 - best).sum() / (len(pool) * C * T)).item() / VAR)
        # omega-step: weighted PCA of u = ghat^T Y (exact, no-grad)
        S = torch.zeros(T, T, dtype=torch.float64)
        with torch.no_grad():
            for s0 in range(0, len(pool), 32768):
                Y = Yc[s0:s0 + 32768].to(DEV).float()
                g = gb.ghat_sel(OFFp[s0:s0 + len(Y)].to(DEV), pick[s0:s0 + len(Y)])
                U = torch.einsum("bc,bct->bt", g, Y)
                S += (U.T @ U).cpu().double()
        ev, Vv = torch.linalg.eigh(S)
        omega = Vv[:, -a.M:].flip(1).T.float().contiguous().to(DEV).detach()
        # g-step: gradient on (mu, sigma) with assignment and omega fixed
        mu0 = gb.mu.detach().clone()
        opt = torch.optim.Adam(gb.parameters(), lr=a.g_lr)
        for st in range(a.g_steps):
            sel = rng.choice(len(pool), min(a.g_batch, len(pool)), replace=False)
            sel.sort()
            Y = Yc[torch.as_tensor(sel)].to(DEV).float()
            g = gb.ghat_sel(OFFp[torch.as_tensor(sel)].to(DEV),
                            pick[torch.as_tensor(sel, device=DEV)])
            score = torch.einsum("bc,bcq->bq", g, Y @ omega.T).pow(2).sum(1)
            loss = ((Y * Y).sum((1, 2)) - score).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            used = torch.unique(pick)
            dmu = (gb.mu.detach() - mu0).norm(dim=1)[used]
        hist.append({"step": it, "nmse": nmse, "wall_s": time.perf_counter() - t0,
                     "used": float(len(used)),
                     "mu_shift_med": float(dmu.median()),
                     "mu_shift_p95": float(dmu.quantile(0.95)),
                     "sigma_med": float(gb.sigma()[used].median())})
        print(f"  iter {it:2d}  nMSE {nmse:.4f}  {len(used)}/{a.N} used  "
              f"µ moved {hist[-1]['mu_shift_med']:.2f}/{hist[-1]['mu_shift_p95']:.2f} µm "
              f"(med/p95)  σ med {hist[-1]['sigma_med']:.1f}  "
              f"{hist[-1]['wall_s']:.0f}s", flush=True)
        if prev - nmse < a.tol:
            print(f"  converged (gain {prev - nmse:.2e} < {a.tol:g})", flush=True)
            break
        prev = nmse

    print("final pass over all spikes...", flush=True)
    conf_all = cfg_id[rec.spike_channels]
    def all_src(s0, s1):
        Y, _ = load_batch(rec, np.arange(s0, s1), off_cent, DEV)
        return Y
    pick, best, y2 = assign(conf_all, all_src)
    full = float(((y2 - best).sum() / (rec.n_spikes * C * T)).item() / VAR)
    KS = pick.cpu().numpy().astype(np.int32)
    used = len(np.unique(KS))
    print(f"FULL-DATA nMSE {full:.4f}  ({used}/{a.N} elements used, "
          f"free rank-1 {ref['free_rank1']:.4f})", flush=True)

    V = np.zeros((rec.n_spikes, a.M), np.float32)
    with torch.no_grad():
        for s0 in range(0, rec.n_spikes, 16384):
            Y, _ = load_batch(rec, np.arange(s0, min(s0 + 16384, rec.n_spikes)),
                              off_cent, DEV)
            g = gb.ghat_sel(torch.as_tensor(
                off_cent[rec.spike_channels[s0:s0 + len(Y)]], device=DEV),
                pick[s0:s0 + len(Y)])
            V[s0:s0 + len(Y)] = torch.einsum(
                "bc,bct,qt->bq", g, Y, omega).cpu().numpy()

    mu_f = gb.mu.detach().cpu().numpy()
    sig_f = gb.sigma().detach().cpu().numpy()
    # explicit per-spike positions: centroid-anchored, so anchor+mu does NOT reproduce
    # them and every downstream loader must prefer this array when present
    pos = np.empty((rec.n_spikes, 3), np.float32)
    pos[:, :2] = (rec.anchors[rec.spike_channels][:, :2]
                  + shift[rec.spike_channels] + mu_f[KS][:, :2])
    pos[:, 2] = mu_f[KS][:, 2]
    np.savez_compressed(a.out / f"pi_{tag}.npz", k=KS, v=V,
                        mu_site=mu_f, S=np.int32(1),
                        prof_sigma=np.array([float(np.median(sig_f))], np.float32),
                        site_sigma=sig_f, pos=pos,
                        anchor_shift=shift.astype(np.float32))
    torch.save({"a": omega.cpu(), "n": a.N, "Q": a.M, "kernel": a.kernel,
                "model": "lattice", "learned_basis": True, "K": a.N, "S": 1,
                "KS": a.N, "span_xy": SPAN_XY, "z_lo": Z_LO, "z_hi": Z_HI,
                "profiles": [(a.kernel, (float(np.median(sig_f)),))],
                "site_sigma": sig_f, "kernels": [a.kernel],
                "sigmas": sorted(set(np.round(sig_f, 1).tolist()))[:12],
                "init": a.init, "dataset": a.dataset, "var": VAR},
               a.out / f"codebook_{tag}.pt")
    (a.out / f"summary_{tag}.json").write_text(json.dumps(
        {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
         "model": "lattice", "kernel": a.kernel, "n": a.N, "K": a.N, "S": 1, "KS": a.N,
         "Q": a.M, "M": a.M, "N": a.N, "learned_basis": True, "learn_mu": True,
         "init": a.init, "references": ref, "full_nmse": full, "history": hist,
         "sites_used": int(used), "cands_used": int(used),
         "mu_shift_med": float(hist[-1]["mu_shift_med"]) if hist else 0.0,
         "sigma_range": [float(sig_f.min()), float(sig_f.max())],
         "pos_used_mean": 1.0, "pos_mass_top1": 1.0,
         "amp_median": float(np.median(np.linalg.norm(V, axis=1)))},
        indent=2, default=float))
    print(f"wrote {a.out}/pi_{tag}.npz, codebook_{tag}.pt, summary_{tag}.json")


if __name__ == "__main__":
    main()
