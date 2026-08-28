"""Fit the spatial-footprint codebook to every spike, and score it honestly.

Block-coordinate: the pi-step is solved exactly (non-negative least squares, K unknowns
per spike) every iteration; the shared codebook {mu, scale, a} takes an Adam step on the
resulting residual. Minibatched over spikes, so the cost per step is independent of S.

References it reports against, all recomputed on the SAME DC-removed target so the
numbers are comparable (the sweep's `recon` column is on RAW waveforms and is NOT):
    free rank-1 SVD      the best any separable (one time course per spike) model can do
    free rank-K SVD      the best with K time courses per spike, still per-spike free
    global mean          predicting one waveform for everything
    per-channel mean     predicting a fixed waveform per neighbourhood slot
The codebook is strictly harder than free rank-K: its time courses are SHARED across all
spikes and its spatial factors are constrained to a K-element parametric family.

Usage:
    python3 zncc/tensor/fit.py --K 32 --kernel monopole --steps 4000
    python3 zncc/tensor/fit.py --K 32 --kernel gauss --n_fit 400000
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

from spiketensor import data as D                                    # noqa: E402
from spiketensor.model import (Codebook, GridCodebook, batch_terms,  # noqa: E402
                               batch_terms_grid, residual_sq, residual_sq_grid,
                               solve_pi, solve_pi_grid, solve_pi_grid_ent)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_batch(rec, idx, off_all, dev):
    """DC-removed waveforms and anchor-relative offsets for a set of spikes."""
    w = torch.as_tensor(np.asarray(rec.waveforms[idx]), dtype=torch.float32, device=dev)
    Y = w / 100.0
    Y = Y - Y.mean(dim=2, keepdim=True)          # per-channel, per-spike DC
    off = torch.as_tensor(off_all[rec.spike_channels[idx]], dtype=torch.float32,
                          device=dev)
    return Y, off


def references(rec, off_all, var, n=20000, K=32, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(rec.n_spikes, n, replace=False))
    Y, _ = load_batch(rec, idx, off_all, "cpu")
    A = Y.numpy().astype(np.float64)
    out = {"n_ref": n}
    out["global_mean"] = float(((A - A.mean((0, 1), keepdims=True)) ** 2).mean() / var)
    out["per_slot_mean"] = float(((A - A.mean(0, keepdims=True)) ** 2).mean() / var)
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    for r in (1, 2, 3, 4, min(8, K)):
        rec_r = (u[:, :, :r] * s[:, None, :r]) @ vt[:, :r]
        out[f"free_rank{r}"] = float(((A - rec_r) ** 2).mean() / var)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cp", "grid"], default="cp")
    ap.add_argument("--K", type=int, default=32, help="cp only")
    ap.add_argument("--P", type=int, default=16, help="grid only: positions")
    ap.add_argument("--Q", type=int, default=8, help="grid only: shapes")
    ap.add_argument("--kernel", choices=["monopole", "gauss"], default="monopole")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pi_iters", type=int, default=60)
    ap.add_argument("--ent", type=float, default=0.0,
                    help="entropy penalty on the position marginal (grid only); "
                         "scale-free, unlike --l1")
    ap.add_argument("--l1", type=float, default=0.0,
                    help="L1 on pi; with pi>=0 this is just rho*sum(pi)")
    ap.add_argument("--n_fit", type=int, default=0, help="0 = all spikes")
    ap.add_argument("--dataset", default="np1")
    ap.add_argument("--out", type=Path, default=REPO / "zncc/runs/tensor")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    rec = D.load(a.dataset)
    off_all = rec.channel_offsets().astype(np.float32)          # (C, 10, 2) um
    var = 4.242133873e-4 if a.dataset == "np1" else float(
        np.var(np.asarray(rec.waveforms[:20000]) / 100.0))
    pool = (np.arange(rec.n_spikes) if a.n_fit <= 0 else
            np.sort(np.random.default_rng(1).choice(rec.n_spikes, a.n_fit,
                                                    replace=False)))
    print(f"{rec.name}: {rec.n_spikes:,} spikes, fitting on {len(pool):,}  "
          f"K={a.K} kernel={a.kernel} dev={DEV}", flush=True)

    ref = references(rec, off_all, var, K=max(a.K, a.Q))
    print("references on the SAME DC-removed target:", flush=True)
    for k, v in ref.items():
        if k != "n_ref":
            print(f"    {k:16s} {v:.4f}", flush=True)

    grid = a.model == "grid"
    nT = rec.waveforms.shape[2]
    cb = (GridCodebook(a.P, a.Q, a.kernel, n_t=nT) if grid
          else Codebook(a.K, a.kernel, n_t=nT)).to(DEV)
    nA = a.Q if grid else a.K
    # temporal atoms initialized from the top-K PCs of the DC-removed data
    Yi, _ = load_batch(rec, np.sort(np.random.default_rng(2).choice(
        rec.n_spikes, 20000, replace=False)), off_all, "cpu")
    F = Yi.reshape(-1, Yi.shape[2]).numpy().astype(np.float64)
    _, _, vt = np.linalg.svd(F - F.mean(0), full_matrices=False)
    with torch.no_grad():
        cb.a.copy_(torch.as_tensor(vt[:nA], dtype=torch.float32, device=DEV))

    opt = torch.optim.Adam(cb.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    hist, t0 = [], time.perf_counter()
    for step in range(1, a.steps + 1):
        idx = np.sort(rng.choice(pool, a.batch, replace=False))
        Y, off = load_batch(rec, idx, off_all, DEV)
        y2 = (Y * Y).sum((1, 2))
        if grid:
            g, at, Sg, Sa, b = batch_terms_grid(cb, Y, off)
            with torch.no_grad():                   # exact block minimization
                pi = (solve_pi_grid_ent(Sg.detach(), Sa.detach(), b.detach(),
                                        a.ent, a.pi_iters) if a.ent else
                      solve_pi_grid(Sg.detach(), Sa.detach(), b.detach(),
                                    a.pi_iters, a.l1))
            loss = (residual_sq_grid(Sg, Sa, b, pi, y2).sum() / Y.numel() / var)
        else:
            g, at, G, b = batch_terms(cb, Y, off)
            with torch.no_grad():
                pi = solve_pi(G.detach(), b.detach(), a.pi_iters, a.l1)
            loss = (residual_sq(G, b, pi, y2).sum() / Y.numel() / var)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 50 == 0 or step == 1:
            hist.append({"step": step, "nmse": float(loss.detach()),
                         "used": float((pi > 1e-6).float().flatten(1).sum(1).mean()),
                         "wall_s": time.perf_counter() - t0})
            if step % 250 == 0 or step == 1:
                print(f"  step {step:5d}  nMSE {float(loss):.4f}  "
                      f"used/spike {hist[-1]['used']:.1f}  "
                      f"{hist[-1]['wall_s']:.0f}s", flush=True)

    # ---- final pass: pi for EVERY spike, and the honest full-data nMSE ----
    print("final pass over all spikes...", flush=True)
    shape = (a.P, a.Q) if grid else (a.K,)
    PI = np.zeros((rec.n_spikes, *shape), np.float32)
    num = den = 0.0
    with torch.no_grad():
        for a0 in range(0, rec.n_spikes, a.batch):
            idx = np.arange(a0, min(a0 + a.batch, rec.n_spikes))
            Y, off = load_batch(rec, idx, off_all, DEV)
            y2 = (Y * Y).sum((1, 2))
            if grid:
                g, at, Sg, Sa, b = batch_terms_grid(cb, Y, off)
                pi = (solve_pi_grid_ent(Sg, Sa, b, a.ent, a.pi_iters) if a.ent
                      else solve_pi_grid(Sg, Sa, b, a.pi_iters, a.l1))
                num += float(residual_sq_grid(Sg, Sa, b, pi, y2).sum())
            else:
                g, at, G, b = batch_terms(cb, Y, off)
                pi = solve_pi(G, b, a.pi_iters, a.l1)
                num += float(residual_sq(G, b, pi, y2).sum())
            PI[idx] = pi.cpu().numpy(); den += Y.numel()
    full = num / den / var
    print(f"FULL-DATA nMSE {full:.4f}   (free rank-1 {ref['free_rank1']:.4f}, "
          f"per-slot mean {ref['per_slot_mean']:.4f})", flush=True)

    tag = (f"grid_{a.kernel}_P{a.P}Q{a.Q}" if grid else f"{a.kernel}_K{a.K}")
    tag += f"_l1{a.l1:g}" if a.l1 else ""
    tag += f"_ent{a.ent:g}" if a.ent else ""
    np.save(a.out / f"pi_{tag}.npy", PI)
    torch.save({"state": cb.state_dict(), "K": a.K, "P": a.P, "Q": a.Q,
                "model": a.model, "kernel": a.kernel,
                "dataset": a.dataset, "var": var}, a.out / f"codebook_{tag}.pt")
    (a.out / f"summary_{tag}.json").write_text(json.dumps(
        {"args": {k: (str(v) if isinstance(v, Path) else v)
                  for k, v in vars(a).items()},
         "references": ref, "full_nmse": full, "history": hist,
         "mu": cb.mu.detach().cpu().numpy().tolist(),
         "scale": cb.scales().detach().cpu().numpy().tolist(),
         "pi_used_mean": float((PI > 1e-6).reshape(len(PI), -1).sum(1).mean()),
         "pi_mass_top1": float(
             (PI.reshape(len(PI), -1).max(1)
              / np.maximum(PI.reshape(len(PI), -1).sum(1), 1e-12)).mean()),
         # for the grid model the POSITION MARGINAL is the object of interest: a spike
         # may legitimately need several shapes, but should need only one place
         "pos_used_mean": (float((PI.sum(2) > 1e-6).sum(1).mean()) if grid else None),
         "pos_mass_top1": (float((PI.sum(2).max(1)
                                  / np.maximum(PI.sum((1, 2)), 1e-12)).mean())
                           if grid else None)},
        indent=2, default=float))
    print(f"wrote {a.out}/codebook_{tag}.pt, pi_{tag}.npy, summary_{tag}.json")


if __name__ == "__main__":
    main()
