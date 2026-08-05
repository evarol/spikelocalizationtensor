"""Wide-lattice single-source model, fitted by exact alternating minimization.

MODEL. Spike s has waveform Y_s (C=10 x T=90, DC-removed) and channel offsets r_{s,c}
relative to its anchor. A codebook of candidates n = (k, j) pairs a lattice site mu_k with
a spherical scale sigma_j; the lattice spans +-150 um in x and y and is GEOMETRIC in z over
[1, 300] um (a symmetric z range would be pure mirror-degeneracy, and uniform z spends half
its samples past 150 um where footprints are nearly flat). Exactly ONE candidate is chosen
per spike:

    Yhat_{s,c,t} = g^{(k_s,j_s)}_{s,c} * (v_s^T a)_t
    g^{(k,j)}_{s,c} = phi(||r_{s,c} - mu_k||; sigma_j)          peak-normalized to 1
    phi_mono(d;s) = s/sqrt(d^2+s^2)          phi_gauss(d;s) = exp(-d^2/2s^2)

    minimize  sum_s || Y_s - g_s (v_s^T a)^T ||_F^2  /  (N C T var)

over the shared time basis a (Q x T), the per-spike selection (k_s, j_s), and the free
per-spike coefficients v_s. No regularizer -- one-of-10K is the only constraint.

WHY IT IS AFFORDABLE. Writing u_s = g_s^T Y_s / ||g_s||^2 gives the identity

    ||Y - g w^T||^2 = ||Y||^2 - ||g||^2 ||u||^2 + ||g||^2 ||w - u||^2

so with a orthonormal the best-candidate score is a RAYLEIGH QUOTIENT in ghat = g/||g||:

    score_s(n) = ghat_n^T (M_s M_s^T) ghat_n,     M_s = Y_s a^T  (C x Q)
    residual_s = ||Y_s||^2 - max_n score_s(n)

M_s M_s^T is 10x10 regardless of Q, so Q drops out of the inner loop entirely. Vectorizing
that symmetric matrix to 55 entries turns the whole argmax into one GEMM, Phi (|N| x 55) @
p_s (55 x B) -- 110 FLOPs per spike-candidate, independent of Q. That is what makes the
64^3 lattice (2.62M candidates x 2.47M spikes) tractable at all.

ALGORITHM -- both blocks exact, so the objective is monotone non-increasing:
    1. init a = top-Q PCA of the raw waveforms
    2. ASSIGN   argmax the Rayleigh quotient per spike (no gradient, exact)
    3. REFIT    with selections fixed, minimizing over a AND all v_s jointly is weighted
                PCA: a = top-Q eigenvectors of  S = sum_s ||g_s||^2 u_s u_s^T   (90x90)
    4. repeat until the objective stops improving

This replaces the Adam-on-a scheme the earlier hard fits used, which descended a
reconstruction term evaluated at a penalized pi and was not monotone.

Usage:
    python -m spiketensor.fit_lattice --n 32 --Q 8 --kernel monopole
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

from spiketensor import data as D                        # noqa: E402
from spiketensor.waveforms import load_batch, references     # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SIGMAS = 2.0 ** np.arange(10)                          # 1, 2, 4, ... 512 um
SPAN_XY, Z_LO, Z_HI = 150.0, 1.0, 300.0
VAR = 4.242133873e-4

# Spatial profiles. Every one is a function of the LATERAL and AXIAL squared offsets
# separately, so anisotropic forms fit the same signature as radial ones. All are
# peak-normalized at d=0 (the overall scale is free anyway -- the score is a Rayleigh
# quotient, invariant to ||g||).
def _k_monopole(dxy2, dz2, p):
    return p[0] / torch.sqrt(dxy2 + dz2 + p[0] ** 2)


def _k_gauss(dxy2, dz2, p):
    return torch.exp(-(dxy2 + dz2) / (2 * p[0] ** 2))


def _k_exp(dxy2, dz2, p):
    return torch.exp(-torch.sqrt(dxy2 + dz2) / p[0])


def _k_lorentz(dxy2, dz2, p):
    return p[0] ** 2 / (dxy2 + dz2 + p[0] ** 2)


def _k_power(dxy2, dz2, p):
    return (p[0] / torch.sqrt(dxy2 + dz2 + p[0] ** 2)) ** p[1]


def _k_student(dxy2, dz2, p):
    return (1.0 + (dxy2 + dz2) / p[0] ** 2) ** (-p[1])


def _k_yukawa(dxy2, dz2, p):
    d = torch.sqrt(dxy2 + dz2 + 1e-6)
    return (p[0] / torch.sqrt(dxy2 + dz2 + p[0] ** 2)) * torch.exp(-d / p[1])


def _k_gauss_aniso(dxy2, dz2, p):
    return torch.exp(-(dxy2 / (2 * p[0] ** 2) + dz2 / (2 * p[1] ** 2)))


def _k_mono_aniso(dxy2, dz2, p):
    return p[0] / torch.sqrt(dxy2 + dz2 * (p[0] / p[1]) ** 2 + p[0] ** 2)


def _k_dog(dxy2, dz2, p):
    """Difference of Gaussians -- the one non-monotonic profile in the set. A spike whose
    footprint has a genuine surround cannot be represented by any monotone kernel."""
    d2 = dxy2 + dz2
    a = torch.exp(-d2 / (2 * p[0] ** 2))
    b = torch.exp(-d2 / (2 * (p[0] * p[1]) ** 2))
    return a - b / (p[1] ** 2)


KERNELS = {"monopole": _k_monopole, "gauss": _k_gauss, "exp": _k_exp,
           "lorentz": _k_lorentz, "power": _k_power, "student": _k_student,
           "yukawa": _k_yukawa, "gauss_aniso": _k_gauss_aniso,
           "mono_aniso": _k_mono_aniso, "dog": _k_dog}
NEEDS_2 = {"power", "student", "yukawa", "gauss_aniso", "mono_aniso", "dog"}


def build_dict(kernels, n_scales=10, n_aniso=0, extra=None):
    """The profile dictionary: a list of (kernel_name, params). Candidate = site x profile.

    Isotropic kernels get n_scales log-spaced sigmas over [1, 512] um. Anisotropic ones get
    an n_aniso x n_aniso grid of (lateral, axial) scales. Two-parameter radial kernels get
    each sigma crossed with the `extra` shape values."""
    sig = np.geomspace(1.0, 512.0, n_scales)
    out = []
    for k in kernels:
        if k in ("gauss_aniso", "mono_aniso"):
            a = np.geomspace(1.0, 512.0, n_aniso or n_scales)
            out += [(k, (float(x), float(z))) for x in a for z in a]
        elif k in NEEDS_2:
            for s_ in sig:
                out += [(k, (float(s_), float(e))) for e in (extra or [2.0])]
        else:
            out += [(k, (float(s_),)) for s_ in sig]
    return out


def lattice(n):
    """n^3 sites: uniform in x,y over +-150 um, geometric in z over [1, 300] um."""
    ax = np.linspace(-SPAN_XY, SPAN_XY, n, dtype=np.float64)
    az = np.geomspace(Z_LO, Z_HI, n)
    MX, MY, MZ = np.meshgrid(ax, ax, az, indexing="ij")
    return np.stack([MX.ravel(), MY.ravel(), MZ.ravel()], 1).astype(np.float32)


def sym_index(C):
    """(rows, cols, weight) for the 55 upper-triangular entries of a CxC symmetric form."""
    r, c = np.triu_indices(C)
    w = np.where(r == c, 1.0, 2.0).astype(np.float32)
    return torch.as_tensor(r), torch.as_tensor(c), torch.as_tensor(w)


def phi(d, s, kernel):
    return s / torch.sqrt(d * d + s * s) if kernel == "monopole" \
        else torch.exp(-d * d / (2.0 * s * s))


class Candidates:
    """The (site x profile) codebook, materialized per channel-config in chunks."""

    def __init__(self, mu, profiles, dev):
        self.mu = torch.as_tensor(mu, device=dev)               # (K, 3)
        self.prof = list(profiles)
        self.K, self.S = len(mu), len(self.prof)
        self.KS, self.dev = self.K * self.S, dev
        self.kernel = self.prof[0][0]
        # one (name -> rows, params) group so a chunk evaluates each kernel once
        self.groups = []
        for j, (nm, pr) in enumerate(self.prof):
            if self.groups and self.groups[-1][0] == nm:
                self.groups[-1][1].append(j); self.groups[-1][2].append(pr)
            else:
                self.groups.append((nm, [j], [pr]))

    def phi_chunk(self, off, lo, hi, ri, ci, wt):
        """Phi rows [lo,hi) for one channel config: (hi-lo, 55), rows already normalized.

        Candidate index is k*S + j, so a chunk spans whole sites and is sliced -- cheaper
        than gathering millions of rows individually."""
        k0, k1 = lo // self.S, (hi + self.S - 1) // self.S
        m = self.mu[k0:k1]                                                   # (k,3)
        dxy2 = ((off[None, :, 0] - m[:, None, 0]) ** 2
                + (off[None, :, 1] - m[:, None, 1]) ** 2)                    # (k,C)
        dz2 = (m[:, None, 2] ** 2).expand_as(dxy2)                           # channels z=0
        g = torch.empty(k1 - k0, self.S, len(off), device=self.dev)
        for nm, rows, prs in self.groups:
            f = KERNELS[nm]
            for j, pr in zip(rows, prs):
                g[:, j] = f(dxy2, dz2, pr)
        g = g.reshape(-1, len(off))[lo - k0 * self.S: hi - k0 * self.S]
        g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return g[:, ri] * g[:, ci] * wt, g


def footprint(cand, off, k):
    """Normalized g for each spike's OWN chosen candidate. O(C) per spike.

    The chunked phi_chunk path is for scanning all candidates; using it here meant
    building every candidate between min(k) and max(k) for each block, which is most of
    the codebook and was dominating the iteration."""
    site, prof = torch.div(k, cand.S, rounding_mode="floor"), k % cand.S
    m = cand.mu[site]                                                      # (b,3)
    dxy2 = ((off[:, :, 0] - m[:, None, 0]) ** 2
            + (off[:, :, 1] - m[:, None, 1]) ** 2)                         # (b,C)
    dz2 = (m[:, None, 2] ** 2).expand_as(dxy2)
    g = torch.empty_like(dxy2)
    for j in torch.unique(prof).tolist():
        sel = prof == j
        nm, pr = cand.prof[j]
        g[sel] = KERNELS[nm](dxy2[sel], dz2[sel], pr)
    return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)


class Cache:
    """The fit pool's waveforms, held once in fp16. Re-reading 1.4 GB from the memmap on
    every iteration cost more than the argmax it was feeding."""

    def __init__(self, rec, idx, off_all, dev, blk=32768):
        self.Y = torch.empty(len(idx), rec.waveforms.shape[1], rec.waveforms.shape[2],
                             dtype=torch.float16)
        self.off = torch.as_tensor(off_all[rec.spike_channels[idx]])
        for s0 in range(0, len(idx), blk):
            Y, _ = load_batch(rec, idx[s0:s0 + blk], off_all, "cpu")
            self.Y[s0:s0 + len(Y)] = Y.half()
        self.dev = dev

    def batch(self, s0, s1):
        return (self.Y[s0:s1].to(self.dev).float(), self.off[s0:s1].to(self.dev))


def compute_P(rec, idx, off_all, a, dev, blk=16384, cache=None):
    """Stream the waveforms ONCE, contiguously, and reduce each spike to 55 floats.

    The assignment step needs only P_s = M_s M_s^T (10x10 symmetric, 55 unique entries)
    with M_s = Y_s a^T. Reducing first means the waveform array is touched sequentially
    and exactly once per iteration, instead of being gathered 106 times in config order --
    which is what made the first version unusable."""
    C = rec.waveforms.shape[1]
    ri, ci, _ = (t.to(dev) for t in sym_index(C))
    P = torch.empty(len(idx), len(ri), device=dev)
    y2 = torch.empty(len(idx), device=dev)
    for s0 in range(0, len(idx), blk):
        sub = idx[s0:s0 + blk]
        Y = (cache.batch(s0, s0 + len(sub))[0] if cache is not None
             else load_batch(rec, sub, off_all, dev)[0])
        y2[s0:s0 + len(sub)] = (Y * Y).sum((1, 2))
        M = Y @ a.T                                        # (b, C, Q)
        Ps = M @ M.transpose(1, 2)                         # (b, C, C)
        P[s0:s0 + len(sub)] = Ps[:, ri, ci]
        del Y, M, Ps
    return P, y2


def assign_from_P(cand, P, conf, off_cfg, ks_chunk, spk_chunk, verbose=""):
    """Exact argmax of the Rayleigh quotient over every candidate, waveform-free.

    Loops config-outer / candidate-chunk-inner so Phi is built once per (config, chunk)
    and reused across that config's spikes."""
    ri, ci, wt = (t.to(cand.dev) for t in sym_index(off_cfg.shape[1]))
    n = P.shape[0]
    best = torch.full((n,), -float("inf"), device=cand.dev)
    pick = torch.zeros(n, dtype=torch.long, device=cand.dev)
    order = np.argsort(conf, kind="stable")
    bnd = np.searchsorted(conf[order], np.arange(len(off_cfg) + 1))
    t0 = time.perf_counter()
    for ic in range(len(off_cfg)):
        rows = order[bnd[ic]:bnd[ic + 1]]
        if not len(rows):
            continue
        rt = torch.as_tensor(rows, device=cand.dev)
        Pc = P[rt]                                          # (n_cfg, 55)
        bc = torch.full((len(rows),), -float("inf"), device=cand.dev)
        kc = torch.zeros(len(rows), dtype=torch.long, device=cand.dev)
        for lo in range(0, cand.KS, ks_chunk):
            hi = min(lo + ks_chunk, cand.KS)
            Phi, _ = cand.phi_chunk(off_cfg[ic], lo, hi, ri, ci, wt)
            for s0 in range(0, len(rows), spk_chunk):
                sc = Phi @ Pc[s0:s0 + spk_chunk].T          # (chunk, b)
                m, k = sc.max(0)
                sl = slice(s0, s0 + len(m))
                upd = m > bc[sl]
                bc[sl] = torch.where(upd, m, bc[sl])
                kc[sl] = torch.where(upd, k + lo, kc[sl])
                del sc
            del Phi
        best[rt] = bc
        pick[rt] = kc
        if verbose and ic % 25 == 0:
            print(f"    {verbose}: config {ic}/{len(off_cfg)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return pick, best


def basis_scatter(cand, rec, idx, off_all, conf, off_cfg, pick, dev, blk=16384,
                  cache=None):
    """One streaming pass: u_s = ghat_s^T Y_s, accumulating S = sum_s u_s u_s^T.

    ghat has unit norm by construction, so the ||g||^2 weight of the weighted-PCA step is
    already folded in and S is a plain scatter of the u's."""
    T = rec.waveforms.shape[2]
    # MPS has no float64: reduce each block in fp32 on device, fold into a CPU fp64 sum
    S = torch.zeros(T, T, dtype=torch.float64)
    for s0 in range(0, len(idx), blk):
        sub = idx[s0:s0 + blk]
        if cache is not None:
            Y, off = cache.batch(s0, s0 + len(sub))
        else:
            Y, off = load_batch(rec, sub, off_all, dev)
        g = footprint(cand, off, pick[s0:s0 + len(sub)])
        U = torch.einsum("bc,bct->bt", g, Y)
        S += (U.T @ U).cpu().double()
        del Y, U, g
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="lattice side; K = n^3")
    ap.add_argument("--Q", type=int, default=8)
    ap.add_argument("--kernel", default="monopole",
                    help="comma-separated profile families to put in ONE dictionary; the "
                         "spike picks a (site, profile) jointly. e.g. monopole / "
                         "monopole,gauss,exp / gauss_aniso")
    ap.add_argument("--n_scales", type=int, default=10,
                    help="log-spaced sigmas over [1,512] um per isotropic family")
    ap.add_argument("--n_aniso", type=int, default=0,
                    help="grid side for anisotropic families: n_aniso^2 (lateral, axial) "
                         "pairs instead of n_scales isotropic ones")
    ap.add_argument("--extra", default="2",
                    help="second shape parameter for 2-param families (power exponent, "
                         "student nu, yukawa screening length, DoG surround ratio)")
    ap.add_argument("--tag", default="", help="override the auto tag")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--n_fit", type=int, default=400000)
    ap.add_argument("--ks_chunk", type=int, default=16384,
                    help="candidates per GEMM block. 16384 measured fastest (13-14 G "
                         "spike-candidates/s) at the batch shapes that actually occur; "
                         "larger blocks are memory-bound, smaller ones under-fill the GPU")
    ap.add_argument("--spk_chunk", type=int, default=8192,
                    help="must exceed the typical spikes-per-config or the GEMM goes "
                         "narrow and throughput collapses (0.3 G/s at B=470)")
    ap.add_argument("--tol", type=float, default=1e-5, help="stop when nMSE gain < tol")
    ap.add_argument("--dataset", default="np1")
    ap.add_argument("--out", type=Path, default=REPO / "runs")
    ap.add_argument("--seed", type=int, default=0)
    a_ = ap.parse_args()
    a_.out.mkdir(parents=True, exist_ok=True)

    rec = D.load(a_.dataset)
    off_all = rec.channel_offsets().astype(np.float32)
    cfg_u, cfg_id = np.unique(off_all.reshape(len(off_all), -1), axis=0,
                              return_inverse=True)
    off_cfg = torch.as_tensor(cfg_u.reshape(-1, off_all.shape[1], 2), device=DEV)
    mu = lattice(a_.n)
    kerns = [k for k in a_.kernel.split(",") if k]
    extra = [float(x) for x in a_.extra.split(",") if x]
    profiles = build_dict(kerns, a_.n_scales, a_.n_aniso, extra)
    cand = Candidates(mu, profiles, DEV)
    C, T = rec.waveforms.shape[1], rec.waveforms.shape[2]
    tag = a_.tag or (f"lat{a_.n}_{'+'.join(kerns)}_Q{a_.Q}"
                     + (f"_a{a_.n_aniso}" if a_.n_aniso else "")
                     + (f"_s{a_.n_scales}" if a_.n_scales != 10 else "")
                     + (f"_e{a_.extra.replace(',', '-')}"
                        if any(k in NEEDS_2 for k in kerns) and not a_.n_aniso else ""))
    print(f"{tag}: K={cand.K:,} sites × {cand.S} profiles = {cand.KS:,} candidates · "
          f"Q={a_.Q} · {len(off_cfg)} channel configs · {DEV}", flush=True)
    print(f"  lattice x,y ±{SPAN_XY:g} µm uniform · z {Z_LO:g}–{Z_HI:g} µm geometric "
          f"({a_.n} levels) · dictionary: {', '.join(kerns)}", flush=True)
    ref = references(rec, off_all, VAR, K=a_.Q)
    print("  refs: " + "  ".join(f"{k} {v:.4f}" for k, v in ref.items() if k != "n_ref"),
          flush=True)

    rng = np.random.default_rng(a_.seed)
    pool = np.sort(rng.choice(rec.n_spikes, min(a_.n_fit, rec.n_spikes), replace=False))

    # init a: top-Q PCA of raw DC-removed waveforms
    Yi, _ = load_batch(rec, np.sort(rng.choice(rec.n_spikes, 20000, replace=False)),
                       off_all, "cpu")
    F = Yi.reshape(-1, T).numpy().astype(np.float64)
    _, _, vt = np.linalg.svd(F - F.mean(0), full_matrices=False)
    a = torch.as_tensor(vt[:a_.Q], dtype=torch.float32, device=DEV)

    conf_pool = cfg_id[rec.spike_channels[pool]]
    print(f"  caching {len(pool):,} fit waveforms "
          f"({len(pool)*C*T*2/1e9:.2f} GB fp16)...", flush=True)
    cache = Cache(rec, pool, off_all, DEV)
    hist, t0, prev = [], time.perf_counter(), np.inf
    for it in range(1, a_.iters + 1):
        P, y2 = compute_P(rec, pool, off_all, a, DEV, cache=cache)
        pick, best = assign_from_P(cand, P, conf_pool, off_cfg, a_.ks_chunk,
                                   a_.spk_chunk,
                                   verbose=f"iter {it}" if cand.KS > 3e5 else "")
        del P
        nmse = float(((y2 - best).sum() / (len(pool) * C * T)).item() / VAR)
        S = basis_scatter(cand, rec, pool, off_all, conf_pool, off_cfg, pick, DEV,
                          cache=cache)
        ev, V = torch.linalg.eigh(S)
        a = V[:, -a_.Q:].flip(1).T.float().contiguous().to(DEV)   # orthonormal, top-Q
        hist.append({"step": it, "nmse": nmse,
                     "wall_s": time.perf_counter() - t0,
                     "used": float(len(torch.unique(pick)))})
        print(f"  iter {it:2d}  nMSE {nmse:.4f}  "
              f"{len(torch.unique(pick)):,} distinct candidates  "
              f"{hist[-1]['wall_s']:.0f}s", flush=True)
        if prev - nmse < a_.tol:
            print(f"  converged (gain {prev-nmse:.2e} < {a_.tol:g})", flush=True)
            break
        prev = nmse

    print("final pass over all spikes...", flush=True)
    allx = np.arange(rec.n_spikes)
    P, y2 = compute_P(rec, allx, off_all, a, DEV)
    pick, best = assign_from_P(cand, P, cfg_id[rec.spike_channels], off_cfg,
                               a_.ks_chunk, a_.spk_chunk, verbose="final")
    del P
    full = float(((y2 - best).sum() / (rec.n_spikes * C * T)).item() / VAR)
    KS = pick.cpu().numpy().astype(np.int64)
    site, scale = KS // cand.S, KS % cand.S
    print(f"FULL-DATA nMSE {full:.4f}   ({len(np.unique(KS)):,}/{cand.KS:,} candidates "
          f"used, {len(np.unique(site)):,}/{cand.K:,} sites, free rank-1 "
          f"{ref['free_rank1']:.4f})", flush=True)

    # v_s for the chosen candidate, so downstream viz has an amplitude
    V = np.zeros((rec.n_spikes, a_.Q), np.float32)
    for s0 in range(0, rec.n_spikes, 16384):
        sub = allx[s0:s0 + 16384]
        Y, off = load_batch(rec, sub, off_all, DEV)
        g = footprint(cand, off, pick[s0:s0 + len(sub)])
        V[s0:s0 + len(sub)] = torch.einsum("bc,bct,qt->bq", g, Y, a).cpu().numpy()

    # store the SITE lattice plus the profile count, not the expanded candidate array:
    # a 6-kernel dictionary at 64^3 would be 15.7M rows of (x, y, z)
    np.savez_compressed(a_.out / f"pi_{tag}.npz", k=KS.astype(np.int32), v=V,
                        mu_site=mu, S=np.int32(cand.S),
                        prof_sigma=np.array([p[1][0] for p in cand.prof], np.float32))
    torch.save({"a": a.cpu(), "n": a_.n, "Q": a_.Q, "kernel": a_.kernel,
                "model": "lattice", "K": cand.K, "S": cand.S, "KS": cand.KS,
                "span_xy": SPAN_XY, "z_lo": Z_LO, "z_hi": Z_HI,
                "profiles": cand.prof, "kernels": kerns,
                "sigmas": sorted({p[1][0] for p in cand.prof}),
                "dataset": a_.dataset, "var": VAR},
               a_.out / f"codebook_{tag}.pt")
    (a_.out / f"summary_{tag}.json").write_text(json.dumps(
        {"args": {k: (str(v) if isinstance(v, Path) else v)
                  for k, v in vars(a_).items()},
         "model": "lattice", "kernel": a_.kernel, "n": a_.n, "K": cand.K, "S": cand.S,
         "KS": cand.KS, "Q": a_.Q, "references": ref, "full_nmse": full,
         "history": hist, "sites_used": int(len(np.unique(site))),
         "cands_used": int(len(np.unique(KS))),
         "scale_hist": np.bincount(scale, minlength=cand.S).tolist(),
         "profiles": [[p[0], list(p[1])] for p in cand.prof],
         "kernels": kerns, "n_profiles": cand.S,
         "sigmas": sorted({p[1][0] for p in cand.prof}),
         "pos_used_mean": 1.0, "pos_mass_top1": 1.0,
         "amp_median": float(np.median(np.linalg.norm(V, axis=1)))},
        indent=2, default=float))
    print(f"wrote {a_.out}/pi_{tag}.npz, codebook_{tag}.pt, summary_{tag}.json")


if __name__ == "__main__":
    main()
