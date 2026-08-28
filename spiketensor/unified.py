"""One model for every spike factorization in this repository.

    Yhat_s = sum_{r=1..R}  g(r_c ; mu_{n_r}, sigma_{n_r})  *  w_r(t)

A spike is at most `R` point sources. Each source picks a spatial atom from a learned
dictionary (analytic kernel, learned center and scale) and contributes a temporal
waveform `w_r` drawn from a shared codebook `Omega in R^{M x T}`. Every model previously
written as its own module is a configuration of this one:

    shape="free"     w_r = b_r^T Omega,  b_r in R^M free      (M0/M1/M2, analytic sparse)
    shape="onehot"   w_r = a_r S_tau psi_q,  one atom + lag   (M3/M4/M5)

Config axes, and the models they recover:

    R           sources per spike            R=1 is the classic rank-one template
    M           codebook size
    kernel      monopole | gauss | exp | ... any entry of fit_lattice.KERNELS
    max_shift   0 disables lags; >0 makes the shape selection shift-INVARIANT
    shape       "free" | "onehot"
    nonneg      amplitudes constrained >= 0 (one-hot only; free shapes are signed)
    P           learned action-potential prototypes, 0 = no prior
    cone_deg    hard angle each atom may deviate from its prototype
    orthonormal keep Omega row-orthonormal (impossible together with P>0 -- see below)

    M0  Config(R=1, shape="free",   orthonormal=True)
    M2  Config(R=4, shape="free",   orthonormal=True)
    M3  Config(R=4, shape="onehot", nonneg=True, max_shift=0,  orthonormal=True)
    M4  Config(R=4, shape="onehot", nonneg=True, max_shift=10, orthonormal=True)
    M5  Config(R=8, shape="onehot", nonneg=True, max_shift=10, P=2, cone_deg=35,
               orthonormal=False)

ORTHONORMALITY AND THE PRIOR ARE MUTUALLY EXCLUSIVE, and this is not an implementation
limit: M unit vectors cannot all lie within a small cone of P << M prototypes and also be
mutually orthogonal. Requesting both raises rather than silently dropping one. That
orthogonality is also what makes a large free codebook drift into Fourier-like atoms --
the spike-like directions get used first -- which is the reason the prior exists.

Inference is matching pursuit over the product dictionary (place x shape x lag), with an
exact coefficient refit after every selection: ridge least squares for free shapes,
non-negative least squares for one-hot. Sources on different temporal atoms at equal lag
are exactly orthogonal, so only same-atom sources ever interact in the refit.

Learning alternates inference with an exact codebook block, chosen by config:
    free  + orthonormal   weighted PCA of the residual scatter (eigendecomposition)
    onehot+ orthonormal   shift-aligned orthogonal Procrustes
    onehot+ prototypes    closed-form projection onto the cones, then prototype refit
Every basis step is a PROPOSAL evaluated on the fit pool and rolled back if it does not
improve, because only the first is exact once shifted copies of an atom overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor import data as D                                      # noqa: E402
from spiketensor.fit import load_batch                               # noqa: E402
from spiketensor.fit_lattice import KERNELS                          # noqa: E402
from spiketensor.dictionary import load_dictionary         # noqa: E402
from spiketensor.state_io import validate_multipole_state        # noqa: E402

VAR_NP1 = 4.242133873e-4


@dataclass
class Config:
    """Every knob of the unified model. Defaults reproduce the M5 headline row."""
    R: int = 4
    M: int = 16
    kernel: str = "monopole"          # any key of fit_lattice.KERNELS
    max_shift: int = 10               # 0 = shift-variant
    shape: str = "onehot"             # "onehot" | "free"
    nonneg: bool = None               # None = auto (True for one-hot, False for free)
    P: int = 2                        # 0 = no prototype prior
    cone_deg: float = 35.0
    orthonormal: bool = False
    dictionary: str = ""              # learned spatial dictionary tag; "" = by kernel
    shortlist: int = 24
    min_separation: float = 4.0
    min_gain: float = 0.0
    ridge: float = 1e-6
    n_fit: int = 40000
    n_eval: int = 0                   # 0 = all spikes
    outer_iters: int = 12
    batch: int = 512
    seed: int = 0
    dataset: str = "np1"

    def __post_init__(self):
        # Flags that are MEANINGLESS for the chosen shape resolve automatically; flags
        # that are CONTRADICTORY still raise in validate(). Silently reinterpreting a
        # contradiction would hide a modelling mistake, but demanding --no-nonneg for a
        # free-coefficient model is just noise.
        if self.nonneg is None:
            self.nonneg = (self.shape == "onehot")
        if self.shape == "free":
            if self.P:
                print(f"  note: the cone prior constrains one-hot atoms; P {self.P} -> 0",
                      flush=True)
                self.P = 0
            if self.max_shift:
                print(f"  note: shifts are redundant with free coefficients; "
                      f"max_shift {self.max_shift} -> 0", flush=True)
                self.max_shift = 0
            if self.nonneg:
                raise ValueError("nonneg=True has no meaning for free coefficient "
                                 "vectors; drop it or use shape='onehot'")

    def validate(self) -> None:
        if self.kernel not in KERNELS:
            raise ValueError(f"kernel {self.kernel!r} not in {sorted(KERNELS)}")
        if self.shape not in ("free", "onehot"):
            raise ValueError("shape must be 'free' or 'onehot'")
        if self.P > 0 and self.orthonormal:
            raise ValueError(
                "orthonormal=True is incompatible with a prototype prior (P>0): M unit "
                "vectors cannot lie in cones around P<<M prototypes AND be mutually "
                "orthogonal. Choose one.")



    def tag(self) -> str:
        bits = [f"u{self.shape[0]}", self.kernel[:5], f"M{self.M}", f"R{self.R}"]
        if self.max_shift:
            bits.append(f"s{self.max_shift}")
        if self.P:
            bits.append(f"P{self.P}c{int(self.cone_deg)}")
        if self.orthonormal:
            bits.append("orth")
        if self.shape == "onehot" and not self.nonneg:
            bits.append("signed")
        return "_".join(bits)


# --------------------------------------------------------------------------- #
# temporal dictionary
# --------------------------------------------------------------------------- #
def shift_bank(omega: torch.Tensor, max_shift: int):
    """(M*nshift, T) unit-norm shifted atoms plus their (atom, lag) labels.

    Zero padding, then renormalisation: a spike shifted partly out of the window loses
    that energy, which is the honest choice, and unit norm keeps `captured` on the same
    scale for every atom.
    """
    M, T = omega.shape
    taus = list(range(-max_shift, max_shift + 1))
    rows, q_of, tau_of = [], [], []
    for q in range(M):
        for tau in taus:
            v = torch.zeros(T, dtype=omega.dtype)
            if tau >= 0:
                v[tau:] = omega[q, :T - tau] if tau else omega[q]
            else:
                v[:T + tau] = omega[q, -tau:]
            rows.append(v / v.norm().clamp_min(1e-8))
            q_of.append(q); tau_of.append(tau)
    return torch.stack(rows), torch.tensor(q_of), torch.tensor(tau_of)


def _nnls(G: torch.Tensor, rhs: torch.Tensor, sweeps: int = 12) -> torch.Tensor:
    """Batched nonnegative least squares for tiny systems (projected coordinate descent).
    Exact when the unconstrained solution is already nonnegative, the common case here."""
    b, k = rhs.shape
    a = torch.zeros_like(rhs)
    diag = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(1e-9)
    for _ in range(sweeps):
        for j in range(k):
            resid = rhs[:, j] - torch.einsum("bk,bk->b", G[:, j, :], a) + G[:, j, j] * a[:, j]
            a[:, j] = (resid / diag[:, j]).clamp_min(0.0)
    return a


def _lstsq(G: torch.Tensor, rhs: torch.Tensor, ridge: float) -> torch.Tensor:
    n = G.shape[-1]
    eye = torch.eye(n, dtype=G.dtype, device=G.device)[None]
    return torch.linalg.solve(G + ridge * eye, rhs.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# inference: matching pursuit over place x shape x lag
# --------------------------------------------------------------------------- #
def infer(Hb: torch.Tensor, Y: torch.Tensor, omega: torch.Tensor, bank, gram,
          pos: torch.Tensor, cfg: Config):
    """One batch of spikes -> support, coefficients, captured energy.

    Returns (idx_n, idx_k, coeff, amp, captured, active) where `idx_k` indexes the
    shifted-atom bank for one-hot shapes and is -1 for free shapes.
    """
    B, C, N = Hb.shape
    dev = Y.device
    onehot = cfg.shape == "onehot"
    n_shift = (2 * cfg.max_shift + 1) if onehot else 1

    # spatial shortlist on unshifted basis energy: the product dictionary is
    # N x M x nshift, too large to correlate in full at M=64 with 21 lags
    Z0 = torch.einsum("bct,mt->bcm", Y, omega)
    if cfg.shortlist and cfg.shortlist < N:
        rank = torch.einsum("bcn,bcm->bnm", Hb, Z0).square().sum(-1)
        top = rank.topk(cfg.shortlist, dim=1).indices
        Hb = torch.gather(Hb, 2, top[:, None, :].expand(B, C, cfg.shortlist))
        pos_b = pos[top]
        N = cfg.shortlist
    else:
        top, pos_b = None, pos[None].expand(B, N, 3)

    if onehot:
        W = torch.einsum("bct,kt->bck", Y, bank)          # (B,C,K)
        A = torch.einsum("bcn,bck->bnk", Hb, W)           # (B,N,K)
        K = bank.shape[0]
    else:
        A = torch.einsum("bcn,bcm->bnm", Hb, Z0)          # (B,N,M)
        K = omega.shape[0]
    resid = A.clone()
    idx_n = torch.full((B, cfg.R), -1, dtype=torch.long, device=dev)
    idx_k = torch.full((B, cfg.R), -1, dtype=torch.long, device=dev)
    amp = torch.zeros(B, cfg.R, dtype=Y.dtype, device=dev)
    coeff = torch.zeros(B, cfg.R, omega.shape[0], dtype=Y.dtype, device=dev)
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    blocked = torch.zeros(B, N, dtype=torch.bool, device=dev)
    captured = torch.zeros(B, dtype=Y.dtype, device=dev)
    ar = torch.arange(B, device=dev)

    for r in range(cfg.R):
        if onehot:
            # a >= 0 means only a positive correlation can reduce the residual
            sc = (resid.clamp_min(0.0) if cfg.nonneg else resid.abs()).square()
            sc = sc.masked_fill(blocked[:, :, None], -1.0)
            best_val, best = sc.reshape(B, N * K).max(1)
            prop_n, prop_k = best // K, best % K
        else:
            sc = resid.square().sum(-1).masked_fill(blocked, -1.0)   # ||h^T Z||^2
            best_val, prop_n = sc.max(1)
            prop_k = torch.full_like(prop_n, -1)
        cand_n = torch.cat([idx_n[:, :r], prop_n[:, None]], 1)
        cand_k = torch.cat([idx_k[:, :r], prop_k[:, None]], 1)
        act = cand_n >= 0
        act[:, r] = True
        sel = cand_n.clamp_min(0)
        Hsel = torch.gather(Hb, 2, sel[:, None, :].expand(B, C, r + 1))
        Gs = torch.einsum("bcr,bcs->brs", Hsel, Hsel)

        if onehot:
            Gt = gram[cand_k.clamp_min(0)[:, :, None], cand_k.clamp_min(0)[:, None, :]]
            G = Gs * Gt * act[:, :, None] * act[:, None, :]
            torch.diagonal(G, dim1=-2, dim2=-1).add_((~act).to(Y.dtype))
            rhs = A[ar[:, None], sel, cand_k.clamp_min(0)] * act
            a_new = (_nnls(G, rhs) if cfg.nonneg else _lstsq(G, rhs, cfg.ridge)) * act
            new_cap = 2.0 * (a_new * rhs).sum(1) - torch.einsum("br,brs,bs->b",
                                                                a_new, G, a_new)
        else:
            # free coefficients: one M-vector per source, joint ridge solve
            G = Gs * act[:, :, None] * act[:, None, :]
            torch.diagonal(G, dim1=-2, dim2=-1).add_((~act).to(Y.dtype))
            Rhs = torch.gather(A, 1, sel[:, :, None].expand(B, r + 1, A.shape[2]))
            Rhs = Rhs * act[:, :, None]
            eye = torch.eye(r + 1, dtype=Y.dtype, device=dev)[None]
            Bc = torch.linalg.solve(G + cfg.ridge * eye, Rhs)        # (B,r+1,M)
            new_cap = (2.0 * (Bc * Rhs).sum((1, 2))
                       - torch.einsum("brm,brs,bsm->b", Bc, G, Bc))
            a_new = torch.linalg.vector_norm(Bc, dim=-1) * act

        gain = new_cap - captured
        accept = (alive & (best_val > 0) & (gain > cfg.min_gain)) if r else alive
        idx_n[:, r] = torch.where(accept, prop_n, torch.full_like(prop_n, -1))
        idx_k[:, r] = torch.where(accept, prop_k, torch.full_like(prop_k, -1))
        amp[:, :r + 1] = torch.where(accept[:, None], a_new, amp[:, :r + 1])
        if onehot:
            # one-hot: the coefficient vector is the amplitude at the selected atom
            q_of = cand_k.clamp_min(0) // n_shift
            newc = torch.zeros_like(coeff[:, :r + 1]).scatter_(
                2, q_of[:, :, None], (a_new * act)[:, :, None])
            coeff[:, :r + 1] = torch.where(accept[:, None, None], newc,
                                           coeff[:, :r + 1])
        else:
            coeff[:, :r + 1] = torch.where(accept[:, None, None], Bc,
                                           coeff[:, :r + 1])
        captured = torch.where(accept, new_cap, captured)
        alive = alive & accept
        if r + 1 == cfg.R or not bool(alive.any()):
            break

        live = idx_n[:, :r + 1] >= 0
        sel = idx_n[:, :r + 1].clamp_min(0)
        Hsel = torch.gather(Hb, 2, sel[:, None, :].expand(B, C, r + 1))
        Gall = torch.einsum("bcn,bcr->bnr", Hb, Hsel)
        resid = A.clone()
        for j in range(r + 1):
            if onehot:
                w = Gall[:, :, j] * (amp[:, j] * live[:, j])[:, None]
                resid -= w[:, :, None] * gram[idx_k[:, j].clamp_min(0)][:, None, :]
            else:
                resid -= (Gall[:, :, j] * live[:, j][:, None])[:, :, None] \
                    * coeff[:, j][:, None, :]
        selpos = torch.gather(pos_b, 1, sel[:, :, None].expand(B, r + 1, 3))
        d = torch.linalg.vector_norm(pos_b[:, :, None, :] - selpos[:, None, :, :], dim=-1)
        blocked |= ((d < cfg.min_separation) & live[:, None, :]).any(-1)

    active = idx_n >= 0
    if top is not None:
        idx_n = torch.where(active, torch.gather(top, 1, idx_n.clamp_min(0)),
                            torch.full_like(idx_n, -1))
    return idx_n, idx_k, coeff, amp * active, captured, active


# --------------------------------------------------------------------------- #
# codebook learning
# --------------------------------------------------------------------------- #
def _orthonormalise(W: torch.Tensor) -> torch.Tensor:
    """Nearest row-orthonormal matrix (polar factor)."""
    U, _, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return (U @ Vh).float()


def project_cone(c: torch.Tensor, phi: torch.Tensor, cos_max: float) -> torch.Tensor:
    """Closest unit vector to `c` within the cone of half-angle acos(cos_max) about phi."""
    c = c / c.norm().clamp_min(1e-9)
    if float(c @ phi) < 0:                    # a cone is one-sided: keep the polarity
        c = -c
    cosang = float(c @ phi)
    if cosang >= cos_max:
        return c
    perp = c - cosang * phi
    n = perp.norm()
    if float(n) < 1e-9:
        return phi.clone()
    sin_max = float(np.sqrt(max(1.0 - cos_max ** 2, 0.0)))
    return cos_max * phi + sin_max * (perp / n)


def fix_polarity(protos: torch.Tensor) -> torch.Tensor:
    """Prototype 0 positive-going, 1 negative-going; others as found."""
    out = protos.clone()
    for p in range(len(out)):
        ext = out[p][out[p].abs().argmax()]
        if (ext < 0) == (p % 2 == 0):
            out[p] = -out[p]
    return out / out.norm(dim=1, keepdim=True).clamp_min(1e-9)


def basis_proposal(rec, idx, off_all, dic, omega, protos, assign, bank, gram,
                   q_of, tau_of, cfg, dev):
    """One codebook update, by config. Returns (omega, protos) -- a PROPOSAL."""
    M, T = omega.shape
    foot = torch.as_tensor(dic.footprints, device=dev)
    pos = torch.as_tensor(dic.candidate_pos, device=dev)
    onehot = cfg.shape == "onehot"
    Cacc = torch.zeros(M, T, dtype=torch.float64)
    Wsum = torch.zeros(M, dtype=torch.float64)
    scatter = torch.zeros(T, T, dtype=torch.float64)

    for i in range(0, len(idx), cfg.batch):
        sub = idx[i:i + cfg.batch]
        Y, _ = load_batch(rec, sub, off_all, dev)
        conf = torch.as_tensor(dic.cfg_id_by_channel[rec.spike_channels[sub]],
                               dtype=torch.long, device=dev)
        Hb = foot[conf]
        idx_n, idx_k, coeff, amp, _, act = infer(Hb, Y, omega, bank, gram, pos, cfg)
        B = len(sub)
        if not onehot:
            # free + orthonormal: accumulate the residual-explaining time scatter and
            # take its leading M eigenvectors -- the exact weighted-PCA block
            for r in range(cfg.R):
                live = act[:, r]
                if not bool(live.any()):
                    continue
                h = torch.gather(Hb, 2, idx_n[:, r].clamp_min(0)[:, None, None]
                                 .expand(B, Hb.shape[1], 1))[:, :, 0]
                proj = torch.einsum("bc,bct->bt", h, Y) * live[:, None]
                scatter += (proj.T.double() @ proj.double())
            continue
        for r in range(cfg.R):
            live = act[:, r]
            if not bool(live.any()):
                continue
            h = torch.gather(Hb, 2, idx_n[:, r].clamp_min(0)[:, None, None]
                             .expand(B, Hb.shape[1], 1))[:, :, 0]
            proj = torch.einsum("bc,bct->bt", h, Y) * (amp[:, r] * live)[:, None]
            tau = tau_of[idx_k[:, r].clamp_min(0)]
            q = q_of[idx_k[:, r].clamp_min(0)]
            for t_shift in torch.unique(tau).tolist():
                m = (tau == t_shift) & live
                if not bool(m.any()):
                    continue
                blk = proj[m]
                al = torch.zeros_like(blk)
                if t_shift >= 0:                      # undo this source's own lag
                    al[:, :T - t_shift] = blk[:, t_shift:]
                else:
                    al[:, -t_shift:] = blk[:, :T + t_shift]
                Cacc.index_add_(0, q[m], al.double())
                Wsum.index_add_(0, q[m], amp[m, r].double())

    if not onehot:
        evals, evecs = torch.linalg.eigh(scatter)
        return evecs[:, -M:].flip(1).T.contiguous().float(), protos

    if cfg.P > 0:
        cos_max = float(np.cos(np.radians(cfg.cone_deg)))
        new_atoms = omega.clone()
        for q in range(M):
            if float(Cacc[q].norm()) > 0:
                new_atoms[q] = project_cone(Cacc[q].float(), protos[assign[q]], cos_max)
        new_p = protos.clone()
        for p in range(len(protos)):
            grp = torch.nonzero(assign == p, as_tuple=False).squeeze(1)
            if len(grp) == 0:
                continue
            Wg = (new_atoms[grp] * Wsum[grp].float().clamp_min(1e-9)[:, None]).double()
            _, _, Vh = torch.linalg.svd(Wg, full_matrices=False)
            new_p[p] = Vh[0].float()
        new_p = fix_polarity(new_p)
        for q in range(M):                         # re-project after prototypes moved
            new_atoms[q] = project_cone(new_atoms[q], new_p[assign[q]], cos_max)
        return new_atoms, new_p

    # one-hot, no prior: shift-aligned orthogonal Procrustes
    U, _, Vh = torch.linalg.svd(Cacc, full_matrices=False)
    prop = (U @ Vh).float()
    return (prop if cfg.orthonormal else
            prop / prop.norm(dim=1, keepdim=True).clamp_min(1e-9)), protos


# --------------------------------------------------------------------------- #
# initialisation, evaluation, fit
# --------------------------------------------------------------------------- #
def init_omega_pca(rec, idx, off_all, M, batch, dev):
    """Top-M eigenvectors of the fit pool's time-time scatter."""
    T = rec.waveforms.shape[2]
    sc = torch.zeros(T, T, dtype=torch.float64)
    for i in range(0, len(idx), batch):
        Y, _ = load_batch(rec, idx[i:i + batch], off_all, dev)
        Yf = Y.reshape(-1, T).double()
        sc += Yf.T @ Yf
    ev, evec = torch.linalg.eigh(sc)
    return evec[:, -M:].flip(1).T.contiguous().float()


def init_prototypes(rec, idx, off_all, P, batch, dev):
    """Mean peak-aligned waveform of each polarity: the data's own mother spikes."""
    T = rec.waveforms.shape[2]
    acc = torch.zeros(2, T, dtype=torch.float64)
    cnt = torch.zeros(2, dtype=torch.float64)
    for i in range(0, len(idx), batch):
        Y, _ = load_batch(rec, idx[i:i + batch], off_all, dev)
        ch = Y.abs().amax(2).argmax(1)
        w = Y[torch.arange(len(Y)), ch]
        peak = w.abs().argmax(1)
        sign = torch.sign(w[torch.arange(len(w)), peak])
        rolled = torch.stack([torch.roll(w[j], int(T // 2 - peak[j]))
                              for j in range(len(w))])
        for s, row in ((0, sign > 0), (1, sign <= 0)):
            if bool(row.any()):
                acc[s] += rolled[row].sum(0).double(); cnt[s] += float(row.sum())
    pr = (acc / cnt.clamp_min(1)[:, None]).float()
    pr = pr / pr.norm(dim=1, keepdim=True).clamp_min(1e-9)
    if P > 2:
        pr = torch.cat([pr, pr[:1].repeat(P - 2, 1) + 0.01 * torch.randn(P - 2, T)])
    return fix_polarity(pr[:P])


def _spherical_kmeans(X, k, seed, iters=25):
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    if n <= k:
        return X.clone() if n else torch.zeros(k, X.shape[1])
    cs = [X[torch.randint(n, (1,), generator=g).item()]]
    for _ in range(k - 1):
        d = 1.0 - (X @ torch.stack(cs).T).abs().max(1).values
        p = (d.clamp_min(0) ** 2)
        cs.append(X[int(torch.multinomial(p / p.sum().clamp_min(1e-12), 1,
                                          generator=g))])
    Cm = torch.stack(cs)
    for _ in range(iters):
        lab = (X @ Cm.T).argmax(1)
        for j in range(k):
            m = lab == j
            if bool(m.any()):
                v = X[m].mean(0); Cm[j] = v / v.norm().clamp_min(1e-9)
    return Cm


def init_atoms_cone(rec, idx, off_all, protos, M, cos_max, seed, batch, dev):
    """Seed atoms by clustering REAL peak-aligned waveforms of the matching polarity.

    Random jitter inside the cone made the fits non-monotone in M, because the cone
    projection pins most atoms to the boundary and the initial direction then decides
    which deformation each atom commits to.
    """
    T = rec.waveforms.shape[2]
    P = len(protos)
    assign = torch.arange(M) % P
    groups = [[], []]
    for i in range(0, len(idx), batch):
        Y, _ = load_batch(rec, idx[i:i + batch], off_all, dev)
        ch = Y.abs().amax(2).argmax(1)
        w = Y[torch.arange(len(Y)), ch]
        peak = w.abs().argmax(1)
        sign = torch.sign(w[torch.arange(len(w)), peak])
        rolled = torch.stack([torch.roll(w[j], int(T // 2 - peak[j]))
                              for j in range(len(w))])
        rolled = rolled / rolled.norm(dim=1, keepdim=True).clamp_min(1e-9)
        groups[0].append(rolled[sign > 0]); groups[1].append(rolled[sign <= 0])
    groups = [torch.cat(g) if g else torch.zeros(0, T) for g in groups]
    atoms = torch.zeros(M, T)
    for p in range(P):
        rows = torch.nonzero(assign == p, as_tuple=False).squeeze(1)
        src = groups[p] if p < len(groups) and len(groups[p]) else protos[p][None]
        Cm = _spherical_kmeans(src, len(rows), seed + p)
        for j, q in enumerate(rows.tolist()):
            atoms[q] = project_cone(Cm[j % len(Cm)], protos[p], cos_max)
    return atoms, assign


def evaluate(rec, idx, off_all, dic, omega, bank, gram, q_of, tau_of, cfg, dev,
             collect=False, progress=False):
    foot = torch.as_tensor(dic.footprints, device=dev)
    pos = torch.as_tensor(dic.candidate_pos, device=dev)
    keys = ("source_index", "source_atom", "source_shift", "source_coeff",
            "source_amp", "support_size", "captured", "sse")
    out = {k: [] for k in keys}
    tot_sse = tot_y2 = 0.0
    t0 = time.perf_counter()
    for i in range(0, len(idx), cfg.batch):
        sub = idx[i:i + cfg.batch]
        Y, _ = load_batch(rec, sub, off_all, dev)
        conf = torch.as_tensor(dic.cfg_id_by_channel[rec.spike_channels[sub]],
                               dtype=torch.long, device=dev)
        idx_n, idx_k, coeff, amp, cap, act = infer(foot[conf], Y, omega, bank, gram,
                                                   pos, cfg)
        y2 = (Y * Y).sum((1, 2))
        sse = (y2 - cap).clamp_min(0.0)
        tot_sse += float(sse.sum()); tot_y2 += float(y2.sum())
        if collect:
            n_shift = (2 * cfg.max_shift + 1) if cfg.shape == "onehot" else 1
            q = (idx_k.clamp_min(0) // n_shift) if cfg.shape == "onehot" \
                else coeff.abs().argmax(2)
            tau = (tau_of[idx_k.clamp_min(0)] if cfg.shape == "onehot"
                   else torch.zeros_like(idx_n))
            for k, v in (("source_index", torch.where(act, idx_n, -torch.ones_like(idx_n))),
                         ("source_atom", torch.where(act, q, -torch.ones_like(q))),
                         ("source_shift", torch.where(act, tau, torch.zeros_like(tau))),
                         ("source_coeff", coeff), ("source_amp", amp),
                         ("support_size", act.sum(1)), ("captured", cap), ("sse", sse)):
                out[k].append(np.asarray(v.detach().cpu()))
        if progress and (i // cfg.batch) % 200 == 0:
            print(f"    {min(i + cfg.batch, len(idx)):,}/{len(idx):,} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    res = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
    res["nmse"] = tot_sse / (len(idx) * rec.waveforms.shape[1]
                             * rec.waveforms.shape[2]) / VAR_NP1
    res["ve"] = 1.0 - tot_sse / max(tot_y2, 1e-12)
    res["wall_s"] = time.perf_counter() - t0
    return res


DEFAULT_DICT = {"monopole": "lrn512_monopole_M8_kmeans",
                "gauss": "lrn512_gauss_M8_kmeans"}


def fit(cfg: Config, out_dir: Path, progress: bool = False) -> dict:
    """Fit one configuration and write the standard multipole state triplet."""
    cfg.validate()
    dev = "cpu"
    rec = D.load(cfg.dataset)
    dict_tag = cfg.dictionary or DEFAULT_DICT.get(cfg.kernel)
    if not dict_tag:
        raise ValueError(f"no default learned dictionary for kernel {cfg.kernel!r}; "
                         f"pass dictionary=<tag> (fit one with fit_learned.py)")
    dic = load_dictionary(REPO / "zncc/runs/lattice", dict_tag, cfg.dataset,
                          device=dev, cache=REPO / "zncc/runs/multipole/cache")
    off_all = (rec.channel_offsets().astype(np.float32)
               - dic.anchor_shift[:, None, :])
    rng = np.random.default_rng(cfg.seed)
    fit_idx = np.sort(rng.choice(rec.n_spikes, min(cfg.n_fit, rec.n_spikes),
                                 replace=False))
    tag = cfg.tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{tag}: {cfg.shape} shapes, R={cfg.R} M={cfg.M} kernel={cfg.kernel} "
          f"shift=±{cfg.max_shift} P={cfg.P} orth={cfg.orthonormal}", flush=True)

    protos = (init_prototypes(rec, fit_idx, off_all, cfg.P, cfg.batch, dev)
              if cfg.P else torch.zeros(0, rec.waveforms.shape[2]))
    if cfg.P:
        cos_max = float(np.cos(np.radians(cfg.cone_deg)))
        omega, assign = init_atoms_cone(rec, fit_idx, off_all, protos, cfg.M,
                                        cos_max, cfg.seed, cfg.batch, dev)
    else:
        omega = init_omega_pca(rec, fit_idx, off_all, cfg.M, cfg.batch, dev)
        assign = torch.zeros(cfg.M, dtype=torch.long)

    def banks(om):
        if cfg.shape != "onehot":
            return None, None, None, None
        b, q, t = shift_bank(om, cfg.max_shift)
        return b, b @ b.T, q, t

    history = []
    for it in range(1, cfg.outer_iters + 1):
        t0 = time.perf_counter()
        bank, gram, q_of, tau_of = banks(omega)
        before = evaluate(rec, fit_idx, off_all, dic, omega, bank, gram, q_of, tau_of,
                          cfg, dev)
        prop_o, prop_p = basis_proposal(rec, fit_idx, off_all, dic, omega, protos,
                                        assign, bank, gram, q_of, tau_of, cfg, dev)
        # Only the free+orthonormal eigensolve is exact; the shift-aligned and cone
        # steps are approximate once shifted copies overlap, so every proposal is
        # tested and backtracked rather than applied blindly.
        accepted, best = False, float(before["nmse"])
        for alpha in (1.0, 0.5, 0.25):
            cand = prop_o if alpha == 1.0 else (
                _orthonormalise((1 - alpha) * omega + alpha * prop_o) if cfg.orthonormal
                else torch.stack([project_cone((1 - alpha) * omega[q] + alpha * prop_o[q],
                                               prop_p[assign[q]], cos_max)
                                  for q in range(cfg.M)]) if cfg.P else
                ((1 - alpha) * omega + alpha * prop_o))
            cb, cg, cq, ct = banks(cand)
            trial = evaluate(rec, fit_idx, off_all, dic, cand, cb, cg, cq, ct, cfg, dev)
            if trial["nmse"] <= best:
                omega, protos, best, accepted = cand, prop_p, float(trial["nmse"]), True
                break
        history.append({"step": it, "nmse": float(before["nmse"]),
                        "nmse_after_basis": best, "basis_accepted": accepted,
                        "wall_s": time.perf_counter() - t0})
        print(f"  outer {it}/{cfg.outer_iters}: nMSE {before['nmse']:.5f} -> {best:.5f}"
              f" [{'accepted' if accepted else 'ROLLED BACK'}]", flush=True)
        if not accepted:
            break

    bank, gram, q_of, tau_of = banks(omega)
    eval_idx = (np.arange(rec.n_spikes) if cfg.n_eval <= 0
                else np.sort(rng.choice(rec.n_spikes, cfg.n_eval, replace=False)))
    print(f"  evaluating {len(eval_idx):,} spikes", flush=True)
    res = evaluate(rec, eval_idx, off_all, dic, omega, bank, gram, q_of, tau_of, cfg,
                   dev, collect=True, progress=progress)

    src = res["source_index"]; active = src >= 0
    pos = dic.source_positions(src, rec.spike_channels[eval_idx], rec.anchors)
    amp = res["source_amp"] * active
    tot = amp.sum(1, keepdims=True)
    w = amp / np.maximum(tot, 1e-12)
    dead = (tot[:, 0] <= 0) & active.any(1)
    w[dead, 0] = 1.0                      # shares undefined when every amplitude is 0
    dom = np.take_along_axis(pos, np.argmax(amp, 1)[:, None, None].repeat(3, 2), 1)[:, 0]
    bary = (np.where(active[:, :, None], np.nan_to_num(pos), 0.0) * w[:, :, None]).sum(1)
    from spiketensor.unified import _min_pair_separation
    state = {
        "spike_index": eval_idx.astype(np.int64),
        "source_index": src.astype(np.int32),
        "source_pos": pos.astype(np.float32),
        "source_coeff": res["source_coeff"].astype(np.float32),
        "source_amp": amp.astype(np.float32),
        "source_weight": w.astype(np.float32),
        "support_size": res["support_size"].astype(np.int32),
        "condition": np.ones(len(eval_idx), np.float32),
        "leaveout_delta": (amp ** 2).astype(np.float32),
        "captured": res["captured"].astype(np.float32),
        "sse": res["sse"].astype(np.float32),
        "pos_dominant": dom.astype(np.float32),
        "pos_barycenter": bary.astype(np.float32),
        "source_temporal_atom": res["source_atom"].astype(np.int16),
        "source_shift": res["source_shift"].astype(np.int16),
        "pair_separation": _min_pair_separation(pos, active).astype(np.float32),
        "omega": omega.detach().cpu().numpy().astype(np.float32),
        "basis_orthonormal": np.asarray(bool(cfg.orthonormal)),
        "candidate_pos": dic.candidate_pos.astype(np.float32),
        "anchor_shift": dic.anchor_shift.astype(np.float32),
        "model": np.asarray(tag),
        "dictionary_tag": np.asarray(dic.tag),
        "dictionary_fingerprint": np.asarray(dic.fingerprint()),
        "args_json": np.asarray(json.dumps(asdict(cfg), sort_keys=True)),
    }
    if cfg.P:
        state["prototypes"] = protos.detach().cpu().numpy().astype(np.float32)
        state["atom_prototype"] = assign.numpy().astype(np.int16)
    validate_multipole_state(state)
    np.savez_compressed(out_dir / f"multipole_{tag}.npz", **state)
    torch.save({"omega": omega.detach().cpu(), "prototypes": protos.detach().cpu(),
                "candidate_pos": dic.candidate_pos,
                "dictionary_metadata": dic.metadata,
                "dictionary_fingerprint": dic.fingerprint(),
                "model": tag, "config": asdict(cfg)},
               out_dir / f"codebook_{tag}.pt")
    summary = {
        "tag": tag, "model": tag, "config": asdict(cfg),
        "dictionary": dic.metadata, "dictionary_fingerprint": dic.fingerprint(),
        "fit_history": history,
        "evaluation": {"nmse": float(res["nmse"]), "ve": float(res["ve"]),
                       "wall_s": float(res["wall_s"])},
        "full_data": bool(len(eval_idx) == rec.n_spikes),
        "M": cfg.M, "R": cfg.R, "P": cfg.P, "max_shift": cfg.max_shift,
        "support_mean": float(state["support_size"].mean()),
        "multi_source_fraction": float((state["support_size"] > 1).mean()),
        "shift_nonzero_fraction": float((res["source_shift"][active] != 0).mean())
        if cfg.max_shift else 0.0,
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2,
                                                            default=float))
    print(f"  final nMSE {res['nmse']:.5f}  VE {100*res['ve']:.1f}%  "
          f"support {summary['support_mean']:.2f}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for f in Config.__dataclass_fields__.values():
        if isinstance(f.default, bool):
            ap.add_argument(f"--{f.name.replace('_','-')}", dest=f.name,
                            action="store_true", default=f.default)
            ap.add_argument(f"--no-{f.name.replace('_','-')}", dest=f.name,
                            action="store_false")
        else:
            t = type(f.default) if f.default is not None else str
            ap.add_argument(f"--{f.name.replace('_','-')}", dest=f.name,
                            type=t, default=f.default)
    ap.add_argument("--out", type=Path, default=REPO / "zncc/runs/unified")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    cfg = Config(**{k: getattr(a, k) for k in Config.__dataclass_fields__})
    fit(cfg, a.out, a.progress)


if __name__ == "__main__":
    main()
