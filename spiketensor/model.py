"""Tensor factorization of spike waveforms into a shared spatial-footprint codebook.

    Y[s,c,t]  ~=  sum_k  pi[s,k] * g_k(D[s,c]) * a[k,t]

with pi >= 0, ||a_k|| = 1, and g_k peak-normalized to 1 at its own centre, so pi carries
all amplitude in waveform units. D[s,c] is the offset of neighbourhood slot c from spike
s's anchor, in micrometres.

THIS IS NOT PLAIN CP. The channel mode indexes a neighbourhood SLOT, and slot c is a
different physical contact for every spike, so the spatial factor cannot be a free
S-independent vector -- it has to be a FUNCTION evaluated at spike-specific coordinates.
That is what makes {mu_k} a shared, anchor-relative codebook instead of 384 unrelated
positions, and it is where the model's identifiability comes from: K components are tied
across millions of spikes. A single spike's 10 amplitudes remain underdetermined in
isolation (measured elsewhere in this project: a free non-negative 1/r mixture over 8960
voxels fits ANY 10-channel pattern exactly, including geometrically impossible permuted
ones, so it localizes nothing). Sharing is the constraint, not the kernel.

TWO PREPROCESSING FACTS THAT ARE NOT OPTIONAL, both measured on 20k np1 spikes:

  DC. 65.5% of raw waveform energy is per-channel, per-spike DC offset, and 89.1% of
  spikes carry BOTH signs of it across their 10 channels. Since pi >= 0 and g >= 0, the
  sign of a component can only come from a_k -- one time course shared by all channels --
  which cannot produce channel-dependent sign. So per-channel DC must be removed. It also
  more than halves the separable-model floor: free rank-1 nMSE 0.2285 raw -> 0.1029
  DC-removed.

  DEPTH IS NOT A FREE 3rd COORDINATE. channel_locations is (384, 2): every contact lies
  on the z=0 plane. For a Gaussian with block-diagonal Sigma, mu_z enters the exponent
  only as the constant mu_z^2/sigma_z^2, identical for every channel, which pi absorbs
  (verified: footprint shape identical for mu_z = 5, 20, 50 um; only the peak scale moves,
  2.9e-1 -> 2.1e-9 -> 5.2e-55). So a planar Gaussian has TWO location parameters and depth
  lives in Sigma. Two kernels are provided:

    "gauss"    g_k(d) = exp(-0.5 * ||L_k^T (d - mu_k)||^2)     mu in R^2, L_k lower 2x2.
               Elliptical, 5 params. Depth is implicit in the width.
    "monopole" g_k(d) = z_k / sqrt(||d - mu_k||^2 + z_k^2)      mu in R^2, z_k > 0.
               Isotropic, 3 params, depth IDENTIFIABLE because it sets the falloff shape
               rather than a scale. Peak-normalized by construction (g = 1 at d = mu).
"""
from __future__ import annotations

import numpy as np
import torch


class Codebook(torch.nn.Module):
    """The shared part of the factorization: K spatial kernels and K time courses."""

    def __init__(self, K: int, kernel: str = "monopole", n_t: int = 90,
                 span=(30.0, 45.0), depth0=20.0, width0=15.0):
        super().__init__()
        self.K, self.kernel, self.n_t = K, kernel, n_t
        # centres on a coarse grid over the neighbourhood span (the 10 nearest contacts
        # reach +-25.6 um in x and +-40 um in y), deliberately a little wider so a
        # component can sit outside the footprint it explains
        nx = int(round(np.sqrt(K * span[0] / span[1]))) or 1
        ny = int(np.ceil(K / nx))
        gx = np.linspace(-span[0], span[0], nx)
        gy = np.linspace(-span[1], span[1], ny)
        mu = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)[:K]
        self.mu = torch.nn.Parameter(torch.as_tensor(mu, dtype=torch.float32))
        # softplus-parameterized so the scale stays positive without a constraint
        self.raw_z = torch.nn.Parameter(
            torch.full((K,), float(np.log(np.expm1(depth0)))))
        # lower-triangular Cholesky of the inverse covariance: [l11, l21, l22]
        l0 = float(np.log(np.expm1(1.0 / width0)))
        self.raw_L = torch.nn.Parameter(
            torch.tensor([[l0, 0.0, l0]] * K, dtype=torch.float32))
        self.a = torch.nn.Parameter(torch.zeros(K, n_t))

    # ------------------------------------------------------------------ #
    def scales(self):
        """Per-component depth (monopole) or (sx, sy, corr-ish) width (gauss), in um."""
        if self.kernel == "monopole":
            return torch.nn.functional.softplus(self.raw_z)
        L = self._chol()
        return torch.linalg.inv(L @ L.transpose(1, 2)).diagonal(dim1=1, dim2=2).sqrt()

    def _chol(self):
        d = torch.nn.functional.softplus(self.raw_L[:, [0, 2]])
        L = torch.zeros(self.K, 2, 2, device=self.raw_L.device, dtype=self.raw_L.dtype)
        L[:, 0, 0] = d[:, 0]
        L[:, 1, 0] = self.raw_L[:, 1]
        L[:, 1, 1] = d[:, 1]
        return L

    def footprint(self, off: torch.Tensor) -> torch.Tensor:
        """off (B,C,2) offsets in um -> g (B,C,K), each column peaking at 1."""
        d = off[:, :, None, :] - self.mu[None, None, :, :]          # (B,C,K,2)
        if self.kernel == "monopole":
            z = torch.nn.functional.softplus(self.raw_z)
            return z / torch.sqrt((d * d).sum(-1) + z * z)
        e = torch.einsum("bckj,kji->bcki", d, self._chol())
        return torch.exp(-0.5 * (e * e).sum(-1))

    def unit_a(self) -> torch.Tensor:
        return self.a / self.a.norm(dim=1, keepdim=True).clamp_min(1e-8)


# ---------------------------------------------------------------------------- #
def solve_pi(G: torch.Tensor, b: torch.Tensor, iters: int = 60,
             l1: float = 0.0) -> torch.Tensor:
    """argmin_{pi>=0} 0.5 pi^T G pi - b^T pi, batched, by FISTA.

    The pi-step is the one block that can be minimized EXACTLY rather than descended, so
    it is solved to convergence every step instead of being given a learning rate. The
    Gram factorizes as G = (g^T g) * (a^T a) -- a per-spike spatial Gram times a shared
    temporal one -- so it costs a (B,K,K) einsum, not a 900-row least squares.
    Lipschitz constant is bounded by the Frobenius norm, which upper-bounds the spectral
    norm and needs no eigen-solve.

    l1 > 0 adds rho * sum(pi). Under pi >= 0 the L1 norm IS the sum, so the whole penalty
    folds into the linear term as b - rho and costs nothing; the non-negative projection
    already does the soft-thresholding. It shrinks amplitude as well as support, so read
    the reconstruction error with that in mind.
    """
    if l1:
        b = b - l1
    B, K = b.shape
    L = G.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None]
    x = torch.zeros_like(b)
    y = x.clone()
    t = torch.ones(B, 1, device=b.device, dtype=b.dtype)
    for _ in range(iters):
        grad = torch.bmm(G, y[:, :, None])[:, :, 0] - b
        xn = torch.clamp(y - grad / L, min=0.0)
        tn = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
    return x


def batch_terms(cb: Codebook, Y: torch.Tensor, off: torch.Tensor):
    """Everything the pi-step and the loss need, without materializing the prediction.

    ||Y - sum_k pi_k g_k a_k||^2 = ||Y||^2 - 2 pi.b + pi^T G pi, so the residual is
    available in closed form from (G, b) and there is never a (B,10,90) prediction
    tensor in memory or in the graph.
    """
    g = cb.footprint(off)                      # (B,C,K)
    a = cb.unit_a()                            # (K,T)
    Sa = a @ a.t()                             # (K,K)
    Sg = torch.einsum("bck,bcl->bkl", g, g)    # (B,K,K)
    G = Sg * Sa[None]
    P = torch.einsum("bct,kt->bck", Y, a)
    b = torch.einsum("bck,bck->bk", g, P)
    return g, a, G, b


def residual_sq(G, b, pi, y2):
    return (y2 - 2.0 * (pi * b).sum(1)
            + torch.einsum("bk,bkl,bl->b", pi, G, pi))


# ---------------------------------------------------------------------------- #
class GridCodebook(torch.nn.Module):
    """Position x shape, chosen independently:

        Y[s,c,t] ~= sum_p sum_q pi[s,p,q] g_p(D[s,c]) a_q[t]

    WHY THIS RATHER THAN CP. In the CP form every atom welds one position to one time
    course, so a spike that needs three temporal components to describe its waveform must
    borrow three atoms -- which, unless they happen to share a centre, drags its position
    estimate across the codebook. Measured on the CP fits: pi support 29/32 and 56/64, and
    the pi-weighted position collapses to slope 0.05-0.10 against the monopole. Here the
    same spike can put all its mass on ONE p and spread freely over q, so temporal
    richness no longer costs positional sharpness. The position marginal sum_q pi[s,p,q]
    is the object to watch.

    Per-spike freedom rises (P*Q instead of K) but the SHARED parameter count falls:
    3P + 90Q instead of 93K.
    """

    def __init__(self, P: int, Q: int, kernel: str = "monopole", n_t: int = 90,
                 span=(30.0, 45.0), depth0=20.0, width0=15.0):
        super().__init__()
        self.P, self.Q, self.kernel, self.n_t = P, Q, kernel, n_t
        self.space = Codebook(P, kernel, n_t, span, depth0, width0)
        self.a = torch.nn.Parameter(torch.zeros(Q, n_t))

    @property
    def mu(self):
        return self.space.mu

    def scales(self):
        return self.space.scales()

    def footprint(self, off):
        return self.space.footprint(off)

    def unit_a(self):
        return self.a / self.a.norm(dim=1, keepdim=True).clamp_min(1e-8)


def batch_terms_grid(cb: GridCodebook, Y, off):
    """(g, a, Sg, Sa, b) for the grid model.

    The Gram is a KRONECKER product, Sg (x) Sa, so it is never formed: the quadratic form
    is applied as Sg @ Pi @ Sa on the (P,Q) coefficient matrix, which costs O(P^2 Q + P
    Q^2) instead of O(P^2 Q^2).
    """
    g = cb.footprint(off)                        # (B,C,P)
    a = cb.unit_a()                              # (Q,T)
    Sg = torch.einsum("bcp,bcr->bpr", g, g)      # (B,P,P)
    Sa = a @ a.t()                               # (Q,Q)
    Pt = torch.einsum("bct,qt->bcq", Y, a)       # (B,C,Q)
    b = torch.einsum("bcp,bcq->bpq", g, Pt)      # (B,P,Q)
    return g, a, Sg, Sa, b


def solve_pi_grid(Sg, Sa, b, iters: int = 60, l1: float = 0.0):
    """argmin_{Pi>=0} 0.5 <Pi, Sg Pi Sa> - <b, Pi>, batched FISTA on the (P,Q) matrix.

    ||Sg (x) Sa||_2 = ||Sg||_2 ||Sa||_2 <= ||Sg||_F ||Sa||_F, which is the step size used.
    """
    if l1:
        b = b - l1
    L = (Sg.flatten(1).norm(dim=1) * Sa.norm()).clamp_min(1e-8)[:, None, None]
    x = torch.zeros_like(b); y = x.clone()
    t = torch.ones(b.shape[0], 1, 1, device=b.device, dtype=b.dtype)
    for _ in range(iters):
        grad = torch.bmm(torch.bmm(Sg, y), Sa.expand(b.shape[0], -1, -1)) - b
        xn = torch.clamp(y - grad / L, min=0.0)
        tn = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
    return x


def residual_sq_grid(Sg, Sa, b, pi, y2):
    Gp = torch.bmm(torch.bmm(Sg, pi), Sa.expand(pi.shape[0], -1, -1))
    return y2 - 2.0 * (pi * b).sum((1, 2)) + (pi * Gp).sum((1, 2))


def solve_pi_grid_ent(Sg, Sa, b, beta: float, iters: int = 120, eps: float = 1e-9):
    """Grid pi-step with an ENTROPY penalty on the position marginal.

        min_{Pi>=0}  0.5 <Pi, Sg Pi Sa> - <b, Pi>  +  beta * H(w),
        w_p = (sum_q Pi[p,q]) / sum_pq Pi,     H(w) = -sum_p w_p log w_p

    WHY ENTROPY AND NOT L1. w is NORMALIZED, so H is scale-invariant: it says nothing
    about how loud a spike is, only about how spread its position is. L1 cannot separate
    those -- with Pi >= 0 it is just sum(Pi), so it shrinks amplitude and support
    together, which is why the rho sweep degraded reconstruction from 0.106 to 0.214
    while support fell. Entropy concentrates position at (in principle) no amplitude cost.

    Gradient of the penalty, derived rather than autograd'd so the inner loop stays cheap.
    With m_p = sum_q Pi[p,q] and M = sum_p m_p, dw_p/dm_j = (delta_pj - w_p)/M, and since
    sum_p dw_p = 0 the log-plus-one collapses:

        dH/dm_j = (-log w_j - H) / M ,   and  dH/dPi[p,q] = dH/dm_p  for every q.

    That gradient is large and positive where w_p is small, so descent drives small
    positions toward zero and leaves the dominant ones alone -- concentration, not decay.
    """
    B, P, Q = b.shape
    L = (Sg.flatten(1).norm(dim=1) * Sa.norm()).clamp_min(1e-8)[:, None, None]
    # start from the unpenalized solution: the entropy term is non-convex, so a warm
    # start at the least-squares optimum matters
    x = solve_pi_grid(Sg, Sa, b, iters=40)
    for _ in range(iters):
        m = x.sum(2)                                   # (B,P)
        M = m.sum(1, keepdim=True).clamp_min(eps)
        w = (m / M).clamp_min(eps)
        H = -(w * w.log()).sum(1, keepdim=True)        # (B,1)
        dH = (-w.log() - H) / M                        # (B,P)
        grad = (torch.bmm(torch.bmm(Sg, x), Sa.expand(B, -1, -1)) - b
                + beta * dH[:, :, None])
        x = torch.clamp(x - grad / L, min=0.0)
    return x


def pos_entropy(pi, eps: float = 1e-9):
    m = pi.sum(2) if pi.ndim == 3 else pi
    w = (m / m.sum(1, keepdims=True).clip(eps)).clip(eps)
    return -(w * np.log(w)).sum(1) if isinstance(w, np.ndarray) else \
        -(w * w.log()).sum(1)


# ---------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# ONE FIXED SOURCE VOLUME for every hard-grid fit, so K is the only lattice knob and
# fits are comparable. Chosen from the monopole reference on np1, measured over 200k
# spikes: x p1..p99 = [-75.6, +70.8], y = [-43.1, +37.7], z median 13.7 with p99 107.8.
# The earlier fits used 5 different span_x, 5 span_y, 5 z-ranges and 12 sigmas, so K
# meant 3 um spacing in one run and 7 um in another and nothing could be compared.
VOL_X, VOL_Y, VOL_Z = 90.0, 60.0, (4.0, 128.0)


def lattice_for_K(K, lx=VOL_X, ly=VOL_Y, lz=VOL_Z):
    """Split the fixed volume into ~K sites as uniformly as possible.

    Cell counts are taken proportional to the extents so cells are as near-cubic as the
    volume allows, then nudged to land closest to the requested K. z is spaced
    GEOMETRICALLY (the footprint changes far faster from 4->8 um than from 120->128), so
    its extent enters the aspect ratio linearly but its levels do not.
    """
    Lx, Ly, Lz = 2 * lx, 2 * ly, lz[1] - lz[0]
    best = None
    h0 = (Lx * Ly * Lz / max(K, 1)) ** (1 / 3)
    for f in np.linspace(0.55, 1.8, 260):
        h = h0 * f
        nx, ny, nz = (max(2, int(round(L / h)) + 1) for L in (Lx, Ly, Lz))
        d = abs(nx * ny * nz - K)
        if best is None or d < best[0]:
            best = (d, nx, ny, nz)
    return best[1], best[2], best[3]


class HardCodebook(torch.nn.Module):
    """Fixed lattice of candidate sources; each spike picks exactly ONE.

        Y[s,c,t] ~= g_{k(s)}(D[s,c]) * sum_q v[s,q] a_q[t]

    mu_k is a UNIFORM lattice and the width is FIXED, so nothing spatial is learned --
    only the Q shared time courses. A spike contributes a hard assignment k(s) and Q
    shape coefficients. That removes every route by which the position estimate could be
    a weighted average, which is what collapsed it in the soft models: there the implied
    position was sum_p w_p mu_p over 29 of 32 active components and regressed to the
    codebook centre (slope 0.05-0.10 against the monopole). A one-hot cannot regress.

    v IS LEFT SIGNED. With a hard assignment the non-negativity that mattered -- that
    position is a choice, not a blend -- is enforced structurally, and free v makes the
    selection exact in closed form (below) instead of requiring K non-negative solves.
    Amplitude is then ||sum_q v_q a_q||.

    LATTICE. mu spans (x, y, z); z is depth below the probe plane.
      monopole  g_k(d) = z_k / sqrt(||d - mu_k^xy||^2 + z_k^2),  peak 1 at d = mu^xy.
                z varies over the lattice, so a 3-D lattice gives K genuinely distinct
                kernels and depth is swept rather than assumed.
      gauss     g_k(d) = exp(-||d - mu_k^xy||^2 / (2 sigma^2)).
                A z level would enter as exp(-z^2/2 sigma^2), a per-k CONSTANT, which the
                free v absorbs exactly -- so z slices would be perfectly degenerate and
                the argmax among them arbitrary. nz is therefore forced to 1 for gauss
                and depth lives in sigma.
    """

    def __init__(self, nx: int, ny: int, nz: int, sigma: float, kernel: str = "monopole",
                 n_t: int = 90, Q: int = 8, span=(28.0, 44.0), z_range=(5.0, 60.0),
                 pexp: float = 1.0):
        """Lattice over (x, y, s). s is the kernel's SCALE and is now swept for every
        kernel, not just the monopole:

            monopole  s / sqrt(d^2 + s^2)              s = depth
            power     (s / sqrt(d^2 + s^2)) ** pexp    s = depth
            gauss     exp(-d^2 / 2 s^2)                s = width
            exp       exp(-d / s)                      s = length constant

        WHY gauss NO LONGER FORCES nz=1. The earlier restriction was correct for what it
        addressed: at FIXED sigma, a z offset enters as exp(-z^2/2 sigma^2), a per-site
        constant that the free v absorbs exactly, so z slices were perfectly degenerate.
        Varying sigma ITSELF is a different thing -- it changes the footprint's SHAPE, not
        its scale -- so a sigma lattice is identifiable for every kernel here.

        SPAN. The default (28, 44) um barely exceeds the channel hull (25.6, 40). Measured
        against it, the monopole places 46.9% of spikes OUTSIDE |x| <= 25.6 and reaches
        +-76 um at the 1st/99th percentile, and every fit at the default span piles
        33-89% of its spikes onto the outermost column. A span that cannot reach where
        half the sources are is the binding constraint on slope_x, which sat at ~0.51 in
        every fit regardless of kernel, sigma, K or Q.
        """
        super().__init__()
        gx = np.linspace(-span[0], span[0], nx)
        gy = np.linspace(-span[1], span[1], ny)
        gz = (np.array([sigma]) if nz == 1
              else np.geomspace(z_range[0], z_range[1], nz))
        # NOTE the third coordinate is the source DEPTH for every kernel here; each
        # kernel supplies its own depth -> footprint law (see footprint()). For the
        # Gaussian that means the width IS the depth: a Gaussian with a free sigma and a
        # separate z is unidentifiable, because z would enter only as exp(-z^2/2 sigma^2),
        # a per-site constant the free v absorbs exactly.
        mu = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.K, self.Q, self.kernel, self.sigma = len(mu), Q, kernel, float(sigma)
        self.pexp = float(pexp)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.a = torch.nn.Parameter(torch.zeros(Q, n_t))

    def footprint(self, off: torch.Tensor) -> torch.Tensor:
        """off (B,C,2) -> g (B,C,K). No parameters: this is a constant of the fit."""
        d2 = ((off[:, :, None, :] - self.mu[None, None, :, :2]) ** 2).sum(-1)
        s = self.mu[:, 2]                       # per-site scale, from the lattice
        if self.kernel == "monopole":
            return s / torch.sqrt(d2 + s * s)
        if self.kernel == "power":
            return (s / torch.sqrt(d2 + s * s)) ** self.pexp
        if self.kernel == "exp":
            return torch.exp(-torch.sqrt(d2) / s)
        return torch.exp(-0.5 * d2 / (s * s))

    def unit_a(self):
        return self.a / self.a.norm(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def precompute_g(cb: HardCodebook, off_cfg: torch.Tensor):
    """g for each DISTINCT channel configuration, once.

    channel_offsets() takes only 106 distinct 10-channel configurations across the 384
    channels, and the lattice is fixed, so the whole (config, channel, component) tensor
    is a constant -- computed once instead of per batch. That is what makes K in the
    thousands affordable.
    """
    return cb.footprint(off_cfg)                       # (n_cfg, C, K)


def hard_assign(cb: HardCodebook, Y, gsel):
    """Exact one-hot selection and its shape coefficients.

    For a fixed component k the best v is unconstrained least squares, so the residual
    at the optimum is  ||Y||^2 - b_k^T (S^a)^-1 b_k / ||g_k||^2 . Selecting k is therefore
    just an argmax of that quadratic form -- no combinatorial search, no inner loop.
    """
    a = cb.unit_a()                                    # (Q,T)
    Sa = a @ a.t()
    Sai = torch.linalg.inv(Sa + 1e-6 * torch.eye(cb.Q, device=a.device, dtype=a.dtype))
    Pt = torch.einsum("bct,qt->bcq", Y, a)             # (B,C,Q)
    b = torch.einsum("bck,bcq->bkq", gsel, Pt)         # (B,K,Q)
    gn = (gsel * gsel).sum(1).clamp_min(1e-8)          # (B,K)
    score = torch.einsum("bkq,qr,bkr->bk", b, Sai, b) / gn
    k = score.argmax(1)                                # (B,)
    ar = torch.arange(len(k), device=k.device)
    v = torch.einsum("qr,br->bq", Sai, b[ar, k]) / gn[ar, k][:, None]
    return k, v, score[ar, k]


def residual_sq_hard(y2, best_score):
    return y2 - best_score
