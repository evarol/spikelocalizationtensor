"""Profiled objectives and sparse solvers for multipole localization.

Notation
--------
``Y`` is a C x T DC-removed waveform, ``omega`` is an M x T orthonormal
temporal basis, and ``Z = Y @ omega.T`` is C x M.  ``H`` is the C x R matrix
of unit-norm spatial footprints selected for one spike.

For the multi-shape model, ``Yhat = H @ B @ omega``.  Profiling ``B`` gives

    B* = (H.T H)^-1 H.T Z

and residual ``||Y||^2 - captured``.  For the shared-shape model,
``Yhat = (H pi) @ (v.T omega)`` and profiling ``v`` gives a Rayleigh quotient
in the non-negative simplex weight ``pi``.

All routines below operate on the projected C x M problem.  The energy outside
the temporal subspace is an additive constant and is restored by callers using
the original ``||Y||^2``.
"""
from __future__ import annotations

import itertools
from typing import Iterable, Tuple

import torch


def project_simplex(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Euclidean projection onto ``{x >= 0, sum(x)=1}``, batched.

    Implements the sorting construction of Duchi et al.  It is used only for
    supports of size at most four, so the sort is negligible compared with
    evaluating candidate footprints.
    """
    if x.shape[dim] == 0:
        raise ValueError("cannot project an empty axis onto the simplex")
    z = x.movedim(dim, -1)
    u, _ = torch.sort(z, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1) - 1.0
    j = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    view = (1,) * (z.ndim - 1) + (z.shape[-1],)
    cond = u - cssv / j.view(view) > 0
    rho = cond.sum(dim=-1, keepdim=True).clamp_min(1) - 1
    theta = torch.gather(cssv, -1, rho) / (rho.to(z.dtype) + 1.0)
    return torch.clamp(z - theta, min=0.0).movedim(-1, dim)


def group_soft_threshold(x: torch.Tensor, threshold: torch.Tensor | float,
                         dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Proximal operator for ``threshold * sum_rows ||row||_2``."""
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    t = torch.as_tensor(threshold, dtype=x.dtype, device=x.device)
    while t.ndim < n.ndim:
        t = t.unsqueeze(-1)
    scale = torch.clamp(1.0 - t / n.clamp_min(eps), min=0.0)
    return x * scale


def _gram_rhs(H: torch.Tensor, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``G=H.T H`` and ``R=H.T Z`` for batched or unbatched inputs."""
    return H.transpose(-2, -1) @ H, H.transpose(-2, -1) @ Z


def profile_multishape(H: torch.Tensor, Z: torch.Tensor, ridge: float = 1e-6,
                       eps: float = 1e-12) -> Tuple[torch.Tensor, ...]:
    """Profile source-specific shape coefficients for a fixed support.

    Parameters
    ----------
    H : (..., C, R)
        Selected, normally unit-norm, spatial footprints.
    Z : (..., C, M)
        Waveform projected into the temporal basis.
    ridge : float
        Numerical/regularization ridge added to the R x R Gram solve.

    Returns
    -------
    B : (..., R, M)
        Profiled source-specific temporal coefficients.
    captured : (...,)
        Reduction in the *unpenalized* projected residual, computed as
        ``2<B,R> - <B,G B>``.  This distinction matters when ridge is nonzero.
    condition : (...,)
        Condition number of the unregularized support Gram.
    gram : (..., R, R)
        The unregularized Gram, useful for diagnostics.
    """
    G, Rhs = _gram_rhs(H, Z)
    r = G.shape[-1]
    eye = torch.eye(r, dtype=G.dtype, device=G.device)
    B = torch.linalg.solve(G + ridge * eye, Rhs)
    linear = (B * Rhs).sum(dim=(-2, -1))
    quad = (B * (G @ B)).sum(dim=(-2, -1))
    captured = 2.0 * linear - quad
    ev = torch.linalg.eigvalsh(G)
    condition = ev[..., -1] / ev[..., 0].clamp_min(eps)
    return B, captured, condition, G


def profile_shared(H: torch.Tensor, Z: torch.Tensor, pi: torch.Tensor,
                   eps: float = 1e-12) -> Tuple[torch.Tensor, ...]:
    """Profile the common shape for one non-negative spatial mixture.

    Returns ``v``, captured score, mixture footprint, and its squared norm.
    ``pi`` need not sum to one for the score, but callers use the simplex to
    identify the displayed contribution weights.
    """
    h = (H * pi.unsqueeze(-2)).sum(dim=-1)
    hn = (h * h).sum(dim=-1).clamp_min(eps)
    v = (h.unsqueeze(-2) @ Z).squeeze(-2) / hn.unsqueeze(-1)
    captured = (v * v).sum(dim=-1) * hn
    return v, captured, h, hn


def _ratio_at(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
              d: torch.Tensor, e: torch.Tensor, f: torch.Tensor,
              eps: float) -> torch.Tensor:
    num = (a * x + b) * x + c
    den = ((d * x + e) * x + f).clamp_min(eps)
    return num / den


def _optimal_shared_pair_moments(x_i: torch.Tensor, x_j: torch.Tensor,
                                 cross: torch.Tensor, corr: torch.Tensor,
                                 alpha_min: float = 0.0,
                                 eps: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact shared-pair weight from the three RHS inner products.

    ``x_i=||h_i.T Z||^2``, ``x_j=||h_j.T Z||^2`` and
    ``cross=<h_i.T Z,h_j.T Z>`` may be broadcast tensors.  Keeping this scalar
    form avoids a B x L x N x M allocation in the full-dictionary beam search.
    """
    if not 0.0 <= alpha_min < 0.5:
        raise ValueError("alpha_min must lie in [0, 0.5)")
    # The ratio is the 2-D generalized Rayleigh quotient
    #     p.T [[x_i,cross],[cross,x_j]] p / p.T [[1,corr],[corr,1]] p.
    # Its only interior maximum is the leading generalized eigenvector.  If
    # that direction is outside the bounded positive cone, the constrained
    # maximum is at an endpoint.  This closed form is equivalent to evaluating
    # the roots of the derivative quadratic, but avoids materializing five
    # candidate-score tensors in the L-by-N beam.
    a = x_i + x_j - 2.0 * cross
    b = 2.0 * (cross - x_j)
    c = x_j
    d = 2.0 * (1.0 - corr)
    e = -d
    f = torch.ones_like(d)
    lo = torch.full_like(x_i + x_j, float(alpha_min))
    hi = torch.full_like(lo, float(1.0 - alpha_min))
    score_lo = _ratio_at(lo, a, b, c, d, e, f, eps)
    score_hi = _ratio_at(hi, a, b, c, d, e, f, eps)
    use_hi = score_hi > score_lo
    alpha = torch.where(use_hi, hi, lo)
    value = torch.where(use_hi, score_hi, score_lo)

    qa = (1.0 - corr.square()).clamp_min(eps)
    qb = 2.0 * corr * cross - x_i - x_j
    qc = x_i * x_j - cross.square()
    disc = torch.clamp(qb.square() - 4.0 * qa * qc, min=0.0)
    leading = (-qb + torch.sqrt(disc)) / (2.0 * qa)

    # Two algebraically equivalent null vectors; select the better-scaled one.
    off = leading * corr - cross
    p = torch.stack([off, x_i - leading], dim=-1)
    q = torch.stack([x_j - leading, off], dim=-1)
    choose_q = q.square().sum(-1) > p.square().sum(-1)
    vec = torch.where(choose_q.unsqueeze(-1), q, p)
    total = vec.sum(-1)
    candidate = vec[..., 0] / torch.where(
        total.abs() > eps, total, torch.ones_like(total))
    feasible = torch.isfinite(candidate) & torch.isfinite(leading) \
        & (total.abs() > eps) & (candidate >= lo - eps) & (candidate <= hi + eps)
    candidate = candidate.clamp(min=float(alpha_min), max=float(1.0 - alpha_min))
    # Re-evaluate the quotient at the recovered direction.  This avoids a small
    # float32 eigenvalue/vector inconsistency for nearly singular two-footprint
    # Grams while retaining the lower-memory closed form.
    candidate_score = _ratio_at(candidate, a, b, c, d, e, f, eps)
    improve = feasible & (candidate_score > value)
    alpha = torch.where(improve, candidate, alpha)
    value = torch.where(improve, candidate_score, value)
    return alpha, value


def optimal_shared_pair(r_i: torch.Tensor, r_j: torch.Tensor, corr: torch.Tensor,
                        alpha_min: float = 0.0,
                        eps: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact bounded weight for a two-source shared-shape mixture.

    ``r_i = h_i.T Z``, ``r_j = h_j.T Z``, and ``corr = h_i.T h_j`` for
    unit-norm footprints.  With ``h(alpha)=alpha*h_i+(1-alpha)*h_j``, the
    profiled captured score is a two-dimensional generalized Rayleigh quotient.
    The leading generalized eigenvector is its only unconstrained interior
    maximum; evaluating that feasible direction and both interval endpoints is
    exact up to floating-point arithmetic.

    ``alpha_min=0`` gives the nested at-most-two model.  A positive value gives
    an exact-two ablation with both weights bounded away from zero.
    """
    return _optimal_shared_pair_moments(
        (r_i * r_i).sum(dim=-1), (r_j * r_j).sum(dim=-1),
        (r_i * r_j).sum(dim=-1), corr, alpha_min=alpha_min, eps=eps)


def optimize_shared_weights(H: torch.Tensor, Z: torch.Tensor, steps: int = 40,
                            step_size: float = 0.25,
                            eps: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projected-gradient maximization of the shared-shape Rayleigh quotient.

    This is used for R=3/4 shortlist pursuit.  R=2 uses
    :func:`optimal_shared_pair`, which is exact.
    """
    G, Rhs = _gram_rhs(H, Z)
    A = Rhs @ Rhs.transpose(-2, -1)
    r = H.shape[-1]
    pi = torch.full(H.shape[:-2] + (r,), 1.0 / r,
                    dtype=H.dtype, device=H.device)
    best_pi = pi.clone()
    best = torch.full(H.shape[:-2], -float("inf"), dtype=H.dtype, device=H.device)
    for _ in range(steps):
        Ap = (A @ pi.unsqueeze(-1)).squeeze(-1)
        Gp = (G @ pi.unsqueeze(-1)).squeeze(-1)
        num = (pi * Ap).sum(-1)
        den = (pi * Gp).sum(-1).clamp_min(eps)
        val = num / den
        upd = val > best
        best = torch.where(upd, val, best)
        best_pi = torch.where(upd.unsqueeze(-1), pi, best_pi)
        grad = 2.0 * (Ap * den.unsqueeze(-1) - Gp * num.unsqueeze(-1)) \
            / den.square().unsqueeze(-1)
        # Normalize the ascent direction so one ill-conditioned support cannot
        # produce an arbitrarily large step before projection.
        grad = grad / torch.linalg.vector_norm(grad, dim=-1, keepdim=True).clamp_min(eps)
        pi = project_simplex(pi + step_size * grad)
    _, final, _, _ = profile_shared(H, Z, best_pi, eps=eps)
    return best_pi, final


def group_lasso_fista(H: torch.Tensor, Z: torch.Tensor, lam: float,
                      max_iter: int = 250, tol: float = 1e-6,
                      initial: torch.Tensor | None = None) -> Tuple[torch.Tensor, dict]:
    """Solve ``0.5||Z-HB||^2 + lam sum_n ||B_n||_2`` by FISTA.

    ``H`` is C x N and ``Z`` is C x M.  The Lipschitz constant is computed from
    the much smaller C x C matrix ``H H.T``, which is important because N may be
    5,120 while C is only 10.
    """
    if H.ndim != 2 or Z.ndim != 2:
        raise ValueError("group_lasso_fista currently expects one H and one Z")
    if H.shape[0] != Z.shape[0]:
        raise ValueError("H and Z must share the channel dimension")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    n, m = H.shape[1], Z.shape[1]
    x = (torch.zeros(n, m, dtype=H.dtype, device=H.device)
         if initial is None else initial.clone())
    y = x.clone()
    t = torch.tensor(1.0, dtype=H.dtype, device=H.device)
    lipschitz = torch.linalg.eigvalsh(H @ H.T)[-1].clamp_min(1e-8)
    history = []
    prev = None
    for it in range(max_iter):
        grad = H.T @ (H @ y - Z)
        xn = group_soft_threshold(y - grad / lipschitz, lam / lipschitz, dim=-1)
        tn = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
        if it % 5 == 0 or it == max_iter - 1:
            resid = Z - H @ x
            obj = 0.5 * (resid * resid).sum() + lam * torch.linalg.vector_norm(
                x, dim=-1).sum()
            value = float(obj.detach().cpu())
            history.append(value)
            if prev is not None and abs(prev - value) <= tol * max(1.0, abs(prev)):
                return x, {"iterations": it + 1, "objective": history,
                           "lipschitz": float(lipschitz.detach().cpu())}
            prev = value
    return x, {"iterations": max_iter, "objective": history,
               "lipschitz": float(lipschitz.detach().cpu())}


def support_combinations(n: int, r: int) -> Iterable[Tuple[int, ...]]:
    """Deterministic support enumeration, kept separate for auditability."""
    return itertools.combinations(range(n), r)


def canonicalize_sources(indices: torch.Tensor, coefficients: torch.Tensor,
                         inactive: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort each spike's active sources by coefficient norm for serialization.

    This ordering is a storage convention only.  The model and metrics remain
    invariant to source permutation.
    """
    amp = torch.linalg.vector_norm(coefficients, dim=-1)
    active = indices >= 0
    key = torch.where(active, amp, torch.full_like(amp, -float("inf")))
    order = key.argsort(dim=-1, descending=True)
    idx = torch.gather(indices, -1, order)
    coef = torch.gather(coefficients, -2,
                        order.unsqueeze(-1).expand(*order.shape, coefficients.shape[-1]))
    idx = torch.where(torch.gather(active, -1, order), idx,
                      torch.full_like(idx, inactive))
    return idx, coef

def selected_footprints(footprints, conf, indices):
    """Gather selected footprint columns as (B, C, R), zeroing inactive slots.

    Lives here because the research tree keeps it in the old per-model fitter, which this
    package does not ship.
    """
    import torch
    out = torch.zeros(len(indices), footprints.shape[1], indices.shape[1],
                      dtype=footprints.dtype, device=footprints.device)
    for ic in torch.unique(conf).tolist():
        rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
        for r in range(indices.shape[1]):
            idx = indices[rows, r]
            use = idx >= 0
            if bool(use.any()):
                out[rows[use], :, r] = footprints[int(ic)].T[idx[use]]
    return out
