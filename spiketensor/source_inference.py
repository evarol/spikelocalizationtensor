"""Screened and exact support inference for one- and two-source models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from spiketensor.source_core import (_optimal_shared_pair_moments, canonicalize_sources,
                   optimal_shared_pair,
                   optimize_shared_weights, profile_multishape)


@dataclass
class InferenceResult:
    indices: torch.Tensor            # (B,R), -1 inactive
    coefficients: torch.Tensor       # (B,R,M), source rows in the shared omega basis
    contributions: torch.Tensor      # (B,R), row-norm normalized over active sources
    amplitudes: torch.Tensor         # (B,R), coefficient row norms
    support_size: torch.Tensor       # (B,)
    captured: torch.Tensor           # (B,), reduction from ||Y||^2
    sse: torch.Tensor                # (B,), full waveform SSE if y2 was supplied
    condition: torch.Tensor          # (B,), support Gram condition
    leaveout_delta: torch.Tensor     # (B,R), SSE increase after deleting one source
    shortlist_indices: torch.Tensor  # (B,L)
    shortlist_scores: torch.Tensor   # (B,L), one-source captured scores
    diagnostics: Dict[str, torch.Tensor]


def topk_single(footprints: torch.Tensor, conf: torch.Tensor, Z: torch.Tensor,
                k: int = 16, candidate_chunk: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k one-source scores without materializing B x N x M globally.

    ``footprints`` has shape ``(n_cfg,C,N)`` and each column is unit norm.
    ``conf`` maps each spike to a configuration.  Candidate chunks bound memory
    for the 5,120-candidate fixed lattice.
    """
    if footprints.ndim != 3 or Z.ndim != 3:
        raise ValueError("footprints must be (cfg,C,N) and Z must be (B,C,M)")
    if len(conf) != len(Z):
        raise ValueError("one configuration id is required per spike")
    n = footprints.shape[2]
    k = min(int(k), int(n))
    if k < 1:
        raise ValueError("k must be positive")
    best_s = torch.full((len(Z), k), -float("inf"), dtype=Z.dtype, device=Z.device)
    best_i = torch.full((len(Z), k), -1, dtype=torch.long, device=Z.device)
    for ic in torch.unique(conf).tolist():
        rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
        zc = Z[rows]
        hs = footprints[int(ic)]
        bs = torch.full((len(rows), k), -float("inf"), dtype=Z.dtype, device=Z.device)
        bi = torch.full((len(rows), k), -1, dtype=torch.long, device=Z.device)
        for lo in range(0, n, candidate_chunk):
            hi = min(lo + candidate_chunk, n)
            # H.T Z -> (B,n_chunk,M).
            rhs = torch.einsum("cn,bcm->bnm", hs[:, lo:hi], zc)
            score = (rhs * rhs).sum(-1)
            take = min(k, hi - lo)
            sv, si = torch.topk(score, take, dim=1)
            si = si + lo
            all_s = torch.cat([bs, sv], dim=1)
            all_i = torch.cat([bi, si], dim=1)
            bs, order = torch.topk(all_s, k, dim=1)
            bi = torch.gather(all_i, 1, order)
        best_s[rows], best_i[rows] = bs, bi
    return best_i, best_s


def gather_shortlist(footprints: torch.Tensor, conf: torch.Tensor,
                     shortlist: torch.Tensor) -> torch.Tensor:
    """Gather per-spike shortlist footprints as ``(B,C,L)``."""
    b, l = shortlist.shape
    c = footprints.shape[1]
    out = torch.empty(b, c, l, dtype=footprints.dtype, device=footprints.device)
    for ic in torch.unique(conf).tolist():
        rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
        # H_cfg.T[index] -> (rows,L,C).
        out[rows] = footprints[int(ic)].T[shortlist[rows]].transpose(1, 2)
    return out


def _pair_grid(l: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(torch.triu_indices(l, l, offset=1, device=device))


def _pair_separation(candidate_pos: torch.Tensor, shortlist: torch.Tensor,
                     i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    pi = candidate_pos[shortlist[:, i]]
    pj = candidate_pos[shortlist[:, j]]
    return torch.linalg.vector_norm(pi - pj, dim=-1)


def _multishape_pair_terms(rhs: torch.Tensor, corr: torch.Tensor,
                           i: torch.Tensor, j: torch.Tensor,
                           ridge: float) -> Tuple[torch.Tensor, ...]:
    """Closed-form two-row solve for every shortlisted pair."""
    ri, rj = rhs[:, i], rhs[:, j]                  # (B,P,M)
    cc = corr[:, i, j]                             # (B,P)
    a = 1.0 + float(ridge)
    det = (a * a - cc.square()).clamp_min(1e-10)
    bi = (a * ri - cc.unsqueeze(-1) * rj) / det.unsqueeze(-1)
    bj = (a * rj - cc.unsqueeze(-1) * ri) / det.unsqueeze(-1)
    linear = (bi * ri).sum(-1) + (bj * rj).sum(-1)
    quad = ((bi * bi).sum(-1) + (bj * bj).sum(-1)
            + 2.0 * cc * (bi * bj).sum(-1))
    captured = 2.0 * linear - quad
    condition = (1.0 + cc.abs()) / (1.0 - cc.abs()).clamp_min(1e-10)
    return bi, bj, captured, condition, cc


def _source_diagnostics(H: torch.Tensor, Z: torch.Tensor, indices: torch.Tensor,
                        coefficients: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Contribution weights and leave-one-source-out deltas for selected rows."""
    amp = torch.linalg.vector_norm(coefficients, dim=-1)
    active = indices >= 0
    amp = torch.where(active, amp, torch.zeros_like(amp))
    weight = amp / amp.sum(-1, keepdim=True).clamp_min(1e-12)
    pred = H @ coefficients
    resid = Z - pred
    delta = torch.zeros_like(amp)
    for r in range(indices.shape[1]):
        add = H[:, :, r:r + 1] @ coefficients[:, r:r + 1]
        d = ((resid + add).square().sum((-2, -1)) - resid.square().sum((-2, -1)))
        delta[:, r] = torch.where(active[:, r], d, torch.zeros_like(d))
    return amp, weight, delta


def _single_result(shortlist: torch.Tensor, scores: torch.Tensor, Hs: torch.Tensor,
                   Z: torch.Tensor, y2: torch.Tensor | None, rmax: int = 2) -> InferenceResult:
    idx = torch.full((len(Z), rmax), -1, dtype=torch.long, device=Z.device)
    coef = torch.zeros(len(Z), rmax, Z.shape[-1], dtype=Z.dtype, device=Z.device)
    idx[:, 0] = shortlist[:, 0]
    coef[:, 0] = torch.einsum("bc,bcm->bm", Hs[:, :, 0], Z)
    Hout = torch.zeros(len(Z), Z.shape[1], rmax, dtype=Z.dtype, device=Z.device)
    Hout[:, :, 0] = Hs[:, :, 0]
    amp, q, delta = _source_diagnostics(Hout, Z, idx, coef)
    captured = scores[:, 0]
    sse = -captured if y2 is None else y2 - captured
    return InferenceResult(idx, coef, q, amp, torch.ones(len(Z), dtype=torch.int16,
                            device=Z.device), captured, sse, torch.ones_like(captured),
                            delta, shortlist, scores, {"pair_gain": torch.zeros_like(captured)})


def _gather_by_local(Hs: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Gather shortlist columns, ``Hs=(B,C,L)``, ``local=(B,R)``."""
    return torch.gather(Hs, 2, local.unsqueeze(1).expand(
        len(Hs), Hs.shape[1], local.shape[1]))


def _greedy_multishape(Hs: torch.Tensor, Z: torch.Tensor,
                       shortlist: torch.Tensor, shortlisted_pos: torch.Tensor,
                       rmax: int, ridge: float, min_separation: float,
                       max_condition: float, min_weight: float,
                       min_gain: float) -> Tuple[torch.Tensor, ...]:
    """Greedy forward selection with an exact coefficient refit after each addition."""
    b, _, l = Hs.shape
    m = Z.shape[-1]
    local = torch.full((b, rmax), -1, dtype=torch.long, device=Z.device)
    local[:, 0] = 0
    coeff = torch.zeros(b, rmax, m, dtype=Z.dtype, device=Z.device)
    coeff[:, 0] = torch.einsum("bc,bcm->bm", Hs[:, :, 0], Z)
    captured = coeff[:, 0].square().sum(-1)
    condition = torch.ones_like(captured)
    alive = torch.ones(b, dtype=torch.bool, device=Z.device)
    gain_history = torch.zeros(b, rmax - 1, dtype=Z.dtype, device=Z.device)

    for r in range(1, rmax):
        current_local = local[:, :r].clamp_min(0)
        current_H = _gather_by_local(Hs, current_local)
        proposal_score = torch.full((b, l), -float("inf"), dtype=Z.dtype, device=Z.device)
        proposal_cond = torch.full((b, l), float("inf"), dtype=Z.dtype, device=Z.device)
        proposal_coeff = torch.zeros(b, l, r + 1, m, dtype=Z.dtype, device=Z.device)
        selected_pos = torch.gather(shortlisted_pos, 1,
                                    current_local.unsqueeze(-1).expand(b, r, 3))
        for q in range(l):
            duplicate = (current_local == q).any(1)
            sep = torch.linalg.vector_norm(
                selected_pos - shortlisted_pos[:, q:q + 1], dim=-1).min(1).values
            Hq = torch.cat([current_H, Hs[:, :, q:q + 1]], dim=2)
            Bq, score, cond, _ = profile_multishape(Hq, Z, ridge=ridge)
            valid = (~duplicate) & (sep >= float(min_separation)) \
                & (cond <= float(max_condition))
            proposal_score[:, q] = torch.where(valid, score,
                                                torch.full_like(score, -float("inf")))
            proposal_cond[:, q] = cond
            proposal_coeff[:, q] = Bq
        best_score, best = proposal_score.max(1)
        ar = torch.arange(b, device=Z.device)
        Bbest = proposal_coeff[ar, best]
        amp = torch.linalg.vector_norm(Bbest, dim=-1)
        weakest = amp.min(1).values / amp.sum(1).clamp_min(1e-12)
        gain = best_score - captured
        accept = alive & torch.isfinite(best_score) & (gain > float(min_gain)) \
            & (weakest >= float(min_weight))
        gain_history[:, r - 1] = gain
        local[:, r] = torch.where(accept, best, torch.full_like(best, -1))
        coeff[:, :r + 1] = torch.where(accept[:, None, None], Bbest,
                                       coeff[:, :r + 1])
        captured = torch.where(accept, best_score, captured)
        condition = torch.where(accept, proposal_cond[ar, best], condition)
        alive &= accept

    safe = local.clamp_min(0)
    global_idx = torch.gather(shortlist, 1, safe)
    global_idx = torch.where(local >= 0, global_idx, torch.full_like(global_idx, -1))
    return global_idx, coeff, captured, condition, gain_history


def _greedy_shared(Hs: torch.Tensor, Z: torch.Tensor, shortlist: torch.Tensor,
                   shortlisted_pos: torch.Tensor, rmax: int,
                   min_separation: float, max_condition: float,
                   min_weight: float, min_gain: float,
                   weight_steps: int = 32) -> Tuple[torch.Tensor, ...]:
    """Greedy support growth for the shared-shape model with simplex refits."""
    b, _, l = Hs.shape
    m = Z.shape[-1]
    local = torch.full((b, rmax), -1, dtype=torch.long, device=Z.device)
    local[:, 0] = 0
    coeff = torch.zeros(b, rmax, m, dtype=Z.dtype, device=Z.device)
    v0 = torch.einsum("bc,bcm->bm", Hs[:, :, 0], Z)
    coeff[:, 0] = v0
    captured = v0.square().sum(-1)
    condition = torch.ones_like(captured)
    alive = torch.ones(b, dtype=torch.bool, device=Z.device)
    gain_history = torch.zeros(b, rmax - 1, dtype=Z.dtype, device=Z.device)

    for r in range(1, rmax):
        current_local = local[:, :r].clamp_min(0)
        current_H = _gather_by_local(Hs, current_local)
        selected_pos = torch.gather(shortlisted_pos, 1,
                                    current_local.unsqueeze(-1).expand(b, r, 3))
        proposal_score = torch.full((b, l), -float("inf"), dtype=Z.dtype, device=Z.device)
        proposal_cond = torch.full((b, l), float("inf"), dtype=Z.dtype, device=Z.device)
        proposal_pi = torch.zeros(b, l, r + 1, dtype=Z.dtype, device=Z.device)
        for q in range(l):
            duplicate = (current_local == q).any(1)
            sep = torch.linalg.vector_norm(
                selected_pos - shortlisted_pos[:, q:q + 1], dim=-1).min(1).values
            Hq = torch.cat([current_H, Hs[:, :, q:q + 1]], dim=2)
            piq, score = optimize_shared_weights(Hq, Z, steps=weight_steps)
            ev = torch.linalg.eigvalsh(Hq.transpose(1, 2) @ Hq)
            cond = ev[:, -1] / ev[:, 0].clamp_min(1e-10)
            valid = (~duplicate) & (sep >= float(min_separation)) \
                & (cond <= float(max_condition))
            proposal_score[:, q] = torch.where(valid, score,
                                                torch.full_like(score, -float("inf")))
            proposal_cond[:, q] = cond
            proposal_pi[:, q] = piq
        best_score, best = proposal_score.max(1)
        ar = torch.arange(b, device=Z.device)
        pibest = proposal_pi[ar, best]
        gain = best_score - captured
        accept = alive & torch.isfinite(best_score) & (gain > float(min_gain)) \
            & (pibest.min(1).values >= float(min_weight))
        gain_history[:, r - 1] = gain
        local[:, r] = torch.where(accept, best, torch.full_like(best, -1))
        Hbest = torch.cat([current_H, Hs[ar, :, best].unsqueeze(-1)], dim=2)
        h = (Hbest * pibest.unsqueeze(1)).sum(2)
        hn = h.square().sum(1).clamp_min(1e-10)
        v = torch.einsum("bc,bcm->bm", h, Z) / hn[:, None]
        Bbest = pibest.unsqueeze(-1) * v.unsqueeze(1)
        coeff[:, :r + 1] = torch.where(accept[:, None, None], Bbest,
                                       coeff[:, :r + 1])
        captured = torch.where(accept, best_score, captured)
        condition = torch.where(accept, proposal_cond[ar, best], condition)
        alive &= accept

    safe = local.clamp_min(0)
    global_idx = torch.gather(shortlist, 1, safe)
    global_idx = torch.where(local >= 0, global_idx, torch.full_like(global_idx, -1))
    return global_idx, coeff, captured, condition, gain_history


def _infer_pair_beam(footprints: torch.Tensor, candidate_pos: torch.Tensor,
                     conf: torch.Tensor, Z: torch.Tensor, y2: torch.Tensor,
                     shortlist: torch.Tensor, single_scores: torch.Tensor,
                     model: str, ridge: float, min_separation: float,
                     max_condition: float, min_weight: float, min_gain: float,
                     exact_alpha_min: float,
                     candidate_chunk: int) -> InferenceResult:
    """Beam-conditioned pair search: L first sources x every second source.

    This is more expensive than pairing only within the top-L singles, but the
    latter can miss the exact pair even in noiseless data.  Complexity is O(LN)
    rather than O(N^2), and every proposed pair receives the exact profiled refit.
    """
    b, m = len(Z), Z.shape[-1]
    pair_idx = torch.full((b, 2), -1, dtype=torch.long, device=Z.device)
    pair_coef = torch.zeros(b, 2, m, dtype=Z.dtype, device=Z.device)
    best_score = torch.full((b,), -float("inf"), dtype=Z.dtype, device=Z.device)
    pair_condition = torch.ones(b, dtype=Z.dtype, device=Z.device)
    pair_sep = torch.full((b,), float("nan"), dtype=Z.dtype, device=Z.device)
    secondary = torch.zeros(b, dtype=Z.dtype, device=Z.device)

    for ic in torch.unique(conf).tolist():
        rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
        H = footprints[int(ic)]                         # (C,N)
        zc = Z[rows]
        first = shortlist[rows]                         # (Bc,L)
        Hfirst = H.T[first]                             # (Bc,L,C)
        rfirst = torch.einsum("blc,bcm->blm", Hfirst, zc)
        n = H.shape[1]
        pfirst = candidate_pos[first]
        local_score = torch.full((len(rows),), -float("inf"), dtype=Z.dtype,
                                 device=Z.device)
        local_first = torch.full((len(rows),), -1, dtype=torch.long, device=Z.device)
        local_second = torch.full_like(local_first, -1)
        local_coef = torch.zeros(len(rows), 2, m, dtype=Z.dtype, device=Z.device)
        local_cond = torch.ones(len(rows), dtype=Z.dtype, device=Z.device)
        local_sep = torch.full_like(local_cond, float("nan"))
        local_secondary = torch.zeros_like(local_cond)
        ar = torch.arange(len(rows), device=Z.device)
        # The B x L x N x M coefficient tensors are too large for the 5,120-
        # candidate fixed lattice.  Scan the second source in bounded chunks and
        # retain only each spike's current best exact profiled refit.
        pair_chunk = max(1, min(int(candidate_chunk), 256))
        for lo in range(0, n, pair_chunk):
            hi = min(lo + pair_chunk, n)
            Hsecond = H[:, lo:hi]
            rhs_second = torch.einsum("cj,bcm->bjm", Hsecond, zc)
            cc = torch.einsum("blc,cj->blj", Hfirst, Hsecond)
            cand = torch.arange(lo, hi, device=Z.device)
            same = first[:, :, None] == cand[None, None, :]
            sep = torch.linalg.vector_norm(
                pfirst[:, :, None, :] - candidate_pos[None, None, lo:hi, :], dim=-1)
            x_i = rfirst.square().sum(-1)[:, :, None]
            x_j = rhs_second.square().sum(-1)[:, None, :]
            cross = torch.einsum("blm,bjm->blj", rfirst, rhs_second)
            if model == "m2_r2":
                a = 1.0 + float(ridge)
                det = (a * a - cc.square()).clamp_min(1e-10)
                linear = (a * (x_i + x_j) - 2.0 * cc * cross) / det
                coef_norm2 = (((a * a + cc.square()) * (x_i + x_j)
                               - 4.0 * a * cc * cross) / det.square())
                # Since (G + ridge I)B = RHS, the unpenalized captured
                # reduction is <B,RHS> + ridge ||B||^2.
                score = linear + float(ridge) * coef_norm2
            else:
                amin = exact_alpha_min if model == "m1_x2" else 0.0
                if model == "m1_eq2":
                    alpha = torch.full_like(cc, 0.5)
                    mix_norm = 0.5 * (1.0 + cc)
                    mix_rhs2 = 0.25 * (x_i + x_j + 2.0 * cross)
                    score = mix_rhs2 / mix_norm.clamp_min(1e-10)
                else:
                    alpha, score = _optimal_shared_pair_moments(
                        x_i, x_j, cross, cc, alpha_min=amin)
            cond = (1.0 + cc.abs()) / (1.0 - cc.abs()).clamp_min(1e-10)
            valid = (~same) & (sep >= float(min_separation)) \
                & (cond <= float(max_condition))
            score = torch.where(valid, score, torch.full_like(score, -float("inf")))
            sc, flat = score.flatten(1).max(1)
            width = hi - lo
            beam = torch.div(flat, width, rounding_mode="floor")
            second_local = flat % width
            second = second_local + lo
            first_best = first[ar, beam]
            if model == "m2_r2":
                aa = 1.0 + float(ridge)
                csel = cc[ar, beam, second_local]
                dsel = (aa * aa - csel.square()).clamp_min(1e-10)
                risel = rfirst[ar, beam]
                rjsel = rhs_second[ar, second_local]
                cb = torch.stack([
                    (aa * risel - csel[:, None] * rjsel) / dsel[:, None],
                    (aa * rjsel - csel[:, None] * risel) / dsel[:, None],
                ], 1)
                amp = torch.linalg.vector_norm(cb, dim=-1)
                sec = amp.min(1).values / amp.sum(1).clamp_min(1e-12)
            else:
                aa = alpha[ar, beam, second_local]
                h = (aa[:, None] * H[:, first_best].T
                     + (1.0 - aa)[:, None] * H[:, second].T)
                hn = h.square().sum(1).clamp_min(1e-10)
                v = torch.einsum("bc,bcm->bm", h, zc) / hn[:, None]
                cb = torch.stack([aa[:, None] * v,
                                  (1.0 - aa)[:, None] * v], 1)
                sec = torch.minimum(aa, 1.0 - aa)
            improve = sc > local_score
            local_score = torch.where(improve, sc, local_score)
            local_first = torch.where(improve, first_best, local_first)
            local_second = torch.where(improve, second, local_second)
            local_coef = torch.where(improve[:, None, None], cb, local_coef)
            local_cond = torch.where(improve, cond[ar, beam, second_local], local_cond)
            local_sep = torch.where(improve, sep[ar, beam, second_local], local_sep)
            local_secondary = torch.where(improve, sec, local_secondary)

        sc = local_score
        first_best = local_first
        second = local_second
        cb = local_coef
        sec = local_secondary
        pair_idx[rows] = torch.stack([first_best, second], 1)
        pair_coef[rows] = cb
        best_score[rows] = sc
        pair_condition[rows] = local_cond
        pair_sep[rows] = local_sep
        secondary[rows] = sec

    gain = best_score - single_scores[:, 0]
    if model in {"m1_eq2", "m1_x2"}:
        accept = torch.isfinite(best_score)
    else:
        accept = torch.isfinite(best_score) & (gain > float(min_gain)) \
            & (secondary >= float(min_weight))
    single_idx = torch.stack([shortlist[:, 0], torch.full_like(shortlist[:, 0], -1)], 1)
    Htop = torch.empty(len(Z), Z.shape[1], dtype=Z.dtype, device=Z.device)
    for ic in torch.unique(conf).tolist():
        rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
        Htop[rows] = footprints[int(ic)].T[shortlist[rows, 0]]
    single_coef = torch.zeros_like(pair_coef)
    single_coef[:, 0] = torch.einsum("bc,bcm->bm", Htop, Z)
    indices = torch.where(accept[:, None], pair_idx, single_idx)
    coefficients = torch.where(accept[:, None, None], pair_coef, single_coef)
    captured = torch.where(accept, best_score, single_scores[:, 0])
    condition = torch.where(accept, pair_condition, torch.ones_like(pair_condition))
    indices, coefficients = canonicalize_sources(indices, coefficients)
    Hout = torch.zeros(b, Z.shape[1], 2, dtype=Z.dtype, device=Z.device)
    for ic in torch.unique(conf).tolist():
        rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
        for rr in range(2):
            active = indices[rows, rr] >= 0
            if active.any():
                use = rows[active]
                Hout[use, :, rr] = footprints[int(ic)].T[indices[use, rr]]
    amp, q, delta = _source_diagnostics(Hout, Z, indices, coefficients)
    return InferenceResult(
        indices, coefficients, q, amp, (indices >= 0).sum(1).to(torch.int16),
        captured, y2 - captured, condition, delta, shortlist, single_scores,
        {"pair_gain": gain, "pair_separation": pair_sep,
         "secondary_weight_proposed": secondary, "pair_accepted": accept,
         "pair_search_evaluations": torch.full_like(captured,
                                                     shortlist.shape[1] * footprints.shape[2])},
    )


def _gather_global_support(footprints: torch.Tensor, conf: torch.Tensor,
                           indices: torch.Tensor) -> torch.Tensor:
    """Gather global dictionary columns as B x C x R."""
    out = torch.zeros(len(indices), footprints.shape[1], indices.shape[1],
                      dtype=footprints.dtype, device=footprints.device)
    for ic in torch.unique(conf).tolist():
        rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
        for rr in range(indices.shape[1]):
            active = indices[rows, rr] >= 0
            if active.any():
                use = rows[active]
                out[use, :, rr] = footprints[int(ic)].T[indices[use, rr]]
    return out


def _extend_multishape_full(footprints: torch.Tensor, candidate_pos: torch.Tensor,
                            conf: torch.Tensor, Z: torch.Tensor, y2: torch.Tensor,
                            pair: InferenceResult, rmax: int, ridge: float,
                            min_separation: float, max_condition: float,
                            min_weight: float, min_gain: float,
                            candidate_chunk: int) -> InferenceResult:
    """Extend a beam-selected M2 pair by full-dictionary hard pursuit."""
    b, m = len(Z), Z.shape[-1]
    indices = torch.full((b, rmax), -1, dtype=torch.long, device=Z.device)
    coefficients = torch.zeros(b, rmax, m, dtype=Z.dtype, device=Z.device)
    indices[:, :2] = pair.indices
    coefficients[:, :2] = pair.coefficients
    support = pair.support_size.clone()
    captured = pair.captured.clone()
    condition = pair.condition.clone()
    gains = torch.zeros(b, rmax - 1, dtype=Z.dtype, device=Z.device)
    gains[:, 0] = pair.diagnostics["pair_gain"]
    alive = support == 2
    pair_chunk = max(1, min(int(candidate_chunk), 128))

    for slot in range(2, rmax):
        for ic in torch.unique(conf[alive]).tolist() if alive.any() else []:
            rows = torch.nonzero(alive & (conf == int(ic)), as_tuple=False).squeeze(1)
            Hcfg = footprints[int(ic)]
            cur_idx = indices[rows, :slot]
            Hcur = Hcfg.T[cur_idx].transpose(1, 2)              # Br,C,slot
            G = Hcur.transpose(1, 2) @ Hcur
            Rhs = Hcur.transpose(1, 2) @ Z[rows]
            pos_cur = candidate_pos[cur_idx]
            local_score = torch.full((len(rows),), -float("inf"), dtype=Z.dtype,
                                     device=Z.device)
            local_idx = torch.full((len(rows),), -1, dtype=torch.long, device=Z.device)
            local_coef = torch.zeros(len(rows), slot + 1, m, dtype=Z.dtype,
                                     device=Z.device)
            local_cond = torch.full_like(local_score, float("inf"))
            ar = torch.arange(len(rows), device=Z.device)
            for lo in range(0, Hcfg.shape[1], pair_chunk):
                hi = min(lo + pair_chunk, Hcfg.shape[1])
                width = hi - lo
                Hj = Hcfg[:, lo:hi]
                cross = torch.einsum("bcr,cj->brj", Hcur, Hj)
                rhs_j = torch.einsum("cj,bcm->bjm", Hj, Z[rows])
                Gq = torch.zeros(len(rows), width, slot + 1, slot + 1,
                                 dtype=Z.dtype, device=Z.device)
                Gq[:, :, :slot, :slot] = G[:, None]
                Gq[:, :, :slot, slot] = cross.transpose(1, 2)
                Gq[:, :, slot, :slot] = cross.transpose(1, 2)
                Gq[:, :, slot, slot] = 1.0
                Rq = torch.empty(len(rows), width, slot + 1, m,
                                 dtype=Z.dtype, device=Z.device)
                Rq[:, :, :slot] = Rhs[:, None]
                Rq[:, :, slot] = rhs_j
                eye = torch.eye(slot + 1, dtype=Z.dtype, device=Z.device)
                Bq = torch.linalg.solve(Gq + float(ridge) * eye, Rq)
                linear = (Bq * Rq).sum((-2, -1))
                quad = (Bq * (Gq @ Bq)).sum((-2, -1))
                score = 2.0 * linear - quad
                ev = torch.linalg.eigvalsh(Gq)
                cond = ev[:, :, -1] / ev[:, :, 0].clamp_min(1e-10)
                cand = torch.arange(lo, hi, device=Z.device)
                duplicate = (cur_idx[:, :, None] == cand[None, None]).any(1)
                sep = torch.linalg.vector_norm(
                    pos_cur[:, :, None] - candidate_pos[None, None, lo:hi], dim=-1
                ).min(1).values
                valid = (~duplicate) & (sep >= float(min_separation)) \
                    & (cond <= float(max_condition))
                score = torch.where(valid, score, torch.full_like(score, -float("inf")))
                sc, jj = score.max(1)
                improve = sc > local_score
                local_score = torch.where(improve, sc, local_score)
                local_idx = torch.where(improve, jj + lo, local_idx)
                local_coef = torch.where(improve[:, None, None], Bq[ar, jj], local_coef)
                local_cond = torch.where(improve, cond[ar, jj], local_cond)
            gain = local_score - captured[rows]
            amp = torch.linalg.vector_norm(local_coef, dim=-1)
            weakest = amp.min(1).values / amp.sum(1).clamp_min(1e-12)
            accept = torch.isfinite(local_score) & (gain > float(min_gain)) \
                & (weakest >= float(min_weight))
            gains[rows, slot - 1] = gain
            use = rows[accept]
            if len(use):
                indices[use, slot] = local_idx[accept]
                coefficients[use, :slot + 1] = local_coef[accept]
                captured[use] = local_score[accept]
                condition[use] = local_cond[accept]
                support[use] = slot + 1
            alive[rows] = accept

    indices, coefficients = canonicalize_sources(indices, coefficients)
    Hout = _gather_global_support(footprints, conf, indices)
    amp, q, delta = _source_diagnostics(Hout, Z, indices, coefficients)
    return InferenceResult(
        indices, coefficients, q, amp, support, captured, y2 - captured,
        condition, delta, pair.shortlist_indices, pair.shortlist_scores,
        {"gain_history": gains, "pair_gain": gains[:, 0],
         "pair_accepted": pair.diagnostics["pair_accepted"],
         "pair_search_evaluations": pair.diagnostics["pair_search_evaluations"]},
    )


def _extend_shared_shortlist(footprints: torch.Tensor, candidate_pos: torch.Tensor,
                             conf: torch.Tensor, Z: torch.Tensor, y2: torch.Tensor,
                             pair: InferenceResult, rmax: int,
                             min_separation: float, max_condition: float,
                             min_weight: float, min_gain: float) -> InferenceResult:
    """Extend a beam-selected M1 pair over the declared single-source shortlist."""
    b, m = len(Z), Z.shape[-1]
    shortlist = pair.shortlist_indices
    indices = torch.full((b, rmax), -1, dtype=torch.long, device=Z.device)
    coefficients = torch.zeros(b, rmax, m, dtype=Z.dtype, device=Z.device)
    indices[:, :2] = pair.indices
    coefficients[:, :2] = pair.coefficients
    support = pair.support_size.clone()
    captured = pair.captured.clone()
    condition = pair.condition.clone()
    gains = torch.zeros(b, rmax - 1, dtype=Z.dtype, device=Z.device)
    gains[:, 0] = pair.diagnostics["pair_gain"]
    alive = support == 2

    for slot in range(2, rmax):
        rows = torch.nonzero(alive, as_tuple=False).squeeze(1)
        if not len(rows):
            break
        Hcur = _gather_global_support(footprints, conf[rows], indices[rows, :slot])
        pos_cur = candidate_pos[indices[rows, :slot]]
        local_score = torch.full((len(rows),), -float("inf"), dtype=Z.dtype,
                                 device=Z.device)
        local_idx = torch.full((len(rows),), -1, dtype=torch.long, device=Z.device)
        local_coef = torch.zeros(len(rows), slot + 1, m, dtype=Z.dtype, device=Z.device)
        local_cond = torch.full_like(local_score, float("inf"))
        ar = torch.arange(len(rows), device=Z.device)
        for qq in range(shortlist.shape[1]):
            candidate = shortlist[rows, qq]
            Hqcol = _gather_global_support(
                footprints, conf[rows], candidate[:, None])[:, :, 0]
            duplicate = (indices[rows, :slot] == candidate[:, None]).any(1)
            sep = torch.linalg.vector_norm(
                pos_cur - candidate_pos[candidate, None], dim=-1).min(1).values
            Hq = torch.cat([Hcur, Hqcol[:, :, None]], dim=2)
            piq, score = optimize_shared_weights(Hq, Z[rows], steps=40)
            ev = torch.linalg.eigvalsh(Hq.transpose(1, 2) @ Hq)
            cond = ev[:, -1] / ev[:, 0].clamp_min(1e-10)
            valid = (~duplicate) & (sep >= float(min_separation)) \
                & (cond <= float(max_condition))
            score = torch.where(valid, score, torch.full_like(score, -float("inf")))
            improve = score > local_score
            h = (Hq * piq[:, None]).sum(2)
            v = torch.einsum("bc,bcm->bm", h, Z[rows]) \
                / h.square().sum(1).clamp_min(1e-10)[:, None]
            Bq = piq[:, :, None] * v[:, None]
            local_score = torch.where(improve, score, local_score)
            local_idx = torch.where(improve, candidate, local_idx)
            local_coef = torch.where(improve[:, None, None], Bq, local_coef)
            local_cond = torch.where(improve, cond, local_cond)
        gain = local_score - captured[rows]
        amp = torch.linalg.vector_norm(local_coef, dim=-1)
        weakest = amp.min(1).values / amp.sum(1).clamp_min(1e-12)
        accept = torch.isfinite(local_score) & (gain > float(min_gain)) \
            & (weakest >= float(min_weight))
        gains[rows, slot - 1] = gain
        use = rows[accept]
        if len(use):
            indices[use, slot] = local_idx[accept]
            coefficients[use, :slot + 1] = local_coef[accept]
            captured[use] = local_score[accept]
            condition[use] = local_cond[accept]
            support[use] = slot + 1
        alive[rows] = accept

    indices, coefficients = canonicalize_sources(indices, coefficients)
    Hout = _gather_global_support(footprints, conf, indices)
    amp, q, delta = _source_diagnostics(Hout, Z, indices, coefficients)
    return InferenceResult(
        indices, coefficients, q, amp, support, captured, y2 - captured,
        condition, delta, shortlist, pair.shortlist_scores,
        {"gain_history": gains, "pair_gain": gains[:, 0],
         "pair_accepted": pair.diagnostics["pair_accepted"],
         "pair_search_evaluations": pair.diagnostics["pair_search_evaluations"]},
    )


def _infer_beam_model(footprints: torch.Tensor, candidate_pos: torch.Tensor,
                      conf: torch.Tensor, Z: torch.Tensor, y2: torch.Tensor,
                      shortlist: torch.Tensor, single_scores: torch.Tensor,
                      model: str, ridge: float, min_separation: float,
                      max_condition: float, min_weight: float, min_gain: float,
                      exact_alpha_min: float,
                      candidate_chunk: int) -> InferenceResult:
    """Dispatch a beam-conditioned model after singleton screening."""
    if model in {"m1_eq2", "m1_w2", "m1_x2", "m2_r2"}:
        return _infer_pair_beam(
            footprints, candidate_pos, conf, Z, y2, shortlist, single_scores,
            model, ridge, min_separation, max_condition, min_weight, min_gain,
            exact_alpha_min, candidate_chunk)
    if model in {"m2_r3", "m2_r4"}:
        pair = _infer_pair_beam(
            footprints, candidate_pos, conf, Z, y2, shortlist, single_scores,
            "m2_r2", ridge, min_separation, max_condition, min_weight, min_gain,
            exact_alpha_min, candidate_chunk)
        return _extend_multishape_full(
            footprints, candidate_pos, conf, Z, y2, pair, int(model[-1]), ridge,
            min_separation, max_condition, min_weight, min_gain, candidate_chunk)
    if model in {"m1_r3", "m1_r4"}:
        pair = _infer_pair_beam(
            footprints, candidate_pos, conf, Z, y2, shortlist, single_scores,
            "m1_w2", ridge, min_separation, max_condition, min_weight, min_gain,
            exact_alpha_min, candidate_chunk)
        return _extend_shared_shortlist(
            footprints, candidate_pos, conf, Z, y2, pair, int(model[-1]),
            min_separation, max_condition, min_weight, min_gain)
    raise ValueError(f"unsupported beam model {model}")


def _merge_inference_subset(base: InferenceResult, sub: InferenceResult,
                            rows: torch.Tensor,
                            skipped: torch.Tensor,
                            gain_upper: torch.Tensor) -> InferenceResult:
    """Merge costly support searches into exact singleton fallbacks."""
    for name in ("indices", "coefficients", "contributions", "amplitudes",
                 "support_size", "captured", "sse", "condition",
                 "leaveout_delta"):
        getattr(base, name)[rows] = getattr(sub, name)
    # The singleton shortlist was already computed for every row and is retained
    # in ``base``. Diagnostics from searched rows are scattered into zero/false
    # defaults, so skipped rows are explicit rather than silently absent.
    for key, value in sub.diagnostics.items():
        shape = (len(base.indices),) + tuple(value.shape[1:])
        target = torch.zeros(shape, dtype=value.dtype, device=value.device)
        target[rows] = value
        base.diagnostics[key] = target
    base.diagnostics["pair_search_skipped"] = skipped
    base.diagnostics["pair_gain_upper_bound"] = gain_upper
    return base


def infer_batch(footprints: torch.Tensor, candidate_pos: torch.Tensor,
                conf: torch.Tensor, Y: torch.Tensor, omega: torch.Tensor,
                model: str, shortlist_size: int = 16,
                candidate_chunk: int = 1024, ridge: float = 1e-6,
                min_separation: float = 4.0, max_condition: float = 1e4,
                min_weight: float = 0.02, min_gain: float = 0.0,
                exact_alpha_min: float = 0.10,
                pair_search: str = "beam") -> InferenceResult:
    """Infer M0 or one of the two-source models for a waveform batch.

    Supported ``model`` values are ``m0``, ``m1_eq2``, ``m1_w2`` (nested,
    learned weights), ``m1_x2`` (exact two with bounded weights), ``m2_r2``,
    and greedy nested ``m1_r3/m1_r4/m2_r3/m2_r4`` variants. Beam pair search
    crosses the one-source shortlist with the full dictionary; exact all-pair
    validation is provided separately.
    """
    allowed = {"m0", "m1_eq2", "m1_w2", "m1_x2", "m2_r2",
               "m1_r3", "m1_r4", "m2_r3", "m2_r4"}
    if model not in allowed:
        raise ValueError(f"unsupported model {model!r}; choose from {sorted(allowed)}")
    if pair_search not in {"beam", "shortlist"}:
        raise ValueError("pair_search must be 'beam' or 'shortlist'")
    Z = Y @ omega.T
    y2 = (Y * Y).sum((-2, -1))
    shortlist, single_scores = topk_single(footprints, conf, Z, shortlist_size,
                                            candidate_chunk)
    Hs = gather_shortlist(footprints, conf, shortlist)
    if model == "m0":
        return _single_result(shortlist, single_scores, Hs, Z, y2)

    if pair_search == "beam":
        # A support of any size cannot capture more than all energy in Z.  Thus
        # ||Z||^2 - best_single is a safe upper bound on any additional-source
        # gain.  For nested thresholded models this lets us skip the expensive
        # L-by-N beam exactly, without changing a single accept/reject decision.
        nested = model not in {"m1_eq2", "m1_x2"}
        if nested and min_gain >= 0.0:
            gain_upper = Z.square().sum((-2, -1)) - single_scores[:, 0]
            tol = 2e-6 * torch.maximum(torch.ones_like(gain_upper),
                                       Z.square().sum((-2, -1)))
            potential = gain_upper + tol > float(min_gain)
            if not potential.all():
                rmax = 2 if model in {"m1_w2", "m2_r2"} else int(model[-1])
                base = _single_result(shortlist, single_scores, Hs, Z, y2, rmax=rmax)
                rows = torch.nonzero(potential, as_tuple=False).squeeze(1)
                if len(rows):
                    sub = _infer_beam_model(
                        footprints, candidate_pos, conf[rows], Z[rows], y2[rows],
                        shortlist[rows], single_scores[rows], model, ridge,
                        min_separation, max_condition, min_weight, min_gain,
                        exact_alpha_min, candidate_chunk)
                    return _merge_inference_subset(base, sub, rows, ~potential,
                                                   gain_upper)
                base.diagnostics["pair_search_skipped"] = ~potential
                base.diagnostics["pair_gain_upper_bound"] = gain_upper
                return base
        return _infer_beam_model(
            footprints, candidate_pos, conf, Z, y2, shortlist, single_scores,
            model, ridge, min_separation, max_condition, min_weight, min_gain,
            exact_alpha_min, candidate_chunk)

    if model in {"m1_r3", "m1_r4", "m2_r3", "m2_r4"}:
        rmax = int(model[-1])
        shortlist_pos = candidate_pos[shortlist]
        if model.startswith("m1"):
            indices, coefficients, captured, condition, gain_history = _greedy_shared(
                Hs, Z, shortlist, shortlist_pos, rmax, min_separation,
                max_condition, min_weight, min_gain)
        else:
            indices, coefficients, captured, condition, gain_history = _greedy_multishape(
                Hs, Z, shortlist, shortlist_pos, rmax, ridge, min_separation,
                max_condition, min_weight, min_gain)
        indices, coefficients = canonicalize_sources(indices, coefficients)
        Hout = torch.zeros(len(Z), Y.shape[1], rmax, dtype=Y.dtype, device=Y.device)
        for ic in torch.unique(conf).tolist():
            rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
            for rr in range(rmax):
                active = indices[rows, rr] >= 0
                if active.any():
                    use = rows[active]
                    Hout[use, :, rr] = footprints[int(ic)].T[indices[use, rr]]
        amp, q, delta = _source_diagnostics(Hout, Z, indices, coefficients)
        support = (indices >= 0).sum(1).to(torch.int16)
        return InferenceResult(
            indices, coefficients, q, amp, support, captured, y2 - captured,
            condition, delta, shortlist, single_scores,
            {"gain_history": gain_history,
             "pair_gain": gain_history[:, 0] if gain_history.shape[1] else
             torch.zeros_like(captured)},
        )

    rhs = Hs.transpose(1, 2) @ Z                    # (B,L,M)
    corr = Hs.transpose(1, 2) @ Hs                 # (B,L,L)
    i, j = _pair_grid(shortlist.shape[1], Z.device)
    separation = _pair_separation(candidate_pos, shortlist, i, j)
    valid_sep = separation >= float(min_separation)
    b = len(Z)

    if model == "m2_r2":
        bi, bj, pair_score, pair_cond, _ = _multishape_pair_terms(rhs, corr, i, j, ridge)
        valid = valid_sep & (pair_cond <= float(max_condition))
        pair_score = torch.where(valid, pair_score,
                                 torch.full_like(pair_score, -float("inf")))
        best_score, best = pair_score.max(1)
        ar = torch.arange(b, device=Z.device)
        pi, pj = i[best], j[best]
        pair_idx = torch.stack([shortlist[ar, pi], shortlist[ar, pj]], 1)
        pair_coef = torch.stack([bi[ar, best], bj[ar, best]], 1)
        pair_condition = pair_cond[ar, best]
        pair_sep = separation[ar, best]
        pair_amp = torch.linalg.vector_norm(pair_coef, dim=-1)
        secondary = pair_amp.min(1).values / pair_amp.sum(1).clamp_min(1e-12)
        gain = best_score - single_scores[:, 0]
        accept = torch.isfinite(best_score) & (gain > float(min_gain)) \
            & (secondary >= float(min_weight))
    else:
        ri, rj = rhs[:, i], rhs[:, j]
        cc = corr[:, i, j]
        if model == "m1_eq2":
            alpha = torch.full_like(cc, 0.5)
            mix_rhs = 0.5 * (ri + rj)
            mix_norm = 0.5 * (1.0 + cc)
            pair_score = mix_rhs.square().sum(-1) / mix_norm.clamp_min(1e-10)
        else:
            amin = exact_alpha_min if model == "m1_x2" else 0.0
            alpha, pair_score = optimal_shared_pair(ri, rj, cc, alpha_min=amin)
        pair_cond_all = (1.0 + cc.abs()) / (1.0 - cc.abs()).clamp_min(1e-10)
        valid = valid_sep & (pair_cond_all <= float(max_condition))
        pair_score = torch.where(valid, pair_score,
                                 torch.full_like(pair_score, -float("inf")))
        best_score, best = pair_score.max(1)
        ar = torch.arange(b, device=Z.device)
        pi, pj = i[best], j[best]
        aa = alpha[ar, best]
        hi, hj = Hs[ar, :, pi], Hs[ar, :, pj]
        h = aa[:, None] * hi + (1.0 - aa)[:, None] * hj
        hn = h.square().sum(1).clamp_min(1e-10)
        v = torch.einsum("bc,bcm->bm", h, Z) / hn[:, None]
        pair_idx = torch.stack([shortlist[ar, pi], shortlist[ar, pj]], 1)
        pair_coef = torch.stack([aa[:, None] * v, (1.0 - aa)[:, None] * v], 1)
        pair_condition = pair_cond_all[ar, best]
        pair_sep = separation[ar, best]
        secondary = torch.minimum(aa, 1.0 - aa)
        gain = best_score - single_scores[:, 0]
        if model in {"m1_eq2", "m1_x2"}:
            accept = torch.isfinite(best_score)
        else:
            accept = torch.isfinite(best_score) & (gain > float(min_gain)) \
                & (secondary >= float(min_weight))

    # Nested models fall back to the exact one-source solution when the second
    # source is too weak or does not clear the predeclared gain threshold.
    single_idx = torch.stack([shortlist[:, 0], torch.full_like(shortlist[:, 0], -1)], 1)
    single_coef = torch.zeros_like(pair_coef)
    single_coef[:, 0] = rhs[:, 0]
    indices = torch.where(accept[:, None], pair_idx, single_idx)
    coefficients = torch.where(accept[:, None, None], pair_coef, single_coef)
    captured = torch.where(accept, best_score, single_scores[:, 0])
    condition = torch.where(accept, pair_condition, torch.ones_like(pair_condition))
    support = 1 + accept.to(torch.int16)

    indices, coefficients = canonicalize_sources(indices, coefficients)
    # Gather H after canonicalization, avoiding candidate tensors for inactive rows.
    Hout = torch.zeros(b, Y.shape[1], 2, dtype=Y.dtype, device=Y.device)
    for ic in torch.unique(conf).tolist():
        rows = torch.nonzero(conf == int(ic), as_tuple=False).squeeze(1)
        for r in range(2):
            active = indices[rows, r] >= 0
            if active.any():
                rr = rows[active]
                Hout[rr, :, r] = footprints[int(ic)].T[indices[rr, r]]
    amp, q, delta = _source_diagnostics(Hout, Z, indices, coefficients)
    return InferenceResult(
        indices=indices,
        coefficients=coefficients,
        contributions=q,
        amplitudes=amp,
        support_size=support,
        captured=captured,
        sse=y2 - captured,
        condition=condition,
        leaveout_delta=delta,
        shortlist_indices=shortlist,
        shortlist_scores=single_scores,
        diagnostics={"pair_gain": gain, "pair_separation": pair_sep,
                     "secondary_weight_proposed": secondary, "pair_accepted": accept},
    )


def exact_pair_multishape(H: torch.Tensor, Z: torch.Tensor, ridge: float = 1e-6,
                          min_separation_mask: torch.Tensor | None = None,
                          max_condition: float = 1e4) -> Tuple[torch.Tensor, ...]:
    """Exact all-pair M2 search for one spike; intended for bounded validation."""
    if H.ndim != 2 or Z.ndim != 2:
        raise ValueError("exact validation expects H=(C,N), Z=(C,M)")
    rhs = H.T @ Z
    corr = H.T @ H
    i, j = _pair_grid(H.shape[1], H.device)
    bi, bj, score, cond, _ = _multishape_pair_terms(
        rhs.unsqueeze(0), corr.unsqueeze(0), i, j, ridge)
    score, cond = score[0], cond[0]
    valid = cond <= float(max_condition)
    if min_separation_mask is not None:
        valid &= min_separation_mask[i, j]
    score = torch.where(valid, score, torch.full_like(score, -float("inf")))
    best = score.argmax()
    idx = torch.stack([i[best], j[best]])
    coef = torch.stack([bi[0, best], bj[0, best]])
    return idx, coef, score[best], cond[best]


def exact_pair_shared(H: torch.Tensor, Z: torch.Tensor, alpha_min: float = 0.0,
                      min_separation_mask: torch.Tensor | None = None,
                      max_condition: float = 1e4) -> Tuple[torch.Tensor, ...]:
    """Exact all-pair M1 search for one spike; intended for bounded validation."""
    rhs = H.T @ Z
    corr = H.T @ H
    i, j = _pair_grid(H.shape[1], H.device)
    alpha, score = optimal_shared_pair(rhs[i], rhs[j], corr[i, j], alpha_min=alpha_min)
    cond = (1.0 + corr[i, j].abs()) / (1.0 - corr[i, j].abs()).clamp_min(1e-10)
    valid = cond <= float(max_condition)
    if min_separation_mask is not None:
        valid &= min_separation_mask[i, j]
    score = torch.where(valid, score, torch.full_like(score, -float("inf")))
    best = score.argmax()
    idx = torch.stack([i[best], j[best]])
    aa = alpha[best]
    h = aa * H[:, idx[0]] + (1.0 - aa) * H[:, idx[1]]
    v = h @ Z / h.square().sum().clamp_min(1e-10)
    coef = torch.stack([aa * v, (1.0 - aa) * v])
    return idx, coef, score[best], cond[best], aa
