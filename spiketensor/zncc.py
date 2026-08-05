"""Differentiable shift-max ZNCC between per-time-bin 3-D localization volumes.

    rho(t,t') = max_delta ZNCC(I_t, shift_delta(I_t'))

computed for every pair in a batch with a single FFT, and made differentiable by
replacing the max with a softmax-weighted average over shifts.

Conventions
-----------
Volumes are (B, nx, ny, nz) with y the drift axis (along the probe shank).
The shift search is along y only by default; `search_xz` enables a small 3-D
search when the lateral/depth displacement matrices D_x, D_z are also wanted.

ZNCC uses the GLOBAL mean and norm of each volume. Because we zero-pad along y
by `max_shift` before the FFT, the circular correlation of the padded, centered
signals equals the LINEAR correlation for every |delta| <= max_shift, so there is
no wrap-around across the ends of the probe. Padding after centering is correct
because zero is exactly the mean of a centered signal; mass shifted outside the
volume simply stops contributing, which is the physically sensible behaviour.

Displacement sign convention (unit-tested in tests/test_zncc.py)
---------------------------------------------------------------
    D(t,t') = argmax_delta  ==  p_t - p_t'
i.e. if the pattern in volume t sits `d` voxels HIGHER in y than the same pattern
in t', then D(t,t') = +d. This is the convention the DREDge solve expects:
    minimise sum_{t,t'} w_{t,t'} (p_t - p_t' - D(t,t'))^2 .

MPS notes
---------
- torch.fft.rfft/irfft support autograd on MPS (verified, torch 2.8).
- Complex einsum on MPS is ~37x slower than a batched complex matmul over the
  frequency axis, so the cross-power spectrum is computed with matmul.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ZNCCConfig:
    max_shift_y_vox: int = 20          # +/- search range along y
    max_shift_x_vox: int = 0           # 0 disables the lateral search
    max_shift_z_vox: int = 0           # 0 disables the depth search
    tau: float = 0.05                  # softmax temperature over shifts (rho is in [-1,1])
    eps: float = 1e-8
    pad: bool = True                   # zero-pad y to make the correlation linear


def _center_normalize(V: torch.Tensor, eps: float):
    """(B,nx,ny,nz) -> centered, unit-L2-norm volumes + the norms (for masking)."""
    B = V.shape[0]
    flat = V.reshape(B, -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    nrm = flat.norm(dim=1, keepdim=True)
    out = flat / nrm.clamp_min(eps)
    return out.reshape(V.shape), nrm.squeeze(1)


def _cross_power(F: torch.Tensor) -> torch.Tensor:
    """(B, C, K) complex spectra -> (B, B, K) cross-power  P[t,u,k] = sum_c F[t,c,k] conj(F[u,c,k]).

    Computed as four REAL batched matmuls rather than one complex matmul, because
    complex `torch.matmul` is BROKEN on the MPS backend in torch 2.8 -- it returns
    silently wrong values (measured max abs error 11.1 on a 5x4x3 random complex
    tensor, versus 4.8e-07 for the real matmul). `rfft`, `irfft`, `conj` and
    permutes are all correct; only the complex matmul is bad. Do not "simplify"
    this back to torch.matmul on a complex tensor without re-running
    tests/test_zncc.py::test_matches_bruteforce on MPS.
    """
    Fr = F.real.permute(2, 0, 1).contiguous()               # (K, B, C)
    Fi = F.imag.permute(2, 0, 1).contiguous()
    Pr = torch.matmul(Fr, Fr.transpose(1, 2)) + torch.matmul(Fi, Fi.transpose(1, 2))
    Pi = torch.matmul(Fi, Fr.transpose(1, 2)) - torch.matmul(Fr, Fi.transpose(1, 2))
    return torch.complex(Pr, Pi).permute(1, 2, 0).contiguous()   # (B, B, K)


def _lag_index(n: int, max_shift: int, device) -> torch.Tensor:
    """Indices into an FFT lag axis of length n selecting lags -max..+max, in order."""
    neg = torch.arange(n - max_shift, n, device=device)
    pos = torch.arange(0, max_shift + 1, device=device)
    return torch.cat([neg, pos])            # lag values: -max..-1, 0..+max


def _lag_values(max_shift: int, device, dtype=torch.float32) -> torch.Tensor:
    return torch.arange(-max_shift, max_shift + 1, device=device, dtype=dtype)


class ShiftMaxZNCC(nn.Module):
    """All-pairs shift-max ZNCC for a batch of volumes."""

    def __init__(self, cfg: ZNCCConfig = ZNCCConfig()):
        super().__init__()
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    def _corr_y(self, V: torch.Tensor) -> torch.Tensor:
        """(B,nx,ny,nz) -> (B,B,n_lag) linear cross-correlation over y lags."""
        c = self.cfg
        B, nx, ny, nz = V.shape
        pad = c.max_shift_y_vox if c.pad else 0
        n = ny + 2 * pad

        # (B, C, ny) with C = nx*nz "channels" that are summed over after correlation
        A = V.permute(0, 1, 3, 2).reshape(B, nx * nz, ny)
        if pad:
            A = torch.nn.functional.pad(A, (pad, pad))

        F = torch.fft.rfft(A.contiguous(), n=n, dim=-1)     # (B, C, K)
        P = _cross_power(F)                                 # (B, B, K)
        corr = torch.fft.irfft(P, n=n, dim=-1)              # (B, B, n)

        idx = _lag_index(n, c.max_shift_y_vox, V.device)
        return corr.index_select(-1, idx)                    # (B, B, 2*max+1)

    def _corr_3d(self, V: torch.Tensor) -> torch.Tensor:
        """(B,nx,ny,nz) -> (B,B,nlx,nly,nlz) for a small 3-D shift search."""
        c = self.cfg
        B, nx, ny, nz = V.shape
        px, py, pz = (c.max_shift_x_vox, c.max_shift_y_vox, c.max_shift_z_vox) if c.pad else (0, 0, 0)
        A = torch.nn.functional.pad(V, (pz, pz, py, py, px, px))
        sx, sy, sz = A.shape[1:]

        F = torch.fft.rfftn(A.contiguous(), dim=(1, 2, 3))  # (B, sx, sy, sz//2+1)
        P = _cross_power(F.reshape(B, 1, -1))               # (B, B, M), M = spatial freqs
        P = P.reshape(B, B, sx, sy, sz // 2 + 1).contiguous()
        corr = torch.fft.irfftn(P, s=(sx, sy, sz), dim=(2, 3, 4))

        ix = _lag_index(sx, c.max_shift_x_vox, V.device)
        iy = _lag_index(sy, c.max_shift_y_vox, V.device)
        iz = _lag_index(sz, c.max_shift_z_vox, V.device)
        return corr.index_select(2, ix).index_select(3, iy).index_select(4, iz)

    # ------------------------------------------------------------------ #
    def forward(self, V: torch.Tensor, return_hard: bool = True):
        """Volumes (B,nx,ny,nz), non-negative. Returns a dict.

        rho_soft : (B,B) softmax-over-shifts ZNCC   -- the differentiable objective
        rho_hard : (B,B) true max-over-shifts ZNCC  -- for logging/selection
        d_y      : (B,B) argmax shift in VOXELS along y  (= p_t - p_t')
        valid    : (B,)  volumes with non-degenerate norm
        """
        c = self.cfg
        Vn, nrm = _center_normalize(V, c.eps)
        valid = nrm > c.eps

        if c.max_shift_x_vox == 0 and c.max_shift_z_vox == 0:
            R = self._corr_y(Vn)                                   # (B,B,L)
            lags = _lag_values(c.max_shift_y_vox, V.device)
            w = torch.softmax(R / c.tau, dim=-1)
            rho_soft = (w * R).sum(-1)
            out = {"rho_soft": rho_soft}
            if return_hard:
                hard, j = R.max(dim=-1)
                out["rho_hard"] = hard.detach()
                out["d_y"] = lags[j].detach()
                out["d_y_soft"] = (w * lags).sum(-1).detach()
        else:
            R = self._corr_3d(Vn)                                  # (B,B,Lx,Ly,Lz)
            B = R.shape[0]
            Rf = R.reshape(B, B, -1)
            w = torch.softmax(Rf / c.tau, dim=-1)
            rho_soft = (w * Rf).sum(-1)
            out = {"rho_soft": rho_soft}
            if return_hard:
                hard, j = Rf.max(dim=-1)
                Lx, Ly, Lz = R.shape[2], R.shape[3], R.shape[4]
                out["rho_hard"] = hard.detach()
                out["d_x"] = (j // (Ly * Lz) - c.max_shift_x_vox).detach()
                out["d_y"] = ((j // Lz) % Ly - c.max_shift_y_vox).detach()
                out["d_z"] = (j % Lz - c.max_shift_z_vox).detach()

        m = valid[:, None] & valid[None, :]
        out["valid_pair"] = m
        out["rho_soft"] = torch.where(m, out["rho_soft"], torch.zeros_like(out["rho_soft"]))
        # Invalid (zero-norm) volumes give R == 0 at every lag. argmax over an
        # all-equal axis returns index 0, i.e. d_y = -max_shift = a confident
        # -80 um reading; and the soft-argmax over a symmetric lag set returns
        # EXACTLY 0.0, i.e. a confident "no motion". Both are worse than missing,
        # because they are the two readings we are specifically looking for. So
        # invalid entries are NaN: a consumer that forgets to mask now produces NaN
        # loudly instead of a plausible lie.
        for k in ("rho_hard", "d_y", "d_y_soft", "d_x", "d_z"):
            if k in out:
                v = out[k].to(torch.float32)
                out[k] = torch.where(m, v, torch.full_like(v, float("nan")))
        return out


def offdiag_mean(M: torch.Tensor, valid_pair: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over off-diagonal pairs, honouring a validity mask."""
    B = M.shape[0]
    off = ~torch.eye(B, dtype=torch.bool, device=M.device)
    if valid_pair is not None:
        off = off & valid_pair
    n = off.sum().clamp(min=1)
    return (M * off).sum() / n


def offdiag_sum(M: torch.Tensor, valid_pair: torch.Tensor | None = None) -> torch.Tensor:
    """Sum over valid off-diagonal pairs, matching the stated Σ C objective."""
    B = M.shape[0]
    off = ~torch.eye(B, dtype=torch.bool, device=M.device)
    if valid_pair is not None:
        off = off & valid_pair
    return (M * off).sum()
