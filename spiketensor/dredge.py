"""DREDge-style decentralized registration: pairwise displacements -> a motion trace.

The ZNCC stage produces, for every pair of time bins, the argmax shift
D(t,t') and its correlation C(t,t'). Following Windolf et al.'s decentralized
registration (the estimator behind DREDge), the motion p(t) is the weighted
least-squares solution of the over-determined system p_t - p_t' = D(t,t'):

    min_p  sum_{t,t'} w_{t,t'} ( p_t - p_t' - D_{t,t'} )^2  +  lam * || diff(p) ||^2

With w symmetric and D antisymmetric this is a weighted graph Laplacian system

    ( 2 (diag(W 1) - W) + lam * Dif^T Dif + eps I ) p  =  2 (W . D) 1

which is solved directly (T ~ 2000 for NP1, so a dense Cholesky is instant). The
gauge freedom (p is determined only up to a constant) is fixed by the eps ridge
plus an explicit mean-centering of the solution.

Robustness: correlations below `c_thresh` are dropped outright, remaining weights
are (C - c_thresh)^p, and optional IRLS reweighting with a Huber loss suppresses
pairs whose residual disagrees with the consensus -- this is what makes the
estimator tolerant of the spurious lattice-periodicity matches that a 20 um-pitch
probe can produce.

Nonrigid mode splits the probe into overlapping depth windows, solves each
independently, and interpolates, producing the (n_t, n_y) `disp` array that the
existing viz package's motion.npz loader expects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DredgeConfig:
    c_thresh: float = 0.0       # drop pairs with correlation at or below this
    c_power: float = 1.0        # w = (C - c_thresh)^c_power
    lam_smooth: float = 1.0     # temporal smoothness weight
    ridge: float = 1e-6         # gauge-fixing ridge
    irls_iters: int = 5         # 0 disables robust reweighting
    huber_delta: float = 10.0   # um
    max_pair_lag: int | None = None   # only use pairs with |t - t'| <= this


def _weights(C: np.ndarray, cfg: DredgeConfig) -> np.ndarray:
    W = np.clip(C - cfg.c_thresh, 0.0, None) ** cfg.c_power
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)                                # enforce symmetry
    if cfg.max_pair_lag is not None:
        T = W.shape[0]
        lag = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
        W = np.where(lag <= cfg.max_pair_lag, W, 0.0)
    return W


def _solve_once(D: np.ndarray, W: np.ndarray, cfg: DredgeConfig) -> np.ndarray:
    T = D.shape[0]
    L = 2.0 * (np.diag(W.sum(1)) - W)
    b = 2.0 * (W * D).sum(1)
    if cfg.lam_smooth > 0 and T > 1:
        Dif = np.eye(T, k=1) - np.eye(T)
        Dif = Dif[:-1]
        L = L + cfg.lam_smooth * (Dif.T @ Dif)
    L = L + cfg.ridge * np.eye(T)
    p = np.linalg.solve(L, b)
    return p - p.mean()


def solve_rigid(D: np.ndarray, C: np.ndarray, cfg: DredgeConfig = DredgeConfig()) -> dict:
    """(T,T) displacement + correlation -> (T,) motion trace in um.

    D must follow the convention D[t,t'] = p_t - p_t' (unit-tested in the ZNCC
    kernel as tests/test_zncc.py::test_displacement_sign).
    """
    D = np.asarray(D, np.float64)
    D = 0.5 * (D - D.T)                                # enforce antisymmetry
    W0 = _weights(np.asarray(C, np.float64), cfg)
    W = W0.copy()
    p = _solve_once(D, W, cfg)
    hist = [p.copy()]
    for _ in range(cfg.irls_iters):
        R = np.abs((p[:, None] - p[None, :]) - D)
        hub = np.where(R <= cfg.huber_delta, 1.0, cfg.huber_delta / np.maximum(R, 1e-9))
        W = W0 * hub
        p = _solve_once(D, W, cfg)
        hist.append(p.copy())
    resid = (p[:, None] - p[None, :]) - D
    m = W0 > 0
    return {
        "p": p,
        "weighted_rmse_um": float(np.sqrt((W0[m] * resid[m] ** 2).sum() / W0[m].sum()))
        if m.any() else float("nan"),
        "frac_pairs_used": float(m.mean()),
        "n_irls": cfg.irls_iters,
        "history": np.array(hist),
    }


def solve_nonrigid(D_win: np.ndarray, C_win: np.ndarray, y_anchors: np.ndarray,
                   cfg: DredgeConfig = DredgeConfig()) -> np.ndarray:
    """Per-depth-window solve. D_win/C_win are (n_win, T, T). Returns (T, n_win)."""
    out = np.stack([solve_rigid(D_win[w], C_win[w], cfg)["p"]
                    for w in range(len(y_anchors))], axis=1)
    return out


# --------------------------------------------------------------------------- #
def save_motion_npz(path, p: np.ndarray, t_anchors: np.ndarray,
                    y_anchors: np.ndarray | None = None):
    """Write the motion in the format the existing viz/eval package consumes.

    handoff/viz/anchor_null_baseline.py::load_motion expects
        disp (n_t, n_y), t_anchors (n_t,), y_anchors (n_y,)
    and bilinearly interpolates Delta_y(t, y).
    """
    disp = p[:, None] if p.ndim == 1 else p
    if y_anchors is None:
        y_anchors = np.array([0.0])
    np.savez(path, disp=np.asarray(disp, np.float64),
             t_anchors=np.asarray(t_anchors, np.float64),
             y_anchors=np.asarray(y_anchors, np.float64))


def apply_motion(y: np.ndarray, t_sec: np.ndarray, p: np.ndarray,
                 t_anchors: np.ndarray) -> np.ndarray:
    """Drift-correct depths: y_corrected = y - p(t)."""
    return y - np.interp(t_sec, t_anchors, p)


# --------------------------------------------------------------------------- #
def score_against_gt(p: np.ndarray, t_anchors: np.ndarray, gt: np.ndarray) -> dict:
    """Compare a recovered trace to the imposed square wave.

    Both traces are mean-centred first: the estimator has an arbitrary additive
    gauge, so only the SHAPE is meaningful. Reported:
      r          Pearson correlation (the headline selection metric)
      rmse_um    RMSE after removing the constant offset
      scale      least-squares amplitude of the recovery (1.0 = correct 50 um swing)
      step_r     correlation of the temporal derivative (step localisation)
    """
    p = np.asarray(p, np.float64); gt = np.asarray(gt, np.float64)
    pc, gc = p - p.mean(), gt - gt.mean()
    r = float(np.corrcoef(pc, gc)[0, 1]) if pc.std() > 0 and gc.std() > 0 else 0.0
    scale = float((pc @ gc) / (gc @ gc)) if gc @ gc > 0 else 0.0
    dp, dg = np.diff(pc), np.diff(gc)
    step_r = float(np.corrcoef(dp, dg)[0, 1]) if dp.std() > 0 and dg.std() > 0 else 0.0
    return {"r": r, "rmse_um": float(np.sqrt(((pc - gc) ** 2).mean())),
            "scale": scale, "step_r": step_r,
            "recovered_range_um": float(p.max() - p.min()),
            "gt_range_um": float(gt.max() - gt.min())}
