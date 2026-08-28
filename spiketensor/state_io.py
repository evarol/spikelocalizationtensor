"""Validated serialization helpers for explicit multipole localization state."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

REQUIRED = (
    "spike_index", "source_index", "source_pos", "source_coeff", "source_amp",
    "source_weight", "support_size", "leaveout_delta", "condition", "captured",
    "sse", "pos_dominant", "pos_barycenter", "omega",
)


def validate_multipole_state(state: Mapping[str, np.ndarray]) -> None:
    missing = [key for key in REQUIRED if key not in state]
    if missing:
        raise ValueError(f"multipole state lacks {missing}")
    n = len(state["spike_index"])
    idx = np.asarray(state["source_index"])
    if idx.ndim != 2 or len(idx) != n:
        raise ValueError("source_index must have shape (n_spikes,Rmax)")
    r = idx.shape[1]
    expected = {
        "source_pos": (n, r, 3),
        "source_amp": (n, r),
        "source_weight": (n, r),
        "leaveout_delta": (n, r),
        "support_size": (n,),
        "condition": (n,),
        "captured": (n,),
        "sse": (n,),
        "pos_dominant": (n, 3),
        "pos_barycenter": (n, 3),
    }
    for key, shape in expected.items():
        if np.shape(state[key]) != shape:
            raise ValueError(f"{key} has shape {np.shape(state[key])}, expected {shape}")
    coeff = np.asarray(state["source_coeff"])
    if coeff.ndim != 3 or coeff.shape[:2] != (n, r):
        raise ValueError("source_coeff must have shape (n_spikes,Rmax,M)")
    omega = np.asarray(state["omega"])
    if omega.ndim != 2 or coeff.shape[2] != omega.shape[0]:
        raise ValueError("coefficient and temporal-basis dimensions disagree")
    # Orthonormality is a property of the PCA-style bases, not of the model class. A
    # prototype-constrained basis (M5) is deliberately non-orthonormal -- M unit vectors
    # cannot all sit close to P<<M prototypes AND be mutually orthogonal -- so such a
    # state declares basis_orthonormal=False and is checked for unit-norm rows instead.
    if bool(np.asarray(state.get("basis_orthonormal", True))):
        if np.max(np.abs(omega @ omega.T - np.eye(len(omega)))) > 1e-3:
            raise ValueError("omega rows are not orthonormal")
    elif np.max(np.abs(np.linalg.norm(omega, axis=1) - 1.0)) > 1e-3:
        raise ValueError("non-orthonormal basis rows must still be unit norm")
    active = idx >= 0
    if not np.array_equal(active.sum(1), np.asarray(state["support_size"]).astype(int)):
        raise ValueError("support_size disagrees with active source indices")
    weight = np.asarray(state["source_weight"])
    if np.any(weight < -1e-7) or np.any(np.abs(weight[~active]) > 1e-6):
        raise ValueError("source weights violate nonnegativity/inactive-zero convention")
    if np.max(np.abs(weight.sum(1) - 1.0)) > 1e-4:
        raise ValueError("active source weights must sum to one")
    pos = np.asarray(state["source_pos"])
    if np.any(np.isfinite(pos[~active])):
        raise ValueError("inactive source positions must be NaN")
    if np.any(~np.isfinite(pos[active])):
        raise ValueError("active source positions must be finite")


def save_multipole_state(path: str | Path, state: Mapping[str, np.ndarray]) -> Path:
    path = Path(path)
    validate_multipole_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state)
    return path


def load_multipole_state(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    validate_multipole_state(state)
    return state
