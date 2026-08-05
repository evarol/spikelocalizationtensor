"""Waveform access and the reconstruction reference bounds.

Targets are DC-removed per channel per spike and divided by 100, which is the
convention every nMSE in this repo is quoted against. `references` computes the
bounds a fit is judged by -- most importantly `free_rank1`, the error when every
spike gets its own unconstrained rank-1 factorization with no spatial structure
at all. That is the floor a single-source model is trying to approach.
"""
from __future__ import annotations

import numpy as np
import torch


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
