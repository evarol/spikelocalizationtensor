"""Per-spike reconstruction error, for the error-coloured panels.

Two quantities, both per spike:
    sse    = ||Y - Yhat||^2      the model's squared reconstruction error
    energy = ||Y||^2             the spike's own energy, model-independent

The panels colour by sse/energy (the fraction of the spike the model FAILS to explain)
rather than raw sse, because raw sse is dominated by amplitude -- loud spikes carry more
absolute error wherever they sit, so an absolute map largely redraws the amplitude map
and cannot answer "does the model fit worse in some places". `--error-metric absolute`
switches the panels back to raw MSE.

`energy` depends only on the recording, so it is computed once and cached. `sse` takes
~12 s for the full 2.48 M spikes on a lattice fit, cheap enough to recompute per model.
Multi-source states already store `sse`, and it is used as-is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ENERGY_CACHE = REPO / "zncc/runs/spike_energy_np1.npy"
BATCH = 20000


def spike_energy(rec, cache: Path = ENERGY_CACHE) -> np.ndarray:
    """||Y||^2 per spike. Model-independent, so computed once and cached."""
    if cache.exists():
        out = np.load(cache)
        if len(out) == rec.n_spikes:
            return out
    from spiketensor.fit import load_batch
    off_all = rec.channel_offsets().astype(np.float32)
    out = np.empty(rec.n_spikes, np.float32)
    for i in range(0, rec.n_spikes, BATCH):
        idx = np.arange(i, min(i + BATCH, rec.n_spikes))
        Y, _ = load_batch(rec, idx, off_all, "cpu")
        out[idx] = (Y * Y).sum((1, 2)).numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def lattice_spike_sse(runs: Path, tag: str, rec) -> np.ndarray:
    """||Y - Yhat||^2 per spike for a single-source (pi_*.npz) fit.

    Rebuilds the reconstruction the same way `viz_lattice.fig_spikes` does, so the
    numbers in the error panels agree with the rel.err printed on the example spikes.
    """
    from spiketensor.fit import load_batch
    from spiketensor.fit_lattice import Candidates, footprint, KERNELS
    from spiketensor.viz_lattice import load

    ck, K, V, mu, sig, site, prof, musite = load(runs, tag)
    off_all = rec.channel_offsets().astype(np.float32)
    learned = ck.get("site_sigma") is not None
    if not learned:
        prf = ck.get("profiles") or [(ck.get("kernel", "monopole"), (s_,))
                                     for s_ in ck["sigmas"]]
        cand = Candidates(musite, [(p[0], tuple(p[1])) for p in prf], "cpu")
    else:
        ss = np.asarray(ck["site_sigma"])
    out = np.empty(rec.n_spikes, np.float32)
    a = ck["a"]
    for i in range(0, rec.n_spikes, BATCH):
        idx = np.arange(i, min(i + BATCH, rec.n_spikes))
        Y, off = load_batch(rec, idx, off_all, "cpu")
        if learned:
            m_ = torch.as_tensor(mu[idx]); sg = torch.as_tensor(ss[K[idx]])[:, None]
            dxy2 = ((off[:, :, 0] - m_[:, None, 0]) ** 2
                    + (off[:, :, 1] - m_[:, None, 1]) ** 2)
            dz2 = (m_[:, None, 2] ** 2).expand_as(dxy2)
            g = KERNELS[ck.get("kernel", "monopole")](dxy2, dz2, (sg,))
            g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            g = footprint(cand, off, torch.as_tensor(K[idx]))
        Yh = g[:, :, None] * (torch.as_tensor(V[idx]) @ a)[:, None, :]
        d = Y - Yh
        out[idx] = (d * d).sum((1, 2)).numpy()
    return out


def error_metric(sse: np.ndarray, energy: np.ndarray, metric: str) -> np.ndarray:
    """Per-spike colour value: unexplained energy fraction, or raw MSE."""
    if metric == "absolute":
        return np.asarray(sse, np.float32)
    return (np.asarray(sse, np.float64)
            / np.maximum(np.asarray(energy, np.float64), 1e-12)).astype(np.float32)
