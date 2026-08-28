"""The one canonical set of example spikes used by every reconstruction panel.

Every model in the browser reconstructs the SAME spikes, so `spikes.png` (and the
multi-source `spike_decomposition.png`) can be read side by side: any difference in the
panel is a difference in the model, not a difference in which spikes were drawn.

The set is drawn from spikes that BOTH selective multi-source models keep a second
source on -- `MULTIPOLE_learned_m2_r2` (at most 2 sources, 5.2% multi) and the group-lasso
`analytic_l1_mono_N512_M8_group9975_R4` (1.53 sources/spike, 25.7% multi). Their
intersection is 75,362 spikes (3.0%). Requiring both keeps genuinely multi-source events
rather than one model's selection noise, and the point is that a single-source model
physically cannot represent them -- its shortcomings show up directly in the overlay.

The selection is FIXED and model-independent: the ids live in
`zncc/figures/canonical_example_spikes.json` alongside their provenance, and are read
from there rather than recomputed, so adding or refitting a model can never move them.
Never make the selection depend on the model being plotted (its own support size,
amplitude, or nMSE) -- the moment it does, the panels stop being comparable, which is
the entire point.

CAVEAT, worth knowing when reading the panel: multi-source spikes are about twice the
amplitude of a typical spike, so on `lat64_monopole_Q32` they carry 2.5x the ABSOLUTE
residual (median SSE 0.10 vs 0.04) but a LOWER relative residual (median 0.275 vs
0.387). The `rel.err` printed above each column will therefore look better on these
spikes, not worse; the unmodelled second source is visible in the waveform overlay, not
in that number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "zncc/figures/canonical_example_spikes.json"

N_EXAMPLE_SPIKES = 10
EXAMPLE_SEED = 11


def example_spike_ids(n_total: int, n: int = N_EXAMPLE_SPIKES,
                      seed: int = EXAMPLE_SEED) -> np.ndarray:
    """Sorted global spike indices, identical for every caller and every model.

    Falls back to a plain deterministic draw only if the canonical file is missing, so a
    fresh checkout still renders rather than crashing -- but then the panels are NOT the
    curated multi-source set. Regenerate the file to restore it.
    """
    if CANONICAL.exists():
        ids = np.asarray(json.loads(CANONICAL.read_text())["spike_ids"], np.int64)
        if len(ids) >= n:
            return np.sort(ids[:n])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(int(n_total), min(int(n), int(n_total)), replace=False))
