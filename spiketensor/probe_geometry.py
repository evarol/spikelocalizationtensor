"""Probe geometry helper — the ONLY non-plotting dependency of the viz package.

For each channel, find its `n_neighbors` nearest channels (the "neighborhood")
and that neighborhood's centroid (the per-spike "anchor"). Localization models in
this project predict a RESIDUAL from this anchor, so the anchor is needed to map
local <-> global coordinates.
"""
from __future__ import annotations
import numpy as np

N_NEIGHBORS = 10


def build_neighborhood_lookup(channel_locations: np.ndarray, n_neighbors: int = N_NEIGHBORS):
    n_channels = channel_locations.shape[0]
    channel_lookup = np.zeros((n_channels, n_neighbors), dtype=np.int32)
    anchor_lookup = np.zeros((n_channels, 3), dtype=np.float32)

    for peak_channel in range(n_channels):
        dists = np.linalg.norm(channel_locations - channel_locations[peak_channel], axis=1)
        nearest = np.argsort(dists)[:n_neighbors]
        locs = channel_locations[nearest]
        order = np.lexsort((locs[:, 0], locs[:, 1]))
        ordered = nearest[order]
        channel_lookup[peak_channel] = ordered.astype(np.int32)
        anchor_lookup[peak_channel, :2] = channel_locations[ordered].mean(axis=0).astype(np.float32)
        anchor_lookup[peak_channel, 2] = 0.0

    return channel_lookup, anchor_lookup
