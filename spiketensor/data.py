"""Dataset access for both recordings, behind one interface.

Nothing here re-derives spikes: detection, waveform extraction and the monopole
fit were already done by the previous pipeline (see handoff/DATASETS.md) and are
reused as inputs, exactly as the brief specifies.

    NP1   dataset1 zigzag : 2,475,738 spikes, 1957 s, 4 columns @ 20 um pitch,
                            imposed 0<->50 um square-wave ground truth.
    ULTRA ZYE-0021        : 16,162 spikes, 60 s, 8 columns @ 6 um pitch, ~no drift.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
from spiketensor.probe_geometry import build_neighborhood_lookup  # noqa: E402

N_NEIGHBORS = 10


# --------------------------------------------------------------------------- #
# Channel ordering inside the stored waveform arrays.
#
# MEASURED, NOT ASSUMED. The two recordings were extracted by different code and
# do NOT share a convention:
#
#   dataset      ordering            peak-slot agreement   corr(slot dist, slot ptp)
#   NP1          argsort(distance)          64.1%                  -0.319
#   NP-Ultra     lexsort by (y, x)          46.7%                  -0.170
#
# Under the WRONG ordering the physical distance-vs-amplitude relationship
# disappears entirely (corr +0.02 for NP1 under lexsort), so every spike's spatial
# context would be permuted while the loss still looked healthy. Locked by
# tests/test_data.py::test_channel_ordering_is_correct.
# --------------------------------------------------------------------------- #
def neighborhood_lookup(channel_locations: np.ndarray, order: str,
                        n_neighbors: int = N_NEIGHBORS):
    """(C, n) neighbour ids in the stored waveform's slot order, and (C, 3) anchors.

    The anchor is the neighbourhood centroid and is order-independent, so it
    matches handoff/viz/probe_geometry.build_neighborhood_lookup either way.
    """
    C = channel_locations.shape[0]
    look = np.zeros((C, n_neighbors), np.int32)
    anch = np.zeros((C, 3), np.float64)
    for c in range(C):
        d = np.linalg.norm(channel_locations - channel_locations[c], axis=1)
        # plain np.argsort, matching probe_geometry.build_neighborhood_lookup exactly.
        # NP-Ultra's 6 um pitch produces many exactly-tied distances, so a stable
        # sort selects a DIFFERENT set of 10 and hence a different anchor centroid.
        near = np.argsort(d)[:n_neighbors]
        if order == "distance":
            sel = near
        elif order == "lexsort_yx":
            loc = channel_locations[near]
            sel = near[np.lexsort((loc[:, 0], loc[:, 1]))]
        else:
            raise ValueError(order)
        look[c] = sel
        anch[c, :2] = channel_locations[sel].mean(0)
    return look, anch


@dataclass
class Recording:
    name: str
    waveforms: np.ndarray          # (N, 10, 90) float32 memmap
    spike_times: np.ndarray        # (N,) int64 sample index
    spike_channels: np.ndarray     # (N,) int64 peak channel
    channel_locations: np.ndarray  # (C, 2) float64 um
    fs: float
    waveform_scale: float
    mp_xyz: np.ndarray             # (N, 3) monopole localization (global um)
    mp_alpha: np.ndarray           # (N,)
    channel_lookup: np.ndarray     # (C, 10) neighbour ids in the stored slot order
    anchors: np.ndarray            # (C, 3) neighbourhood centroid per channel
    has_gt_motion: bool
    channel_order: str             # "distance" | "lexsort_yx" -- see neighborhood_lookup

    @property
    def n_spikes(self):
        return len(self.spike_times)

    @property
    def duration_s(self):
        return float(self.spike_times.max() / self.fs)

    @property
    def n_channels(self):
        return self.channel_locations.shape[0]

    def time_bins(self, bin_s: float = 1.0):
        """(N,) int bin index and the number of bins."""
        b = np.floor(self.spike_times / (self.fs * bin_s)).astype(np.int64)
        return b, int(b.max()) + 1

    def channel_offsets(self) -> np.ndarray:
        """(C, 10, 2) position of each neighbour channel relative to the anchor, um."""
        pos = self.channel_locations[self.channel_lookup]      # (C, 10, 2)
        return (pos - self.anchors[:, None, :2]).astype(np.float32)

    def summary(self):
        return (f"{self.name}: {self.n_spikes:,} spikes, {self.duration_s:.0f} s, "
                f"{self.n_channels} ch, fs={self.fs:.0f}, "
                f"x={sorted(set(self.channel_locations[:,0].tolist()))[:8]}, "
                f"y pitch={np.diff(np.unique(self.channel_locations[:,1]))[0]:.0f} um, "
                f"GT motion={self.has_gt_motion}")


# --------------------------------------------------------------------------- #
def load_np1() -> Recording:
    R = REPO / "results"
    ch = np.load(R / "channel_locations.npy").astype(np.float64)
    lookup, anchors = neighborhood_lookup(ch, "distance")
    mp = np.stack([np.load(R / f"tpca_monopolar_full/spike_locs_{a}.npy") for a in "xyz"], 1)
    return Recording(
        name="np1",
        waveforms=np.load(R / "all_spike_waveforms/waveforms_all.npy", mmap_mode="r"),
        spike_times=np.load(R / "spike_times.npy"),
        spike_channels=np.load(R / "spike_channels.npy"),
        channel_locations=ch,
        fs=float(np.load(R / "fs.npy").reshape(-1)[0]),
        waveform_scale=9.5577,
        mp_xyz=mp,
        # Paired with mp_xyz above. The top-level results/spike_locs_alpha.npy is a
        # DIFFERENT localization run from tpca_monopolar_full/ -- measured corr 0.645
        # against the paired array, median ratio 2.04 -- so alpha/r with that one is
        # not the fit that produced these positions. Only the two viz scripts read
        # mp_alpha (for marker colour/size), so no result depended on the old pairing.
        mp_alpha=np.load(R / "tpca_monopolar_full/spike_locs_alpha.npy"),
        channel_lookup=lookup, anchors=anchors.astype(np.float64),
        has_gt_motion=True, channel_order="distance",
    )


def load_ultra() -> Recording:
    R = REPO / "results/np_ultra"
    ch = np.load(R / "raw/channel_locations.npy").astype(np.float64)
    lookup, anchors = neighborhood_lookup(ch, "lexsort_yx")
    mp = np.stack([np.load(R / f"raw/spike_locs_{a}.npy") for a in "xyz"], 1)
    return Recording(
        name="ultra",
        waveforms=np.load(R / "dataset/waveforms.npy", mmap_mode="r"),
        spike_times=np.load(R / "raw/spike_times.npy"),
        spike_channels=np.load(R / "raw/spike_channels.npy"),
        channel_locations=ch,
        fs=float(np.load(R / "raw/fs.npy").reshape(-1)[0]),
        waveform_scale=7.285867415954456,
        mp_xyz=mp,
        mp_alpha=np.load(R / "raw/spike_locs_alpha.npy"),
        channel_lookup=lookup, anchors=anchors.astype(np.float64),
        has_gt_motion=False, channel_order="lexsort_yx",
    )


LOADERS = {"np1": load_np1, "ultra": load_ultra}


def load(name: str) -> Recording:
    return LOADERS[name]()


# --------------------------------------------------------------------------- #
def gt_motion_trace(t_sec: np.ndarray) -> np.ndarray:
    """Imposed square wave on dataset1, evaluated at arbitrary times (um).

    22 steps alternating 0 and 50 um; first transition at t=578.77 s, last at
    t=1583.48 s. Position is 0 outside that window.
    """
    pos = np.load(REPO / "data/dataset1/manip.positions.npy").reshape(-1)
    ts = np.load(REPO / "data/dataset1/manip.timestamps_p1.npy").reshape(-1)
    # LINEAR interpolation, not zero-order hold. The entries are manipulator
    # WAYPOINTS and the stage ramps between them -- the recording is named
    # "zigzag" for that reason. Reading them as a square wave is wrong and badly
    # understates every method: scored against the shipped MP+DREDge reference,
    # the ZOH square gives r=+0.448 gain=0.133 while the linear triangle gives
    # r=+0.923 gain=0.468 (verified). The manipulator clock is file-relative;
    # the legacy motion.npz t_anchors additionally carry the SpikeGLX firstSample
    # offset of 197820/30000 = 6.594 s, which our own traces do not.
    return np.interp(t_sec, ts, pos).astype(np.float64)


def gt_motion_window():
    ts = np.load(REPO / "data/dataset1/manip.timestamps_p1.npy").reshape(-1)
    return float(ts[0]), float(ts[-1])


# --------------------------------------------------------------------------- #
class BinSampler:
    """Samples (time bins) x (spikes within each bin) for one training step."""

    def __init__(self, bin_idx: np.ndarray, n_bins: int, seed: int = 0,
                 t_lo_bin: int = 0, t_hi_bin: int | None = None,
                 spike_mask: np.ndarray | None = None,
                 eligible_bins: np.ndarray | None = None):
        self.rng = np.random.default_rng(seed)
        if spike_mask is None:
            use = np.arange(len(bin_idx))
        else:
            spike_mask = np.asarray(spike_mask, dtype=bool)
            if spike_mask.shape != bin_idx.shape:
                raise ValueError("spike_mask must have one entry per spike")
            use = np.flatnonzero(spike_mask)
        order = use[np.argsort(bin_idx[use], kind="stable")]
        self.sorted_spikes = order
        counts = np.bincount(bin_idx[use], minlength=n_bins)
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        self.counts = counts
        hi = n_bins if t_hi_bin is None else t_hi_bin
        self.eligible = np.flatnonzero((counts > 0)[:hi])
        self.eligible = self.eligible[self.eligible >= t_lo_bin]
        if eligible_bins is not None:
            self.eligible = np.intersect1d(
                self.eligible, np.asarray(eligible_bins, dtype=np.int64),
                assume_unique=False)
        if not len(self.eligible):
            raise ValueError("no eligible time bins after applying split")

    def sample(self, n_bins_per_step: int, spikes_per_bin: int, contiguous: bool = False):
        """Returns (spike_indices, local_bin_index, chosen_bins)."""
        if contiguous:
            n = min(n_bins_per_step, len(self.eligible))
            starts = np.flatnonzero(
                self.eligible[n - 1:] - self.eligible[:len(self.eligible) - n + 1]
                == n - 1)
            if len(starts):
                start = int(self.rng.choice(starts))
                bins = self.eligible[start:start + n]
            else:
                # A heavily blocked CV split may contain no run this long. Fall
                # back to distinct eligible bins rather than crossing into test data.
                bins = np.sort(self.rng.choice(self.eligible, size=n, replace=False))
        else:
            bins = self.rng.choice(self.eligible,
                                   size=min(n_bins_per_step, len(self.eligible)),
                                   replace=False)
            bins.sort()
        idx, loc = [], []
        for j, b in enumerate(bins):
            a, e = self.offsets[b], self.offsets[b + 1]
            n = e - a
            take = self.sorted_spikes[a:e] if n <= spikes_per_bin else \
                self.sorted_spikes[self.rng.choice(np.arange(a, e), spikes_per_bin,
                                                   replace=False)]
            idx.append(take)
            loc.append(np.full(len(take), j, np.int64))
        return np.concatenate(idx), np.concatenate(loc), bins
