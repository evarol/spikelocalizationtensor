"""Voxel grid, anchor quantization, and the differentiable local -> global scatter.

Each spike s gets a local probability heatmap z_s over a small patch of voxels
centred on its ANCHOR (the centroid of its 10 nearest channels). The per-time-bin
volume is  I_t = sum_{s in t} scatter(z_s, anchor_s).

Key efficiency fact
-------------------
The anchor is a function of the PEAK CHANNEL only, and there are just 384 of them.
So instead of scattering S spikes individually (which needs an (S, P) int64 index
tensor -- hundreds of MB), we:

    1. segment-sum z_s into a dense (n_bins, n_channels, P) tensor  [index_add on dim 0]
    2. scatter that with a (n_channels * P,) index precomputed ONCE  [index_add on dim 1]

Step 2's index is reused every training step, so per-step index memory is zero.

Anchor quantization
-------------------
Anchors do not land exactly on voxel centres. Rather than interpolate, the local
patch is centred on the ROUNDED anchor voxel and the model's residual is defined
relative to that. The rounding is therefore exact by construction, not an
approximation -- global position = anchor_voxel * bin_size + local_offset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridSpec:
    """Global voxel grid. Extents are [lo, lo + n*bin)."""
    x_lo: float = -64.0
    y_lo: float = -64.0
    z_lo: float = 0.0
    x_bin: float = 8.0
    y_bin: float = 4.0
    z_bin: float = 8.0
    nx: int = 24
    ny: int = 992
    nz: int = 16

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def n_vox(self):
        return self.nx * self.ny * self.nz

    @property
    def bins(self):
        return (self.x_bin, self.y_bin, self.z_bin)

    @property
    def lo(self):
        return (self.x_lo, self.y_lo, self.z_lo)

    def describe(self):
        return (f"global grid {self.nx}x{self.ny}x{self.nz} = {self.n_vox:,} vox | "
                f"bins {self.x_bin}/{self.y_bin}/{self.z_bin} um | "
                f"x[{self.x_lo},{self.x_lo+self.nx*self.x_bin}) "
                f"y[{self.y_lo},{self.y_lo+self.ny*self.y_bin}) "
                f"z[{self.z_lo},{self.z_lo+self.nz*self.z_bin}) um")

    @staticmethod
    def for_probe(channel_locations: np.ndarray, *, x_pad=64.0, y_pad=64.0, z_hi=128.0,
                  x_bin=8.0, y_bin=4.0, z_bin=8.0) -> "GridSpec":
        cx, cy = channel_locations[:, 0], channel_locations[:, 1]
        x_lo = float(np.floor((cx.min() - x_pad) / x_bin) * x_bin)
        y_lo = float(np.floor((cy.min() - y_pad) / y_bin) * y_bin)
        nx = int(np.ceil((cx.max() + x_pad - x_lo) / x_bin))
        ny = int(np.ceil((cy.max() + y_pad - y_lo) / y_bin))
        nz = int(np.ceil(z_hi / z_bin))
        return GridSpec(x_lo, y_lo, 0.0, x_bin, y_bin, z_bin, nx, ny, nz)


@dataclass(frozen=True)
class PatchSpec:
    """Local heatmap emitted per spike, in voxels, centred on the anchor voxel.

    z starts at the probe face (z_lo of the global grid) rather than being centred,
    because z >= 0 by convention -- a source cannot be behind the probe.
    """
    nlx: int = 16
    nly: int = 20
    nlz: int = 16

    @property
    def shape(self):
        return (self.nlx, self.nly, self.nlz)

    @property
    def n_vox(self):
        return self.nlx * self.nly * self.nlz

    def extent_um(self, g: GridSpec):
        return (self.nlx * g.x_bin, self.nly * g.y_bin, self.nlz * g.z_bin)


# --------------------------------------------------------------------------- #
def anchor_voxels(anchors: np.ndarray, g: GridSpec) -> np.ndarray:
    """(n_channels, 3) anchor coords -> (n_channels, 3) int voxel indices of the
    patch ORIGIN (its low corner), such that the patch is centred on the anchor."""
    ax = np.round((anchors[:, 0] - g.x_lo) / g.x_bin).astype(np.int64)
    ay = np.round((anchors[:, 1] - g.y_lo) / g.y_bin).astype(np.int64)
    az = np.zeros_like(ax)
    return np.stack([ax, ay, az], 1)


def patch_origin(anchor_vox: np.ndarray, p: PatchSpec) -> np.ndarray:
    """Low corner of the patch so that the anchor sits at the patch centre in x,y
    and at z = 0 (the probe face) in z."""
    out = anchor_vox.copy()
    out[:, 0] -= p.nlx // 2
    out[:, 1] -= p.nly // 2
    out[:, 2] = 0
    return out


def local_offsets_um(p: PatchSpec, g: GridSpec, device="cpu") -> torch.Tensor:
    """(P, 3) physical offset of each local voxel CENTRE from the anchor, in um.

    Used to turn a heatmap into a coordinate: pos = anchor + sum_v z_v * offset_v.
    """
    ix = (torch.arange(p.nlx, device=device) - p.nlx // 2 + 0.5) * g.x_bin
    iy = (torch.arange(p.nly, device=device) - p.nly // 2 + 0.5) * g.y_bin
    iz = (torch.arange(p.nlz, device=device) + 0.5) * g.z_bin + g.z_lo
    ox, oy, oz = torch.meshgrid(ix, iy, iz, indexing="ij")
    return torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], 1)


# --------------------------------------------------------------------------- #
class VolumeScatter(torch.nn.Module):
    """Scatters per-spike local heatmaps into per-time-bin global volumes.

    The (n_channels * P,) flat index is built once at construction. Out-of-bounds
    local voxels are routed to a trash slot that is discarded, so a patch hanging
    off the end of the probe loses that mass rather than wrapping.
    """

    def __init__(self, anchors_per_channel: np.ndarray, g: GridSpec, p: PatchSpec,
                 device="cpu"):
        super().__init__()
        self.g, self.p = g, p
        self.n_channels = anchors_per_channel.shape[0]

        origin = patch_origin(anchor_voxels(anchors_per_channel, g), p)   # (C,3)
        li, lj, lk = np.meshgrid(np.arange(p.nlx), np.arange(p.nly), np.arange(p.nlz),
                                 indexing="ij")
        li, lj, lk = li.ravel(), lj.ravel(), lk.ravel()                    # (P,)

        gi = origin[:, 0:1] + li[None, :]      # (C, P)
        gj = origin[:, 1:2] + lj[None, :]
        gk = origin[:, 2:3] + lk[None, :]
        ok = ((gi >= 0) & (gi < g.nx) & (gj >= 0) & (gj < g.ny) & (gk >= 0) & (gk < g.nz))

        flat = (gi * (g.ny * g.nz) + gj * g.nz + gk)
        flat = np.where(ok, flat, g.n_vox)                                 # trash slot
        self.register_buffer("flat_idx", torch.as_tensor(flat.ravel(), dtype=torch.long,
                                                         device=device))
        self.oob_fraction = float(1.0 - ok.mean())

    def forward(self, z: torch.Tensor, bin_idx: torch.Tensor, chan_idx: torch.Tensor,
                n_bins: int, weights: torch.Tensor | None = None) -> torch.Tensor:
        """z (S, P) heatmaps -> (n_bins, nx, ny, nz) volumes.

        bin_idx  (S,) time-bin index in [0, n_bins)
        chan_idx (S,) peak channel in [0, n_channels)
        weights  (S,) optional per-spike weight (e.g. amplitude weighting)
        """
        g, p = self.g, self.p
        S, P = z.shape
        if weights is not None:
            z = z * weights[:, None]

        # 1) segment-sum over spikes sharing a (time bin, channel)
        seg = bin_idx * self.n_channels + chan_idx
        A = z.new_zeros(n_bins * self.n_channels, P)
        A = A.index_add(0, seg, z)
        A = A.reshape(n_bins, self.n_channels * P)

        # 2) scatter the per-channel patches with the precomputed index
        V = z.new_zeros(n_bins, g.n_vox + 1)
        V = V.index_add(1, self.flat_idx, A)
        return V[:, :g.n_vox].reshape(n_bins, g.nx, g.ny, g.nz)


# --------------------------------------------------------------------------- #
def _gauss1d(sigma_vox: float, device, dtype=torch.float32):
    r = max(1, int(round(3 * sigma_vox)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma_vox ** 2))
    return k / k.sum()


class VolumeSmoother(torch.nn.Module):
    """Separable Gaussian blur of the per-bin volumes.

    Smoothing every z_s and then summing is identical to summing and then
    smoothing (the scatter-sum is linear), so blurring the ~12 volumes is
    equivalent to blurring thousands of per-spike heatmaps -- and about four
    orders of magnitude cheaper. This is what supplies the spatial-smoothness
    prior that a 3-D transposed-convolution head would have provided; that head
    measured 82x slower than a dense head on MPS, which made it unaffordable.

    It also implements the soft-histogram bandwidth: sigma is specified in um and
    converted per axis, so the blur is isotropic in physical space even though
    the voxels are not.
    """

    def __init__(self, sigma_um: float, g: GridSpec, device="cpu"):
        super().__init__()
        self.sigma_um = sigma_um
        self.enabled = sigma_um > 0
        if not self.enabled:
            return
        for ax, b in zip("xyz", g.bins):
            s = sigma_um / b
            self.register_buffer(f"k{ax}", _gauss1d(s, device) if s > 0.05
                                 else torch.ones(1, device=device))

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return V
        h = V.unsqueeze(1)                                     # (B,1,nx,ny,nz)
        for ax, dim in zip("xyz", (2, 3, 4)):
            k = getattr(self, f"k{ax}")
            if k.numel() == 1:
                continue
            shape = [1, 1, 1, 1, 1]; shape[dim] = k.numel()
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (4 - dim)] = pad[2 * (4 - dim) + 1] = k.numel() // 2
            h = torch.nn.functional.conv3d(
                torch.nn.functional.pad(h, pad), k.reshape(shape))
        return h.squeeze(1)


def heatmap_to_coords(z: torch.Tensor, offsets: torch.Tensor,
                      anchor_xyz: torch.Tensor) -> torch.Tensor:
    """(S,P) heatmap + (P,3) local offsets + (S,3) anchor -> (S,3) expected position.

    Uses the expectation (not the argmax) so it is differentiable and sub-voxel.
    """
    return anchor_xyz + z @ offsets


def heatmap_to_coords_robust(z: torch.Tensor, offsets: torch.Tensor,
                             anchor_xyz: torch.Tensor, patch: PatchSpec,
                             win=(2, 3, 2)) -> torch.Tensor:
    """Mode-local expectation: expectation restricted to a window around the argmax.

    The plain expectation is the wrong readout for a MULTIMODAL heatmap. Measured
    on the pretrained NP1 model: predicted E[z] carried 2.06x the target's spread
    with r_z = 0.29, the signature of a bimodal hedge (mass split between a near
    and a far solution, with the mixture weight flipping on input noise) rather
    than of regression to the mean. Because the z axis spans 160 um, a few percent
    of tail mass moves E[z] by tens of um. Restricting the expectation to the
    dominant mode keeps sub-voxel precision without the tail leverage.

    Only a READOUT: the training objective always consumes the full heatmap.
    """
    S, P = z.shape
    nlx, nly, nlz = patch.shape
    dev = z.device
    am = z.argmax(-1)
    ai, aj, ak = am // (nly * nlz), (am // nlz) % nly, am % nlz
    p = torch.arange(P, device=dev)
    gi, gj, gk = p // (nly * nlz), (p // nlz) % nly, p % nlz
    m = (((gi[None, :] - ai[:, None]).abs() <= win[0]) &
         ((gj[None, :] - aj[:, None]).abs() <= win[1]) &
         ((gk[None, :] - ak[:, None]).abs() <= win[2])).to(z.dtype)
    zm = z * m
    zm = zm / zm.sum(-1, keepdim=True).clamp_min(1e-12)
    return anchor_xyz + zm @ offsets


def quantized_anchor_xyz(anchors_per_channel: np.ndarray, g: GridSpec) -> np.ndarray:
    """The anchor actually used by the scatter, i.e. snapped to the voxel lattice.

    Localizations must be reported relative to THIS, not the raw anchor, or a
    per-channel bias of up to half a voxel is introduced.
    """
    av = anchor_voxels(anchors_per_channel, g)
    return np.stack([av[:, 0] * g.x_bin + g.x_lo,
                     av[:, 1] * g.y_bin + g.y_lo,
                     np.zeros(len(av))], 1).astype(np.float32)


def grid_config_dict(g: GridSpec, p: PatchSpec):
    return {"grid": asdict(g), "patch": asdict(p)}
