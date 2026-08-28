"""Load learned and fixed spatial dictionaries without touching production outputs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from spiketensor import data as D
from spiketensor.fit_lattice import KERNELS


@dataclass
class SpatialDictionary:
    """A finite spatial dictionary evaluated for every channel configuration."""

    tag: str
    kind: str
    omega: np.ndarray                 # (M,T), row-orthonormal
    footprints: np.ndarray            # (n_cfg,C,N), unit norm over C
    candidate_pos: np.ndarray         # (N,3), anchor/centroid-relative
    cfg_id_by_channel: np.ndarray      # (n_channels,)
    anchor_shift: np.ndarray           # (n_channels,2), zero for fixed lattice
    metadata: Dict[str, Any]

    @property
    def n_candidates(self) -> int:
        return int(self.footprints.shape[2])

    @property
    def n_shapes(self) -> int:
        return int(self.omega.shape[0])

    @property
    def n_configs(self) -> int:
        return int(self.footprints.shape[0])

    def source_positions(self, indices: np.ndarray,
                         spike_channels: np.ndarray,
                         anchors: np.ndarray) -> np.ndarray:
        """Absolute xyz positions for ``indices`` shaped ``(B,R)``.

        Inactive entries (-1) become NaN.  Learned dictionaries are centered on
        the ten-channel centroid and therefore add their saved per-channel shift;
        fixed dictionaries use the ordinary anchor-relative convention.
        """
        indices = np.asarray(indices)
        ch = np.asarray(spike_channels)
        if indices.ndim != 2 or len(indices) != len(ch):
            raise ValueError("indices must be (B,R) with one channel per row")
        safe = np.maximum(indices, 0)
        local = self.candidate_pos[safe].astype(np.float32, copy=True)
        base = anchors[ch, :2].astype(np.float32) + self.anchor_shift[ch]
        local[..., :2] += base[:, None, :]
        local[indices < 0] = np.nan
        return local

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for arr in (self.omega, self.footprints, self.candidate_pos,
                    self.cfg_id_by_channel, self.anchor_shift):
            h.update(np.ascontiguousarray(arr).view(np.uint8))
        h.update(json.dumps(self.metadata, sort_keys=True, default=str).encode())
        return h.hexdigest()


def _unique_configs(off_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cfg, inv = np.unique(off_all.reshape(len(off_all), -1), axis=0,
                         return_inverse=True)
    return cfg.reshape(-1, off_all.shape[1], 2).astype(np.float32), inv.astype(np.int32)


def _normalize(g: torch.Tensor) -> torch.Tensor:
    return g / torch.linalg.vector_norm(g, dim=-1, keepdim=True).clamp_min(1e-12)


def _learned_footprints(mu: np.ndarray, sigma: np.ndarray, configs: np.ndarray,
                        kernel: str, device: str) -> np.ndarray:
    mt = torch.as_tensor(mu, dtype=torch.float32, device=device)
    st = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    out = np.empty((len(configs), configs.shape[1], len(mu)), np.float32)
    with torch.no_grad():
        for ic, off in enumerate(configs):
            ot = torch.as_tensor(off, dtype=torch.float32, device=device)
            dxy2 = ((ot[None, :, 0] - mt[:, None, 0]) ** 2
                    + (ot[None, :, 1] - mt[:, None, 1]) ** 2)
            dz2 = mt[:, None, 2].square().expand_as(dxy2)
            g = KERNELS[kernel](dxy2, dz2, (st[:, None],))
            out[ic] = _normalize(g).T.cpu().numpy()
    return out


def _fixed_footprints(mu_site: np.ndarray, profiles: Sequence[Tuple[str, Sequence[float]]],
                      configs: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    mt = torch.as_tensor(mu_site, dtype=torch.float32, device=device)
    s = len(profiles)
    n = len(mu_site) * s
    out = np.empty((len(configs), configs.shape[1], n), np.float32)
    with torch.no_grad():
        for ic, off in enumerate(configs):
            ot = torch.as_tensor(off, dtype=torch.float32, device=device)
            dxy2 = ((ot[None, :, 0] - mt[:, None, 0]) ** 2
                    + (ot[None, :, 1] - mt[:, None, 1]) ** 2)
            dz2 = mt[:, None, 2].square().expand_as(dxy2)
            rows: List[torch.Tensor] = []
            for name, params in profiles:
                p = tuple(float(x) for x in params)
                rows.append(KERNELS[name](dxy2, dz2, p))
            # Candidate order is site*S + profile, matching fit_lattice.py.
            g = torch.stack(rows, dim=1).reshape(n, configs.shape[1])
            out[ic] = _normalize(g).T.cpu().numpy()
    pos = np.repeat(mu_site.astype(np.float32), s, axis=0)
    return out, pos


def load_dictionary(runs: Path, tag: str, dataset: str = "np1",
                    device: str = "cpu", cache: Path | None = None) -> SpatialDictionary:
    """Load one production baseline as a read-only multipole dictionary.

    A cache, when requested, is written only beneath the caller-provided isolated
    path.  Production run files are never modified.
    """
    runs = Path(runs)
    rec = D.load(dataset)
    ck_path = runs / f"codebook_{tag}.pt"
    pi_path = runs / f"pi_{tag}.npz"
    summary_path = runs / f"summary_{tag}.json"
    if not (ck_path.exists() and pi_path.exists() and summary_path.exists()):
        raise FileNotFoundError(f"incomplete baseline triplet for {tag} in {runs}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    summary = json.loads(summary_path.read_text())
    with np.load(pi_path) as z:
        zfiles = set(z.files)
        learned = bool(ck.get("learned_basis") or summary.get("learned_basis"))
        omega = np.asarray(ck["a"], np.float32)
        if learned:
            required = {"mu_site", "site_sigma", "anchor_shift"}
            missing = required.difference(zfiles)
            if missing:
                raise ValueError(f"learned baseline {tag} lacks {sorted(missing)}")
            mu = z["mu_site"].astype(np.float32)
            sigma = z["site_sigma"].astype(np.float32)
            shift = z["anchor_shift"].astype(np.float32)
            off = rec.channel_offsets().astype(np.float32) - shift[:, None, :]
            cfg, cfg_id = _unique_configs(off)
            kind = "learned"
            candidate_pos = mu
            profiles = [[str(ck.get("kernel", "monopole")), [float(x)]]
                        for x in sigma]
        else:
            if "mu_site" in zfiles:
                mu_site = z["mu_site"].astype(np.float32)
            elif "mu" in zfiles:
                # Legacy lat* files store the expanded site x profile table.  Recover
                # the site lattice without changing candidate ordering.
                expanded = z["mu"].astype(np.float32)
                s_legacy = int(ck.get("S", summary.get("S", 1)))
                if len(expanded) % s_legacy:
                    raise ValueError(f"{tag}: legacy mu rows are not divisible by S")
                mu_site = expanded[::s_legacy]
                if not np.allclose(np.repeat(mu_site, s_legacy, axis=0), expanded):
                    raise ValueError(f"{tag}: legacy mu layout is not site-major")
            else:
                raise ValueError(f"fixed-lattice baseline {tag} has no candidate positions")
            shift = np.zeros((rec.n_channels, 2), np.float32)
            cfg, cfg_id = _unique_configs(rec.channel_offsets().astype(np.float32))
            profiles = ck.get("profiles")
            if not profiles:
                profiles = [(str(ck.get("kernel", "monopole")), (float(s),))
                            for s in ck.get("sigmas", [])]
            profiles = [(str(name), tuple(float(x) for x in params))
                        for name, params in profiles]
            kind = "fixed"
            candidate_pos = None

    cache_path = None
    if cache is not None:
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        cache_path = cache / f"dictionary_{tag}.npz"
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as q:
            footprints = q["footprints"].astype(np.float32)
            cached_pos = q["candidate_pos"].astype(np.float32)
        if candidate_pos is None:
            candidate_pos = cached_pos
    else:
        if learned:
            footprints = _learned_footprints(mu, sigma, cfg,
                                               str(ck.get("kernel", "monopole")), device)
        else:
            footprints, candidate_pos = _fixed_footprints(mu_site, profiles, cfg, device)
        if cache_path is not None:
            np.savez_compressed(cache_path, footprints=footprints,
                                candidate_pos=candidate_pos)

    expected = int(summary.get("KS", summary.get("N", len(candidate_pos))))
    if footprints.shape[2] != expected:
        raise ValueError(f"{tag}: built {footprints.shape[2]} candidates, metadata says {expected}")
    ident = np.eye(omega.shape[0], dtype=np.float32)
    err = float(np.max(np.abs(omega @ omega.T - ident)))
    if err > 5e-4:
        raise ValueError(f"{tag}: temporal basis is not orthonormal (max error {err:g})")
    metadata = {
        "tag": tag,
        "kind": kind,
        "dataset": dataset,
        "kernel": ck.get("kernel", summary.get("kernel")),
        "K_sites": int(summary.get("K", summary.get("N", len(candidate_pos)))),
        "S_profiles": int(summary.get("S", 1)),
        "N_candidates": int(footprints.shape[2]),
        "M": int(omega.shape[0]),
        "T": int(omega.shape[1]),
        "profiles": [[p[0], list(p[1])] if isinstance(p, tuple) else p for p in profiles],
        "source_codebook": str(ck_path),
        "source_assignments": str(pi_path),
        "source_summary": str(summary_path),
        "baseline_full_nmse": float(summary.get("full_nmse", float("nan"))),
    }
    return SpatialDictionary(tag, kind, omega, footprints, candidate_pos,
                             cfg_id, shift, metadata)
