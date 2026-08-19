"""
Analytic approach to Spike Localization.

We work on the primary assumption that all channels recorded during a spike carry
the same shape, scaled only by an amplitude factor depending on the distance from
the spike source to the recording channel.

        Y_c^s = a(X^c, C^s) * b^s        (a: amplitude scalar, b: shape row)

Each spike is reconstructed as a gained spatial footprint times one temporal cookbook row:

        Y[s, c, t]  ~=  alpha_s g_s(c) * (Pi_s Omega)_t

The spatial part is a single discrete choice (one-of-K, no soft mixture). The
alternating solve searches a coarse subset of an implicit 1 um voxel grid, then
performs a final hierarchical local refinement. Omega is a Q x T temporal
cookbook and each Pi_s is a binary one-hot 1 x Q selector. There is no neural
network and no gradient descent.

For fixed Omega, the joint spatial/temporal assignment solves alpha_s in closed
form and minimizes

        ||Y_s - alpha_s ghat_n Omega_q||_F^2

The assignment is evaluated in candidate and spike chunks so the temporary score
matrix remains bounded on the GPU.

The solver core is implemented in torch and runs on whatever device is passed
("cuda" when available); the data accessors and the drift extension stay numpy.
"""

import numpy as np
import torch
from pathlib import Path

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Spatial amplitude profiles (kernels).
# Every kernel maps separately the squared LATERAL (in-plane) and AXIAL (depth)
# offsets, dxy2 and dz2, so anisotropic forms share a signature with radial ones.
# Their absolute scale is free because the candidate score is normalized by
# ||g||.                                               (freed from the alpha/d
# blow-up of the original monopole, which needed a noise floor.)
# --------------------------------------------------------------------------- #
def _k_monopole(dxy2, dz2, p):
    return p[0] / torch.sqrt(dxy2 + dz2 + p[0] ** 2)


def _k_exponential(dxy2, dz2, p):
    return torch.exp(-torch.sqrt(dxy2 + dz2) / p[0])


def _k_gauss(dxy2, dz2, p):
    return torch.exp(-(dxy2 + dz2) / (2 * p[0] ** 2))


def _k_lorentz(dxy2, dz2, p):
    return p[0] ** 2 / (dxy2 + dz2 + p[0] ** 2)


def _k_power(dxy2, dz2, p):
    return (p[0] / torch.sqrt(dxy2 + dz2 + p[0] ** 2)) ** p[1]


def _k_student(dxy2, dz2, p):
    return (1.0 + (dxy2 + dz2) / p[0] ** 2) ** (-p[1])


def _k_yukawa(dxy2, dz2, p):
    d = torch.sqrt(dxy2 + dz2 + _EPS)
    return (p[0] / torch.sqrt(dxy2 + dz2 + p[0] ** 2)) * torch.exp(-d / p[1])


def _k_dog(dxy2, dz2, p):
    """Difference of Gaussians -- the single non-monotonic profile. A spike whose
    footprint has a genuine surround cannot be represented by any monotone kernel."""
    d2 = dxy2 + dz2
    a = torch.exp(-d2 / (2 * p[0] ** 2))
    b = torch.exp(-d2 / (2 * (p[0] * p[1]) ** 2))
    return a - b / (p[1] ** 2)


def _k_gauss_aniso(dxy2, dz2, p):
    return torch.exp(-(dxy2 / (2 * p[0] ** 2) + dz2 / (2 * p[1] ** 2)))


def _k_mono_aniso(dxy2, dz2, p):
    return p[0] / torch.sqrt(dxy2 + dz2 * (p[0] / p[1]) ** 2 + p[0] ** 2)


KERNELS = {
    "monopole": _k_monopole,
    "exponential": _k_exponential,
    "gauss": _k_gauss,
    "lorentz": _k_lorentz,
    "power": _k_power,
    "student": _k_student,
    "yukawa": _k_yukawa,
    "dog": _k_dog,
    "gauss_aniso": _k_gauss_aniso,
    "mono_aniso": _k_mono_aniso,
}
NEEDS_2 = {"power", "student", "yukawa", "dog"}


def build_profiles(kernels=("monopole",), n_scales=10, sig_lo=1.0, sig_hi=512.0,
                   n_aniso=0, extra=(2.0,)):
    """A list of (kernel_name, params); the candidate is (lattice site, profile).

    Isotropic families get n_scales log-spaced sigmas over [sig_lo, sig_hi] um.
    Anisotropic ones get an n_aniso x n_aniso grid of (lateral, axial) scales.
    Two-parameter radial families cross each sigma with the `extra` shape value.

    Returns:
        list of (name, tuple) profile parameters, in index order.
    """
    sig = np.geomspace(sig_lo, sig_hi, n_scales)
    out = []
    for k in kernels:
        if k in ("gauss_aniso", "mono_aniso"):
            s = np.geomspace(sig_lo, sig_hi, n_aniso or n_scales)
            out += [(k, (float(x), float(z))) for x in s for z in s]
        elif k in NEEDS_2:
            for s_ in sig:
                out += [(k, (float(s_), float(e))) for e in extra]
        else:
            out += [(k, (float(s_),)) for s_ in sig]
    return out


# --------------------------------------------------------------------------- #
# Spatial codebook.
# The full codebook is the implicit 1 um voxel grid. A small uniform subset is
# materialized for the coarse assignment, then each spike is refined on the
# integer grid without ever allocating all 301^3 locations.
# --------------------------------------------------------------------------- #
VOXEL_LO = (-150, -150, 0)
VOXEL_HI = (150, 150, 300)
VOXEL_SIZE_UM = 1


def _coarse_axis(n, lo, hi):
    if not 2 <= n <= hi - lo + 1:
        raise ValueError(f"n must be between 2 and {hi - lo + 1}")
    axis = np.rint(np.linspace(lo, hi, n)).astype(np.float32)
    if len(np.unique(axis)) != n:
        raise ValueError("coarse sites must occupy distinct voxels")
    return axis


def lattice(n, bounds=(VOXEL_LO, VOXEL_HI)):
    """Uniform coarse subset of the implicit 1 um voxel grid: (n^3, 3)."""
    lo, hi = bounds
    ax = [_coarse_axis(n, lo[d], hi[d]) for d in range(3)]
    MX, MY, MZ = np.meshgrid(*ax, indexing="ij")
    return torch.as_tensor(
        np.stack([MX.ravel(), MY.ravel(), MZ.ravel()], -1).astype(np.float32))


def _sym_index(C):
    """(rows, cols, weights) for the C(C+1)/2 unique entries of a CxC symmetric form."""
    r, c = np.triu_indices(C)
    w = np.where(r == c, 1.0, 2.0)
    return r, c, w


def _normalize(g):
    """Unit-norm along last dim (rows) -- peaks the Rayleigh quotient."""
    return g / g.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def _profile_block(off_cfg, sites, name, params, device):
    """Raw footprint for a set of sites on one channel config: (len(sites), C).

    off_cfg: (C, 2) lateral channel offsets relative to the anchor; channels lie
    in the plane z=0, a site's axial offset is its own depth.
    sites: torch (n_sites, 3) on device.
    """
    dxy2 = ((off_cfg[:, None, 0] - sites[None, :, 0]) ** 2
            + (off_cfg[:, None, 1] - sites[None, :, 1]) ** 2)
    dz2 = sites[None, :, 2] ** 2
    return KERNELS[name](dxy2, dz2, params).movedim(0, 1)   # (n_sites, C)


# --------------------------------------------------------------------------- #
# Data accessors (numpy)
# --------------------------------------------------------------------------- #
def _make_spike_vector(waveform_path, times_path, n_channels=None):
    """Load spike waveforms into a tensor.

    Args:
        waveform_path: path to (N, C, T) spike waveform array.
        times_path: path to (N,) spike sample indices.

    Returns:
        Y: (N, C, T) waveform tensor.
        TMAP: (N,) spike index -> recording timestamp.
    """
    Y = np.load(waveform_path)
    TMAP = np.load(times_path)
    if n_channels is not None:
        Y = Y[:, :n_channels, :]
    return Y, TMAP


def _get_spike(Y, s):
    """Return spike s as (C, T)."""
    return Y[s]


def _get_channel(Y, s, c):
    """Return the full waveform of spike s on channel c as (1, 1, T)."""
    return Y[s][c][None, None]


def load_session(session_path, max_spikes=None):
    """Load a session's saved neighborhoods for the fit.

    Reads the arrays written by src/preprocessing/extract_neighborhoods.py:
      neighborhood_waveforms.npy  (N, C, T)
      local_coords.npy            (N, C, 2) lateral offsets vs the spike centroid
      neighbor_ids.npy            (N, C) global channel ids, -1 = padding
      neighbor_counts.npy         (N,)
      centroids.npy               (N, 2) global centroid per spike
      spike_times.npy             (N,)

    Returns:
        Y:   (N, C, T) waveforms.
        off: (N, C, 2) lateral channel offsets relative to each spike's anchor.
        mask: (N, C) bool, True for the real (non-padded) neighbors.
        meta: dict with centroids, neighbor_ids, neighbor_counts, times.
    """
    session_path = Path(session_path)
    Y = np.load(session_path / "neighborhood_waveforms.npy")
    off = np.load(session_path / "local_coords.npy").astype(np.float32)
    ids = np.load(session_path / "neighbor_ids.npy")
    nbr_counts = np.load(session_path / "neighbor_counts.npy")
    centroids = np.load(session_path / "centroids.npy")
    times = np.load(session_path / "spike_times.npy")

    if max_spikes is not None:
        Y, off, ids, nbr_counts, centroids, times = (
            Y[:max_spikes], off[:max_spikes], ids[:max_spikes],
            nbr_counts[:max_spikes], centroids[:max_spikes], times[:max_spikes],
        )

    mask = ids >= 0
    meta = {
        "centroids": centroids,
        "neighbor_ids": ids,
        "neighbor_counts": nbr_counts,
        "times": times,
    }
    return Y, off, mask, meta


def load_channel_map(recording_path):
    """Absolute channel geometry via probeinterface (e.g. Neuropixels).

    Only needed to reconstruct ABSOLUTE source positions (centroids + tool
    offset) or to inspect the probe; the fit itself uses the relative
    local_coords already saved by extract_neighborhoods.py. Accepts either the
    recording folder or the path to a SpikeGLX .ap.meta / .bin file.

    Returns:
        probe object exposing .contact_positions (n_ch, D) in um and
        .device_channel_indices.
    """
    import probeinterface as pi

    p = Path(recording_path)
    if p.is_dir() or p.suffix not in (".meta", ".bin"):
        meta = next(p.glob("*.ap.meta"), None) if p.is_dir() else None
        if meta is None:
            raise FileNotFoundError(f"no *.ap.meta found in {p}")
        return pi.read_spikeglx(str(meta))
    meta = p if p.suffix == ".meta" else p.with_suffix(".meta")
    return pi.read_spikeglx(str(meta))


# --------------------------------------------------------------------------- #
# Model primitives (torch)
# --------------------------------------------------------------------------- #
def _model_wf(amplitude, shape):
    """Model waveform for a spike: outer product of amplitude profile and shape."""
    return torch.outer(amplitude, shape)


def _compute_P(Y, a):
    """Reduce every spike to the C(C+1)/2 scalars of its CxC symmetric M_s M_s^T.

    With a orthonormal and M_s = Y_s a^T (C x Q), the best-candidate Rayleigh
    quotient score_s(n) needs only M_s M_s^T, not the raw waveform.

    Args:
        Y: (N, C, T) torch tensor on device.
        a: (Q, T) orthonormal time basis on device.

    Returns:
        P: (N, n_pairs) unique entries of M_s M_s^T.
        y2: (N,) squared Frobenius energy ||Y_s||^2.
    """
    C = Y.shape[1]
    r, c, _ = _sym_index(C)
    M = torch.einsum("nct,tq->ncq", Y, a.T)          # (N, C, Q)
    Ps = torch.einsum("ncq,niq->nci", M, M)          # (N, C, C)
    P = Ps[:, r, c]
    y2 = (Y * Y).sum((1, 2))
    return P, y2


def _assign_hard_temporal(
    sites,
    profiles,
    off_cfg,
    mask_cfg,
    cfg_id,
    M,
    omega,
    device,
    candidate_block=4096,
    spike_block=512,
):
    """Jointly choose spatial candidate, Omega row, and closed-form gain."""
    C, Q = M.shape[1:]
    S = len(profiles)
    n_cand = len(sites) * S
    best = torch.full((len(M),), float("-inf"), device=device)
    pick = torch.zeros(len(M), dtype=torch.long, device=device)
    temporal_idx = torch.zeros(len(M), dtype=torch.long, device=device)
    alpha = torch.zeros(len(M), device=device)
    omega_energy = omega.square().sum(1).clamp_min(_EPS)

    for ic in range(len(off_cfg)):
        rows = np.flatnonzero(cfg_id == ic)
        for b0 in range(0, len(rows), spike_block):
            row = rows[b0:b0 + spike_block]
            Mc = M[row]
            bc = torch.full((len(row),), float("-inf"), device=device)
            kc = torch.zeros(len(row), dtype=torch.long, device=device)
            qc = torch.zeros(len(row), dtype=torch.long, device=device)
            ac = torch.zeros(len(row), device=device)
            for lo in range(0, n_cand, candidate_block):
                hi = min(n_cand, lo + candidate_block)
                s0, s1 = lo // S, (hi + S - 1) // S
                footprints = torch.empty((s1 - s0, S, C), device=device)
                for j, (name, params) in enumerate(profiles):
                    g = _profile_block(
                        off_cfg[ic], sites[s0:s1], name, params, device)
                    footprints[:, j] = _normalize(g * mask_cfg[ic][None])
                footprints = footprints.reshape(-1, C)[lo - s0 * S:hi - s0 * S]
                response = torch.einsum("kc,bcq->kbq", footprints, Mc)
                response = response.permute(1, 0, 2).reshape(len(row), -1)
                score = (
                    response.reshape(len(row), -1, Q).square()
                    / omega_energy[None, None]
                ).reshape(len(row), -1)
                block_best, flat_idx = score.max(1)
                block_q = flat_idx % Q
                block_response = response.gather(1, flat_idx[:, None]).squeeze(1)
                block_alpha = block_response / omega_energy[block_q]
                update = block_best > bc
                bc = torch.where(update, block_best, bc)
                kc = torch.where(update, lo + flat_idx // Q, kc)
                qc = torch.where(update, block_q, qc)
                ac = torch.where(update, block_alpha, ac)
                del footprints, response, score
            best[row] = bc
            pick[row] = kc
            temporal_idx[row] = qc
            alpha[row] = ac
    return pick, temporal_idx, alpha, best


def _assign_from_P(sites, profiles, off_cfg, cfg_id, P, device, ks_block=16384):
    """Exact argmax of the Rayleigh quotient over every candidate, waveform-free.

    score_s(n) = ghat_n^T P_s ghat_n = sum_pairs Phi[n, pair] P_s[pair], so the
    whole argmax is a GEMM (candidates x pairs) @ (pairs x spikes) per channel
    config. Loops config-outer so each profile block is built once per config.

    Args:
        sites: torch (K, 3) lattice on device.
        profiles: list of (name, params).
        off_cfg: torch (n_configs, C, 2) unique lateral offsets per config.
        cfg_id: numpy (N,) config index per spike.
        P: torch (N, n_pairs) precomputed scalar reduction.

    Returns:
        pick: (N,) best candidate index k*S + j.
        best: (N,) best score (max over n of score_s(n)).
    """
    C = off_cfg.shape[1]
    r, c, w = _sym_index(C)
    r = torch.as_tensor(r, device=device, dtype=torch.long)
    c = torch.as_tensor(c, device=device, dtype=torch.long)
    w = torch.as_tensor(w, device=device, dtype=torch.float32)
    S = len(profiles)
    n_cand = len(sites) * S
    best = torch.full((P.shape[0],), float("-inf"), device=device)
    pick = torch.zeros(P.shape[0], dtype=torch.long, device=device)

    for ic in range(len(off_cfg)):
        rows = np.flatnonzero(cfg_id == ic)
        if len(rows) == 0:
            continue
        Pc = P[rows]                                   # (n_spk, pairs)
        bc = torch.full((len(rows),), float("-inf"), device=device)
        kc = torch.zeros(len(rows), dtype=torch.long, device=device)
        # each profile is (site x current-profile); candidate = site*S + profile
        for lo in range(0, n_cand, ks_block):
            hi = min(n_cand, lo + ks_block)           # candidates [lo, hi)
            s0, s1 = lo // S, (hi + S - 1) // S        # whole-site slice
            Phi = torch.empty((s1 - s0, S, C), device=device)
            for j, (nm, pr) in enumerate(profiles):
                g = _profile_block(off_cfg[ic], sites[s0:s1], nm, pr, device)
                Phi[:, j] = _normalize(g)
            Phi = Phi.reshape(-1, C)[lo - s0 * S: hi - s0 * S]   # (block, C)
            Phi_w = Phi[:, r] * Phi[:, c] * w          # (block, pairs)
            sc = Phi_w @ Pc.T                          # (block, n_spk)
            m = sc.max(0).values
            k = sc.argmax(0)
            upd = m > bc
            bc = torch.where(upd, m, bc)
            kc = torch.where(upd, k + lo, kc)
            del Phi, Phi_w, sc
        best[rows] = bc
        pick[rows] = kc
    return pick, best


def _footprint_per_spike(off_cfg, sites, profiles, k, device, mask_cfg=None):
    """Unit-norm footprint of each spike's OWN chosen candidate. O(C) per spike.

    off_cfg: (C, 2) lateral offsets (single shared config in this row grouping).
    k: (N,) candidate index; candidate = site*len(profiles) + profile.
    Returns: (N, C) normalized footprints.
    """
    S = len(profiles)
    site = k // S
    prof = k % S
    g = torch.empty((len(k), off_cfg.shape[0]), device=device)
    for j in torch.unique(prof).tolist():
        sel = prof == j
        nm, pr = profiles[j]
        g[sel] = _profile_block(off_cfg, sites[site[sel]], nm, pr, device)
    if mask_cfg is not None:
        g *= mask_cfg[None]
    return _normalize(g)


def _basis_scatter(Y, off_cfg, cfg_id, sites, profiles, pick, device):
    """Weighted-PCA refit scatter: u_s = ghat_s^T Y_s, S = sum_s u_s u_s^T.

    ghat is unit-norm, so the ||g||^2 weight of the weighted-PCA step is already
    folded in and S is a plain (T x T) scatter of the u's.
    """
    T = Y.shape[2]
    S = torch.zeros((T, T), device=device)
    for ic in range(len(off_cfg)):
        rows = np.flatnonzero(cfg_id == ic)
        if len(rows) == 0:
            continue
        g = _footprint_per_spike(off_cfg[ic], sites, profiles, pick[rows], device)
        U = torch.einsum("bc,bct->bt", g, Y[rows])     # (n, T)
        S += U.T @ U
    return S


def _refit_omega(
    Y, off_cfg, mask_cfg, cfg_id, sites, profiles, pick, temporal_idx, alpha,
    omega, device,
    spike_block=65536,
):
    """Refit each Omega row by weighted least squares with gains fixed."""
    Q = len(omega)
    T = Y.shape[2]
    total = torch.zeros((Q, T), device=device)
    count = torch.zeros(Q, device=device)
    weight = torch.zeros(Q, device=device)
    for ic in range(len(off_cfg)):
        rows = np.flatnonzero(cfg_id == ic)
        for b0 in range(0, len(rows), spike_block):
            row = rows[b0:b0 + spike_block]
            g = _footprint_per_spike(
                off_cfg[ic], sites, profiles, pick[row], device, mask_cfg[ic])
            projected = torch.einsum("bc,bct->bt", g, Y[row])
            q = temporal_idx[row]
            a = alpha[row]
            total.index_add_(0, q, a[:, None] * projected)
            count += torch.bincount(q, minlength=Q)
            weight += torch.bincount(q, weights=a.square(), minlength=Q)
    updated = omega.clone()
    used = weight > _EPS
    updated[used] = _normalize(total[used] / weight[used, None])
    return updated, count


def _refine_sources(
    Y,
    M,
    omega,
    sites,
    profiles,
    off_cfg,
    mask_cfg,
    cfg_id,
    pick,
    temporal_idx,
    alpha,
    best,
    n_sites,
    device,
    max_levels=6,
    spike_block=8192,
):
    """Refine each spike through shrinking grids to one integer 1 um voxel."""
    n_spatial_profiles = len(profiles)
    site_idx = pick // n_spatial_profiles
    profile_idx = (pick % n_spatial_profiles).to("cpu").numpy()
    current = sites[site_idx].clone()
    coarse = current.clone()
    levels = np.zeros(len(Y), dtype=np.uint8)
    offsets = torch.cartesian_prod(
        torch.tensor([-1.0, 0.0, 1.0], device=device),
        torch.tensor([-1.0, 0.0, 1.0], device=device),
        torch.tensor([-1.0, 0.0, 1.0], device=device),
    )

    grid = sites.reshape(n_sites, n_sites, n_sites, 3)
    axes = (grid[:, 0, 0, 0], grid[0, :, 0, 1], grid[0, 0, :, 2])
    coarse_indices = torch.stack((
        site_idx // (n_sites * n_sites),
        (site_idx // n_sites) % n_sites,
        site_idx % n_sites,
    ), dim=1)
    initial_steps = []
    for dim, axis in enumerate(axes):
        index = coarse_indices[:, dim]
        left = axis[index] - axis[(index - 1).clamp_min(0)]
        right = axis[(index + 1).clamp_max(n_sites - 1)] - axis[index]
        initial_steps.append(torch.ceil(0.5 * torch.maximum(left, right)))
    step = torch.stack(initial_steps, dim=1).clamp_min(VOXEL_SIZE_UM)
    omega_norm = omega.square().sum(1).clamp_min(_EPS)

    groups = []
    for config in range(len(off_cfg)):
        for profile in range(n_spatial_profiles):
            rows = np.flatnonzero((cfg_id == config) & (profile_idx == profile))
            if len(rows):
                groups.append((config, profile, rows))

    for _ in range(max_levels):
        for config, profile, group_rows in groups:
            rows = group_rows
            for b0 in range(0, len(rows), spike_block):
                row = rows[b0:b0 + spike_block]
                candidates = current[row, None, :] + step[row, None, :] * offsets[None]
                for dim in range(3):
                    candidates[..., dim].clamp_(VOXEL_LO[dim], VOXEL_HI[dim])
                channel_xy = off_cfg[config]
                dxy2 = (
                    (channel_xy[None, None, :, 0] - candidates[:, :, None, 0]).square()
                    + (channel_xy[None, None, :, 1] - candidates[:, :, None, 1]).square()
                )
                dz2 = candidates[:, :, None, 2].square()
                name, params = profiles[profile]
                footprints = KERNELS[name](dxy2, dz2, params)
                footprints *= mask_cfg[config][None, None]
                footprints = _normalize(footprints)
                response = torch.einsum("blc,bcq->blq", footprints, M[row])
                response = response.reshape(len(row), -1)
                score = (
                    response.reshape(len(row), -1, M.shape[2]).square()
                    / omega_norm[None, None]
                ).reshape(len(row), -1)
                block_best, flat = score.max(1)
                local_idx = flat // M.shape[2]
                q = flat % M.shape[2]
                selected_response = response.gather(1, flat[:, None]).squeeze(1)
                updated = candidates[torch.arange(len(row), device=device), local_idx]
                current[row] = updated
                temporal_idx[row] = q
                alpha[row] = selected_response / omega_norm[q]
                best[row] = block_best
                levels[row] += 1
        if bool((step == VOXEL_SIZE_UM).all()):
            break
        step = torch.floor(step / 2).clamp_min(VOXEL_SIZE_UM)
    return current, coarse, temporal_idx, alpha, best, levels


# --------------------------------------------------------------------------- #
# Main alternating minimization
# --------------------------------------------------------------------------- #
def fit_spike_model(
    off,
    Y,
    Q=8,
    kernels=("monopole",),
    n_scales=10,
    n_sites=16,
    n_iters=8,
    tol=1e-5,
    refine_levels=6,
    refine_stop_um=3.0,
    device=None,
    mask=None,
):
    """Alternating minimization with hard spatial and temporal selections.

    Omega is a Q x T temporal cookbook. Each spike chooses one spatial candidate,
    a binary one-hot selector Pi_s, and a closed-form scalar alpha_s:
        min sum_s || Y_s - alpha_s g_s (Pi_s Omega) ||_F^2
    The assignment is exact over the materialized coarse candidates and the
    temporal update is exact for fixed assignments, so the coarse loss is
    monotone non-increasing up to floating-point error.

    Args:
        off: (N, C, 2) lateral channel offsets relative to each spike's anchor.
        Y: (N, C, T) waveforms.
        mask: (N, C) True for real channels and False for padding.
        Q: number of rows in Omega; Pi selects exactly one per spike.
        kernels: profile families in the dictionary.
        n_scales: sigmas per isotropic family.
        n_sites: lattice side (K = n_sites^3).
        n_iters: alternating iterations.
        tol: stop when normalized nMSE gain < tol.
        refine_levels: maximum shrinking 3x3x3 local-grid refinements; the
            default reaches the 1 um neighborhood from the default coarse grid.
        refine_stop_um: retained for command-line compatibility. With enough
            refinement levels, the local search evaluates a 1 um step.
        device: torch device; defaults to cuda if available else cpu.

    Returns:
        dict with explicit coarse and refined candidate identifiers, source
        coordinates, profile parameters, temporal cookbook, and scores.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    N, C, T = Y.shape
    sites = lattice(n_sites).to(device)
    profiles = build_profiles(kernels, n_scales)
    ns_prof = len(profiles)
    n_cand = len(sites) * ns_prof

    # config grouping in numpy (torch has no unique over rows cheaply)
    off_np = np.asarray(off.cpu().numpy() if torch.is_tensor(off) else off)
    Y_np = np.asarray(Y.cpu().numpy() if torch.is_tensor(Y) else Y)
    if mask is None:
        mask_np = np.ones((N, C), dtype=bool)
    else:
        mask_np = np.asarray(
            mask.cpu().numpy() if torch.is_tensor(mask) else mask, dtype=bool)
        if mask_np.shape != (N, C):
            raise ValueError(f"mask must have shape {(N, C)}, got {mask_np.shape}")
    cfg_rows = np.concatenate(
        (off_np.reshape(N, -1), mask_np.astype(np.float32)), axis=1)
    cfg_u, cfg_id = np.unique(cfg_rows, axis=0, return_inverse=True)
    off_cfg = torch.as_tensor(
        cfg_u[:, :2 * C].reshape(-1, C, 2), dtype=torch.float32, device=device)
    mask_cfg = torch.as_tensor(
        cfg_u[:, 2 * C:], dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y_np, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
    Yt = Yt.masked_fill(~mask_t[:, :, None], 0)

    # Initialize Omega from real high-energy-channel spike waveforms.
    F = Y_np.reshape(-1, T) - Y_np.reshape(-1, T).mean(-1, keepdims=True)
    per_spike = F.reshape(N, C, T)
    per_spike[~mask_np] = 0
    strongest = np.square(per_spike).sum(2).argmax(1)
    representative = per_spike[np.arange(N), strongest]
    rng = np.random.default_rng(0)
    init_rows = rng.choice(N, Q, replace=N < Q)
    omega = _normalize(torch.as_tensor(
        representative[init_rows], dtype=torch.float32, device=device))

    var = float(np.mean(F ** 2))
    hist, prev = [], np.inf
    omega_final = omega
    for it in range(1, n_iters + 1):
        omega_final = omega
        M = torch.einsum("nct,tq->ncq", Yt, omega.T)
        y2 = (Yt * Yt).sum((1, 2))
        pick, temporal_idx, alpha, best = _assign_hard_temporal(
            sites, profiles, off_cfg, mask_cfg, cfg_id, M, omega, device
        )
        del M
        nmse = float((y2 - best).mean().item() / (C * T) / var)
        updated, temporal_count = _refit_omega(
            Yt, off_cfg, mask_cfg, cfg_id, sites, profiles, pick, temporal_idx,
            alpha, omega, device
        )
        hist.append({"step": it, "nmse": nmse,
                     "used": int(len(torch.unique(pick))),
                     "temporal_used": int(len(torch.unique(temporal_idx)))})
        print(f"  iter {it:2d}  nMSE {nmse:.4f}  "
              f"{len(torch.unique(pick)):,}/{n_cand:,} candidates used", flush=True)
        if prev - nmse < tol:
            print(f"  converged (gain {prev-nmse:.2e} < {tol:g})", flush=True)
            break
        prev = nmse
        omega = updated

    M = torch.einsum("nct,tq->ncq", Yt, omega_final.T)
    refined, coarse, temporal_idx, alpha, best, refinement_levels = _refine_sources(
        Yt, M, omega_final, sites, profiles, off_cfg, mask_cfg, cfg_id, pick,
        temporal_idx, alpha, best, n_sites, device, max_levels=refine_levels,
    )
    del M
    nmse_coarse = nmse
    nmse = float((y2 - best).mean().item() / (C * T) / var)

    device_cpu = torch.device("cpu")
    coarse_pick = pick.to(device_cpu).numpy()
    coarse_site_idx = coarse_pick // ns_prof
    profile_idx = coarse_pick % ns_prof
    pole_source = refined.to(device_cpu).numpy().astype(np.float64)
    coarse_source = coarse.to(device_cpu).numpy().astype(np.float64)
    voxel_coordinates = np.rint(pole_source).astype(np.int16)
    if not np.array_equal(pole_source, voxel_coordinates):
        raise RuntimeError("refined sources left the 1 um voxel grid")
    voxel_shape = np.subtract(VOXEL_HI, VOXEL_LO) + 1
    voxel_offset = voxel_coordinates.astype(np.int32) - np.asarray(VOXEL_LO)
    voxel_idx = np.ravel_multi_index(voxel_offset.T, voxel_shape)
    refined_pick = (
        voxel_idx.astype(np.int64) * ns_prof + profile_idx.astype(np.int64))

    one_hot = np.zeros((N, Q), np.uint8)
    temporal_idx_np = temporal_idx.to(device_cpu).numpy()
    alpha_np = alpha.to(device_cpu).numpy()
    one_hot[np.arange(N), temporal_idx_np] = 1
    return {
        "pick": coarse_pick,
        "site_idx": coarse_site_idx,
        "coarse_pick": coarse_pick,
        "coarse_site_idx": coarse_site_idx,
        "refined_pick": refined_pick,
        "profile_idx": profile_idx,
        "sources": pole_source,
        "voxel_coordinates": voxel_coordinates,
        "voxel_idx": voxel_idx,
        "voxel_bounds_um": np.asarray((VOXEL_LO, VOXEL_HI), dtype=np.int16),
        "voxel_size_um": VOXEL_SIZE_UM,
        "coarse_sources": coarse_source,
        "refinement_levels": refinement_levels,
        "refinement_displacement_um": np.linalg.norm(
            pole_source - coarse_source, axis=1).astype(np.float32),
        "sigma": np.array([profiles[p][1][0] for p in profile_idx], np.float32),
        "omega": omega_final.to("cpu").numpy(),
        "a": omega_final.to("cpu").numpy(),
        "pi": one_hot,
        "alpha": alpha_np,
        "temporal_idx": temporal_idx_np,
        "temporal_one_hot": one_hot,
        "v": alpha_np[:, None] * one_hot,
        "nmse": nmse,
        "nmse_coarse": nmse_coarse,
        "history": hist,
        "lattice": sites.to(device_cpu).numpy(),
        "profiles": profiles,
        "n_candidates": int(np.prod(voxel_shape)) * ns_prof,
        "n_coarse_candidates": n_cand,
    }


# --------------------------------------------------------------------------- #
# Extension 1: Spike collision detection (greedy matching pursuit)
# --------------------------------------------------------------------------- #
def greedy_matching_pursuit(
    off,
    Y,
    Q=8,
    kernels=("monopole",),
    n_scales=10,
    n_sites=16,
    epsilon=1e-3,
    tol_gain=1e-4,
    max_sources=4,
    device=None,
):
    """Decompose a single waveform (or batch) into multiple spatial sources.

    Repeatedly fit a single source to the residual Y - Yhat. When the gain from
    the next source falls below tol_gain * ||Y||^2, stop adding sources.

    Returns:
        dict with per-source pick/v_s, source counts, and residual energy.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Y = np.atleast_3d(Y)
    N, T = Y.shape[0], Y.shape[2]
    R = Y.copy()
    pick_src = np.zeros((N, max_sources), dtype=np.int64)
    v_src = np.zeros((N, max_sources, Q), np.float32)
    n_found = np.zeros(N, dtype=np.int64)
    resid = np.zeros(N)

    for s in range(N):
        for j in range(max_sources):
            y = R[s:s + 1]
            res = fit_spike_model(off[s:s + 1], y, Q=Q, kernels=kernels,
                                  n_scales=n_scales, n_sites=n_sites, n_iters=5,
                                  device=device)
            pj = res["refined_pick"][0]
            pred = _predict_waveform(off[s], res, pj)           # (C, T)
            yE = float(np.sum(y * y))
            gain = yE - float(np.sum((y[0] - pred) ** 2))
            pick_src[s, j] = pj
            v_src[s, j] = res["v"][0]
            R[s] = y[0] - pred
            n_found[s] = j + 1
            resid[s] = float(np.sum(R[s] ** 2))
            if resid[s] < epsilon or gain < tol_gain * yE:
                break

    return {
        "pick": pick_src,
        "v": v_src,
        "n_sources": n_found,
        "residual": resid,
    }


def _predict_waveform(off_row, res, k, device="cpu"):
    """Reconstruct a one-spike fit at its refined source location."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    S = len(res["profiles"])
    prof = k % S
    nm, pr = res["profiles"][prof]
    sites = torch.as_tensor(
        np.asarray(res["sources"][[0]]), dtype=torch.float32, device=dev)
    off_t = torch.as_tensor(np.asarray(off_row), dtype=torch.float32, device=dev)
    g = _normalize(_profile_block(off_t, sites, nm, pr, dev))[0]
    q = int(res["temporal_idx"][0])
    shape = torch.as_tensor(res["omega"][q], device=dev)
    alpha = float(res.get("alpha", np.ones(1))[0])
    return (alpha * torch.outer(g, shape)).cpu().numpy()


# --------------------------------------------------------------------------- #
# Extension 3: Non-rigid drift correction (numpy)
# --------------------------------------------------------------------------- #
def depth_histogram(z_s, alpha_s, tau_s, z_edges, n_timebins):
    """Amplitude-weighted depth histogram per time frame.

    Args:
        z_s: (N,) spike depths.
        alpha_s: (N,) spike amplitudes.
        tau_s: (N,) spike time frames (bins in [0, T)).
        z_edges: (n_z+1,) depth bin edges.
        n_timebins: number of time frames.

    Returns:
        (n_timebins, n_z) histograms.
    """
    n_z = len(z_edges) - 1
    H = np.zeros((n_timebins, n_z))
    zi = np.searchsorted(z_edges, z_s, side="right") - 1
    valid = (zi >= 0) & (zi < n_z - 1) & (tau_s >= 0) & (tau_s < n_timebins)
    zi = zi[valid]
    frac = (z_s[valid] - z_edges[zi]) / (z_edges[zi + 1] - z_edges[zi] + _EPS)
    w = np.log1p(np.abs(alpha_s[valid]))
    H[tau_s[valid], zi] += w * (1 - frac)
    H[tau_s[valid], zi + 1] += w * frac
    return H


def gaussian_smooth(H, sigma_z=1.0):
    """1D Gaussian smoothing of depth histograms along the depth axis."""
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(H, sigma=sigma_z, axis=-1)


def pairwise_displacement(h1, h0, d_max, v_max, dt):
    """Cross-correlation displacement between two depth profiles.

    Args:
        h1, h0: depth profiles (mean-centered, normalised).
        d_max: max absolute displacement (spatial bins).
        v_max: max drift velocity (bins per frame).
        dt: time gap between frames.

    Returns:
        delta (displacement), rho (peak correlation) or (0,0) if admissable
        set is empty.
    """
    bound = min(d_max, v_max * dt)
    if bound < 1:
        return 0.0, 0.0
    n = len(h1)
    lag_axis = np.arange(-(n - 1), n)
    admiss = np.abs(lag_axis) <= bound
    corr = np.array([np.sum(h1 * np.roll(h0, d)) for d in lag_axis])
    peak = np.argmax(corr[admiss])
    return lag_axis[admiss][peak], corr[admiss][peak]


def recover_drift_trajectory(deltas, weights, n_frames, anchor=0):
    """Weighted least-squares displacement integration.

    Args:
        deltas: list of (i, j, delta) pair constraints.
        weights: list of weights per constraint.
        n_frames: number of time frames.
        anchor: index fixed at 0 displacement.

    Returns:
        (n_frames,) drift trajectory D such that D_j - D_i ~= delta.
    """
    n = n_frames
    A = np.zeros((len(deltas), n))
    b = np.zeros(len(deltas))
    W = np.array(weights)
    for r, (i, j, d) in enumerate(deltas):
        A[r, j] = 1.0
        A[r, i] = -1.0
        b[r] = d
    A = np.vstack([A, np.eye(n)[anchor]])
    b = np.concatenate([b, [0.0]])
    W = np.concatenate([W, [1.0]])
    Aw = A * W[:, None]
    D = np.linalg.lstsq(Aw, W * b, rcond=None)[0]
    return D - D[anchor]


# --------------------------------------------------------------------------- #
# __main__ smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # toy: 10 contacts on an ASYMMETRIC grid in the plane z=0, source below them
    C, T = 10, 90
    base = np.array([[0, 0], [30, 0], [60, 0], [0, 40], [40, 40], [80, 40],
                     [0, 80], [40, 80], [80, 80], [40, 120]], np.float32)   # (C,2)
    src_true = np.array([35.0, 52.0, 60.0])
    spike_shape = np.exp(-((np.arange(T) - 40.0) ** 2) / 30.0).astype(np.float32)

    sites = lattice(16)
    g = _profile_block(torch.as_tensor(base),
                       torch.as_tensor(np.array([src_true], np.float32)),
                       "monopole", (40.0,), "cpu")[0].numpy()
    Y = (np.outer(g, spike_shape) + 0.02 * rng.normal(size=(C, T)))[None].astype(np.float32)
    off = base[None]

    fit = fit_spike_model(off, Y, Q=2, kernels=("monopole",), n_scales=4,
                          n_sites=16, n_iters=8)
    print("true source  :", src_true)
    print("estimated    :", fit["sources"][0])
    print("sigma        :", fit["sigma"][0])
    print("final nMSE   :", fit["nmse"])
    print("v (Q=2)       :", np.round(fit["v"][0], 3))
    print("history      :", [round(h["nmse"], 5) for h in fit["history"]])
