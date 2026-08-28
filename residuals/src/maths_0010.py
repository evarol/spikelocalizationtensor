"""Whitening-coordinate localizer for session 0010.

For ``Y_white = Y_raw @ W``, every raw footprint ``g`` is evaluated as
``g_white = g @ W`` before it is normalized or compared with data.
"""

import numpy as np
import torch

from maths import KERNELS, VOXEL_HI, VOXEL_LO, VOXEL_SIZE_UM, build_profiles, lattice


EPS = 1e-12


def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def _continuous_refine(off, projected, temporal, source_grid, profile_index,
                       profile_sigmas, transform, mask, max_iterations, backtracks):
    """Refine fixed whitened monopole fits inside their final 1 um cells."""
    n_events = len(source_grid)
    rows = torch.arange(n_events, device=off.device)
    sigma = profile_sigmas[profile_index]
    lower = (source_grid - 0.5).clamp(
        min=torch.as_tensor(VOXEL_LO, dtype=off.dtype, device=off.device),
        max=torch.as_tensor(VOXEL_HI, dtype=off.dtype, device=off.device),
    )
    upper = (source_grid + 0.5).clamp(
        min=torch.as_tensor(VOXEL_LO, dtype=off.dtype, device=off.device),
        max=torch.as_tensor(VOXEL_HI, dtype=off.dtype, device=off.device),
    )
    chosen_projection = projected[rows, :, temporal]
    temporal_energy = torch.ones(n_events, dtype=off.dtype, device=off.device)

    def evaluate(source):
        dxy2 = (
            (off[:, :, 0] - source[:, None, 0]).square()
            + (off[:, :, 1] - source[:, None, 1]).square()
        )
        raw = sigma[:, None] / torch.sqrt(
            dxy2 + source[:, None, 2].square() + sigma[:, None].square()
        )
        white = raw if transform is None else torch.einsum("bc,bcd->bd", raw, transform)
        atom = _normalize(white * mask)
        response = (atom * chosen_projection).sum(1)
        return atom, response.square() / temporal_energy, response

    source = source_grid.detach().clone()
    _, grid_energy, _ = evaluate(source)
    for _ in range(max_iterations):
        candidate = source.detach().requires_grad_(True)
        _, energy, _ = evaluate(candidate)
        gradient, = torch.autograd.grad(energy.sum(), candidate)
        direction = gradient / gradient.norm(dim=1, keepdim=True).clamp_min(EPS)
        improved = torch.zeros(n_events, dtype=torch.bool, device=off.device)
        best_source = source
        best_energy = energy.detach()
        scale = torch.full((n_events, 1), 0.5, dtype=off.dtype, device=off.device)
        for _ in range(backtracks):
            proposal = torch.maximum(
                torch.minimum(source + scale * direction, upper), lower
            )
            _, proposal_energy, _ = evaluate(proposal)
            accept = (~improved) & (proposal_energy > best_energy)
            best_source = torch.where(accept[:, None], proposal, best_source)
            best_energy = torch.where(accept, proposal_energy, best_energy)
            improved |= accept
            scale = torch.where(accept[:, None], scale, scale * 0.5)
        source = best_source.detach()
        if not bool(improved.any()):
            break
    atom, energy, response = evaluate(source)
    invalid = ~torch.isfinite(energy) | (energy < grid_energy)
    source = torch.where(invalid[:, None], source_grid, source)
    atom, energy, response = evaluate(source)
    alpha = response / temporal_energy
    displacement = torch.linalg.vector_norm(source - source_grid, dim=1)
    return source, alpha, energy, displacement


def _continuous_refine_rho(off, projected, temporal, source_grid, profile_index,
                           profile_sigmas, transform, mask, max_iterations,
                           backtracks):
    """Refine an identifiable monopole state `(x, y, rho)`."""
    n_events = len(source_grid)
    rows = torch.arange(n_events, device=off.device)
    sigma = profile_sigmas[profile_index]
    rho_grid = torch.sqrt(source_grid[:, 2].square() + sigma.square())
    state_grid = torch.cat((source_grid[:, :2], rho_grid[:, None]), dim=1)
    xy_lower = (source_grid[:, :2] - 0.5).clamp(
        min=torch.as_tensor(VOXEL_LO[:2], dtype=off.dtype, device=off.device),
        max=torch.as_tensor(VOXEL_HI[:2], dtype=off.dtype, device=off.device),
    )
    xy_upper = (source_grid[:, :2] + 0.5).clamp(
        min=torch.as_tensor(VOXEL_LO[:2], dtype=off.dtype, device=off.device),
        max=torch.as_tensor(VOXEL_HI[:2], dtype=off.dtype, device=off.device),
    )
    lower = torch.cat((xy_lower, torch.ones((n_events, 1), dtype=off.dtype, device=off.device)), dim=1)
    upper = torch.cat((xy_upper, torch.full((n_events, 1), 600.0, dtype=off.dtype, device=off.device)), dim=1)
    chosen_projection = projected[rows, :, temporal]

    def evaluate(state):
        dxy2 = (
            (off[:, :, 0] - state[:, None, 0]).square()
            + (off[:, :, 1] - state[:, None, 1]).square()
        )
        rho = state[:, 2]
        raw = rho[:, None] / torch.sqrt(dxy2 + rho[:, None].square())
        white = raw if transform is None else torch.einsum("bc,bcd->bd", raw, transform)
        atom = _normalize(white * mask)
        response = (atom * chosen_projection).sum(1)
        return atom, response.square(), response

    state = state_grid.detach().clone()
    _, grid_energy, _ = evaluate(state)
    for _ in range(max_iterations):
        candidate = state.detach().requires_grad_(True)
        _, energy, _ = evaluate(candidate)
        gradient, = torch.autograd.grad(energy.sum(), candidate)
        direction = gradient / gradient.norm(dim=1, keepdim=True).clamp_min(EPS)
        improved = torch.zeros(n_events, dtype=torch.bool, device=off.device)
        best_state = state
        best_energy = energy.detach()
        scale = torch.full((n_events, 1), 0.5, dtype=off.dtype, device=off.device)
        for _ in range(backtracks):
            proposal = torch.maximum(torch.minimum(state + scale * direction, upper), lower)
            _, proposal_energy, _ = evaluate(proposal)
            accept = (~improved) & (proposal_energy > best_energy)
            best_state = torch.where(accept[:, None], proposal, best_state)
            best_energy = torch.where(accept, proposal_energy, best_energy)
            improved |= accept
            scale = torch.where(accept[:, None], scale, scale * 0.5)
        state = best_state.detach()
        if not bool(improved.any()):
            break
    atom, energy, response = evaluate(state)
    invalid = ~torch.isfinite(energy) | (energy < grid_energy)
    state = torch.where(invalid[:, None], state_grid, state)
    atom, energy, response = evaluate(state)
    source = torch.cat((state[:, :2], torch.zeros((n_events, 1), dtype=off.dtype, device=off.device)), dim=1)
    displacement = torch.linalg.vector_norm(state - state_grid, dim=1)
    return source, state[:, 2], response, energy, displacement


def _atoms(off, sources, profiles, transform, mask):
    """Return whitened, unit-norm atoms: (B, K, C, P)."""
    dxy2 = (
        (off[:, None, :, 0] - sources[:, :, None, 0]).square()
        + (off[:, None, :, 1] - sources[:, :, None, 1]).square()
    )
    dz2 = sources[:, :, None, 2].square()
    atoms = []
    for name, params in profiles:
        raw = KERNELS[name](dxy2, dz2, params)
        white = raw if transform is None else torch.einsum("bkc,bcd->bkd", raw, transform)
        atoms.append(_normalize(white * mask[:, None]))
    return torch.stack(atoms, dim=-1)


def _selected_monopole_atoms(off, candidates, profile_sigmas, profile_index,
                             transform, mask):
    dxy2 = (
        (off[:, None, :, 0] - candidates[:, :, None, 0]).square()
        + (off[:, None, :, 1] - candidates[:, :, None, 1]).square()
    )
    sigma = profile_sigmas[profile_index][:, None, None]
    raw = sigma / torch.sqrt(dxy2 + candidates[:, :, None, 2].square() + sigma.square())
    white = raw if transform is None else torch.einsum("bkc,bcd->bkd", raw, transform)
    return _normalize(white * mask[:, None])


def _choose(off, projected, omega, sources, profiles, transform, mask,
            candidate_block=512):
    """Exact batched hard assignment over sources, profiles, and omega rows."""
    batch = len(off)
    q_count = len(omega)
    best = torch.full((batch,), float("-inf"), device=off.device)
    source_index = torch.zeros(batch, dtype=torch.long, device=off.device)
    profile_index = torch.zeros(batch, dtype=torch.long, device=off.device)
    temporal_index = torch.zeros(batch, dtype=torch.long, device=off.device)
    alpha = torch.zeros(batch, device=off.device)
    omega_energy = omega.square().sum(1).clamp_min(EPS)
    for begin in range(0, len(sources), candidate_block):
        stop = min(begin + candidate_block, len(sources))
        candidate = sources[None].expand(batch, -1, -1)[:, begin:stop]
        atoms = _atoms(off, candidate, profiles, transform, mask)
        response = torch.einsum("bkcp,bcq->bkqp", atoms, projected)
        score = response.square() / omega_energy[None, None, :, None]
        flat_score = score.flatten(1)
        value, index = flat_score.max(1)
        update = value > best
        local_source = index // (q_count * len(profiles))
        remainder = index % (q_count * len(profiles))
        local_q = remainder // len(profiles)
        local_profile = remainder % len(profiles)
        selected_response = response[
            torch.arange(batch, device=off.device), local_source, local_q, local_profile
        ]
        best = torch.where(update, value, best)
        source_index = torch.where(update, begin + local_source, source_index)
        profile_index = torch.where(update, local_profile, profile_index)
        temporal_index = torch.where(update, local_q, temporal_index)
        alpha = torch.where(update, selected_response / omega_energy[local_q], alpha)
    return source_index, profile_index, temporal_index, alpha, best


def _refine(off, projected, omega, source, profile_index, transform, mask,
            levels=6):
    offsets = torch.cartesian_prod(
        torch.tensor([-1., 0., 1.], device=off.device),
        torch.tensor([-1., 0., 1.], device=off.device),
        torch.tensor([-1., 0., 1.], device=off.device),
    )
    step = torch.full_like(source, 32.0)
    profiles = build_profiles(("monopole",), 1)  # placeholder for type clarity
    del profiles
    for _ in range(levels):
        candidates = source[:, None] + step[:, None] * offsets[None]
        for dimension in range(3):
            candidates[..., dimension].clamp_(VOXEL_LO[dimension], VOXEL_HI[dimension])
        # The caller supplies one profile per grouped batch.
        yield candidates
        step = torch.floor(step / 2).clamp_min(VOXEL_SIZE_UM)


def localize_spikes_fixed_codebook(
    off, Y, omega, kernels=("monopole",), n_scales=9, n_sites=16,
    refine_levels=6, device=None, mask=None, spatial_transform=None,
    continuous=True, continuous_max_iterations=80, continuous_backtracks=30,
    identifiable_rho=False,
    **_ignored,
):
    """GPU localize/reconstruct in the same whitening coordinates as ``Y``.

    ``spatial_transform[n]`` maps raw local channel amplitudes to the local
    whitened output channels for event ``n``. ``None`` is the identity path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    off_np = np.asarray(off, dtype=np.float32)
    values_np = np.asarray(Y, dtype=np.float32)
    omega_np = np.asarray(omega, dtype=np.float32)
    n_events, n_channels, n_time = values_np.shape
    if mask is None:
        mask_np = np.ones((n_events, n_channels), dtype=bool)
    else:
        mask_np = np.asarray(mask, dtype=bool)
    transform_np = None
    if spatial_transform is not None:
        transform_np = np.asarray(spatial_transform, dtype=np.float32)
        if transform_np.shape != (n_events, n_channels, n_channels):
            raise ValueError("spatial_transform must have shape (N, C, C)")
    if omega_np.shape[1] != n_time:
        raise ValueError("omega and waveform lengths differ")
    profiles = build_profiles(kernels, n_scales)
    profile_sigmas = torch.as_tensor(
        [params[0] for _, params in profiles], dtype=torch.float32, device=device
    )
    monopole_profiles = all(name == "monopole" for name, _ in profiles)
    sites = lattice(n_sites).to(device)
    values = torch.as_tensor(values_np, device=device)
    offsets = torch.as_tensor(off_np, device=device)
    valid = torch.as_tensor(mask_np, device=device)
    transforms = None if transform_np is None else torch.as_tensor(transform_np, device=device)
    values = values.masked_fill(~valid[:, :, None], 0)
    temporal = _normalize(torch.as_tensor(omega_np, device=device))
    projected = torch.einsum("nct,qt->ncq", values, temporal)
    source_index, profile_index, temporal_index, alpha, captured = _choose(
        offsets, projected, temporal, sites, profiles, transforms, valid
    )
    source = sites[source_index].clone()
    step = torch.full_like(source, 32.0)
    refinement_levels = np.zeros(n_events, dtype=np.uint8)
    for _ in range(refine_levels):
        candidates = source[:, None] + step[:, None] * torch.cartesian_prod(
            torch.tensor([-1., 0., 1.], device=device), torch.tensor([-1., 0., 1.], device=device), torch.tensor([-1., 0., 1.], device=device)
        )[None]
        for dimension in range(3):
            candidates[..., dimension].clamp_(VOXEL_LO[dimension], VOXEL_HI[dimension])
        if monopole_profiles:
            atoms = _selected_monopole_atoms(
                offsets, candidates, profile_sigmas, profile_index, transforms, valid
            )
        else:
            chosen_profiles = [profiles[index] for index in profile_index.cpu().tolist()]
            raw = torch.empty((n_events, len(candidates[0]), n_channels), device=device)
            for index, (name, params) in enumerate(chosen_profiles):
                dxy2 = ((offsets[index, :, 0][None] - candidates[index, :, 0, None]).square() + (offsets[index, :, 1][None] - candidates[index, :, 1, None]).square())
                raw[index] = KERNELS[name](dxy2, candidates[index, :, 2, None].square(), params)
            white = raw if transforms is None else torch.einsum("bkc,bcd->bkd", raw, transforms)
            atoms = _normalize(white * valid[:, None])
        response = torch.einsum("bkc,bcq->bkq", atoms, projected)
        score = response.square() / temporal.square().sum(1)[None, None]
        flat = score.flatten(1).argmax(1)
        candidate_index, temporal_index = flat // len(temporal), flat % len(temporal)
        selected = response[torch.arange(n_events, device=device), candidate_index, temporal_index]
        alpha = selected / temporal.square().sum(1)[temporal_index]
        captured = selected.square() / temporal.square().sum(1)[temporal_index]
        source = candidates[torch.arange(n_events, device=device), candidate_index]
        refinement_levels += 1
        if bool((step == VOXEL_SIZE_UM).all()):
            break
        step = torch.floor(step / 2).clamp_min(VOXEL_SIZE_UM)
    sources_grid = source.clone()
    # Full-window scale refit at the refined source, with temporal row fixed.
    atoms = _atoms(offsets, source[:, None], profiles, transforms, valid)
    projected_row = projected[
        torch.arange(n_events, device=device), :, temporal_index
    ]
    response = torch.einsum("bcp,bc->bp", atoms[:, 0], projected_row)
    energy = temporal.square().sum(1)[temporal_index]
    profile_index = response.square().argmax(1)
    selected_response = response[
        torch.arange(n_events, device=device), profile_index
    ]
    alpha = selected_response / energy
    captured = selected_response.square() / energy
    captured_before_continuous = captured.clone()
    atom = atoms[torch.arange(n_events, device=device), 0, :, profile_index]
    continuous_displacement = torch.zeros(n_events, device=device)
    continuous_energy_gain = torch.zeros(n_events, device=device)
    selected_sigma = profile_sigmas[profile_index]
    rho = torch.sqrt(source[:, 2].square() + selected_sigma.square())
    if identifiable_rho:
        source, rho, response, captured, continuous_displacement = _continuous_refine_rho(
            offsets, projected, temporal_index, sources_grid, profile_index,
            profile_sigmas, transforms, valid, continuous_max_iterations,
            continuous_backtracks,
        )
        alpha = response
        continuous_energy_gain = captured - captured_before_continuous
        dxy2 = (
            (offsets[:, :, 0] - source[:, None, 0]).square()
            + (offsets[:, :, 1] - source[:, None, 1]).square()
        )
        raw = rho[:, None] / torch.sqrt(dxy2 + rho[:, None].square())
        white = raw if transforms is None else torch.einsum("bc,bcd->bd", raw, transforms)
        atom = _normalize(white * valid)
    elif continuous:
        source, alpha, captured, continuous_displacement = _continuous_refine(
            offsets, projected, temporal_index, sources_grid, profile_index,
            profile_sigmas, transforms, valid, continuous_max_iterations,
            continuous_backtracks,
        )
        continuous_energy_gain = captured - captured_before_continuous
        dxy2 = (
            (offsets[:, :, 0] - source[:, None, 0]).square()
            + (offsets[:, :, 1] - source[:, None, 1]).square()
        )
        raw = selected_sigma[:, None] / torch.sqrt(
            dxy2 + source[:, None, 2].square() + selected_sigma[:, None].square()
        )
        white = raw if transforms is None else torch.einsum("bc,bcd->bd", raw, transforms)
        atom = _normalize(white * valid)
    prediction = alpha[:, None, None] * atom[:, :, None] * temporal[temporal_index, None]
    input_energy = values.square().sum((1, 2))
    return {
        "sources": source.cpu().numpy().astype(np.float32),
        "sources_grid": sources_grid.cpu().numpy().astype(np.float32),
        "profile_idx": profile_index.cpu().numpy().astype(np.int16),
        "sigma": selected_sigma.cpu().numpy().astype(np.float32),
        "rho": rho.cpu().numpy().astype(np.float32),
        "temporal_idx": temporal_index.cpu().numpy().astype(np.int16),
        "alpha": alpha.cpu().numpy().astype(np.float32),
        "input_energy": input_energy.cpu().numpy().astype(np.float32),
        "captured_energy": captured.cpu().numpy().astype(np.float32),
        "prediction": prediction.cpu().numpy().astype(np.float32),
        "continuous_displacement_um": continuous_displacement.cpu().numpy().astype(np.float32),
        "continuous_energy_gain": continuous_energy_gain.cpu().numpy().astype(np.float32),
        "refinement_levels": refinement_levels,
    }
