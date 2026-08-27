"""Bounded continuous xyz refinement of a frozen masked monopole fit.

The discrete fit supplies a final 1 um voxel, monopole scale, temporal row, and
temporal cookbook for each spike. This module keeps every fitted quantity fixed
and maximizes only the gain-eliminated spatial score inside that voxel.

The final step uses projected gradient ascent, adapted to padded channel masks
and our hard temporal-row objective.
"""

from __future__ import annotations

import numpy as np
import torch


def voxel_cell_bounds(mu_grid, grid_bounds, voxel_size):
    """Return Voronoi-cell bounds around final regularly spaced grid points."""
    mu_grid = np.asarray(mu_grid, dtype=np.float64)
    grid_bounds = np.asarray(grid_bounds, dtype=np.float64)
    if mu_grid.ndim != 2 or mu_grid.shape[1] != 3:
        raise ValueError(f"mu_grid must have shape (N, 3), got {mu_grid.shape}")
    if grid_bounds.shape != (2, 3):
        raise ValueError(
            f"grid_bounds must have shape (2, 3), got {grid_bounds.shape}")
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size:g}")
    half = 0.5 * voxel_size
    lower = np.maximum(mu_grid - half, grid_bounds[0])
    upper = np.minimum(mu_grid + half, grid_bounds[1])
    if np.any(lower > mu_grid) or np.any(mu_grid > upper):
        raise ValueError("grid points fall outside their continuous cells")
    return lower, upper


def monopole_profile(offsets, mu, sigma, mask=None):
    """Return the raw monopole footprint and its derivative intermediates."""
    z = mu[:, 2]
    a = torch.stack(
        (
            offsets[:, :, 0] - mu[:, None, 0],
            offsets[:, :, 1] - mu[:, None, 1],
            -z[:, None].expand(-1, offsets.shape[1]),
        ),
        dim=2,
    )
    d2 = (a * a).sum(dim=2) + (sigma * sigma)[:, None]
    footprint = sigma[:, None] / torch.sqrt(d2)
    if mask is not None:
        footprint = footprint * mask
    return footprint, a, footprint / d2


def score(form, offsets, mu, sigma, mask=None):
    """Gain-eliminated captured energy at continuous source positions."""
    footprint, _, _ = monopole_profile(offsets, mu, sigma, mask)
    product = torch.einsum("nij,nj->ni", form, footprint)
    return (footprint * product).sum(dim=1) / (footprint * footprint).sum(dim=1)


def score_gradient_hessian(form, offsets, mu, sigma, mask=None):
    """Return the score and its analytic xyz gradient and Hessian."""
    footprint, displacement, quotient = monopole_profile(
        offsets, mu, sigma, mask)
    d2 = (sigma * sigma)[:, None] + (displacement * displacement).sum(dim=2)
    jacobian = quotient[:, :, None] * displacement

    product = torch.einsum("nij,nj->ni", form, footprint)
    numerator = (footprint * product).sum(dim=1)
    denominator = (footprint * footprint).sum(dim=1)
    value = numerator / denominator

    grad_numerator = 2.0 * torch.einsum(
        "nci,nc->ni", jacobian, product)
    grad_denominator = 2.0 * torch.einsum(
        "nci,nc->ni", jacobian, footprint)
    gradient = (
        grad_numerator - value[:, None] * grad_denominator
    ) / denominator[:, None]

    eye = torch.eye(3, dtype=mu.dtype, device=mu.device)

    def curvature(weight):
        scaled = weight * quotient / d2
        outer = torch.einsum(
            "nc,nci,ncj->nij", scaled, displacement, displacement)
        trace = (weight * quotient).sum(dim=1)
        return 3.0 * outer - trace[:, None, None] * eye

    form_jacobian = torch.einsum("nij,njk->nik", form, jacobian)
    hess_numerator = 2.0 * torch.einsum(
        "nci,ncj->nij", jacobian, form_jacobian
    ) + 2.0 * curvature(product)
    hess_denominator = 2.0 * torch.einsum(
        "nci,ncj->nij", jacobian, jacobian
    ) + 2.0 * curvature(footprint)
    cross = gradient[:, :, None] * grad_denominator[:, None, :]
    hessian = (
        hess_numerator
        - value[:, None, None] * hess_denominator
        - cross
        - cross.transpose(1, 2)
    ) / denominator[:, None, None]
    return value, gradient, hessian


def score_gradient(form, offsets, mu, sigma, mask=None):
    """Return the score and analytic xyz gradient without forming a Hessian."""
    footprint, displacement, quotient = monopole_profile(
        offsets, mu, sigma, mask)
    jacobian = quotient[:, :, None] * displacement
    product = torch.einsum("nij,nj->ni", form, footprint)
    numerator = (footprint * product).sum(dim=1)
    denominator = (footprint * footprint).sum(dim=1)
    value = numerator / denominator
    grad_numerator = 2.0 * torch.einsum(
        "nci,nc->ni", jacobian, product)
    grad_denominator = 2.0 * torch.einsum(
        "nci,nc->ni", jacobian, footprint)
    gradient = (
        grad_numerator - value[:, None] * grad_denominator
    ) / denominator[:, None]
    return value, gradient


def _line_search(
    form,
    offsets,
    sigma,
    mask,
    mu,
    value,
    gradient,
    step,
    lower,
    upper,
    pending,
    backtracks,
):
    moved = torch.zeros_like(pending)
    scale = torch.ones_like(value)
    for _ in range(backtracks):
        live = pending & ~moved
        if not bool(live.any()):
            break
        candidate = torch.clamp(mu + scale[:, None] * step, lower, upper)
        new_value = score(form, offsets, candidate, sigma, mask)
        expected = (gradient * (candidate - mu)).sum(dim=1)
        accept = live & (new_value > value + 1e-4 * expected) & (
            new_value > value)
        mu = torch.where(accept[:, None], candidate, mu)
        value = torch.where(accept, new_value, value)
        moved |= accept
        scale = torch.where(live & ~accept, scale * 0.5, scale)
    return mu, value, moved


def refine_batch(
    form,
    offsets,
    mu_grid,
    sigma,
    lower,
    upper,
    mask=None,
    max_iterations=80,
    backtracks=30,
):
    """Maximize the continuous score by projected gradient ascent."""
    mu = mu_grid.clone()
    live = torch.arange(len(mu), device=mu.device)

    for _ in range(max_iterations):
        if live.numel() == 0:
            break
        sub_form = form[live]
        sub_offsets = offsets[live]
        sub_sigma = sigma[live]
        sub_mask = None if mask is None else mask[live]
        sub_mu = mu[live]
        sub_lower = lower[live]
        sub_upper = upper[live]
        span = (sub_upper - sub_lower).clamp_min(1e-12)
        edge = 1e-12 * span

        value, gradient = score_gradient(
            sub_form, sub_offsets, sub_mu, sub_sigma, sub_mask)
        frozen = (
            ((sub_mu <= sub_lower + edge) & (gradient < 0))
            | ((sub_mu >= sub_upper - edge) & (gradient > 0))
        )
        free_gradient = gradient.masked_fill(frozen, 0.0)

        direction = free_gradient * span
        reach = (direction / span).norm(
            dim=1, keepdim=True).clamp_min(1e-30)
        direction = direction / reach

        pending = torch.ones(len(live), dtype=torch.bool, device=mu.device)
        sub_mu, value, moved = _line_search(
            sub_form,
            sub_offsets,
            sub_sigma,
            sub_mask,
            sub_mu,
            value,
            gradient,
            direction,
            sub_lower,
            sub_upper,
            pending,
            backtracks,
        )

        mu[live] = sub_mu
        live = live[moved]

    value, gradient, hessian = score_gradient_hessian(
        form, offsets, mu, sigma, mask)
    return mu, value, gradient, hessian


def _symmetric_eigvalsh_3x3(matrix):
    """Closed-form eigenvalues of real symmetric 3x3 matrices."""
    diagonal = torch.diagonal(matrix, dim1=1, dim2=2)
    center = diagonal.mean(dim=1)
    a00 = matrix[:, 0, 0] - center
    a11 = matrix[:, 1, 1] - center
    a22 = matrix[:, 2, 2] - center
    a01 = matrix[:, 0, 1]
    a02 = matrix[:, 0, 2]
    a12 = matrix[:, 1, 2]
    spread2 = (
        a00.square() + a11.square() + a22.square()
        + 2.0 * (a01.square() + a02.square() + a12.square())
    ) / 6.0
    spread = torch.sqrt(spread2.clamp_min(0.0))
    repeated = spread2 == 0
    scale = torch.where(repeated, torch.ones_like(spread), spread)
    b00, b11, b22 = a00 / scale, a11 / scale, a22 / scale
    b01, b02, b12 = a01 / scale, a02 / scale, a12 / scale
    determinant = (
        b00 * (b11 * b22 - b12.square())
        - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02)
    )
    angle = torch.acos((0.5 * determinant).clamp(-1.0, 1.0)) / 3.0
    largest = center + 2.0 * spread * torch.cos(angle)
    smallest = center + 2.0 * spread * torch.cos(
        angle + 2.0 * torch.pi / 3.0)
    middle = 3.0 * center - largest - smallest
    values = torch.sort(
        torch.stack((smallest, middle, largest), dim=1), dim=1).values
    return torch.where(repeated[:, None], center[:, None], values)


def curvature_width(hessian, value, drop=0.01):
    """Return Hessian eigenvalues and the displacement costing `drop` energy."""
    eigenvalues = _symmetric_eigvalsh_3x3(hessian)
    magnitude = eigenvalues.abs().clamp_min(1e-30)
    width = torch.sqrt(
        2.0 * drop * value[:, None].clamp_min(0.0) / magnitude)
    return eigenvalues, width
