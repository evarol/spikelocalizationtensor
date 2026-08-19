"""Lightweight CPU tests for masked continuous monopole refinement."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from continuous_refine import (
    _batched_eigh,
    curvature_width,
    monopole_profile,
    refine_batch,
    score,
    score_gradient_hessian,
    voxel_cell_bounds,
)


DTYPE = torch.float64


def geometry(n):
    base = torch.tensor(
        [
            [-48, -30],
            [-16, -10],
            [16, -30],
            [48, -10],
            [-48, 10],
            [-16, 30],
            [16, 10],
            [48, 30],
        ],
        dtype=DTYPE,
    )
    return base.unsqueeze(0).expand(n, -1, -1).contiguous()


def random_forms(n, channels=8, rank=4):
    generator = torch.Generator().manual_seed(2)
    matrix = torch.randn(
        n, channels, rank, dtype=DTYPE, generator=generator)
    return matrix @ matrix.transpose(1, 2)


class TestVoxelBounds(unittest.TestCase):
    def test_interior_and_edge_cells(self):
        source = np.array([[-150, 0, 0], [12, -8, 90], [150, 150, 300]])
        lower, upper = voxel_cell_bounds(
            source, ((-150, -150, 0), (150, 150, 300)), 1)
        np.testing.assert_allclose(lower[1], source[1] - 0.5)
        np.testing.assert_allclose(upper[1], source[1] + 0.5)
        np.testing.assert_allclose(lower[0], (-150, -0.5, 0))
        np.testing.assert_allclose(upper[2], (150, 150, 300))


class TestMaskedObjective(unittest.TestCase):
    def setUp(self):
        self.n = 12
        self.offsets = geometry(self.n)
        self.mu = torch.tensor(
            np.random.default_rng(3).uniform(
                [-20, -20, 10], [20, 20, 100], (self.n, 3)),
            dtype=DTYPE,
        )
        self.sigma = torch.tensor(
            np.random.default_rng(4).choice([4, 16, 64], self.n),
            dtype=DTYPE,
        )
        self.mask = torch.ones(self.n, 8, dtype=DTYPE)
        self.mask[::2, -2:] = 0
        self.form = random_forms(self.n)

    def test_profile_matches_monopole_definition(self):
        footprint, _, _ = monopole_profile(
            self.offsets, self.mu, self.sigma, self.mask)
        dxy2 = (
            (self.offsets[:, :, 0] - self.mu[:, None, 0]).square()
            + (self.offsets[:, :, 1] - self.mu[:, None, 1]).square()
        )
        expected = self.sigma[:, None] / torch.sqrt(
            dxy2 + self.mu[:, None, 2].square() + self.sigma[:, None].square())
        torch.testing.assert_close(footprint, expected * self.mask)

    def test_masked_form_entries_do_not_change_score(self):
        changed = self.form.clone()
        changed[::2, -2:, :] = 1e9
        changed[::2, :, -2:] = 1e9
        direct = score(
            self.form, self.offsets, self.mu, self.sigma, self.mask)
        modified = score(
            changed, self.offsets, self.mu, self.sigma, self.mask)
        torch.testing.assert_close(direct, modified)

    def test_score_matches_eliminated_gain(self):
        projected = torch.randn(self.n, 8, dtype=DTYPE)
        omega_energy = torch.linspace(0.7, 1.4, self.n, dtype=DTYPE)
        form = (
            projected[:, :, None] * projected[:, None, :]
            / omega_energy[:, None, None]
        )
        footprint, _, _ = monopole_profile(
            self.offsets, self.mu, self.sigma, self.mask)
        normalized = footprint / footprint.norm(dim=1, keepdim=True)
        expected = (normalized * projected).sum(dim=1).square() / omega_energy
        torch.testing.assert_close(
            score(form, self.offsets, self.mu, self.sigma, self.mask),
            expected,
        )

    def test_analytic_derivatives_match_autograd(self):
        _, gradient, hessian = score_gradient_hessian(
            self.form, self.offsets, self.mu, self.sigma, self.mask)
        mu = self.mu.clone().requires_grad_(True)
        exact_gradient, = torch.autograd.grad(
            score(
                self.form, self.offsets, mu, self.sigma, self.mask
            ).sum(),
            mu,
            create_graph=True,
        )
        torch.testing.assert_close(
            gradient, exact_gradient, rtol=1e-11, atol=1e-13)
        for dimension in range(3):
            row, = torch.autograd.grad(
                exact_gradient[:, dimension].sum(), mu, retain_graph=True)
            torch.testing.assert_close(
                hessian[:, dimension], row, rtol=1e-10, atol=1e-12)

    def test_eigh_sub_batches_match_direct_decomposition(self):
        matrices = self.form[:, :3, :3]
        expected_values, expected_vectors = torch.linalg.eigh(matrices)
        values, vectors = _batched_eigh(matrices, batch_size=5)
        torch.testing.assert_close(values, expected_values)
        reconstructed = vectors @ torch.diag_embed(values) @ vectors.transpose(1, 2)
        expected = (
            expected_vectors
            @ torch.diag_embed(expected_values)
            @ expected_vectors.transpose(1, 2)
        )
        torch.testing.assert_close(reconstructed, expected)


class TestRefinement(unittest.TestCase):
    def test_refinement_is_monotone_and_box_respecting(self):
        n = 32
        rng = np.random.default_rng(5)
        mu_grid = torch.tensor(
            rng.integers([-30, -30, 5], [31, 31, 120], (n, 3)),
            dtype=DTYPE,
        )
        lower = mu_grid - 0.5
        upper = mu_grid + 0.5
        offsets = geometry(n)
        sigma = torch.full((n,), 24.0, dtype=DTYPE)
        mask = torch.ones(n, 8, dtype=DTYPE)
        mask[::3, -2:] = 0
        form = random_forms(n)
        before = score(form, offsets, mu_grid, sigma, mask)
        mu, after, _, hessian = refine_batch(
            form, offsets, mu_grid, sigma, lower, upper, mask)
        self.assertGreaterEqual(float((after - before).min()), 0)
        self.assertTrue(bool((mu >= lower).all()))
        self.assertTrue(bool((mu <= upper).all()))
        values, width = curvature_width(hessian, after)
        self.assertTrue(bool(torch.isfinite(values).all()))
        self.assertTrue(bool(torch.isfinite(width).all()))

    def test_recovers_off_grid_monopole_source(self):
        n = 24
        rng = np.random.default_rng(6)
        mu_grid = torch.tensor(
            np.column_stack(
                (
                    rng.integers(-30, 31, n),
                    rng.integers(-30, 31, n),
                    rng.integers(20, 100, n),
                )
            ),
            dtype=DTYPE,
        )
        truth = mu_grid + torch.tensor(
            rng.uniform(-0.4, 0.4, (n, 3)), dtype=DTYPE)
        lower = mu_grid - 0.5
        upper = mu_grid + 0.5
        offsets = geometry(n)
        sigma = torch.full((n,), 24.0, dtype=DTYPE)
        mask = torch.ones(n, 8, dtype=DTYPE)
        footprint, _, _ = monopole_profile(offsets, truth, sigma, mask)
        normalized = footprint / footprint.norm(dim=1, keepdim=True)
        form = normalized[:, :, None] * normalized[:, None, :]
        refined, value, _, _ = refine_batch(
            form, offsets, mu_grid, sigma, lower, upper, mask)
        grid_error = (mu_grid - truth).norm(dim=1)
        refined_error = (refined - truth).norm(dim=1)
        self.assertLess(
            float(refined_error.median()), 0.02 * float(grid_error.median()))
        self.assertGreater(float(value.min()), 0.999999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
