"""Synthetic tests for learned-basis and canonical/corrected visualization plumbing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from spiketensor import browser, dredge_real, drift, viz_centroid_basis, viz_embed
from spiketensor import viz_lattice
from spiketensor.fit_learned import LearnedBasis


class PortedFeatureTests(unittest.TestCase):
    def test_only_canonical_corrections_are_offered(self):
        """The package ships dredge_ap rigid/nonrigid; soft/hard must not silently work."""
        self.assertEqual(drift.WHICH, ("real-rigid", "real-nonrigid"))
        self.assertNotIn("soft", drift.SUFFIX)
        self.assertNotIn("hard", drift.SUFFIX)
        with tempfile.TemporaryDirectory() as tmp:
            figs = Path(tmp); (figs / "fit").mkdir()
            with self.assertRaises(ValueError):
                drift.correction(figs, "fit", "soft", np.array([2.0]), 1.0)

    def test_canonical_rigid_correction_interpolates(self):
        with tempfile.TemporaryDirectory() as tmp:
            figs = Path(tmp); row = figs / "fit"; row.mkdir()
            np.savez(row / "dredge_real.npz", t=np.arange(6.0),
                     p_rigid=np.arange(6.0) + 10.0,
                     t_nonrigid=np.arange(6.0),
                     p_nonrigid=np.tile((np.arange(6.0) + 10.0)[:, None], (1, 3)),
                     win_centers=np.array([0.0, 1000.0, 2000.0]))
            got = drift.correction(figs, "fit", "real-rigid", np.array([2.0, 4.0]), 1.0)
            np.testing.assert_allclose(got, [12.0, 14.0], atol=1e-6)

    def test_model_amplitude_is_rescaled_and_explicit_position_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            expected = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
            coeff = np.array([[3, 4], [0, 10]], np.float32)
            np.savez(
                runs / "pi_fit.npz",
                k=np.array([0, 1]),
                v=coeff,
                pos=expected,
                mu_site=np.zeros((2, 3), np.float32),
                S=np.int32(1),
            )
            rec = SimpleNamespace(n_spikes=2)
            pos, amp = dredge_real.load_positions(runs, "fit", rec)
            np.testing.assert_array_equal(pos, expected)
            self.assertAlmostEqual(float(np.median(amp)), 100.0, places=5)
            self.assertAlmostEqual(float(amp[1] / amp[0]), 2.0, places=5)
            (runs / "summary_fit.json").write_text("{}")

            np.savez(
                runs / "pi_zero.npz",
                k=np.array([0, 1]),
                v=np.zeros((2, 2), np.float32),
                pos=expected,
            )
            with self.assertRaisesRegex(ValueError, "amplitude median"):
                dredge_real.load_positions(runs, "zero", rec)

    def test_learned_footprints_are_differentiable(self):
        basis = LearnedBasis(
            np.array([[0, 0, 10], [2, -3, 20]], np.float32),
            np.array([5, 8], np.float32),
        )
        off = torch.tensor([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
        all_g = basis.ghat_all(off)
        self.assertEqual(tuple(all_g.shape), (2, 3))
        torch.testing.assert_close(all_g.norm(dim=1), torch.ones(2))
        selected = basis.ghat_sel(off[None].repeat(2, 1, 1), torch.tensor([0, 1]))
        selected.sum().backward()
        self.assertIsNotNone(basis.mu.grad)
        self.assertIsNotNone(basis.raw_sig.grad)

    def test_embedding_and_centroid_loaders_prefer_explicit_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            tag = "lrn2_monopole_M2_kmeans"
            torch.save({"Q": 2, "S": 1}, runs / f"codebook_{tag}.pt")
            pos = np.array([[10, 20, 30], [40, 50, 60]], np.float32)
            coeff = np.array([[1, 0], [0, -2]], np.float32)
            np.savez(
                runs / f"pi_{tag}.npz",
                k=np.array([0, 1]),
                v=coeff,
                pos=pos,
                mu_site=np.zeros((2, 3), np.float32),
                S=np.int32(1),
            )
            rec = SimpleNamespace(
                n_spikes=2,
                anchors=np.full((2, 3), 999.0),
                spike_channels=np.array([0, 1]),
            )
            _, loaded_coeff, embed_rel = viz_embed.load_fit(runs, tag, rec)
            np.testing.assert_array_equal(loaded_coeff, coeff)
            # load_fit returns ANCHOR-RELATIVE offsets: the caller reconstructs the
            # absolute position as anchor + offset, so an explicit `pos` array must come
            # back with the anchor removed rather than untouched
            np.testing.assert_allclose(embed_rel[:, :2] + rec.anchors[:, :2],
                                       pos[:, :2])
            np.testing.assert_allclose(embed_rel[:, 2], pos[:, 2])
            centroid_pos, dom, amp, _, _ = viz_centroid_basis.load_learned(
                runs, tag, rec
            )
            # when the fit stores an explicit `pos` the centroid loader hands back the
            # full absolute xyz, not just the lateral pair
            np.testing.assert_array_equal(centroid_pos[:, :2], pos[:, :2])
            np.testing.assert_array_equal(dom, [0, 1])
            np.testing.assert_allclose(amp, [1, 2])

    def test_lattice_loader_converts_explicit_position_to_anchor_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            tag = "lrn2_monopole_M2_kmeans"
            site_sigma = np.array([5, 9], np.float32)
            torch.save(
                {
                    "Q": 2,
                    "S": 1,
                    "K": 2,
                    "dataset": "synthetic",
                    "site_sigma": site_sigma,
                },
                runs / f"codebook_{tag}.pt",
            )
            pos = np.array([[11, 22, 30], [44, 55, 60]], np.float32)
            np.savez(
                runs / f"pi_{tag}.npz",
                k=np.array([0, 1]),
                v=np.ones((2, 2), np.float32),
                pos=pos,
                mu_site=np.zeros((2, 3), np.float32),
                site_sigma=site_sigma,
                prof_sigma=np.array([7], np.float32),
                S=np.int32(1),
            )
            rec = SimpleNamespace(
                anchors=np.array([[1, 2, 0], [4, 5, 0]], float),
                spike_channels=np.array([0, 1]),
            )
            with mock.patch.object(viz_lattice.D, "load", return_value=rec):
                _, _, _, mu, sigma, site, _, _ = viz_lattice.load(runs, tag)
            np.testing.assert_array_equal(mu[:, :2], [[10, 20], [40, 50]])
            np.testing.assert_array_equal(mu[:, 2], [30, 60])
            np.testing.assert_array_equal(sigma, site_sigma[site])

    def test_browser_collects_learned_and_canonical_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs, figs = root / "runs", root / "figures"
            runs.mkdir()
            tag = "lrn512_monopole_M8_kmeans"
            summary = {
                "args": {"kernel": "monopole", "M": 8},
                "K": 512,
                "S": 1,
                "KS": 512,
                "Q": 8,
                "learned_basis": True,
                "learn_mu": True,
                "mu_shift_med": 7.25,
                "full_nmse": 0.12,
                "sites_used": 500,
                "cands_used": 500,
            }
            (runs / f"summary_{tag}.json").write_text(json.dumps(summary))
            row_dir = figs / tag
            row_dir.mkdir(parents=True)
            np.savez(
                row_dir / "dredge_real.npz",
                gt_r_rigid=0.91,
                gt_gain_rigid=0.54,
                gt_r_nonrigid_best=0.89,
                gt_gain_nonrigid_best=0.52,
            )
            (row_dir / "aggregate_1s_drr.png").write_bytes(b"png")
            rows = browser.collect(runs, figs)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["model"], "basis-learned")
            self.assertEqual(row["n"], 512)
            self.assertEqual(row["Q"], 8)
            self.assertEqual(row["real_r"], 0.91)
            # the browser exposes rigid gain and the best nonrigid correlation;
            # there is no separate nonrigid gain column
            self.assertNotIn("real_gain_nr", row)
            self.assertEqual(row["real_r_nr"], 0.89)
            self.assertEqual(
                row["panels"]["aggregate_1s_drr"],
                f"{tag}/aggregate_1s_drr.png",
            )


if __name__ == "__main__":
    unittest.main()
