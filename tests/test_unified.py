"""The unified model must REPRODUCE the dedicated modules it replaces, not merely run."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiketensor.unified import Config, shift_bank, project_cone, fix_polarity


def test_config_rejects_orthonormal_prior():
    """M unit vectors cannot lie in cones about P<<M prototypes AND be orthogonal."""
    with pytest.raises(ValueError, match="incompatible"):
        Config(shape="onehot", P=2, orthonormal=True).validate()


def test_config_auto_relaxes_meaningless_flags():
    c = Config(shape="free", max_shift=10, nonneg=None)
    assert c.max_shift == 0 and c.nonneg is False      # redundant, resolved silently
    assert Config(shape="onehot").nonneg is True        # sensible default kept


def test_config_rejects_contradictions_but_not_defaults():
    with pytest.raises(ValueError):
        Config(shape="free", nonneg=True)
    Config(shape="free", P=0).validate()                # fine


def test_shift_bank_is_unit_norm_and_labelled():
    om = torch.linalg.qr(torch.randn(40, 6))[0].T
    bank, q, tau = shift_bank(om, 3)
    assert bank.shape == (6 * 7, 40)
    assert torch.allclose(bank.norm(dim=1), torch.ones(len(bank)), atol=1e-5)
    # the zero-lag row of each atom is the atom itself
    for qi in range(6):
        row = int(np.flatnonzero((q.numpy() == qi) & (tau.numpy() == 0))[0])
        assert torch.allclose(bank[row], om[qi], atol=1e-5)


def test_cone_projection_respects_the_cap():
    torch.manual_seed(0)
    phi = torch.randn(90); phi /= phi.norm()
    cos_max = float(np.cos(np.radians(35.0)))
    for _ in range(8):
        psi = project_cone(torch.randn(90), phi, cos_max)
        ang = np.degrees(np.arccos(float(torch.clamp(psi @ phi, -1, 1))))
        assert abs(float(psi.norm()) - 1) < 1e-5
        assert ang <= 35.0 + 1e-3


def test_prototype_polarity_convention():
    p = fix_polarity(torch.tensor([[-1.0, 0.2], [0.9, -0.1]]))
    assert p[0][p[0].abs().argmax()] > 0        # prototype 0 depolarizing
    assert p[1][p[1].abs().argmax()] < 0        # prototype 1 hyperpolarizing


@pytest.mark.slow
def test_unified_configs_are_all_constructible():
    """Every configuration named in the README must validate."""
    for kw in (dict(R=1, shape="free", orthonormal=True),
               dict(R=4, shape="free", orthonormal=True),
               dict(R=4, shape="onehot", max_shift=0, P=0, orthonormal=True),
               dict(R=4, shape="onehot", max_shift=10, P=0, orthonormal=True),
               dict(R=8, shape="onehot", max_shift=10, P=2, cone_deg=35)):
        Config(**kw).validate()
