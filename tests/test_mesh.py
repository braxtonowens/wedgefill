import numpy as np
import torch

from wedgefill.mesh import area_volume, icosphere, voxelize, willmore_terms


def test_sphere_willmore_is_4pi_for_any_radius():
    v, f = icosphere(4)
    for r in (3.0, 25.0):
        W, _, _ = willmore_terms(torch.tensor(r * v), torch.tensor(f))
        assert abs(float(W) / (4 * np.pi) - 1) < 0.01


def test_area_volume_of_sphere():
    v, f = icosphere(4)
    a, vol = area_volume(20 * v, f)
    assert abs(a / (4 * np.pi * 400) - 1) < 0.01
    assert abs(vol / (4 / 3 * np.pi * 8000) - 1) < 0.01


def test_voxelize_matches_volume():
    v, f = icosphere(4)
    m = voxelize(20 * v + 31.5, f, (64, 64, 64))
    assert abs(m.sum() / (4 / 3 * np.pi * 8000) - 1) < 0.03
