# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.warp.dirichlet, against its NumPy counterpart."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli import dirichlet
from rivoli.geometry import spherical_to_cartesian

wp = pytest.importorskip("warp")
warp_dirichlet = pytest.importorskip("rivoli.warp.dirichlet")

# The CPU is always tested, a GPU only when there is one
DEVICES = ["cpu", *(str(device) for device in wp.get_cuda_devices())]
RANGE = (380.0, 780.0)


@pytest.fixture
def rng():
    return np.random.default_rng(20261028)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize(
    ("isotropic", "spectral"), [(True, False), (True, True), (False, False), (False, True)]
)
@pytest.mark.parametrize("symmetry", ["none", "reciprocity"])
def test_dirichlet_matrix_matches_numpy(rng, device, metric, isotropic, spectral, symmetry):
    n = 40
    phi_h = np.zeros(n) if isotropic else rng.uniform(-np.pi, np.pi, n)
    centres = [
        spherical_to_cartesian(rng.uniform(0, np.pi / 2, n), phi_h),
        spherical_to_cartesian(rng.uniform(0, np.pi / 2, n), rng.uniform(-np.pi, np.pi, n)),
    ]
    arguments = {
        "widths": (0.3, 0.4),
        "metric": metric,
        "isotropic": isotropic,
        "scales": (1.5, 3.0),
        "symmetry": symmetry,
    }
    if spectral:
        centres.append(rng.uniform(*RANGE, n))
        arguments |= {"widths": (0.3, 0.4, 50.0), "scales": (1.5, 3.0, 400.0)}
        arguments["wavelength_range"] = RANGE
    expected = dirichlet.dirichlet_matrix(centres, **arguments)
    actual = warp_dirichlet.dirichlet_matrix(centres, device=device, **arguments)
    assert actual.shape == (n, n)
    assert_allclose(actual, expected, rtol=0, atol=1e-13 * np.abs(expected).max())


def test_chunks_are_accumulated(rng, monkeypatch):
    # Chunks of a few nodes, to check their accumulation
    monkeypatch.setattr(warp_dirichlet, "_PAIRS_PER_CHUNK", 100)
    n = 10
    centres = (
        spherical_to_cartesian(rng.uniform(0, np.pi / 2, n), 0.0),
        spherical_to_cartesian(rng.uniform(0, np.pi / 2, n), rng.uniform(-np.pi, np.pi, n)),
    )
    arguments = {"widths": (0.5, 0.5), "metric": "chordal", "isotropic": True}
    expected = dirichlet.dirichlet_matrix(centres, **arguments)
    actual = warp_dirichlet.dirichlet_matrix(centres, device="cpu", **arguments)
    assert_allclose(actual, expected, rtol=0, atol=1e-13 * np.abs(expected).max())


def test_invalid_arguments(rng):
    centres = (spherical_to_cartesian(rng.uniform(0, 1, 3), 0.0),) * 2
    with pytest.raises(ValueError, match="if and only if"):
        warp_dirichlet.dirichlet_matrix(
            centres,
            widths=(1.0, 1.0),
            metric="chordal",
            isotropic=True,
            wavelength_range=RANGE,
            device="cpu",
        )
