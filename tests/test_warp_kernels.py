# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

# The bodies of Warp kernels are in the kernel language of Warp, which Warp type
# checks when compiling them: its array annotations are calls
# pyright: reportInvalidTypeForm=false

"""Tests of rivoli.warp.kernels, against their NumPy counterparts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli import kernels
from rivoli.geometry import normalize

wp = pytest.importorskip("warp")
warp_kernels = pytest.importorskip("rivoli.warp.kernels")

# The CPU is always tested, a GPU only when there is one
DEVICES = ["cpu", *(str(device) for device in wp.get_cuda_devices())]


@pytest.fixture
def rng():
    return np.random.default_rng(20261030)


def random_sites(rng, n, spectral=False):
    sites = [normalize(rng.normal(size=(n, 3))), normalize(rng.normal(size=(n, 3)))]
    if spectral:
        sites.append(rng.uniform(380, 780, n))
    return sites


@wp.kernel
def _profile(r: wp.array(dtype=wp.float64), values: wp.array(dtype=wp.float64)):
    i = wp.tid()
    values[i] = warp_kernels.gaussian(r[i])


@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_matches_numpy(device):
    r = np.linspace(0, 6, 100)
    values = wp.empty(100, dtype=wp.float64, device=device)
    wp.launch(
        _profile,
        dim=100,
        inputs=[wp.array(r, dtype=wp.float64, device=device)],
        outputs=[values],
        device=device,
    )
    assert_allclose(values.numpy(), kernels.gaussian(r), rtol=1e-14)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize("spectral", [False, True])
def test_collocation_matrix_matches_numpy(rng, device, metric, spectral):
    sites, centres = random_sites(rng, 70, spectral), random_sites(rng, 50, spectral)
    widths = (0.4, 0.6, 100.0) if spectral else (0.4, 0.6)
    expected = kernels.collocation_matrix(sites, centres, widths=widths, metric=metric)
    actual = warp_kernels.collocation_matrix(
        sites, centres, widths=widths, metric=metric, device=device
    )
    assert actual.shape == (70, 50)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-300)


def test_mismatched_factors(rng):
    with pytest.raises(ValueError, match="same factors, got 2 and 3"):
        warp_kernels.collocation_matrix(
            random_sites(rng, 3),
            random_sites(rng, 3, spectral=True),
            widths=(1.0, 1.0),
            metric="chordal",
            device="cpu",
        )


def test_unknown_kernel(rng):
    centres = random_sites(rng, 3)
    with pytest.raises(ValueError, match="got 'cubic'"):
        warp_kernels.collocation_matrix(
            centres, centres, widths=(1.0, 1.0), metric="chordal", kernel="cubic", device="cpu"
        )
