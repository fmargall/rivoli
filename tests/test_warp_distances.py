# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

# The bodies of Warp functions and kernels are in the kernel language of Warp,
# which Warp type checks when compiling them: its stubs give no operators to
# its vector types, and its array annotations are calls
# pyright: reportOperatorIssue=false, reportInvalidTypeForm=false

"""Tests of rivoli.warp.distances, against their NumPy counterparts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli import distances
from rivoli.geometry import normalize, spherical_to_cartesian
from rivoli.sampling import super_fibonacci_anisotropic, super_fibonacci_isotropic

wp = pytest.importorskip("warp")
warp_distances = pytest.importorskip("rivoli.warp.distances")

# The CPU is always tested, a GPU only when there is one
DEVICES = ["cpu", *(str(device) for device in wp.get_cuda_devices())]


@pytest.fixture
def rng():
    return np.random.default_rng(20261029)


def random_directions(rng, n):
    return normalize(rng.normal(size=(n, 3)))


@wp.kernel
def _pairwise(
    u: wp.array(dtype=wp.vec3d),
    v: wp.array(dtype=wp.vec3d),
    geodesic: wp.array(dtype=wp.float64),
    chordal: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    geodesic[i] = warp_distances.geodesic(u[i], v[i])
    chordal[i] = warp_distances.chordal(u[i], v[i])


@pytest.mark.parametrize("device", DEVICES)
def test_warp_functions_match_numpy(rng, device):
    u, v = random_directions(rng, 1000), random_directions(rng, 1000)
    # Close and opposite vectors, the hardest cases
    v[0] = [np.cos(1e-10), np.sin(1e-10), 0.0]
    u[0] = [1.0, 0.0, 0.0]
    v[1] = -u[1]
    geodesic = wp.empty(1000, dtype=wp.float64, device=device)
    chordal = wp.empty(1000, dtype=wp.float64, device=device)
    wp.launch(
        _pairwise,
        dim=1000,
        inputs=[
            wp.array(u, dtype=wp.vec3d, device=device),
            wp.array(v, dtype=wp.vec3d, device=device),
        ],
        outputs=[geodesic, chordal],
        device=device,
    )
    assert_allclose(geodesic.numpy(), distances.geodesic_distance(u, v), rtol=1e-14, atol=1e-15)
    assert_allclose(chordal.numpy(), distances.chordal_distance(u, v), rtol=1e-14, atol=1e-15)


def isotropic_sites(n):
    theta_h, theta_d, phi_d = super_fibonacci_isotropic(n)
    return spherical_to_cartesian(theta_h, 0.0), spherical_to_cartesian(theta_d, phi_d)


def anisotropic_sites(n):
    theta_h, phi_h, theta_d, phi_d = super_fibonacci_anisotropic(n)
    return spherical_to_cartesian(theta_h, phi_h), spherical_to_cartesian(theta_d, phi_d)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize("spectral", [False, True])
@pytest.mark.parametrize("sites", [isotropic_sites, anisotropic_sites])
@pytest.mark.parametrize("scaled", [False, True])
def test_spacings_match_numpy(device, metric, spectral, sites, scaled):
    h, d = sites(500)
    wavelength = np.linspace(380, 780, 500) if spectral else None
    scales = None
    if scaled:
        scales = distances.diameters(
            isotropic=sites is isotropic_sites,
            metric=metric,
            wavelength_range=(380, 780) if spectral else None,
        )
    expected = distances.nearest_neighbour_spacings(h, d, wavelength, metric=metric, scales=scales)
    actual = warp_distances.nearest_neighbour_spacings(
        h, d, wavelength, metric=metric, scales=scales, device=device
    )
    assert len(actual) == len(expected)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


def test_at_least_two_sites():
    with pytest.raises(ValueError, match="at least 2, got 1"):
        warp_distances.nearest_neighbour_spacings([[0, 0, 1]], [[0, 0, 1]], device="cpu")


def test_unknown_metric():
    with pytest.raises(ValueError, match="'geodesic' or 'chordal', got 'euclid'"):
        warp_distances.nearest_neighbour_spacings(
            [[0, 0, 1], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0]],
            metric="euclid",
            device="cpu",
        )


def test_invalid_scales():
    with pytest.raises(ValueError, match="expected 3 scales, one per factor, got 2"):
        warp_distances.nearest_neighbour_spacings(
            [[0, 0, 1], [1, 0, 0]], [[0, 0, 1], [1, 0, 0]], [0, 1], scales=(1, 1), device="cpu"
        )
