# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.kernels, the radial basis function kernels."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli.distances import nearest_neighbour_spacings, pairwise_distances
from rivoli.geometry import normalize, spherical_to_cartesian
from rivoli.kernels import collocation_matrix, gaussian
from rivoli.sampling import super_fibonacci_isotropic


@pytest.fixture
def rng():
    return np.random.default_rng(20261030)


def random_sites(rng, n, spectral=False):
    sites = [normalize(rng.normal(size=(n, 3))), normalize(rng.normal(size=(n, 3)))]
    if spectral:
        sites.append(rng.uniform(380, 780, n))
    return sites


def test_gaussian_profile():
    assert_allclose(gaussian([0, 1, 2]), [1, np.exp(-1), np.exp(-4)])
    r = np.linspace(0, 5, 100)
    assert np.all(np.diff(gaussian(r)) < 0)


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize("spectral", [False, True])
def test_collocation_matrix_is_the_profile_of_the_distances(rng, metric, spectral):
    sites, centres = random_sites(rng, 30, spectral), random_sites(rng, 20, spectral)
    widths = (0.4, 0.6, 100.0) if spectral else (0.4, 0.6)
    matrix = collocation_matrix(sites, centres, widths=widths, metric=metric, symmetry="none")
    assert matrix.shape == (30, 20)
    r = pairwise_distances(sites, centres, metric=metric, scales=widths)
    assert_allclose(matrix, np.exp(-(r**2)))


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
def test_interpolation_matrix_is_symmetric_with_a_unit_diagonal(rng, metric):
    centres = random_sites(rng, 50)
    matrix = collocation_matrix(centres, centres, widths=(0.5, 0.5), metric=metric, symmetry="none")
    assert_allclose(np.diag(matrix), 1)
    assert_allclose(matrix, matrix.T)
    assert np.all((matrix > 0) & (matrix <= 1))


def test_wider_kernels_decay_slower(rng):
    sites, centres = random_sites(rng, 20), random_sites(rng, 20)
    narrow = collocation_matrix(sites, centres, widths=(0.3, 0.3), metric="chordal")
    wide = collocation_matrix(sites, centres, widths=(0.6, 0.6), metric="chordal")
    assert np.all(wide >= narrow)


@pytest.mark.parametrize("width", [1.0, 2.0, 4.0])
def test_only_the_chordal_gaussian_is_positive_definite_on_the_equator(width):
    # Centres all around the equator of the hemisphere of d, where the
    # geodesic distance reaches pi
    phi = np.linspace(-np.pi, np.pi, 50, endpoint=False)
    d = spherical_to_cartesian(np.pi / 2, phi)
    centres = (np.broadcast_to([0.0, 0.0, 1.0], d.shape), d)

    def smallest_eigenvalue(metric):
        matrix = collocation_matrix(centres, centres, widths=(1.0, width), metric=metric)
        return np.linalg.eigvalsh(matrix)[0]

    assert smallest_eigenvalue("chordal") > -1e-12
    assert smallest_eigenvalue("geodesic") < -1e-5


def test_chordal_gaussian_on_a_super_fibonacci_sampling_is_positive_definite():
    theta_h, theta_d, phi_d = super_fibonacci_isotropic(400)
    centres = (spherical_to_cartesian(theta_h, 0.0), spherical_to_cartesian(theta_d, phi_d))
    spacings = nearest_neighbour_spacings(*centres, metric="chordal")
    widths = tuple(2 * float(np.mean(spacing)) for spacing in spacings)
    matrix = collocation_matrix(centres, centres, widths=widths, metric="chordal")
    assert np.linalg.eigvalsh(matrix)[0] > 1e-3
    # Interpolation reproduces the values at the centres
    values = np.cos(theta_h) * np.sin(theta_d)
    weights = np.linalg.solve(matrix, values)
    assert_allclose(matrix @ weights, values, atol=1e-10)


def test_unknown_kernel(rng):
    centres = random_sites(rng, 3)
    with pytest.raises(ValueError, match="kernel must be one of \\['gaussian'\\], got 'cubic'"):
        collocation_matrix(
            centres,
            centres,
            widths=(1.0, 1.0),
            metric="chordal",
            kernel="cubic",  # type: ignore[arg-type]
        )


def test_invalid_widths(rng):
    centres = random_sites(rng, 3)
    with pytest.raises(ValueError, match="must be positive"):
        collocation_matrix(centres, centres, widths=(1.0, 0.0), metric="chordal")


HALF_TURN = np.array([-1.0, -1.0, 1.0])


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize("spectral", [False, True])
def test_reciprocal_kernels_sum_over_the_orbit(rng, metric, spectral):
    sites, centres = random_sites(rng, 30, spectral), random_sites(rng, 20, spectral)
    widths = (0.4, 0.6, 100.0) if spectral else (0.4, 0.6)
    images = [centres[0], centres[1] * HALF_TURN, *centres[2:]]
    plain = {"widths": widths, "metric": metric, "symmetry": "none"}
    expected = collocation_matrix(sites, centres, **plain) + collocation_matrix(
        sites, images, **plain
    )
    matrix = collocation_matrix(sites, centres, widths=widths, metric=metric)
    assert_allclose(matrix, expected, rtol=1e-15)
    # Swapping the directions of the sites, d -> g d, leaves the kernels unchanged
    swapped = [sites[0], sites[1] * HALF_TURN, *sites[2:]]
    assert_allclose(collocation_matrix(swapped, centres, widths=widths, metric=metric), matrix)


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
def test_reciprocal_interpolation_matrix_is_symmetric_positive_definite(rng, metric):
    centres = random_sites(rng, 60)
    matrix = collocation_matrix(centres, centres, widths=(0.5, 0.5), metric=metric)
    assert_allclose(matrix, matrix.T, rtol=1e-15)
    if metric == "chordal":
        assert np.linalg.eigvalsh(matrix)[0] > 0


def test_invalid_symmetry(rng):
    with pytest.raises(ValueError, match="symmetry must be one of"):
        collocation_matrix(
            random_sites(rng, 3),
            random_sites(rng, 3),
            widths=(1.0, 1.0),
            metric="chordal",
            symmetry="bilateral",  # pyright: ignore[reportArgumentType]
        )
