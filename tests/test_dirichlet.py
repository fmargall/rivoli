# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.dirichlet, the Dirichlet matrix of kernels."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli.dirichlet import (
    _blocks,
    _hemisphere_rule,
    _interval_rule,
    _meridian_rule,
    dirichlet_matrix,
)
from rivoli.distances import diameters, nearest_neighbour_spacings
from rivoli.geometry import rotate, spherical_to_cartesian
from rivoli.kernels import collocation_matrix
from rivoli.sampling import super_fibonacci_anisotropic, super_fibonacci_isotropic

RANGE = (380.0, 780.0)


@pytest.fixture
def rng():
    return np.random.default_rng(20261028)


def isotropic_centres(n, spectral=False):
    if not spectral:
        theta_h, theta_d, phi_d = super_fibonacci_isotropic(n)
        return spherical_to_cartesian(theta_h, 0.0), spherical_to_cartesian(theta_d, phi_d)
    theta_h, theta_d, phi_d, wavelength = super_fibonacci_isotropic(n, wavelength_range=RANGE)
    return (
        spherical_to_cartesian(theta_h, 0.0),
        spherical_to_cartesian(theta_d, phi_d),
        wavelength,
    )


def anisotropic_centres(n):
    theta_h, phi_h, theta_d, phi_d = super_fibonacci_anisotropic(n)
    return spherical_to_cartesian(theta_h, phi_h), spherical_to_cartesian(theta_d, phi_d)


def typical_widths(centres, metric, isotropic):
    # Twice the mean spacing of the centres, compared relative to the diameters
    scales = diameters(
        isotropic=isotropic, metric=metric, wavelength_range=RANGE if len(centres) == 3 else None
    )
    spacings = nearest_neighbour_spacings(*centres, metric=metric, scales=scales)
    return tuple(2 * float(np.mean(s)) * scale for s, scale in zip(spacings, scales, strict=True))


def gauss_legendre(low, high, count):
    x, w = np.polynomial.legendre.leggauss(count)
    return 0.5 * (low + high) + 0.5 * (high - low) * x, 0.5 * (high - low) * w


def gaussian_blocks(centres, width, low, high):
    """Closed forms of the integrals of products of 1D Gaussians and of their derivatives."""
    a = 1 / width**2
    distance = centres[:, np.newaxis] - centres
    middle = 0.5 * (centres[:, np.newaxis] + centres)
    # exp(-a (t - c_i)^2) exp(-a (t - c_j)^2) = exp(-a distance^2 / 2) exp(-2 a s^2),
    # with s = t - middle
    erf = np.vectorize(math.erf)
    s_low, s_high = low - middle, high - middle
    root = np.sqrt(2 * a)
    erfs = erf(root * s_high) - erf(root * s_low)
    zeroth = np.sqrt(np.pi / (8 * a)) * erfs
    second = np.sqrt(np.pi / (128 * a**3)) * erfs - (
        s_high * np.exp(-2 * a * s_high**2) - s_low * np.exp(-2 * a * s_low**2)
    ) / (4 * a)
    factor = np.exp(-0.5 * a * distance**2)
    mass = factor * zeroth
    stiffness = 4 * a**2 * factor * (second - 0.25 * distance**2 * zeroth)
    return mass / (high - low), stiffness / (high - low)


@pytest.mark.parametrize("width", [5.0, 40.0, 300.0])
def test_wavelength_blocks_match_their_closed_form(rng, width):
    centres = rng.uniform(*RANGE, 12)
    mass, stiffness = _blocks(_interval_rule(*RANGE, width, 1), [centres], width, "chordal")
    expected_mass, expected_stiffness = gaussian_blocks(centres, width, *RANGE)
    assert_allclose(mass, expected_mass, rtol=1e-12, atol=1e-15 * expected_mass.max())
    assert_allclose(
        stiffness, expected_stiffness, rtol=1e-11, atol=1e-14 * expected_stiffness.max()
    )


@pytest.mark.parametrize("width", [0.02, 0.3, 2.0])
def test_geodesic_meridian_blocks_match_their_closed_form(rng, width):
    # On the meridian, the geodesic distance is the difference of polar angles
    theta = rng.uniform(0, np.pi / 2, 12)
    centres = spherical_to_cartesian(theta, 0.0)
    mass, stiffness = _blocks(_meridian_rule(width, 1), [centres], width, "geodesic")
    expected_mass, expected_stiffness = gaussian_blocks(theta, width, 0, np.pi / 2)
    assert_allclose(mass, expected_mass, rtol=1e-12, atol=1e-15 * expected_mass.max())
    assert_allclose(
        stiffness, expected_stiffness, rtol=1e-11, atol=1e-14 * expected_stiffness.max()
    )


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize("width", [0.05, 0.4, 1.0])
def test_hemisphere_blocks_of_the_normal(metric, width):
    # The Gaussian of the normal only depends on the polar angle, whose
    # derivative is its gradient
    normal = np.array([[0.0, 0.0, 1.0]])
    mass, stiffness = _blocks(_hemisphere_rule(width, 1), [normal], width, metric)
    theta, weights = gauss_legendre(0, np.pi / 2, 400)
    if metric == "geodesic":
        distance, derivative = theta, 1.0
    else:
        distance, derivative = 2 * np.sin(theta / 2), np.cos(theta / 2)
    values = np.exp(-((distance / width) ** 2))
    gradient = -2 * distance / width**2 * derivative * values
    # The mean over the hemisphere, of area 2 pi, with the area element
    # 2 pi sin(theta) dtheta of an axisymmetric function
    assert_allclose(mass[0, 0], np.sum(weights * np.sin(theta) * values**2), rtol=1e-13)
    assert_allclose(stiffness[0, 0], np.sum(weights * np.sin(theta) * gradient**2), rtol=1e-12)


def test_rules_are_normalized():
    for rule in (_meridian_rule(0.3, 1), _hemisphere_rule(0.3, 2), _interval_rule(*RANGE, 20, 1)):
        assert_allclose(np.sum(rule.weights), 1, rtol=1e-14)
    for rule in (_meridian_rule(0.3, 1), _hemisphere_rule(0.3, 1)):
        assert rule.tangents is not None
        # The tangents are an orthonormal basis of the tangent plane
        products = np.einsum("qkj,qlj->qkl", rule.tangents, rule.tangents)
        assert_allclose(
            products, np.broadcast_to(np.eye(products.shape[1]), products.shape), atol=1e-15
        )
        assert_allclose(np.einsum("qkj,qj->qk", rule.tangents, rule.nodes), 0, atol=1e-15)


def energy_by_finite_differences(centres, weights, widths, metric, scales, isotropic, symmetry):
    """Mean squared gradient of sum_j w_j phi_j, by finite differences of the angles."""
    polar = gauss_legendre(0, np.pi / 2, 24)
    azimuth = gauss_legendre(-np.pi, np.pi, 48)
    wavelength = gauss_legendre(*RANGE, 16)
    # Coordinates, their quadrature, and the area elements of the hemispheres
    grids = [polar] if isotropic else [polar, azimuth]
    grids += [polar, azimuth]
    if len(centres) == 3:
        grids.append(wavelength)
    coordinates = [x.ravel() for x in np.meshgrid(*(x for x, _ in grids), indexing="ij")]
    quadrature = np.ones(())
    for _, w in grids:
        quadrature = np.multiply.outer(quadrature, w)
    quadrature = quadrature.ravel()

    def angles(c):
        h = spherical_to_cartesian(c[0], 0.0) if isotropic else spherical_to_cartesian(c[0], c[1])
        k = 1 if isotropic else 2
        sites = [h, spherical_to_cartesian(c[k], c[k + 1]), *c[k + 2 :]]
        matrix = collocation_matrix(sites, centres, widths=widths, metric=metric, symmetry=symmetry)
        return matrix @ weights

    # Each coordinate, its factor, and the polar angle that divides its
    # derivative, for the azimuths
    k = 1 if isotropic else 2
    layout = [(0, 0, None)] if isotropic else [(0, 0, None), (1, 0, 0)]
    layout += [(k, 1, None), (k + 1, 1, k)]
    if len(centres) == 3:
        layout.append((k + 2, 2, None))
    area = np.ones_like(quadrature)
    if not isotropic:
        area *= np.sin(coordinates[0])
    area *= np.sin(coordinates[k])
    energy = 0.0
    for index, factor, polar_index in layout:
        step = 1e-5 * (100 if factor == 2 else 1)
        plus = [x.copy() for x in coordinates]
        minus = [x.copy() for x in coordinates]
        plus[index] += step
        minus[index] -= step
        derivative = (angles(plus) - angles(minus)) / (2 * step)
        if polar_index is not None:
            derivative /= np.sin(coordinates[polar_index])
        energy += scales[factor] ** 2 * np.sum(quadrature * area * derivative**2)
    volume = (np.pi / 2 if isotropic else 2 * np.pi) * 2 * np.pi
    if len(centres) == 3:
        volume *= RANGE[1] - RANGE[0]
    return energy / volume


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize(("isotropic", "spectral"), [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("symmetry", ["none", "reciprocity"])
def test_energy_matches_finite_differences(rng, metric, isotropic, spectral, symmetry):
    n = 4
    theta_h = rng.uniform(0, np.pi / 2, n)
    phi_h = np.zeros(n) if isotropic else rng.uniform(-np.pi, np.pi, n)
    centres = [
        spherical_to_cartesian(theta_h, phi_h),
        spherical_to_cartesian(rng.uniform(0, np.pi / 2, n), rng.uniform(-np.pi, np.pi, n)),
    ]
    widths, scales = (0.9, 1.1), (1.3, 0.7)
    wavelength_range = None
    if spectral:
        centres.append(rng.uniform(*RANGE, n))
        widths, scales, wavelength_range = (*widths, 150.0), (*scales, 200.0), RANGE
    weights = rng.normal(size=n)
    matrix = dirichlet_matrix(
        centres,
        widths=widths,
        metric=metric,
        isotropic=isotropic,
        wavelength_range=wavelength_range,
        scales=scales,
        symmetry=symmetry,
    )
    expected = energy_by_finite_differences(
        centres, weights, widths, metric, scales, isotropic, symmetry
    )
    assert_allclose(weights @ matrix @ weights, expected, rtol=1e-6)


# The geodesic distance is not smooth at the antipodes of the centres near the
# equator, which limits the accuracy of the quadrature for widths of nearly 1
@pytest.mark.parametrize(("metric", "tolerance"), [("geodesic", 1e-9), ("chordal", 1e-12)])
def test_quadrature_has_converged(metric, tolerance):
    for centres, isotropic in (
        (isotropic_centres(300, spectral=True), True),
        (anisotropic_centres(1000), False),
    ):
        spectral = len(centres) == 3
        widths = typical_widths(centres, metric, isotropic)
        arguments = {
            "widths": widths,
            "metric": metric,
            "isotropic": isotropic,
            "wavelength_range": RANGE if spectral else None,
        }
        matrix = dirichlet_matrix(centres, **arguments)
        refined = dirichlet_matrix(centres, refinement=2, **arguments)
        assert_allclose(refined, matrix, rtol=0, atol=tolerance * np.abs(matrix).max())


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
def test_matrix_is_symmetric_positive_semidefinite(metric):
    centres = isotropic_centres(200)
    matrix = dirichlet_matrix(
        centres, widths=typical_widths(centres, metric, True), metric=metric, isotropic=True
    )
    assert_allclose(matrix, matrix.T, rtol=0, atol=0)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues[0] > -1e-12 * eigenvalues[-1]


def test_invariant_under_rotations_about_the_normal():
    # The domain is invariant under rotations about the normal, of the half
    # and difference vectors independently
    h, d = anisotropic_centres(80)
    arguments = {"widths": (0.4, 0.5), "metric": "chordal", "isotropic": False}
    matrix = dirichlet_matrix((h, d), **arguments)
    normal = [0.0, 0.0, 1.0]
    rotated = dirichlet_matrix((rotate(h, normal, 0.123), rotate(d, normal, -2.0)), **arguments)
    assert_allclose(rotated, matrix, rtol=0, atol=1e-13 * np.abs(matrix).max())


def test_scales_weigh_the_energy_of_each_factor():
    centres = isotropic_centres(60, spectral=True)
    arguments = {
        "widths": (0.4, 0.5, 60.0),
        "metric": "chordal",
        "isotropic": True,
        "wavelength_range": RANGE,
    }
    unit = dirichlet_matrix(centres, **arguments)
    assert_allclose(dirichlet_matrix(centres, scales=(1.0, 1.0, 1.0), **arguments), unit)
    # The energy is a sum of one term per factor, times its squared scale
    terms = []
    for k in range(3):
        scales = [1.0, 1.0, 1.0]
        scales[k] = 2.0
        terms.append((dirichlet_matrix(centres, scales=tuple(scales), **arguments) - unit) / 3)
    assert_allclose(sum(terms), unit, rtol=1e-12, atol=1e-14 * np.abs(unit).max())
    scaled = dirichlet_matrix(centres, scales=(0.5, 3.0, 400.0), **arguments)
    expected = 0.25 * terms[0] + 9 * terms[1] + 160000 * terms[2]
    assert_allclose(scaled, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize(
    ("centres", "arguments", "message"),
    [
        ((2, False), {"wavelength_range": RANGE}, "if and only if"),
        ((3, False), {}, "if and only if"),
        ((3, False), {"wavelength_range": (780.0, 380.0)}, "must be increasing"),
        ((2, False), {"refinement": 0}, "refinement must be a positive integer"),
        ((2, False), {"refinement": 1.5}, "refinement must be a positive integer"),
        ((2, True), {}, "zero azimuth"),
        ((2, False), {"kernel": "cubic"}, "kernel must be one of"),
        ((2, False), {"metric": "manhattan"}, "metric must be"),
        ((2, False), {"widths": (1.0,)}, "expected 2 scales"),
        ((2, False), {"scales": (1.0, -1.0)}, "must be positive"),
        ((2, False), {"symmetry": "bilateral"}, "symmetry must be one of"),
    ],
)
def test_invalid_arguments(rng, centres, arguments, message):
    count, rotated = centres
    h = spherical_to_cartesian(rng.uniform(0, 1, 3), 0.0)
    if rotated:
        h = rotate(h, [0.0, 0.0, 1.0], 0.1)
    factors = [h, spherical_to_cartesian(rng.uniform(0, 1, 3), 0.0), rng.uniform(*RANGE, 3)]
    defaults = {"widths": (1.0,) * count, "metric": "chordal", "isotropic": True}
    with pytest.raises(ValueError, match=message):
        dirichlet_matrix(factors[:count], **(defaults | arguments))


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize(("isotropic", "spectral"), [(True, False), (True, True), (False, False)])
def test_reciprocal_energy_is_that_of_the_sums_over_orbits(metric, isotropic, spectral):
    # The kernels of the centres c and of their images g c, without symmetry
    centres = isotropic_centres(30, spectral) if isotropic else anisotropic_centres(30)
    images = [centres[0], centres[1] * np.array([-1.0, -1.0, 1.0]), *centres[2:]]
    both = [np.concatenate(pair) for pair in zip(centres, images, strict=True)]
    arguments = {
        "widths": (0.4, 0.5, 60.0)[: len(centres)],
        "metric": metric,
        "isotropic": isotropic,
        "wavelength_range": RANGE if spectral else None,
        "scales": (1.3, 0.7, 200.0)[: len(centres)],
    }
    plain = dirichlet_matrix(both, symmetry="none", **arguments)
    n = len(centres[0])
    # Energy of k(., c) + k(., g c): the sum of the four blocks
    expected = plain[:n, :n] + plain[:n, n:] + plain[n:, :n] + plain[n:, n:]
    matrix = dirichlet_matrix(centres, **arguments)
    tolerance = 1e-13 * np.abs(matrix).max()
    assert_allclose(matrix, expected, rtol=0, atol=tolerance)
    # The half turn is an isometry of the domain, which leaves the energy unchanged
    assert_allclose(plain[n:, n:], plain[:n, :n], rtol=0, atol=tolerance)
    assert_allclose(matrix, 2 * (plain[:n, :n] + plain[:n, n:]), rtol=0, atol=tolerance)
