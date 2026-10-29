# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.sampling, the Super-Fibonacci samplers."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli import sampling
from rivoli.parametrization import is_above_horizon, rusinkiewicz_to_directions
from rivoli.sampling import super_fibonacci_anisotropic, super_fibonacci_isotropic

VISIBLE = (380.0, 780.0)


def isotropic_cube(theta_h, theta_d, phi_d, *wavelength):
    # Inverse of the measure-preserving maps of the sampler
    return np.stack(
        [
            theta_h / (np.pi / 2),
            1 - np.cos(theta_d),
            (np.pi - phi_d) / (2 * np.pi),
            *[(w - VISIBLE[0]) / (VISIBLE[1] - VISIBLE[0]) for w in wavelength],
        ]
    )


def anisotropic_cube(theta_h, phi_h, theta_d, phi_d, *wavelength):
    return np.stack(
        [
            1 - np.cos(theta_h),
            (np.pi - phi_h) / (2 * np.pi),
            1 - np.cos(theta_d),
            (np.pi - phi_d) / (2 * np.pi),
            *[(w - VISIBLE[0]) / (VISIBLE[1] - VISIBLE[0]) for w in wavelength],
        ]
    )


def isotropic_spectral(n, **arguments):
    return super_fibonacci_isotropic(n, wavelength_range=VISIBLE, **arguments)


def anisotropic_spectral(n, **arguments):
    return super_fibonacci_anisotropic(n, wavelength_range=VISIBLE, **arguments)


SAMPLERS = [
    (super_fibonacci_isotropic, isotropic_cube),
    (super_fibonacci_anisotropic, anisotropic_cube),
    (isotropic_spectral, isotropic_cube),
    (anisotropic_spectral, anisotropic_cube),
]


def star_discrepancy(u):
    u = np.sort(u)
    i = np.arange(len(u))
    return max(np.max((i + 1) / len(u) - u), np.max(u - i / len(u)))


def grid_deviation(u, k):
    """Largest relative deviation of the counts in a regular grid from their mean."""
    cells = np.minimum((u * k).astype(int), k - 1)
    counts = np.zeros((k,) * len(u), dtype=int)
    np.add.at(counts, tuple(cells), 1)
    expected = u.shape[1] / k ** len(u)
    return np.max(np.abs(counts - expected)) / expected


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_generalized_golden_ratios(k):
    x = sampling._generalized_golden_ratio(k)
    assert_allclose(x ** (k + 1), x + 1, rtol=1e-15)


def test_golden_and_plastic_numbers():
    assert_allclose(sampling._generalized_golden_ratio(1), (1 + np.sqrt(5)) / 2, rtol=1e-15)
    assert_allclose(sampling._generalized_golden_ratio(2), 1.3247179572447460, rtol=1e-15)


@pytest.mark.parametrize("n", [1, 2, 1000])
def test_isotropic_ranges(n):
    theta_h, theta_d, phi_d = super_fibonacci_isotropic(n)
    assert theta_h.shape == theta_d.shape == phi_d.shape == (n,)
    assert np.all((theta_h > 0) & (theta_h < np.pi / 2))
    assert np.all((theta_d >= 0) & (theta_d <= np.pi / 2))
    assert np.all((phi_d > -np.pi) & (phi_d <= np.pi))


@pytest.mark.parametrize("n", [1, 2, 1000])
def test_anisotropic_ranges(n):
    theta_h, phi_h, theta_d, phi_d = super_fibonacci_anisotropic(n)
    assert theta_h.shape == phi_h.shape == theta_d.shape == phi_d.shape == (n,)
    assert np.all((theta_h > 0) & (theta_h < np.pi / 2))
    assert np.all((theta_d >= 0) & (theta_d <= np.pi / 2))
    assert np.all((phi_h > -np.pi) & (phi_h <= np.pi))
    assert np.all((phi_d > -np.pi) & (phi_d <= np.pi))


@pytest.mark.parametrize(("sampler", "cube"), SAMPLERS)
@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_marginals_have_a_low_discrepancy(sampler, cube, n):
    u = cube(*sampler(n))
    # The stratified coordinate is exactly regular
    assert_allclose(star_discrepancy(u[0]), 0.5 / n)
    # A random sampling would have a discrepancy of the order of 1 / sqrt(n),
    # a Kronecker sequence has one of the order of log(n) / n
    assert all(star_discrepancy(x) < 15 / n for x in u[1:])


@pytest.mark.parametrize(
    ("sampler", "cube", "tolerance"),
    [
        (super_fibonacci_isotropic, isotropic_cube, 0.05),
        # The 2D projections of a 3D Kronecker sequence are less regular, but
        # still better than those of a random sampling, of about 0.27 here
        (super_fibonacci_anisotropic, anisotropic_cube, 0.2),
        (isotropic_spectral, isotropic_cube, 0.2),
        (anisotropic_spectral, anisotropic_cube, 0.1),
    ],
)
def test_projections_are_uniform(sampler, cube, tolerance):
    u = cube(*sampler(10000))
    for pair in itertools.combinations(range(len(u)), 2):
        assert grid_deviation(u[list(pair)], 8) < tolerance, pair


@pytest.mark.parametrize(
    ("sampler", "cube", "tolerance"),
    [
        (super_fibonacci_isotropic, isotropic_cube, 0.05),
        # A random sampling has a deviation of about 0.5 here
        (super_fibonacci_anisotropic, anisotropic_cube, 0.15),
        (isotropic_spectral, isotropic_cube, 0.15),
    ],
)
def test_domain_is_uniformly_covered(sampler, cube, tolerance):
    assert grid_deviation(cube(*sampler(10000)), 4) < tolerance


def test_spectral_anisotropic_domain_is_uniformly_covered():
    # A random sampling has a deviation of about 0.47 on this coarser grid
    assert grid_deviation(anisotropic_cube(*anisotropic_spectral(10000)), 3) < 0.3


@pytest.mark.parametrize(("sampler", "count"), [(isotropic_spectral, 4), (anisotropic_spectral, 5)])
def test_spectral_ranges(sampler, count):
    sites = sampler(1000)
    assert len(sites) == count
    assert np.all((sites[-1] >= VISIBLE[0]) & (sites[-1] < VISIBLE[1]))


@pytest.mark.parametrize("sampler", [super_fibonacci_isotropic, super_fibonacci_anisotropic])
@pytest.mark.parametrize("wavelength_range", [(780.0, 380.0), (500.0, 500.0)])
def test_wavelength_range_must_be_increasing(sampler, wavelength_range):
    with pytest.raises(ValueError, match="must be increasing"):
        sampler(10, wavelength_range=wavelength_range)


def test_isotropic_sites_are_valid_rusinkiewicz_angles():
    theta_h, theta_d, phi_d = super_fibonacci_isotropic(1000)
    wi, wo = rusinkiewicz_to_directions(theta_h, 0.0, theta_d, phi_d)
    assert_allclose(np.linalg.norm(wi, axis=-1), 1)
    assert_allclose(np.linalg.norm(wo, axis=-1), 1)


@pytest.mark.parametrize(("sampler", "_"), SAMPLERS)
def test_sampling_is_deterministic(sampler, _):
    assert_allclose(sampler(500), sampler(500))


@pytest.mark.parametrize("sampler", [super_fibonacci_isotropic, super_fibonacci_anisotropic])
@pytest.mark.parametrize("n", [0, -3])
def test_at_least_one_site(sampler, n):
    with pytest.raises(ValueError, match=f"at least 1, got {n}"):
        sampler(n)


@pytest.mark.parametrize(
    ("sampler", "angle_count"),
    [
        (super_fibonacci_isotropic, 3),
        (super_fibonacci_anisotropic, 4),
        (isotropic_spectral, 3),
        (anisotropic_spectral, 4),
    ],
)
@pytest.mark.parametrize("n", [1, 2, 3, 10, 1000])
def test_above_horizon_gives_exactly_n_sites_above_it(sampler, angle_count, n):
    sites = sampler(n, above_horizon=True)
    assert all(x.shape == (n,) for x in sites)
    assert np.all(is_above_horizon(*sites[:angle_count]))


@pytest.mark.parametrize(
    ("sampler", "angle_count"),
    [
        (super_fibonacci_isotropic, 3),
        (super_fibonacci_anisotropic, 4),
        (isotropic_spectral, 3),
        (anisotropic_spectral, 4),
    ],
)
@pytest.mark.parametrize("above_horizon", [False, True])
def test_reciprocity_samples_half_of_the_difference_azimuths(sampler, angle_count, above_horizon):
    n = 2000
    full = sampler(n, above_horizon=above_horizon)
    half = sampler(n, above_horizon=above_horizon, symmetry="reciprocity")
    phi_d = half[angle_count - 1]
    assert all(x.shape == (n,) for x in half)
    assert np.all((phi_d > 0) & (phi_d <= np.pi))
    if above_horizon:
        assert np.all(is_above_horizon(*half[:angle_count]))
    else:
        # The same sites, with the azimuths of d halved onto (0, pi]
        assert_allclose(phi_d, (full[angle_count - 1] + np.pi) / 2)
        for k, x in enumerate(half):
            if k != angle_count - 1:
                assert_allclose(x, full[k])


def test_invalid_symmetry():
    with pytest.raises(ValueError, match="symmetry must be one of"):
        super_fibonacci_isotropic(10, symmetry="bilateral")  # pyright: ignore[reportArgumentType]


def test_above_horizon_sites_are_uniform():
    # Reference proportions of the cells above the horizon, from a dense sampling
    reference = super_fibonacci_isotropic(400000)
    reference = tuple(x[is_above_horizon(*reference)] for x in reference)

    def cells(sites):
        u = np.minimum((isotropic_cube(*sites) * 4).astype(int), 3)
        return np.ravel_multi_index(tuple(u), (4, 4, 4))

    expected = 10000 * np.bincount(cells(reference), minlength=64) / len(reference[0])
    counts = np.bincount(cells(super_fibonacci_isotropic(10000, above_horizon=True)), minlength=64)
    # Cells mostly below the horizon have too few sites to compare. A random
    # subset of the reference has a deviation of about 0.3
    populated = expected > 20
    deviation = np.abs(counts - expected)[populated] / expected[populated]
    assert np.max(deviation) < 0.12
