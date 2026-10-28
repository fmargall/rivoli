# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.distances, the distances and spacings of samplings."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli import distances
from rivoli.distances import (
    chordal_distance,
    diameters,
    geodesic_distance,
    nearest_neighbour_spacings,
)
from rivoli.geometry import normalize, spherical_to_cartesian
from rivoli.sampling import super_fibonacci_anisotropic, super_fibonacci_isotropic

X, Y, Z = np.eye(3)


@pytest.fixture
def rng():
    return np.random.default_rng(20261029)


def random_directions(rng, n):
    return normalize(rng.normal(size=(n, 3)))


@pytest.mark.parametrize(
    ("u", "v", "geodesic", "chordal"),
    [
        (X, X, 0, 0),
        (X, Y, np.pi / 2, np.sqrt(2)),
        (X, -X, np.pi, 2),
        (Z, (X + Z) / np.sqrt(2), np.pi / 4, 2 * np.sin(np.pi / 8)),
    ],
)
def test_known_values(u, v, geodesic, chordal):
    assert_allclose(geodesic_distance(u, v), geodesic, atol=1e-15)
    assert_allclose(chordal_distance(u, v), chordal, atol=1e-15)


@pytest.mark.parametrize("angle", [1e-12, 1e-8, 1e-4])
def test_geodesic_distance_is_accurate_for_close_vectors(angle):
    v = np.array([np.cos(angle), np.sin(angle), 0.0])
    # Where arccos of the dot product is wrong by up to about 1e-8
    assert_allclose(geodesic_distance(X, v), angle, rtol=1e-12)


def test_chordal_distance_is_the_chord_of_the_geodesic_one(rng):
    u, v = random_directions(rng, 1000), random_directions(rng, 1000)
    assert_allclose(chordal_distance(u, v), 2 * np.sin(geodesic_distance(u, v) / 2))


@pytest.mark.parametrize("distance", [geodesic_distance, chordal_distance])
def test_metric_axioms(rng, distance):
    u, v, w = (random_directions(rng, 1000) for _ in range(3))
    assert np.all(distance(u, v) >= 0)
    assert_allclose(distance(u, u), 0, atol=1e-15)
    assert_allclose(distance(u, v), distance(v, u))
    assert np.all(distance(u, w) <= distance(u, v) + distance(v, w) + 1e-12)


@pytest.mark.parametrize("distance", [geodesic_distance, chordal_distance])
def test_distances_broadcast(rng, distance):
    u = random_directions(rng, 4)[:, np.newaxis]
    v = random_directions(rng, 5)
    assert distance(u, v).shape == (4, 5)


def test_isotropic_half_vectors_are_at_their_polar_angle_difference(rng):
    theta_h = rng.uniform(0, np.pi / 2, (2, 1000))
    h = spherical_to_cartesian(theta_h, 0.0)
    assert_allclose(geodesic_distance(h[0], h[1]), np.abs(theta_h[0] - theta_h[1]), atol=1e-14)


def brute_force_spacings(h, d, wavelength, distance):
    components = [distance(h[:, np.newaxis], h), distance(d[:, np.newaxis], d)]
    if wavelength is not None:
        components.append(np.abs(wavelength[:, np.newaxis] - wavelength))
    squared = np.sum(np.stack(components) ** 2, axis=0)
    np.fill_diagonal(squared, np.inf)
    nearest = np.argmin(squared, axis=1)
    return tuple(c[np.arange(len(h)), nearest] for c in components)


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize("spectral", [False, True])
@pytest.mark.parametrize("pairs_per_chunk", [7, 2**20])
def test_spacings_match_a_brute_force_search(rng, monkeypatch, metric, spectral, pairs_per_chunk):
    # A small chunk size exercises the chunks, including a partial last one
    monkeypatch.setattr(distances, "_PAIRS_PER_CHUNK", pairs_per_chunk)
    h, d = random_directions(rng, 50), random_directions(rng, 50)
    wavelength = rng.uniform(0, 1, 50) if spectral else None
    spacings = nearest_neighbour_spacings(h, d, wavelength, metric=metric)
    assert len(spacings) == (3 if spectral else 2)
    distance = {"geodesic": geodesic_distance, "chordal": chordal_distance}[metric]
    assert_allclose(spacings, brute_force_spacings(h, d, wavelength, distance))


def test_spacings_of_a_regular_sampling():
    # Half vectors every 0.1 rad along a meridian, the same difference vector
    theta_h = 0.1 * np.arange(10)
    h = spherical_to_cartesian(theta_h, 0.0)
    d = np.broadcast_to(Z, h.shape)
    spacing_h, spacing_d = nearest_neighbour_spacings(h, d)
    assert_allclose(spacing_h, 0.1)
    assert_allclose(spacing_d, 0, atol=1e-15)


@pytest.mark.parametrize(
    ("sampler", "dimension"),
    [(super_fibonacci_isotropic, 3), (super_fibonacci_anisotropic, 4)],
)
def test_spacings_of_super_fibonacci_shrink_with_the_dimension(sampler, dimension):
    def mean_spacing(n):
        sites = sampler(n)
        if dimension == 3:
            theta_h, theta_d, phi_d = sites
            h = spherical_to_cartesian(theta_h, 0.0)
        else:
            theta_h, phi_h, theta_d, phi_d = sites
            h = spherical_to_cartesian(theta_h, phi_h)
        spacings = nearest_neighbour_spacings(h, spherical_to_cartesian(theta_d, phi_d))
        return np.mean(np.hypot(*spacings))

    # Multiplying the number of sites by 2^dimension halves the spacings
    ratio = mean_spacing(100) / mean_spacing(100 * 2**dimension)
    assert 1.6 < ratio < 2.4


def test_at_least_two_sites():
    with pytest.raises(ValueError, match="at least 2, got 1"):
        nearest_neighbour_spacings([Z], [Z])


def test_unknown_metric():
    with pytest.raises(ValueError, match="'geodesic' or 'chordal', got 'euclid'"):
        nearest_neighbour_spacings([Z, X], [Z, X], metric="euclid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("isotropic", "metric", "expected"),
    [
        (True, "geodesic", (np.pi / 2, np.pi)),
        (False, "geodesic", (np.pi, np.pi)),
        (True, "chordal", (np.sqrt(2), 2)),
        (False, "chordal", (2, 2)),
    ],
)
def test_diameters(isotropic, metric, expected):
    assert_allclose(diameters(isotropic=isotropic, metric=metric), expected)
    spectral = diameters(isotropic=isotropic, metric=metric, wavelength_range=(380, 780))
    assert_allclose(spectral, (*expected, 400))


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
def test_diameters_are_the_largest_distances(metric):
    distance = {"geodesic": geodesic_distance, "chordal": chordal_distance}[metric]
    isotropic_h, hemisphere = diameters(isotropic=True, metric=metric)
    # The ends of a quarter of meridian, and two opposite points of the equator
    assert_allclose(distance(Z, X), isotropic_h)
    assert_allclose(distance(X, -X), hemisphere)
    # No pair of a dense sampling of the hemisphere is farther apart
    theta_h, phi_h, _, _ = super_fibonacci_anisotropic(2000)
    h = spherical_to_cartesian(theta_h, phi_h)
    assert np.max(distance(h[:, np.newaxis], h)) <= hemisphere
    quarter = spherical_to_cartesian(theta_h, 0.0)
    assert np.max(distance(quarter[:, np.newaxis], quarter)) <= isotropic_h


def test_diameters_reject_a_decreasing_wavelength_range():
    with pytest.raises(ValueError, match="must be increasing"):
        diameters(isotropic=True, wavelength_range=(780, 380))


def test_diameters_reject_an_unknown_metric():
    with pytest.raises(ValueError, match="got 'euclid'"):
        diameters(isotropic=True, metric="euclid")  # type: ignore[arg-type]


@pytest.mark.parametrize("metric", ["geodesic", "chordal"])
@pytest.mark.parametrize("spectral", [False, True])
def test_scaled_spacings_match_a_brute_force_search(rng, metric, spectral):
    h, d = random_directions(rng, 50), random_directions(rng, 50)
    wavelength = rng.uniform(380, 780, 50) if spectral else None
    scales = diameters(
        isotropic=False, metric=metric, wavelength_range=(380, 780) if spectral else None
    )
    distance = {"geodesic": geodesic_distance, "chordal": chordal_distance}[metric]
    # Scaling every factor amounts to scaling its distance
    expected = brute_force_spacings(
        h,
        d,
        None if wavelength is None else wavelength / scales[-1],
        lambda u, v: distance(u, v) / scales[0],
    )
    spacings = nearest_neighbour_spacings(h, d, wavelength, metric=metric, scales=scales)
    assert_allclose(spacings, expected)
    assert all(np.all(spacing <= 1) for spacing in spacings)


def test_scales_change_the_nearest_neighbour():
    # The second site is nearer in h, the third in d
    h = spherical_to_cartesian([0.0, 0.1, 0.0], 0.0)
    d = spherical_to_cartesian([0.0, 0.0, 0.15], 0.0)
    assert_allclose(nearest_neighbour_spacings(h, d)[0][0], 0.1)
    # Giving the factor of h a smaller scale stretches its distances
    assert_allclose(nearest_neighbour_spacings(h, d, scales=(0.5, 1.0))[1][0], 0.15)


@pytest.mark.parametrize(
    ("scales", "message"),
    [
        ((1.0,), "expected 2 scales, one per factor, got 1"),
        ((1.0, 1.0, 1.0), "expected 2 scales, one per factor, got 3"),
        ((1.0, 0.0), "must be positive"),
        ((1.0, -2.0), "must be positive"),
    ],
)
def test_invalid_scales(scales, message):
    with pytest.raises(ValueError, match=message):
        nearest_neighbour_spacings([Z, X], [Z, X], scales=scales)
