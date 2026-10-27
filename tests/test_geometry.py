# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.geometry, the primitives on the unit sphere."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli.geometry import cartesian_to_spherical, normalize, rotate, spherical_to_cartesian

X, Y, Z = np.eye(3)


@pytest.fixture
def rng():
    return np.random.default_rng(20261027)


def random_directions(rng, n):
    return normalize(rng.normal(size=(n, 3)))


def test_normalize_gives_unit_vectors(rng):
    v = rng.normal(size=(4, 5, 3)) * 10
    u = normalize(v)
    assert u.shape == v.shape
    assert_allclose(np.linalg.norm(u, axis=-1), 1)
    assert_allclose(np.cross(u, v), 0, atol=1e-12)


@pytest.mark.parametrize(
    ("theta", "phi", "expected"),
    [
        (0, 0, Z),
        (np.pi, 0, -Z),
        (np.pi / 2, 0, X),
        (np.pi / 2, np.pi / 2, Y),
        (np.pi / 2, np.pi, -X),
        (np.pi / 4, 0, (X + Z) / np.sqrt(2)),
    ],
)
def test_spherical_to_cartesian_known_values(theta, phi, expected):
    assert_allclose(spherical_to_cartesian(theta, phi), expected, atol=1e-15)


@pytest.mark.parametrize(
    ("v", "theta", "phi"),
    [
        (Z, 0, 0),
        (-Z, np.pi, 0),
        (X, np.pi / 2, 0),
        (Y, np.pi / 2, np.pi / 2),
        (-Y, np.pi / 2, -np.pi / 2),
        (-X, np.pi / 2, np.pi),
        (3 * (X + Z), np.pi / 4, 0),
    ],
)
def test_cartesian_to_spherical_known_values(v, theta, phi):
    assert_allclose(cartesian_to_spherical(v), (theta, phi), atol=1e-15)


def test_spherical_round_trip(rng):
    v = random_directions(rng, 1000)
    assert_allclose(spherical_to_cartesian(*cartesian_to_spherical(v)), v, atol=1e-15)

    theta = rng.uniform(0, np.pi, 1000)
    phi = rng.uniform(-np.pi, np.pi, 1000)
    assert_allclose(cartesian_to_spherical(spherical_to_cartesian(theta, phi)), (theta, phi))


def test_cartesian_to_spherical_is_accurate_near_the_pole():
    theta = 1e-10
    v = spherical_to_cartesian(theta, 0.3)
    assert_allclose(cartesian_to_spherical(v), (theta, 0.3), rtol=1e-14)


def test_spherical_to_cartesian_broadcasts():
    theta = np.linspace(0, np.pi, 4)[:, np.newaxis]
    phi = np.linspace(-np.pi, np.pi, 5)
    v = spherical_to_cartesian(theta, phi)
    assert v.shape == (4, 5, 3)
    assert_allclose(np.linalg.norm(v, axis=-1), 1)


@pytest.mark.parametrize(
    ("v", "axis", "expected"),
    [
        (X, Z, Y),
        (Y, Z, -X),
        (Y, X, Z),
        (Z, Y, X),
    ],
)
def test_rotate_follows_the_right_hand_rule(v, axis, expected):
    assert_allclose(rotate(v, axis, np.pi / 2), expected, atol=1e-15)


def test_rotate_preserves_norms_and_angles(rng):
    v = rng.normal(size=(100, 3))
    w = rng.normal(size=(100, 3))
    axis = rng.normal(size=(100, 3))
    angle = rng.uniform(-np.pi, np.pi, 100)
    rv, rw = rotate(v, axis, angle), rotate(w, axis, angle)
    assert_allclose(np.linalg.norm(rv, axis=-1), np.linalg.norm(v, axis=-1))
    assert_allclose(np.sum(rv * rw, axis=-1), np.sum(v * w, axis=-1), atol=1e-12)


def test_rotate_leaves_the_axis_unchanged(rng):
    axis = rng.normal(size=(100, 3))
    assert_allclose(rotate(axis, 5 * axis, rng.uniform(-np.pi, np.pi, 100)), axis)


def test_rotate_inverse_and_composition(rng):
    v = rng.normal(size=(100, 3))
    axis = rng.normal(size=3)
    a, b = rng.uniform(-np.pi, np.pi, (2, 100))
    assert_allclose(rotate(rotate(v, axis, a), axis, -a), v, atol=1e-14)
    assert_allclose(rotate(rotate(v, axis, a), axis, b), rotate(v, axis, a + b), atol=1e-14)


def test_rotate_around_the_normal_shifts_the_azimuth(rng):
    theta = rng.uniform(0.1, np.pi - 0.1, 100)
    phi = rng.uniform(-np.pi / 2, np.pi / 2, 100)
    shift = rng.uniform(-np.pi / 2, np.pi / 2, 100)
    rotated = rotate(spherical_to_cartesian(theta, phi), Z, shift)
    assert_allclose(cartesian_to_spherical(rotated), (theta, phi + shift))


def test_rotate_broadcasts():
    v = np.broadcast_to(X, (4, 1, 3))
    angle = np.linspace(0, np.pi, 5)
    assert rotate(v, Z, angle).shape == (4, 5, 3)
