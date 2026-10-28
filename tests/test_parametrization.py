# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.parametrization, the Rusinkiewicz parametrization."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli.geometry import normalize, rotate, spherical_to_cartesian
from rivoli.parametrization import (
    directions_to_half_difference,
    directions_to_rusinkiewicz,
    half_difference_to_directions,
    rusinkiewicz_to_directions,
    rusinkiewicz_to_spherical,
    spherical_to_rusinkiewicz,
)

X, Y, Z = np.eye(3)


@pytest.fixture
def rng():
    return np.random.default_rng(20261028)


def upper_hemisphere(rng, n):
    v = normalize(rng.normal(size=(n, 3)))
    v[:, 2] = np.abs(v[:, 2])
    return v


def assert_angles_close(actual, desired):
    # Compare on the unit circle, so that -pi and pi are equal
    assert_allclose(np.exp(1j * np.asarray(actual)), np.exp(1j * np.asarray(desired)), atol=1e-12)


@pytest.mark.parametrize(
    ("wi", "wo", "expected"),
    [
        (Z, Z, (0, 0, 0, 0)),
        (X, Z, (np.pi / 4, 0, np.pi / 4, 0)),
        (Y, Z, (np.pi / 4, np.pi / 2, np.pi / 4, 0)),
        (Z, X, (np.pi / 4, 0, np.pi / 4, np.pi)),
        ((X + Z) / np.sqrt(2), (Z - X) / np.sqrt(2), (0, 0, np.pi / 4, 0)),
    ],
)
def test_known_values(wi, wo, expected):
    assert_allclose(directions_to_rusinkiewicz(wi, wo), expected, atol=1e-15)


def test_specular_configuration_has_a_normal_half_vector(rng):
    theta = rng.uniform(0, np.pi / 2, 100)
    phi = rng.uniform(-np.pi, np.pi, 100)
    wi = spherical_to_cartesian(theta, phi)
    wo = spherical_to_cartesian(theta, phi + np.pi)
    theta_h, phi_h, theta_d, phi_d = directions_to_rusinkiewicz(wi, wo)
    # theta_h is 0 up to rounding errors, which make phi_h arbitrary: the
    # rotation by -phi_h that it then implies is still exactly undone
    assert_allclose(theta_h, 0, atol=1e-7)
    assert_allclose(theta_d, theta)
    assert_angles_close(phi_d, phi - phi_h)


def test_upper_hemisphere_ranges(rng):
    theta_h, phi_h, theta_d, phi_d = directions_to_rusinkiewicz(
        upper_hemisphere(rng, 1000), upper_hemisphere(rng, 1000)
    )
    assert np.all((theta_h >= 0) & (theta_h <= np.pi / 2))
    assert np.all((theta_d >= 0) & (theta_d <= np.pi / 2))
    assert np.all((phi_h > -np.pi) & (phi_h <= np.pi))
    assert np.all((phi_d > -np.pi) & (phi_d <= np.pi))


def test_difference_angle_is_the_angle_between_half_vector_and_directions(rng):
    wi, wo = upper_hemisphere(rng, 1000), upper_hemisphere(rng, 1000)
    h, d = directions_to_half_difference(wi, wo)
    assert_allclose(np.linalg.norm(h, axis=-1), 1)
    assert_allclose(np.linalg.norm(d, axis=-1), 1)
    assert_allclose(d[:, 2], np.sum(wi * h, axis=-1), atol=1e-15)
    assert_allclose(d[:, 2], np.sum(wo * h, axis=-1), atol=1e-15)


def test_directions_round_trip(rng):
    wi, wo = upper_hemisphere(rng, 1000), upper_hemisphere(rng, 1000)
    angles = directions_to_rusinkiewicz(wi, wo)
    back_wi, back_wo = rusinkiewicz_to_directions(*angles)
    assert_allclose(back_wi, wi, atol=1e-14)
    assert_allclose(back_wo, wo, atol=1e-14)


def test_half_difference_round_trip(rng):
    wi, wo = upper_hemisphere(rng, 1000), upper_hemisphere(rng, 1000)
    back_wi, back_wo = half_difference_to_directions(*directions_to_half_difference(wi, wo))
    assert_allclose(back_wi, wi, atol=1e-14)
    assert_allclose(back_wo, wo, atol=1e-14)


def test_angles_round_trip(rng):
    # Away from the poles, where the azimuths are undefined
    theta_h, theta_d = rng.uniform(0.01, np.pi / 2, (2, 1000))
    phi_h, phi_d = rng.uniform(-np.pi, np.pi, (2, 1000))
    back = directions_to_rusinkiewicz(*rusinkiewicz_to_directions(theta_h, phi_h, theta_d, phi_d))
    assert_allclose(back[0], theta_h)
    assert_angles_close(back[1], phi_h)
    assert_allclose(back[2], theta_d)
    assert_angles_close(back[3], phi_d)


def test_reciprocity_shifts_the_difference_azimuth(rng):
    wi, wo = upper_hemisphere(rng, 1000), upper_hemisphere(rng, 1000)
    theta_h, phi_h, theta_d, phi_d = directions_to_rusinkiewicz(wi, wo)
    swapped = directions_to_rusinkiewicz(wo, wi)
    assert_allclose(swapped[0], theta_h)
    assert_angles_close(swapped[1], phi_h)
    assert_allclose(swapped[2], theta_d)
    assert_angles_close(swapped[3], phi_d + np.pi)


def test_rotation_around_the_normal_only_shifts_the_half_azimuth(rng):
    wi, wo = upper_hemisphere(rng, 1000), upper_hemisphere(rng, 1000)
    alpha = rng.uniform(-np.pi, np.pi, 1000)
    theta_h, phi_h, theta_d, phi_d = directions_to_rusinkiewicz(wi, wo)
    rotated = directions_to_rusinkiewicz(rotate(wi, Z, alpha), rotate(wo, Z, alpha))
    assert_allclose(rotated[0], theta_h)
    assert_angles_close(rotated[1], phi_h + alpha)
    assert_allclose(rotated[2], theta_d)
    assert_angles_close(rotated[3], phi_d)


def test_isotropic_inverse_with_a_zero_half_azimuth(rng):
    theta_h, theta_d = rng.uniform(0, np.pi / 2, (2, 100))
    phi_d = rng.uniform(-np.pi, np.pi, 100)
    wi, wo = rusinkiewicz_to_directions(theta_h, 0, theta_d, phi_d)
    assert wi.shape == wo.shape == (100, 3)
    h = normalize(wi + wo)
    assert_allclose(h[:, 1], 0, atol=1e-15)


def test_broadcasting():
    wi = spherical_to_cartesian(np.linspace(0, 1, 4)[:, np.newaxis], 0.5)
    wo = spherical_to_cartesian(0.3, np.linspace(-1, 1, 5))
    assert all(a.shape == (4, 5) for a in directions_to_rusinkiewicz(wi, wo))
    wi, wo = rusinkiewicz_to_directions(np.linspace(0, 1, 4)[:, np.newaxis], 0, 0.2, 0)
    assert wi.shape == wo.shape == (4, 1, 3)


def random_spherical(rng, n):
    # Upper hemisphere, away from the poles where the azimuths are undefined
    theta_i, theta_o = rng.uniform(0.01, np.pi / 2, (2, n))
    phi_i, phi_o = rng.uniform(-np.pi, np.pi, (2, n))
    return theta_i, phi_i, theta_o, phi_o


def test_spherical_matches_directions(rng):
    theta_i, phi_i, theta_o, phi_o = random_spherical(rng, 1000)
    expected = directions_to_rusinkiewicz(
        spherical_to_cartesian(theta_i, phi_i), spherical_to_cartesian(theta_o, phi_o)
    )
    assert_allclose(spherical_to_rusinkiewicz(theta_i, phi_i, theta_o, phi_o), expected)


def test_spherical_round_trip(rng):
    theta_i, phi_i, theta_o, phi_o = random_spherical(rng, 1000)
    back = rusinkiewicz_to_spherical(*spherical_to_rusinkiewicz(theta_i, phi_i, theta_o, phi_o))
    assert len(back) == 4
    assert_allclose(back[0], theta_i)
    assert_angles_close(back[1], phi_i)
    assert_allclose(back[2], theta_o)
    assert_angles_close(back[3], phi_o)


def test_isotropic_matches_the_anisotropic_conversion(rng):
    theta_i, phi_i, theta_o, phi_o = random_spherical(rng, 1000)
    theta_h, _, theta_d, phi_d = spherical_to_rusinkiewicz(theta_i, phi_i, theta_o, phi_o)
    isotropic = spherical_to_rusinkiewicz(theta_i, theta_o, phi_o - phi_i)
    assert len(isotropic) == 3
    assert_allclose(isotropic[0], theta_h)
    assert_allclose(isotropic[1], theta_d)
    assert_angles_close(isotropic[2], phi_d)


def test_isotropic_round_trip(rng):
    theta_i, _, theta_o, _ = random_spherical(rng, 1000)
    delta_phi = rng.uniform(-np.pi, np.pi, 1000)
    back = rusinkiewicz_to_spherical(*spherical_to_rusinkiewicz(theta_i, theta_o, delta_phi))
    assert_allclose(back[0], theta_i)
    assert_allclose(back[1], theta_o)
    assert_allclose(back[2], delta_phi)


def test_isotropic_inverse_wraps_the_azimuth_difference(rng):
    theta_h, theta_d = rng.uniform(0.01, np.pi / 2, (2, 1000))
    phi_d = rng.uniform(-np.pi, np.pi, 1000)
    theta_i, theta_o, delta_phi = rusinkiewicz_to_spherical(theta_h, theta_d, phi_d)
    assert np.all((delta_phi > -np.pi) & (delta_phi <= np.pi))
    assert_allclose(
        spherical_to_rusinkiewicz(theta_i, theta_o, delta_phi), (theta_h, theta_d, phi_d)
    )


def test_isotropic_known_values():
    # Specular configuration: the half vector is the normal. theta_h is 0
    # up to rounding errors only, which make phi_h and thus phi_d arbitrary
    theta_h, theta_d, _ = spherical_to_rusinkiewicz(0.3, 0.3, np.pi)
    assert_allclose((theta_h, theta_d), (0, 0.3), atol=1e-15)
    assert_allclose(rusinkiewicz_to_spherical(0, 0.3, 0), (0.3, 0.3, np.pi), atol=1e-15)
    assert_allclose(spherical_to_rusinkiewicz(0.2, 0.6, 0), (0.4, 0.2, np.pi), atol=1e-15)


@pytest.mark.parametrize("function", [spherical_to_rusinkiewicz, rusinkiewicz_to_spherical])
@pytest.mark.parametrize("count", [2, 5])
def test_wrong_number_of_angles(function, count):
    with pytest.raises(TypeError, match=f"expected 3 or 4 angles, got {count}"):
        function(*[0.0] * count)
