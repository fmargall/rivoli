# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.colour, the conversions of spectra to colours."""

from importlib.resources import files

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli.colour import spectral_matrix, spectrum_to_rgb, spectrum_to_xyz, xyz_to_rgb

GRID = np.arange(360.0, 831.0)

# Matrix of IEC 61966-2-1, rounded to 4 decimals
SRGB_FROM_XYZ = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


@pytest.fixture
def rng():
    return np.random.default_rng(20261031)


def table(name):
    with (files("rivoli") / "data" / name).open() as stream:
        return np.loadtxt(stream, delimiter=",")


@pytest.mark.parametrize("illuminant", ["D65", "E"])
@pytest.mark.parametrize(
    "wavelengths", [GRID, np.arange(380.0, 781.0, 10.0), np.array([400.0, 650.0]), [550.0]]
)
def test_white_is_white(wavelengths, illuminant):
    white = np.ones(len(wavelengths))
    assert_allclose(spectrum_to_rgb(wavelengths, white, illuminant=illuminant), 1.0, rtol=1e-14)
    assert_allclose(spectrum_to_rgb(wavelengths, 0.3 * white), 0.3, rtol=1e-14)


def test_white_point_of_d65():
    xyz = spectrum_to_xyz(GRID, np.ones(len(GRID)))
    assert_allclose(xyz, [0.95047, 1.0, 1.08883], atol=1e-5)
    assert_allclose(spectrum_to_xyz(GRID, np.ones(len(GRID)), illuminant="E"), 1.0, atol=1e-3)


def test_srgb_matrix():
    assert_allclose(xyz_to_rgb(np.eye(3)).T, SRGB_FROM_XYZ, atol=5e-4)


@pytest.mark.parametrize("illuminant", ["D65", "E"])
def test_monochromatic_spectra_give_the_colour_matching_functions(illuminant):
    observer = table("CIE_xyz_1931_2deg.csv")
    d65 = table("CIE_std_illum_D65.csv")
    power = np.interp(GRID, d65[:, 0], d65[:, 1]) if illuminant == "D65" else np.ones(len(GRID))
    expected = power[:, np.newaxis] * observer[:, 1:] / (power @ observer[:, 2])
    assert_allclose(spectrum_to_xyz(GRID, np.eye(len(GRID)), illuminant=illuminant), expected)


def test_spectra_are_interpolated_linearly(rng):
    wavelengths = np.sort(rng.uniform(400.0, 700.0, size=12))
    spectra = rng.uniform(size=(5, 12))
    # The same spectra on the grid of the CIE tables, extended by their ends
    dense = np.array([np.interp(GRID, wavelengths, spectrum) for spectrum in spectra])
    for illuminant in ("D65", "E"):
        assert_allclose(
            spectrum_to_rgb(wavelengths, spectra, illuminant=illuminant),
            spectrum_to_rgb(GRID, dense, illuminant=illuminant),
            rtol=1e-12,
        )


def test_conversions_are_the_spectral_matrix(rng):
    wavelengths = np.linspace(380.0, 780.0, 41)
    spectra = rng.uniform(size=(2, 3, 41))
    rgb = spectrum_to_rgb(wavelengths, spectra, illuminant="E")
    xyz = spectrum_to_xyz(wavelengths, spectra, illuminant="E")
    assert rgb.shape == xyz.shape == (2, 3, 3)
    assert_allclose(rgb, spectra @ spectral_matrix(wavelengths, illuminant="E"))
    assert_allclose(xyz, spectra @ spectral_matrix(wavelengths, space="xyz", illuminant="E"))
    # Under D65, no adaptation is needed
    xyz = spectrum_to_xyz(wavelengths, spectra)
    assert_allclose(spectrum_to_rgb(wavelengths, spectra), xyz_to_rgb(xyz), atol=1e-14)


def test_saturated_colours():
    # A red spectrum is red, outside the gamut of sRGB for a single line
    wavelengths = np.array([600.0, 610.0, 620.0, 630.0])
    red, green, blue = spectrum_to_rgb(wavelengths, [0.0, 1.0, 1.0, 0.0])
    assert red > 0 and green < red and blue < 0


def test_invalid_arguments():
    with pytest.raises(ValueError, match="shape \\(k,\\)"):
        spectral_matrix([])
    with pytest.raises(ValueError, match="shape \\(k,\\)"):
        spectral_matrix([[400.0, 500.0]])
    with pytest.raises(ValueError, match="finite and increasing"):
        spectral_matrix([500.0, 400.0])
    with pytest.raises(ValueError, match="finite and increasing"):
        spectral_matrix([400.0, np.nan])
    with pytest.raises(ValueError, match="space must be"):
        spectral_matrix([400.0], space="lab")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="illuminant must be"):
        spectrum_to_rgb([400.0], [1.0], illuminant="A")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="shape \\(\\.\\.\\., 2\\)"):
        spectrum_to_rgb([400.0, 500.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape \\(\\.\\.\\., 1\\)"):
        spectrum_to_xyz([400.0], 1.0)
    with pytest.raises(ValueError, match="shape \\(\\.\\.\\., 3\\)"):
        xyz_to_rgb([1.0, 1.0])
