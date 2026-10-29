# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.merl, the files of the MERL BRDF database."""

import math
import struct
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rivoli.geometry import spherical_to_cartesian
from rivoli.merl import (
    SCALES,
    SHAPE,
    bin_angles,
    bin_indices,
    read_merl,
    write_merl,
    write_model,
)
from rivoli.model import RBFModel
from rivoli.parametrization import is_above_horizon
from rivoli.sampling import super_fibonacci_isotropic


@pytest.fixture
def rng():
    return np.random.default_rng(20261030)


@pytest.fixture
def values(rng):
    """A BRDF table with missing bins below the horizon."""
    values = rng.uniform(0.0, 10.0, size=(*SHAPE, 3))
    values[~is_above_horizon(*bin_angles())] = -1.0
    return values


def reference_indices(theta_h, theta_d, phi_d):
    """Lookup of BRDFRead.cpp, the reference implementation of the database."""
    i = 0 if theta_h <= 0.0 else int(math.sqrt(theta_h / (math.pi / 2) * 90 * 90))
    j = int(theta_d / (math.pi / 2) * 90)
    if phi_d < 0.0:
        phi_d += math.pi
    k = int(phi_d / math.pi * 360 / 2)
    return min(max(i, 0), 89), min(max(j, 0), 89), min(max(k, 0), 179)


def test_round_trip(tmp_path, values):
    path = tmp_path / "brdf.binary"
    write_merl(path, values)
    assert path.stat().st_size == 12 + 3 * 8 * 90 * 90 * 180
    assert_allclose(read_merl(path), values, rtol=1e-15)
    # The stored values are kept exactly, despite the rounding of the scales
    copy = tmp_path / "copy.binary"
    write_merl(str(copy), read_merl(str(path)))
    assert copy.read_bytes() == path.read_bytes()


def test_file_layout(tmp_path):
    # The header, then the red, green and blue tables, of scaled values
    tables = np.zeros((3, *SHAPE))
    tables[0, 1, 2, 3] = 1500.0
    tables[1, 89, 0, 179] = 3000.0
    tables[2, 0, 89, 0] = -1.0
    path = tmp_path / "brdf.binary"
    path.write_bytes(struct.pack("<3i", *SHAPE) + tables.astype("<f8").tobytes())
    values = read_merl(path)
    assert values.shape == (*SHAPE, 3)
    assert values[1, 2, 3, 0] == 1.0
    assert_allclose(values[89, 0, 179, 1], 2.3)
    assert_allclose(values[0, 89, 0, 2], -1.66 / 1500)
    assert np.count_nonzero(values) == 3
    write_merl(tmp_path / "copy.binary", values)
    assert (tmp_path / "copy.binary").read_bytes() == path.read_bytes()


def test_scales():
    assert SCALES == (1.0 / 1500, 1.15 / 1500, 1.66 / 1500)


def test_invalid_files(tmp_path):
    path = tmp_path / "brdf.binary"
    path.write_bytes(struct.pack("<2i", 90, 90))
    with pytest.raises(ValueError, match="header of a MERL file is truncated"):
        read_merl(path)
    path.write_bytes(struct.pack("<3i", 90, 90, 360))
    with pytest.raises(ValueError, match="got \\(90, 90, 360\\)"):
        read_merl(path)
    header = struct.pack("<3i", *SHAPE)
    for size in (3 * 8 * 90 * 90 * 180 - 8, 3 * 8 * 90 * 90 * 180 + 1):
        path.write_bytes(header + bytes(size))
        with pytest.raises(ValueError, match="expected 34992012 bytes"):
            read_merl(path)


def test_invalid_values(tmp_path, values):
    path = tmp_path / "brdf.binary"
    with pytest.raises(ValueError, match="expected values of shape \\(90, 90, 180, 3\\)"):
        write_merl(path, values[..., :1])
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        write_merl(path, values)
    assert not path.exists()


def test_bin_angles():
    theta_h, theta_d, phi_d = bin_angles()
    for angle in (theta_h, theta_d, phi_d):
        assert angle.shape == SHAPE
    assert_allclose(theta_h[:, 0, 0], (np.arange(90) / 90) ** 2 * np.pi / 2)
    assert_allclose(theta_d[0, :, 0], np.radians(np.arange(90)))
    assert_allclose(phi_d[0, 0], np.radians(np.arange(180)))


def test_bins_of_the_bin_angles_are_their_own():
    for indices, expected in zip(bin_indices(*bin_angles()), np.indices(SHAPE), strict=True):
        assert_array_equal(indices, expected)


def test_bin_indices_match_the_reference(rng):
    theta_h = rng.uniform(-0.1, np.pi / 2 + 0.1, size=2000)
    theta_d = rng.uniform(-0.1, np.pi / 2 + 0.1, size=2000)
    phi_d = rng.uniform(-np.pi, np.pi, size=2000)
    indices = np.stack(bin_indices(theta_h, theta_d, phi_d), axis=-1)
    angles = zip(theta_h, theta_d, phi_d, strict=True)
    expected = [reference_indices(*angle) for angle in angles]
    assert_array_equal(indices, expected)


def test_bin_indices_use_reciprocity():
    # phi_d and phi_d + pi are the same bin, pi included
    i, j, k = bin_indices(0.3, 0.4, np.array([0.0, np.pi, -np.pi, 0.5, 0.5 - np.pi]))
    assert_array_equal(k, [0, 0, 0, 28, 28])
    assert_array_equal(i, 39)
    assert_array_equal(j, 22)


def test_bin_indices_broadcast():
    indices = bin_indices(np.zeros((4, 1)), np.zeros(5), 0.0)
    for index in indices:
        assert index.shape == (4, 5)
        assert index.dtype == np.intp


def model(rng, channels=(3,), spectral=False):
    """A small model with random weights."""
    if spectral:
        theta_h, theta_d, phi_d, wavelength = super_fibonacci_isotropic(
            30, wavelength_range=(380.0, 780.0)
        )
        wavelengths = (wavelength,)
    else:
        theta_h, theta_d, phi_d = super_fibonacci_isotropic(30)
        wavelengths = ()
    h, d = spherical_to_cartesian(theta_h, 0.0), spherical_to_cartesian(theta_d, phi_d)
    centres = (h, d, *wavelengths)
    return RBFModel(
        centres=centres,
        weights=rng.normal(scale=0.5, size=(30, *channels)) - 2.0,
        widths=(0.4, 0.6, 80.0)[: len(centres)],
        metric="chordal",
        kernel="gaussian",
        log_offset=1e-6,
    )


def grid_sites(above):
    theta_h, theta_d, phi_d = bin_angles()
    return (
        spherical_to_cartesian(theta_h[above], 0.0),
        spherical_to_cartesian(theta_d[above], phi_d[above]),
    )


def test_write_rgb_model(tmp_path, rng):
    rgb = model(rng)
    write_model(tmp_path / "model.binary", rgb)
    values = read_merl(tmp_path / "model.binary")
    above = is_above_horizon(*bin_angles())
    assert np.all(values[~above] == -1.0)
    assert_allclose(values[above], rgb(grid_sites(above)), rtol=1e-14)
    # Negative values of a model fitted in linear space are clamped
    linear = replace(rgb, log_offset=None)
    write_model(tmp_path / "model.binary", linear)
    values = read_merl(tmp_path / "model.binary")
    expected = linear(grid_sites(above))
    assert np.any(expected < 0)
    assert_allclose(values[above], np.maximum(expected, 0.0), rtol=1e-14)


def test_write_spectral_model(tmp_path, rng):
    spectral = model(rng, channels=(), spectral=True)
    wavelengths = np.arange(380.0, 781.0, 10.0)
    write_model(tmp_path / "model.binary", spectral, wavelengths=wavelengths, illuminant="E")
    values = read_merl(tmp_path / "model.binary")
    above = is_above_horizon(*bin_angles())
    assert np.all(values[~above] == -1.0)
    expected = spectral.rgb(grid_sites(above), wavelengths, illuminant="E")
    assert_allclose(values[above], np.maximum(expected, 0.0), rtol=1e-14)


def test_write_invalid_models(tmp_path, rng):
    path = tmp_path / "model.binary"
    wavelengths = np.arange(380.0, 781.0, 10.0)
    with pytest.raises(ValueError, match="if and only if the model is spectral"):
        write_model(path, model(rng), wavelengths=wavelengths)
    with pytest.raises(ValueError, match="if and only if the model is spectral"):
        write_model(path, model(rng, channels=(), spectral=True))
    with pytest.raises(ValueError, match="got weights of shape \\(30,\\)"):
        write_model(path, model(rng, channels=()))
    assert not path.exists()


@pytest.mark.parametrize("spectral", [False, True])
def test_write_model_with_warp(tmp_path, rng, spectral):
    pytest.importorskip("warp")
    written = model(rng, channels=() if spectral else (3,), spectral=spectral)
    wavelengths = np.arange(380.0, 781.0, 20.0) if spectral else None
    write_model(tmp_path / "numpy.binary", written, wavelengths=wavelengths)
    write_model(tmp_path / "warp.binary", written, wavelengths=wavelengths, device="cpu")
    expected = read_merl(tmp_path / "numpy.binary")
    assert_allclose(read_merl(tmp_path / "warp.binary"), expected, rtol=1e-12, atol=1e-300)
