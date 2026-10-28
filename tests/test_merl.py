# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.merl, the files of the MERL BRDF database."""

import math
import struct

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rivoli.merl import SCALES, SHAPE, bin_angles, bin_indices, read_merl, write_merl
from rivoli.parametrization import is_above_horizon


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
