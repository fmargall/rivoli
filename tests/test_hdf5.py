# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.hdf5, the reader of HDF5 files described by a layout."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli.geometry import cartesian_to_spherical, spherical_to_cartesian
from rivoli.parametrization import spherical_to_rusinkiewicz

h5py = pytest.importorskip("h5py")
hdf5 = pytest.importorskip("rivoli.hdf5")

INCIDENCES = [0.0, 30.0, 60.0]
AZIMUTHS = [0.0, 90.0, 180.0]
SPECTRUM = [400.0, 550.0, 700.0, 850.0]

LAYOUT = {
    "angle_unit": "degree",
    "axes": {
        "theta_i": "/@incidences",
        "phi_i": 180.0,
        "theta_o": "/incidence_{theta_i}@zeniths",
        "phi_o": "/@azimuths",
        "wavelength": "/spectrum",
    },
    "values": {"path": "/incidence_{theta_i}/band_{wavelength}", "axes": ["phi_o", "theta_o"]},
    "cosine_weighted": "/@cosine",
}


def brdf(theta_i, theta_o, delta_phi, wavelength):
    """A made-up BRDF, to recognize each value."""
    return 1 + theta_i + 10 * theta_o + 100 * delta_phi + wavelength / 1000


def zeniths(i):
    # A grid of outgoing polar angles that depends on the incidence
    return np.linspace(0.0, 80.0, 4 + i)


@pytest.fixture
def path(tmp_path):
    """A file of tables of values per incidence and per wavelength, in degrees."""
    path = tmp_path / "measurements.h5"
    with h5py.File(path, "w") as file:
        file.attrs["incidences"] = INCIDENCES
        file.attrs["azimuths"] = AZIMUTHS
        file.attrs["cosine"] = "False"
        file["spectrum"] = SPECTRUM
        for i, theta_i in enumerate(INCIDENCES):
            group = file.create_group(f"incidence_{i}")
            group.attrs["zeniths"] = zeniths(i)
            phi_o, theta_o = np.meshgrid(AZIMUTHS, zeniths(i), indexing="ij")
            for k, wavelength in enumerate(SPECTRUM):
                values = brdf(theta_i, theta_o, phi_o - 180.0, wavelength)
                if i == 1 and k == 2:
                    # Missing measurements
                    values[1, 2:] = np.nan
                group[f"band_{k}"] = values
    return path


def expected_rows():
    """Angles in degrees, wavelengths and values of the measurements of the fixture."""
    rows = []
    for i, theta_i in enumerate(INCIDENCES):
        for k, wavelength in enumerate(SPECTRUM):
            for a, phi_o in enumerate(AZIMUTHS):
                for z, theta_o in enumerate(zeniths(i)):
                    if not (i == 1 and k == 2 and a == 1 and z >= 2):
                        value = brdf(theta_i, theta_o, phi_o - 180.0, wavelength)
                        rows.append((theta_i, 180.0, theta_o, phi_o, wavelength, value))
    return np.array(rows).T


def expected_measurements(isotropic=True):
    """Sites and values of the measurements of the file of the fixture."""
    theta_i, phi_i, theta_o, phi_o, wavelength, values = expected_rows()
    angles = np.radians([theta_i, phi_i, theta_o, phi_o])
    if isotropic:
        theta_h, theta_d, phi_d = spherical_to_rusinkiewicz(
            angles[0], angles[2], angles[3] - angles[1]
        )
        phi_h = 0.0
    else:
        theta_h, phi_h, theta_d, phi_d = spherical_to_rusinkiewicz(*angles)
    sites = (spherical_to_cartesian(theta_h, phi_h), spherical_to_cartesian(theta_d, phi_d))
    return (*sites, wavelength), values


@pytest.mark.parametrize("isotropic", [True, False])
def test_read_measurements(path, isotropic):
    measurements = hdf5.read_hdf5(path, LAYOUT, isotropic=isotropic)
    (h, d, wavelength), values = measurements
    (expected_h, expected_d, expected_wavelength), expected_values = expected_measurements(
        isotropic
    )
    # 4 wavelengths, 3 azimuths and 4, 5 and 6 zeniths, of which 3 are missing
    assert len(values) == 4 * 3 * (4 + 5 + 6) - 3
    assert_allclose(values, expected_values)
    assert_allclose(h, expected_h, atol=1e-15)
    assert_allclose(d, expected_d, atol=1e-15)
    assert_allclose(wavelength, expected_wavelength)
    assert measurements.sites[0] is h
    assert measurements.values is values
    if isotropic:
        assert np.all(h[:, 1] == 0)
    else:
        # The incident direction at phi_i = 180 degrees turns the half vectors
        assert np.any(np.abs(cartesian_to_spherical(h)[1]) > 1)


def test_layout_from_a_json_file(path, tmp_path):
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps(LAYOUT), encoding="utf-8")
    from_file = hdf5.read_hdf5(path, layout)
    from_mapping = hdf5.read_hdf5(path, LAYOUT)
    assert_allclose(from_file.values, from_mapping.values)
    assert_allclose(hdf5.read_hdf5(path, str(layout)).values, from_mapping.values)


@pytest.mark.parametrize("flag", ["True", "true", np.bytes_(b"True"), 1, np.bool_(True)])
def test_cosine_weighted_values(path, flag):
    plain = hdf5.read_hdf5(path, LAYOUT)
    with h5py.File(path, "a") as file:
        # Created with its type, as h5py returns plain bytes as strings
        file.attrs.create("cosine", flag)
    weighted = hdf5.read_hdf5(path, LAYOUT)
    theta_o = np.radians(expected_rows()[2])
    assert_allclose(weighted.values, plain.values / np.cos(theta_o))
    assert_allclose(hdf5.read_hdf5(path, LAYOUT | {"cosine_weighted": False}).values, plain.values)
    assert_allclose(
        hdf5.read_hdf5(path, LAYOUT | {"cosine_weighted": True}).values, weighted.values
    )


def test_monochromatic_measurements_in_radians(tmp_path):
    # A single table, of the outgoing directions, with the other angles shared
    path = tmp_path / "measurements.h5"
    theta_o, phi_o = np.linspace(0.1, 1.2, 5), np.linspace(-3.0, 3.0, 7)
    values = np.arange(35.0).reshape(5, 7)
    with h5py.File(path, "w") as file:
        file["brdf"] = values
        file["brdf"].attrs["theta_i"] = 0.5
        file["theta_o"], file["phi_o"] = theta_o, phi_o
    layout = {
        "axes": {
            "theta_i": "/brdf@theta_i",
            "phi_i": 0.0,
            "theta_o": "/theta_o",
            "phi_o": "/phi_o",
        },
        "values": {"path": "/brdf", "axes": ["theta_o", "phi_o"]},
    }
    sites, read = hdf5.read_hdf5(path, layout)
    assert len(sites) == 2
    assert_allclose(read, values.ravel())
    grid = np.meshgrid(theta_o, phi_o, indexing="ij")
    theta_h, theta_d, phi_d = spherical_to_rusinkiewicz(0.5, grid[0].ravel(), grid[1].ravel())
    assert_allclose(sites[0], spherical_to_cartesian(theta_h, 0.0), atol=1e-15)
    assert_allclose(sites[1], spherical_to_cartesian(theta_d, phi_d), atol=1e-15)


def with_changes(axes=None, values=None, **other):
    return (
        LAYOUT
        | other
        | {"axes": LAYOUT["axes"] | (axes or {})}
        | ({"values": LAYOUT["values"] | values} if values else {})
    )


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (LAYOUT | {"unit": "degree"}, "unknown keys"),
        (LAYOUT | {"angle_unit": "gradian"}, "angle_unit must be one of"),
        (LAYOUT | {"axes": {"theta_i": 0.0}}, "the axes must be"),
        (with_changes(axes={"polarisation": "/@p"}), "the axes must be"),
        (LAYOUT | {"values": {"path": "/x"}}, "must have a path and axes"),
        (with_changes(values={"axes": ["phi_o", "phi_o"]}), "must be distinct axes"),
        (with_changes(values={"axes": ["phi_o", "phi_i"]}), "must be read from the file"),
        (with_changes(values={"axes": ["phi_o", "theta_i"]}), "cannot be an index"),
        (with_changes(axes={"theta_o": "/@missing"}), "is not an attribute or a dataset"),
        (with_changes(axes={"theta_o": "/incidence_0"}), "is not an attribute or a dataset"),
        (with_changes(axes={"theta_o": "/missing@zeniths"}), "is not an attribute or a dataset"),
        (with_changes(axes={"theta_o": "/incidence_{phi_d}@zeniths"}), "depends on 'phi_d'"),
        (with_changes(values={"axes": ["theta_o", "phi_o"]}), "has the shape"),
        (with_changes(axes={"phi_i": "/@azimuths"}), "neither an axis of the values"),
        (with_changes(axes={"theta_o": "/incidence_{theta_i}/band_0"}), "one-dimensional"),
    ],
)
def test_invalid_layouts(path, layout, message):
    with pytest.raises(ValueError, match=message):
        hdf5.read_hdf5(path, layout)


def test_layout_file_must_hold_an_object(path, tmp_path):
    layout = tmp_path / "layout.json"
    layout.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        hdf5.read_hdf5(path, layout)
