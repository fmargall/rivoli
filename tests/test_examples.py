# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of the scripts of examples/, run on small synthetic data."""

import json
import re
import runpy
from pathlib import Path

import numpy as np
import pytest

from rivoli.merl import bin_angles, read_merl, write_merl
from rivoli.parametrization import is_above_horizon

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture
def merl_file(tmp_path):
    """A MERL file of a smooth BRDF, a diffuse part and a wide glossy lobe."""
    theta_h, theta_d, _ = bin_angles()
    lobe = np.exp(-((theta_h / 0.4) ** 2)) * np.cos(theta_d)
    values = np.stack([0.1 + lobe, 0.05 + 2 * lobe, 0.02 + 0.5 * lobe], axis=-1)
    values[~is_above_horizon(*bin_angles())] = -1.0
    path = tmp_path / "material.binary"
    write_merl(path, values)
    return path


def test_merl_reconstruction(merl_file, tmp_path, capsys):
    main = runpy.run_path(str(EXAMPLES / "merl_reconstruction.py"))["main"]
    output = tmp_path / "reconstruction.binary"
    figure = tmp_path / "comparison.png"
    main(
        [
            str(merl_file),
            "--sites", "2000",
            "--centres", "300",
            "--test-bins", "5000",
            "--output", str(output),
            "--figure", str(figure),
            "--device", "cpu",
        ]
    )  # fmt: skip
    printed = capsys.readouterr().out
    errors = [float(e) for e in re.findall(r"error: (\S+), (\S+), (\S+) ", printed)[0]]
    assert max(errors) < 0.05
    reconstruction, original = read_merl(output), read_merl(merl_file)
    above = is_above_horizon(*bin_angles())
    assert np.all(reconstruction[~above] == -1.0)
    # Up to the large relative errors of the few grazing bins
    difference = np.linalg.norm(reconstruction[above] - original[above])
    assert difference < 0.05 * np.linalg.norm(original[above])
    assert figure.stat().st_size > 0


LAYOUT = {
    "angle_unit": "degree",
    "axes": {
        "theta_i": "/@incidences",
        "phi_i": 180.0,
        "theta_o": "/@zeniths",
        "phi_o": "/@azimuths",
        "wavelength": "/@wavelengths",
    },
    "values": {"path": "/incidence_{theta_i}/band_{wavelength}", "axes": ["phi_o", "theta_o"]},
}


@pytest.fixture
def hdf5_files(tmp_path):
    """A file of spectral measurements of a smooth BRDF, and its layout."""
    h5py = pytest.importorskip("h5py")
    incidences = np.arange(0.0, 75.0, 10.0)
    zeniths = np.arange(0.0, 81.0, 8.0)
    azimuths = np.arange(0.0, 360.0, 20.0)
    wavelengths = np.arange(400.0, 1000.0, 10.0)
    path = tmp_path / "measurements.h5"
    with h5py.File(path, "w") as file:
        file.attrs["incidences"] = incidences
        file.attrs["zeniths"] = zeniths
        file.attrs["azimuths"] = azimuths
        file.attrs["wavelengths"] = wavelengths
        phi_o, theta_o = np.radians(np.meshgrid(azimuths, zeniths, indexing="ij"))
        for i, theta_i in enumerate(np.radians(incidences)):
            group = file.create_group(f"incidence_{i}")
            # Cosine of the angle between the outgoing and the mirror directions
            mirror = np.cos(theta_i) * np.cos(theta_o) + np.sin(theta_i) * np.sin(theta_o) * np.cos(
                phi_o
            )
            for k, wavelength in enumerate(wavelengths):
                colour = 1 + 0.5 * np.sin(wavelength / 80)
                group[f"band_{k}"] = colour * (0.1 + 0.5 * np.exp(4 * (mirror - 1)))
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps(LAYOUT), encoding="utf-8")
    return path, layout


@pytest.mark.parametrize("mode", ["wavelengths", "rgb", "spectral"])
def test_hdf5_reconstruction(hdf5_files, tmp_path, capsys, mode):
    main = runpy.run_path(str(EXAMPLES / "hdf5_reconstruction.py"))["main"]
    output = tmp_path / "reconstruction.binary"
    centres = "400" if mode == "spectral" else "200"
    main(
        [
            *map(str, hdf5_files),
            "--mode", mode,
            "--centres", centres,
            "--step", "50",
            "--output", str(output),
            "--device", "cpu",
        ]
    )  # fmt: skip
    printed = capsys.readouterr().out
    errors = [float(e) for e in re.findall(r"error: (\S+), (\S+), (\S+) ", printed)[0]]
    assert max(errors) < 0.1
    values = read_merl(output)
    above = is_above_horizon(*bin_angles())
    assert np.all(values[~above] == -1.0)
    assert np.all(values[above] >= 0.0)
