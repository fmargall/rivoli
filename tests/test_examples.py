# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of the scripts of examples/, run on small synthetic data."""

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
            "--centres", "150",
            "--test-bins", "5000",
            "--output", str(output),
            "--figure", str(figure),
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
