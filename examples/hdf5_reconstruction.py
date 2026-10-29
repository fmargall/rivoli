# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Reconstruct the colour of spectral measurements of a BRDF, in three ways.

The measurements are read from an HDF5 file described by a layout, as
documented in :mod:`rivoli.hdf5`, and reconstructed as a MERL file of
three channels, red, green and blue, in one of three ways:

-   ``wavelengths``: each channel is the BRDF at a single wavelength, by
    default 645, 510 and 440 nm as in earlier versions of RIVOLI, or the
    closest measured ones, and is fitted by its own model;
-   ``rgb``: the measured spectra are converted to the linear RGB of sRGB
    before the fit, by :func:`rivoli.colour.spectrum_to_rgb`, and each
    channel is fitted by its own model;
-   ``spectral``: a single spectral model is fitted to the spectra, and
    its colour is computed afterwards, by :meth:`rivoli.model.RBFModel.rgb`.

A random tenth of the measured directions is kept out of the fit, to
measure the errors of the reconstruction where it was not fitted: on the
three channels of ``wavelengths``, and on the linear RGB of the
measurements for ``rgb`` and ``spectral``, which can be compared.

Usage::

    python examples/hdf5_reconstruction.py measurements.h5 layout.json --mode rgb
    python examples/hdf5_reconstruction.py measurements.h5 layout.json --mode spectral

Run it with ``--help`` for the other options.
"""

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rivoli.colour import spectral_matrix
from rivoli.distances import diameters, nearest_neighbour_spacings
from rivoli.geometry import spherical_to_cartesian
from rivoli.hdf5 import read_hdf5
from rivoli.merl import write_model
from rivoli.model import RBFModel, fit
from rivoli.sampling import super_fibonacci_isotropic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# Widths of the kernels on the half and difference vectors, as multiples of
# the mean spacings of the centres, as in examples/merl_reconstruction.py
WIDTH_FACTORS = (0.7, 12.0)
# Upper bound of the wavelengths that contribute to colours, in nm
VISIBLE = 830.0
# Offset of the values in the logarithmic error, in sr^-1
LOG_ERROR_OFFSET = 1e-3


class Spectra:
    """Measured spectra, one row per direction, NaN where missing."""

    def __init__(self, path: Path, layout: Path) -> None:
        (h, d, wavelength), values = read_hdf5(path, layout)
        # Directions measured several times, such as theta_o = 0 at several
        # azimuths, are merged by averaging their values
        directions, row = np.unique(np.round(np.c_[h, d], 10), axis=0, return_inverse=True)
        self.wavelengths, column = np.unique(wavelength, return_inverse=True)
        total = np.zeros((len(directions), len(self.wavelengths)))
        count = np.zeros_like(total)
        np.add.at(total, (row, column), values)
        np.add.at(count, (row, column), 1)
        self.values = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        self.h, self.d = directions[:, :3], directions[:, 3:]

    def rgb(self) -> "NDArray":
        """Linear RGB of the spectra, NaN where the visible spectrum is incomplete."""
        visible = self.wavelengths <= VISIBLE
        return self.values[:, visible] @ spectral_matrix(self.wavelengths[visible])

    def at(self, wavelengths: "Sequence[float]") -> tuple["NDArray", "NDArray"]:
        """Values at the closest measured wavelengths, and these wavelengths."""
        columns = [int(np.abs(self.wavelengths - w).argmin()) for w in wavelengths]
        return self.values[:, columns], self.wavelengths[columns]


def centres_of(count: int, wavelength_range: tuple[float, float] | None = None) -> tuple:
    """Reciprocal centres above the horizon, and the widths of their kernels."""
    if wavelength_range is None:
        theta_h, theta_d, phi_d = super_fibonacci_isotropic(
            count, above_horizon=True, symmetry="reciprocity"
        )
        wavelengths = ()
    else:
        theta_h, theta_d, phi_d, wavelength = super_fibonacci_isotropic(
            count, wavelength_range=wavelength_range, above_horizon=True, symmetry="reciprocity"
        )
        wavelengths = (wavelength,)
    centres = (spherical_to_cartesian(theta_h, 0.0), spherical_to_cartesian(theta_d, phi_d))
    spacings = nearest_neighbour_spacings(*centres, metric="chordal")
    widths = tuple(
        factor * float(np.mean(spacing))
        for factor, spacing in zip(WIDTH_FACTORS, spacings, strict=True)
    )
    return (*centres, *wavelengths), widths


def errors(reconstruction: "NDArray", reference: "NDArray") -> dict[str, "NDArray"]:
    """Relative and logarithmic errors per channel."""
    log_ratio = np.log(
        (np.maximum(reconstruction, 0) + LOG_ERROR_OFFSET) / (reference + LOG_ERROR_OFFSET)
    )
    return {
        "relative error": np.linalg.norm(reconstruction - reference, axis=0)
        / np.linalg.norm(reference, axis=0),
        "log RMS error": np.sqrt(np.mean(log_ratio**2, axis=0)),
    }


def reconstruct(
    spectra: Spectra,
    mode: str,
    train: "NDArray",
    arguments: argparse.Namespace,
) -> tuple[RBFModel, "NDArray | None"]:
    """Fit a model in a mode, and the wavelengths at which to evaluate it if spectral."""
    sites = (spectra.h[train], spectra.d[train])
    if mode in ("wavelengths", "rgb"):
        if mode == "rgb":
            values = spectra.rgb()[train]
        else:
            values, measured = spectra.at(arguments.wavelengths)
            values = values[train]
            print(f"  wavelengths of the channels: {np.round(measured, 1)} nm")
        measured_directions = ~np.isnan(values).any(axis=1)
        centres, widths = centres_of(arguments.centres)
        sites = tuple(x[measured_directions] for x in sites)
        values = np.maximum(values[measured_directions], 0.0)
        model = fit(
            sites,
            values,
            centres,
            widths=widths,
            metric="chordal",
            isotropic=True,
            regularization=arguments.regularization,
        )
        return model, None

    # One site per direction and wavelength, every step nm of the visible
    visible = spectra.wavelengths <= VISIBLE
    grid = spectra.wavelengths[visible]
    kept = np.unique(np.searchsorted(grid, np.arange(grid[0], grid[-1], arguments.step)))
    grid = grid[kept]
    values = spectra.values[train][:, visible][:, kept]
    measured = ~np.isnan(values)
    rows, columns = np.nonzero(measured)
    sites = (sites[0][rows], sites[1][rows], grid[columns])
    wavelength_range = (float(grid[0]), float(grid[-1]))
    centres, widths = centres_of(arguments.centres, wavelength_range)
    widths = (*widths, arguments.spectral_width)
    print(f"  {len(rows)} spectral sites, at {len(grid)} wavelengths")
    model = fit(
        sites,
        np.maximum(values[measured], 0.0),
        centres,
        widths=widths,
        metric="chordal",
        isotropic=True,
        regularization=arguments.regularization,
        wavelength_range=wavelength_range,
        scales=diameters(isotropic=True, metric="chordal", wavelength_range=wavelength_range),
    )
    return model, grid


def main(argv: "Sequence[str] | None" = None) -> None:
    """Run the example from command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("hdf5", type=Path, help="HDF5 file of the measurements")
    parser.add_argument("layout", type=Path, help="JSON layout of the file, see rivoli.hdf5")
    parser.add_argument(
        "--mode",
        choices=("wavelengths", "rgb", "spectral"),
        default="rgb",
        help="what the models reconstruct (default: %(default)s)",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs=3,
        default=(645.0, 510.0, 440.0),
        help="wavelengths of the red, green and blue channels in nm, for the "
        "wavelengths mode (default: %(default)s)",
    )
    parser.add_argument(
        "--centres",
        type=int,
        default=1000,
        help="number of centres of the kernels (default: %(default)s)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-3,
        help="weight of the Dirichlet energy, in [0, 1) (default: %(default)s)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=20.0,
        help="step between the wavelengths fitted in the spectral mode, in nm "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--spectral-width",
        type=float,
        default=120.0,
        help="width of the kernels along the wavelengths in nm, for the spectral "
        "mode (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, help="MERL file of the reconstruction")
    parser.add_argument(
        "--device", help="Warp device on which the model is evaluated, such as cuda:0"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the test directions")
    arguments = parser.parse_args(argv)

    spectra = Spectra(arguments.hdf5, arguments.layout)
    rng = np.random.default_rng(arguments.seed)
    test = rng.random(len(spectra.h)) < 0.1
    print(f"{len(spectra.h)} measured directions, {test.sum()} kept out of the fit")

    start = time.perf_counter()
    model, grid = reconstruct(spectra, arguments.mode, ~test, arguments)
    print(f"model fitted in {time.perf_counter() - start:.1f} s")

    sites = (spectra.h[test], spectra.d[test])
    if arguments.mode == "wavelengths":
        reference = spectra.at(arguments.wavelengths)[0][test]
        reconstruction = model(sites, device=arguments.device)
    else:
        reference = spectra.rgb()[test]
        if grid is None:
            reconstruction = model(sites, device=arguments.device)
        else:
            reconstruction = model.rgb(sites, grid, device=arguments.device)
    measured = ~np.isnan(reference).any(axis=1)
    target = "the channels" if arguments.mode == "wavelengths" else "the linear RGB"
    print(f"errors on {target} of the test directions:")
    for name, error in errors(reconstruction[measured], reference[measured]).items():
        print(f"  {name}: " + ", ".join(f"{e:.4f}" for e in error) + " (red, green, blue)")

    if arguments.output is not None:
        start = time.perf_counter()
        write_model(arguments.output, model, wavelengths=grid, device=arguments.device)
        print(f"{arguments.output} written in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
