# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Decimate a MERL BRDF, reconstruct it and compare the reconstruction with it.

The BRDF is read from a file of the MERL database, and only the bins of a
Super-Fibonacci sampling of the isotropic Rusinkiewicz domain are kept, a
few percent of the grid. A radial basis function model, regularized by
its Dirichlet energy, is fitted to them in log space, then compared with
the whole BRDF on random bins, and optionally written as a MERL file and
plotted.

Usage::

    python examples/merl_reconstruction.py blue-acrylic.binary
    python examples/merl_reconstruction.py blue-acrylic.binary --output out.binary
    python examples/merl_reconstruction.py blue-acrylic.binary --figure comparison.png

Run it with ``--help`` for the other options, such as the numbers of sites
and centres. Writing the reconstruction evaluates the model on the 1.1
million bins of the grid above the horizon, which takes a few minutes.
"""

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rivoli.distances import nearest_neighbour_spacings
from rivoli.geometry import spherical_to_cartesian
from rivoli.merl import SHAPE, bin_angles, bin_indices, read_merl, write_model
from rivoli.model import RBFModel, fit
from rivoli.parametrization import is_above_horizon
from rivoli.sampling import super_fibonacci_isotropic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# Widths of the kernels on the half and difference vectors, as multiples of
# the mean spacings of the centres, those of earlier versions of RIVOLI: the
# BRDFs vary much faster with the half vector, around the specular peak,
# than with the difference vector
WIDTH_FACTORS = (0.7, 12.0)
# Offset of the values in the logarithmic error, in sr^-1, below which
# differences are not significant
LOG_ERROR_OFFSET = 1e-3


def sites_of(theta_h: "NDArray", theta_d: "NDArray", phi_d: "NDArray") -> tuple["NDArray", ...]:
    """Half and difference vectors of an isotropic BRDF."""
    return spherical_to_cartesian(theta_h, 0.0), spherical_to_cartesian(theta_d, phi_d)


def decimate(values: "NDArray", count: int) -> tuple[tuple["NDArray", ...], "NDArray"]:
    """Keep the measured bins of a Super-Fibonacci sampling of the domain.

    The bins are taken at their angles in the grid, with ``phi_d`` in
    ``[0, pi)``, the only half of the azimuths that MERL stores.
    """
    angles = bin_angles()
    measured = is_above_horizon(*angles) & np.all(values >= 0, axis=-1)
    bins = np.ravel_multi_index(
        bin_indices(*super_fibonacci_isotropic(count, above_horizon=True)), SHAPE
    )
    bins = np.unique(bins)
    bins = bins[measured.ravel()[bins]]
    sites = sites_of(*(angle.ravel()[bins] for angle in angles))
    return sites, values.reshape(-1, 3)[bins]


def reconstruct(
    sites: "Sequence[NDArray]", values: "NDArray", centres: int, regularization: float
) -> RBFModel:
    """Fit a model to the measurements, on centres above the horizon."""
    # Reciprocal kernels: centres over the fundamental domain, phi_d in (0, pi]
    centre_sites = sites_of(
        *super_fibonacci_isotropic(centres, above_horizon=True, symmetry="reciprocity")
    )
    spacings = nearest_neighbour_spacings(*centre_sites, metric="chordal")
    widths = tuple(
        factor * float(np.mean(spacing))
        for factor, spacing in zip(WIDTH_FACTORS, spacings, strict=True)
    )
    return fit(
        sites,
        values,
        centre_sites,
        widths=widths,
        metric="chordal",
        isotropic=True,
        regularization=regularization,
    )


def compare(
    model: RBFModel, values: "NDArray", count: int, rng: np.random.Generator
) -> dict[str, "NDArray"]:
    """Errors of the model on random measured bins, per channel."""
    angles = bin_angles()
    measured = is_above_horizon(*angles) & np.all(values >= 0, axis=-1)
    bins = rng.choice(np.flatnonzero(measured), size=min(count, measured.sum()), replace=False)
    reference = values.reshape(-1, 3)[bins]
    reconstruction = model(sites_of(*(angle.ravel()[bins] for angle in angles)))
    log_ratio = np.log((reconstruction + LOG_ERROR_OFFSET) / (reference + LOG_ERROR_OFFSET))
    return {
        "relative error": np.linalg.norm(reconstruction - reference, axis=0)
        / np.linalg.norm(reference, axis=0),
        "log RMS error": np.sqrt(np.mean(log_ratio**2, axis=0)),
    }


def plot(model: RBFModel, values: "NDArray", path: Path) -> None:
    """Plot the BRDF and its reconstruction against theta_h, in the incidence plane."""
    # Imported here, as matplotlib is only needed for the figure
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    theta_h, theta_d, phi_d = bin_angles()
    # phi_d = 90 degrees: the half vector moves in the incidence plane
    k = 90
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for j, ax in zip((10, 40, 70), axes, strict=True):
        above = is_above_horizon(theta_h[:, j, k], theta_d[:, j, k], phi_d[:, j, k])
        angles = np.degrees(theta_h[above, j, k])
        reconstruction = model(
            sites_of(theta_h[above, j, k], theta_d[above, j, k], phi_d[above, j, k])
        )
        for c, colour in enumerate(("red", "green", "blue")):
            ax.plot(angles, values[above, j, k, c], color=colour, label=f"{colour} measured")
            ax.plot(
                angles,
                reconstruction[:, c],
                color=colour,
                linestyle="--",
                label=f"{colour} reconstructed",
            )
        ax.set(
            title=f"theta_d = {np.degrees(theta_d[0, j, 0]):.0f} deg",
            xlabel="theta_h (deg)",
            yscale="log",
        )
    axes[0].set_ylabel("BRDF (1/sr)")
    axes[0].legend(fontsize="small")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: "Sequence[str] | None" = None) -> None:
    """Run the example from command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("merl", type=Path, help="MERL file of the BRDF")
    parser.add_argument(
        "--sites",
        type=int,
        default=20000,
        help="number of Super-Fibonacci sites whose bins are kept (default: %(default)s)",
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
        default=1e-5,
        help="weight of the Dirichlet energy, in [0, 1) (default: %(default)s)",
    )
    parser.add_argument(
        "--test-bins",
        type=int,
        default=100000,
        help="number of random bins on which errors are measured (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, help="MERL file of the reconstruction")
    parser.add_argument("--figure", type=Path, help="image comparing slices, needs matplotlib")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random test bins")
    arguments = parser.parse_args(argv)

    values = read_merl(arguments.merl)
    sites, measurements = decimate(values, arguments.sites)
    print(f"{len(measurements)} measured bins kept, {values[..., 0].size} in the grid")

    start = time.perf_counter()
    model = reconstruct(sites, measurements, arguments.centres, arguments.regularization)
    print(f"model of {arguments.centres} centres fitted in {time.perf_counter() - start:.1f} s")

    errors = compare(model, values, arguments.test_bins, np.random.default_rng(arguments.seed))
    for name, error in errors.items():
        print(f"{name}: " + ", ".join(f"{e:.4f}" for e in error) + " (red, green, blue)")

    if arguments.output is not None:
        start = time.perf_counter()
        write_model(arguments.output, model)
        print(f"{arguments.output} written in {time.perf_counter() - start:.1f} s")
    if arguments.figure is not None:
        plot(model, values, arguments.figure)
        print(f"{arguments.figure} written")


if __name__ == "__main__":
    main()
