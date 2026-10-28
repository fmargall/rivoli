# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

# The bodies of Warp functions and kernels are in the kernel language of Warp,
# which Warp type checks when compiling them: its stubs give no operators nor
# indexing to its vector types, and its array annotations are calls
# pyright: reportOperatorIssue=false, reportIndexIssue=false, reportInvalidTypeForm=false

"""Radial basis function kernels on the Rusinkiewicz domain, with Warp.

This is the parallel counterpart of :mod:`rivoli.kernels`, which
documents the kernels. The Warp function :func:`gaussian` can be called
from other kernels. The computations are in double precision, as in
NumPy.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from rivoli.distances import _check_metric, _check_scales, _factors
from rivoli.kernels import _check_kernel

try:
    import warp as wp
except ModuleNotFoundError as error:  # pragma: no cover
    # Tests always run with Warp installed
    raise ModuleNotFoundError(
        "rivoli.warp requires Warp, installed by `pip install rivoli[warp]`"
    ) from error

from rivoli.warp.distances import _METRICS, _distance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "collocation_matrix",
    "gaussian",
]


@wp.func
def gaussian(r: wp.float64) -> wp.float64:
    """Evaluate the Gaussian profile, ``exp(-r^2)``."""
    return wp.exp(-r * r)


@wp.kernel
def _collocation(
    sites_h: wp.array(dtype=wp.vec3d),
    sites_d: wp.array(dtype=wp.vec3d),
    sites_wavelength: wp.array(dtype=wp.float64),
    centres_h: wp.array(dtype=wp.vec3d),
    centres_d: wp.array(dtype=wp.vec3d),
    centres_wavelength: wp.array(dtype=wp.float64),
    metric: int,
    width: wp.vec3d,
    matrix: wp.array2d(dtype=wp.float64),
) -> None:
    # Warp's stubs do not tell that the thread index of a 2D launch is a pair
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    distance_h = _distance(sites_h[i], centres_h[j], metric) / width[0]
    distance_d = _distance(sites_d[i], centres_d[j], metric) / width[1]
    squared = distance_h * distance_h + distance_d * distance_d
    if sites_wavelength.shape[0] > 0:
        distance_wavelength = wp.abs(sites_wavelength[i] - centres_wavelength[j]) / width[2]
        squared += distance_wavelength * distance_wavelength
    matrix[i, j] = gaussian(wp.sqrt(squared))


def collocation_matrix(
    sites: "Sequence[ArrayLike]",
    centres: "Sequence[ArrayLike]",
    *,
    widths: tuple[float, ...],
    metric: Literal["geodesic", "chordal"],
    kernel: Literal["gaussian"] = "gaussian",
    device: str | None = None,
) -> "NDArray[np.float64]":
    """Evaluate the kernels of centres at sites.

    See :func:`rivoli.kernels.collocation_matrix`, which this function
    computes with one thread per entry of the matrix.

    Parameters
    ----------
    sites : sequence of array_like
        ``(h, d)`` or ``(h, d, wavelength)`` of the sites.
    centres : sequence of array_like
        ``(h, d)`` or ``(h, d, wavelength)`` of the centres.
    widths : tuple of float
        Positive widths of the kernel on each factor.
    metric : {"geodesic", "chordal"}
        Distance between the half vectors and between the difference
        vectors.
    kernel : {"gaussian"}, default "gaussian"
        Profile of the kernel.
    device : str, optional
        Warp device, such as ``"cpu"`` or ``"cuda:0"``. Default: the
        default device of Warp.

    Returns
    -------
    ndarray, shape (m, n)
        Values of the kernels of the centres at the sites.

    Raises
    ------
    ValueError
        If `kernel` or `metric` is unknown, if `sites` and `centres` do
        not both have 2 or 3 factors, or if `widths` are not one positive
        number per factor.
    """
    _check_kernel(kernel)
    _check_metric(metric)
    sites = _factors(sites, "sites")
    centres = _factors(centres, "centres")
    if len(sites) != len(centres):
        raise ValueError(
            f"sites and centres must have the same factors, got {len(sites)} and {len(centres)}"
        )
    spectral = len(sites) == 3
    # Without wavelengths, their width is unused
    width = wp.vec3d(*_check_scales(widths, len(sites)), *(() if spectral else (1.0,)))

    def arrays(factors: list["NDArray[np.float64]"]) -> list[wp.array]:
        wavelength = factors[2] if spectral else np.empty(0)
        return [
            wp.array(np.ascontiguousarray(factors[0]), dtype=wp.vec3d, device=device),
            wp.array(np.ascontiguousarray(factors[1]), dtype=wp.vec3d, device=device),
            wp.array(np.ascontiguousarray(wavelength), dtype=wp.float64, device=device),
        ]

    shape = (len(sites[0]), len(centres[0]))
    # Warp's stubs type dtype as a Python type, not as one of its scalar types
    matrix = wp.empty(shape, dtype=wp.float64, device=device)  # pyright: ignore[reportArgumentType]
    wp.launch(
        _collocation,
        dim=shape,
        inputs=[*arrays(sites), *arrays(centres), _METRICS[metric], width],
        outputs=[matrix],
        device=device,
    )
    return np.asarray(matrix.numpy(), dtype=np.float64)
