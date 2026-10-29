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

:func:`expansion` evaluates expansions on the kernels, such as the
models of :mod:`rivoli.model`, without the collocation matrix, which
would not fit in memory for the million sites of a MERL grid: each
thread accumulates the kernels of all the centres at its site.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from rivoli.distances import _check_metric, _check_scales, _factors
from rivoli.kernels import _check_kernel, _check_symmetry

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
    "expansion",
    "gaussian",
]

# Pairs of a site and a centre evaluated by one launch of a kernel: a launch
# that runs for more than about 2 seconds on a GPU that also drives a
# display is stopped by the watchdog of Windows
_PAIRS_PER_LAUNCH = 2**26


@wp.func
def gaussian(r: wp.float64) -> wp.float64:
    """Evaluate the Gaussian profile, ``exp(-r^2)``."""
    return wp.exp(-r * r)


@wp.func
def _kernel(
    sites_h: wp.array(dtype=wp.vec3d),
    sites_d: wp.array(dtype=wp.vec3d),
    sites_wavelength: wp.array(dtype=wp.float64),
    centres_h: wp.array(dtype=wp.vec3d),
    centres_d: wp.array(dtype=wp.vec3d),
    centres_wavelength: wp.array(dtype=wp.float64),
    i: int,
    j: int,
    metric: int,
    width: wp.vec3d,
    reciprocal: int,
) -> wp.float64:
    """Value at the site ``i`` of the kernel of the centre ``j``."""
    distance_h = _distance(sites_h[i], centres_h[j], metric) / width[0]
    squared = distance_h * distance_h
    if sites_wavelength.shape[0] > 0:
        distance_wavelength = wp.abs(sites_wavelength[i] - centres_wavelength[j]) / width[2]
        squared += distance_wavelength * distance_wavelength
    centre_d = centres_d[j]
    value = wp.float64(0.0)
    # The centre, then its image by reciprocity, a half turn of d
    for image in range(1 + reciprocal):
        if image == 1:
            centre_d = wp.vec3d(-centre_d[0], -centre_d[1], centre_d[2])
        distance_d = _distance(sites_d[i], centre_d, metric) / width[1]
        value += gaussian(wp.sqrt(squared + distance_d * distance_d))
    return value


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
    reciprocal: int,
    matrix: wp.array2d(dtype=wp.float64),
) -> None:
    # Warp's stubs do not tell that the thread index of a 2D launch is a pair
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    matrix[i, j] = _kernel(
        sites_h,
        sites_d,
        sites_wavelength,
        centres_h,
        centres_d,
        centres_wavelength,
        i,
        j,
        metric,
        width,
        reciprocal,
    )


@wp.kernel
def _expansion(
    sites_h: wp.array(dtype=wp.vec3d),
    sites_d: wp.array(dtype=wp.vec3d),
    sites_wavelength: wp.array(dtype=wp.float64),
    centres_h: wp.array(dtype=wp.vec3d),
    centres_d: wp.array(dtype=wp.vec3d),
    centres_wavelength: wp.array(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float64),
    metric: int,
    width: wp.vec3d,
    reciprocal: int,
    values: wp.array2d(dtype=wp.float64),
) -> None:
    # Warp's stubs do not tell that the thread index of a 1D launch is an int
    i = int(wp.tid())  # pyright: ignore[reportArgumentType]
    # Each thread writes the row of its site only, initialized to 0
    for j in range(centres_h.shape[0]):
        value = _kernel(
            sites_h,
            sites_d,
            sites_wavelength,
            centres_h,
            centres_d,
            centres_wavelength,
            i,
            j,
            metric,
            width,
            reciprocal,
        )
        for c in range(weights.shape[1]):
            values[i, c] += value * weights[j, c]


def _arguments(
    sites: "Sequence[ArrayLike]",
    centres: "Sequence[ArrayLike]",
    widths: tuple[float, ...],
    metric: str,
    kernel: str,
    symmetry: str,
) -> tuple[list["NDArray[np.float64]"], list["NDArray[np.float64]"], wp.vec3d]:
    """Check the arguments of the kernels: factors of sites and centres, and widths."""
    _check_kernel(kernel)
    _check_metric(metric)
    _check_symmetry(symmetry)
    sites = _factors(sites, "sites")
    centres = _factors(centres, "centres")
    if len(sites) != len(centres):
        raise ValueError(
            f"sites and centres must have the same factors, got {len(sites)} and {len(centres)}"
        )
    # Without wavelengths, their width is unused
    padding = (1.0,) if len(sites) == 2 else ()
    return sites, centres, wp.vec3d(*_check_scales(widths, len(sites)), *padding)


def _arrays(factors: list["NDArray[np.float64]"], device: str | None) -> list[wp.array]:
    """Device arrays of the factors, with no wavelengths when there are none."""
    wavelength = factors[2] if len(factors) == 3 else np.empty(0)
    return [
        wp.array(np.ascontiguousarray(factors[0]), dtype=wp.vec3d, device=device),
        wp.array(np.ascontiguousarray(factors[1]), dtype=wp.vec3d, device=device),
        wp.array(np.ascontiguousarray(wavelength), dtype=wp.float64, device=device),
    ]


def collocation_matrix(
    sites: "Sequence[ArrayLike]",
    centres: "Sequence[ArrayLike]",
    *,
    widths: tuple[float, ...],
    metric: Literal["geodesic", "chordal"],
    kernel: Literal["gaussian"] = "gaussian",
    symmetry: Literal["none", "reciprocity"] = "reciprocity",
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
    symmetry : {"reciprocity", "none"}, default "reciprocity"
        Symmetry of the BRDF, over whose orbit the kernels are summed.
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
        If `kernel`, `metric` or `symmetry` is unknown, if `sites` and `centres` do
        not both have 2 or 3 factors, or if `widths` are not one positive
        number per factor.
    """
    sites, centres, width = _arguments(sites, centres, widths, metric, kernel, symmetry)
    shape = (len(sites[0]), len(centres[0]))
    # Warp's stubs type dtype as a Python type, not as one of its scalar types
    matrix = wp.empty(shape, dtype=wp.float64, device=device)  # pyright: ignore[reportArgumentType]
    wp.launch(
        _collocation,
        dim=shape,
        inputs=[
            *_arrays(sites, device),
            *_arrays(centres, device),
            _METRICS[metric],
            width,
            int(symmetry == "reciprocity"),
        ],
        outputs=[matrix],
        device=device,
    )
    return np.asarray(matrix.numpy(), dtype=np.float64)


def expansion(
    sites: "Sequence[ArrayLike]",
    centres: "Sequence[ArrayLike]",
    weights: "ArrayLike",
    *,
    widths: tuple[float, ...],
    metric: Literal["geodesic", "chordal"],
    kernel: Literal["gaussian"] = "gaussian",
    symmetry: Literal["none", "reciprocity"] = "reciprocity",
    device: str | None = None,
) -> "NDArray[np.float64]":
    """Evaluate expansions on the kernels of centres at sites.

    This is the product of the collocation matrix of
    :func:`collocation_matrix` with the weights, computed with one thread
    per site, without the matrix.

    Parameters
    ----------
    sites : sequence of array_like
        ``(h, d)`` or ``(h, d, wavelength)`` of the ``m`` sites.
    centres : sequence of array_like
        ``(h, d)`` or ``(h, d, wavelength)`` of the ``n`` centres.
    weights : array_like, shape (n,) or (n, c)
        Weights of the kernels, with a column per expansion.
    widths : tuple of float
        Positive widths of the kernel on each factor.
    metric : {"geodesic", "chordal"}
        Distance between the half vectors and between the difference
        vectors.
    kernel : {"gaussian"}, default "gaussian"
        Profile of the kernel.
    symmetry : {"reciprocity", "none"}, default "reciprocity"
        Symmetry of the BRDF, over whose orbit the kernels are summed.
    device : str, optional
        Warp device, such as ``"cpu"`` or ``"cuda:0"``. Default: the
        default device of Warp.

    Returns
    -------
    ndarray, shape (m,) or (m, c)
        Values of the expansions at the sites.

    Raises
    ------
    ValueError
        As :func:`collocation_matrix`, or if `weights` are not one row
        per centre.
    """
    sites, centres, width = _arguments(sites, centres, widths, metric, kernel, symmetry)
    weights = np.asarray(weights, dtype=np.float64)
    n = len(centres[0])
    if weights.shape[:1] != (n,) or weights.ndim > 2:
        raise ValueError(f"expected weights of shape ({n},) or ({n}, c), got {weights.shape}")
    columns = weights.reshape(n, -1)
    inputs = [
        *_arrays(centres, device),
        wp.array(np.ascontiguousarray(columns), dtype=wp.float64, device=device),
        _METRICS[metric],
        width,
        int(symmetry == "reciprocity"),
    ]
    m = len(sites[0])
    values = np.empty((m, columns.shape[1]))
    step = max(1, _PAIRS_PER_LAUNCH // max(n, 1))
    for start in range(0, m, step):
        chunk = [factor[start : start + step] for factor in sites]
        count = len(chunk[0])
        output = wp.zeros((count, columns.shape[1]), dtype=wp.float64, device=device)
        wp.launch(
            _expansion,
            dim=count,
            inputs=[*_arrays(chunk, device), *inputs],
            outputs=[output],
            device=device,
        )
        values[start : start + count] = output.numpy()
    return values.reshape(m, *weights.shape[1:])
