# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

# The bodies of Warp functions and kernels are in the kernel language of Warp,
# which Warp type checks when compiling them: its stubs give no operators nor
# indexing to its vector types, and its array annotations are calls
# pyright: reportOperatorIssue=false, reportIndexIssue=false, reportInvalidTypeForm=false

"""Distances between directions, and spacings of samplings, with Warp.

This is the parallel counterpart of :mod:`rivoli.distances`, which
documents the quantities. The Warp functions :func:`geodesic` and
:func:`chordal` can be called from other kernels. The computations are
in double precision, as in NumPy.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from rivoli.distances import _check_metric, _check_scales

try:
    import warp as wp
except ModuleNotFoundError as error:  # pragma: no cover
    # Tests always run with Warp installed
    raise ModuleNotFoundError(
        "rivoli.warp requires Warp, installed by `pip install rivoli[warp]`"
    ) from error

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "chordal",
    "geodesic",
    "nearest_neighbour_spacings",
]

_METRICS = {"geodesic": 0, "chordal": 1}
_GEODESIC = wp.constant(_METRICS["geodesic"])


@wp.func
def geodesic(u: wp.vec3d, v: wp.vec3d) -> wp.float64:
    """Compute the geodesic distance between unit vectors."""
    return wp.float64(2.0) * wp.atan2(wp.length(u - v), wp.length(u + v))


@wp.func
def chordal(u: wp.vec3d, v: wp.vec3d) -> wp.float64:
    """Compute the chordal distance between unit vectors."""
    return wp.length(u - v)


@wp.func
def _distance(u: wp.vec3d, v: wp.vec3d, metric: int) -> wp.float64:
    if metric == _GEODESIC:
        return geodesic(u, v)
    return chordal(u, v)


@wp.kernel
def _nearest_neighbours(
    h: wp.array(dtype=wp.vec3d),
    d: wp.array(dtype=wp.vec3d),
    wavelength: wp.array(dtype=wp.float64),
    metric: int,
    scale: wp.vec3d,
    spacing_h: wp.array(dtype=wp.float64),
    spacing_d: wp.array(dtype=wp.float64),
    spacing_wavelength: wp.array(dtype=wp.float64),
) -> None:
    i = wp.tid()
    spectral = wavelength.shape[0] > 0
    best = wp.float64(wp.inf)
    best_h = wp.float64(0.0)
    best_d = wp.float64(0.0)
    best_wavelength = wp.float64(0.0)
    # The first nearest neighbour wins ties, as with numpy.argmin
    for j in range(h.shape[0]):
        if j != i:
            distance_h = _distance(h[i], h[j], metric) / scale[0]
            distance_d = _distance(d[i], d[j], metric) / scale[1]
            distance_wavelength = wp.float64(0.0)
            if spectral:
                distance_wavelength = wp.abs(wavelength[i] - wavelength[j]) / scale[2]
            squared = (
                distance_h * distance_h
                + distance_d * distance_d
                + distance_wavelength * distance_wavelength
            )
            if squared < best:
                best = squared
                best_h = distance_h
                best_d = distance_d
                best_wavelength = distance_wavelength
    spacing_h[i] = best_h
    spacing_d[i] = best_d
    if spectral:
        spacing_wavelength[i] = best_wavelength


def nearest_neighbour_spacings(
    h: "ArrayLike",
    d: "ArrayLike",
    wavelength: "ArrayLike | None" = None,
    *,
    metric: Literal["geodesic", "chordal"] = "geodesic",
    scales: tuple[float, ...] | None = None,
    device: str | None = None,
) -> tuple["NDArray[np.float64]", ...]:
    """Measure how far each site of a sampling is from its nearest neighbour.

    See :func:`rivoli.distances.nearest_neighbour_spacings`, which this
    function computes with one thread per site. The computation is
    exhaustive, in O(n^2) time, but in O(n) memory.

    Parameters
    ----------
    h : array_like, shape (n, 3)
        Half vectors of the sites.
    d : array_like, shape (n, 3)
        Difference vectors of the sites.
    wavelength : array_like, shape (n,), optional
        Wavelengths of the sites, in a unit commensurate with angles or
        given a scale.
    metric : {"geodesic", "chordal"}, default "geodesic"
        Distance between the half vectors and between the difference
        vectors.
    scales : tuple of float, optional
        Positive scales of the distances between the half vectors, the
        difference vectors and, with `wavelength`, the wavelengths.
        Default: 1 for each.
    device : str, optional
        Warp device, such as ``"cpu"`` or ``"cuda:0"``. Default: the
        default device of Warp.

    Returns
    -------
    spacing_h, spacing_d : ndarray, shape (n,)
        Distances between the half vectors and between the difference
        vectors of each site and of its nearest neighbour, divided by
        their scales.
    spacing_wavelength : ndarray, shape (n,)
        Distances between their wavelengths, divided by its scale, only
        when `wavelength` is given.

    Raises
    ------
    ValueError
        If there are fewer than 2 sites, if `metric` is unknown, or if
        `scales` are not one positive number per factor.
    """
    _check_metric(metric)
    h = np.ascontiguousarray(h, dtype=np.float64)
    d = np.ascontiguousarray(d, dtype=np.float64)
    n = len(h)
    if n < 2:
        raise ValueError(f"the number of sites must be at least 2, got {n}")
    spectral = wavelength is not None
    wavelength = np.ascontiguousarray(wavelength if spectral else np.empty(0), dtype=np.float64)
    # Without wavelengths, their scale is unused
    scale = wp.vec3d(*_check_scales(scales, 3 if spectral else 2), *(() if spectral else (1.0,)))

    # Without wavelengths, the kernel writes no spacing of wavelengths
    spacings = [
        # Warp's stubs type dtype as a Python type, not as one of its scalar types
        wp.empty(size, dtype=wp.float64, device=device)  # pyright: ignore[reportArgumentType]
        for size in (n, n, n if spectral else 0)
    ]
    wp.launch(
        _nearest_neighbours,
        dim=n,
        inputs=[
            wp.array(h, dtype=wp.vec3d, device=device),
            wp.array(d, dtype=wp.vec3d, device=device),
            wp.array(wavelength, dtype=wp.float64, device=device),
            _METRICS[metric],
            scale,
        ],
        outputs=spacings,
        device=device,
    )
    return tuple(
        np.asarray(spacing.numpy(), dtype=np.float64)
        for spacing in spacings[: 3 if spectral else 2]
    )
