# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Radial basis function kernels on the Rusinkiewicz domain.

A kernel is a profile ``phi`` applied to the product distance ``r``
between a site and a centre, in which the distance on each factor of the
domain, the half vectors, the difference vectors and the wavelengths, is
divided by the width of the kernel on that factor:
``r = sqrt(sum_k (distance_k / width_k)^2)``. The widths make the kernel
anisotropic, and can be derived from the spacings of the sampling of the
centres given by :func:`rivoli.distances.nearest_neighbour_spacings`.

Interpolation requires the kernel to be positive definite: the matrix of
its values between any distinct centres must be positive definite. This
depends on the metric:

-   With the chordal metric, the hemispheres are seen as subsets of
    ``R^3``, and the product distance is a Euclidean distance between
    points of ``R^6``, or ``R^7`` with the wavelengths. A profile that
    is positive definite on that space, such as the Gaussian, then gives
    a positive definite kernel. On a sphere, the Gaussian of the chordal
    distance is also the von Mises-Fisher kernel, ``exp(2 cos(gamma) /
    width^2)`` up to a constant factor.
-   With the geodesic metric, the Gaussian is not positive definite on
    the sphere [1]_: its matrix can have negative eigenvalues when its
    widths are large compared to the spacing of the centres. It is
    kept as the kernel of earlier versions of RIVOLI.

The kernels of :mod:`rivoli.warp.kernels` compute the same matrices in
parallel, on the CPU or a GPU.

References
----------
.. [1] T. Gneiting, "Strictly and Non-Strictly Positive Definite Functions
    on Spheres", Bernoulli, vol. 19, no. 4, 2013, pp. 1327-1349.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from rivoli.distances import pairwise_distances

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "collocation_matrix",
    "gaussian",
]


def gaussian(r: "ArrayLike") -> "NDArray[np.float64]":
    """Evaluate the Gaussian profile, ``exp(-r^2)``.

    Parameters
    ----------
    r : array_like
        Product distances, divided by the widths of the kernel.

    Returns
    -------
    ndarray
        Values of the profile, 1 at a zero distance.
    """
    r = np.asarray(r, dtype=np.float64)
    return np.exp(-(r**2))


_PROFILES = {"gaussian": gaussian}


def _check_kernel(kernel: str) -> None:
    if kernel not in _PROFILES:
        raise ValueError(f"kernel must be one of {sorted(_PROFILES)}, got {kernel!r}")


def collocation_matrix(
    sites: "Sequence[ArrayLike]",
    centres: "Sequence[ArrayLike]",
    *,
    widths: tuple[float, ...],
    metric: Literal["geodesic", "chordal"],
    kernel: Literal["gaussian"] = "gaussian",
) -> "NDArray[np.float64]":
    """Evaluate the kernels of centres at sites.

    The entry ``(i, j)`` of the matrix is the value at the site ``i`` of
    the kernel of the centre ``j``. With the centres as sites, it is the
    symmetric interpolation matrix, and with other sites, such as
    measurements, the rectangular matrix of a least-squares fit.

    Parameters
    ----------
    sites : sequence of array_like
        ``(h, d)`` or, for a spectral BRDF, ``(h, d, wavelength)``: the
        half vectors, of shape ``(m, 3)``, the difference vectors, of
        shape ``(m, 3)``, and the wavelengths, of shape ``(m,)``, of the
        sites. For an isotropic BRDF, the half vectors have a zero
        azimuth.
    centres : sequence of array_like
        The same for the centres, with ``n`` instead of ``m``.
    widths : tuple of float
        Positive widths of the kernel on each factor, in the units of
        their distances.
    metric : {"geodesic", "chordal"}
        Distance between the half vectors and between the difference
        vectors. Only the chordal one guarantees a positive definite
        kernel.
    kernel : {"gaussian"}, default "gaussian"
        Profile of the kernel.

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
    r = pairwise_distances(sites, centres, metric=metric, scales=widths)
    return _PROFILES[kernel](r)
