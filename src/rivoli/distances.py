# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Distances between directions, and spacings of samplings.

Two distances between unit vectors are provided: the geodesic distance,
the angle between them, which is the length of the great-circle arc that
joins them, and the chordal distance, the length of the straight segment
that joins them. For an angle ``gamma`` between the vectors, the chordal
distance is ``2 sin(gamma / 2)``: both agree for close vectors, and the
chordal one is shorter for distant ones.

The domains of the Rusinkiewicz parametrization are products, whose
factors are the hemispheres of the half vector ``h`` and of the
difference vector ``d``, and possibly an interval of wavelengths. The
distance between two sites is then the Euclidean norm of the distances
on each factor. For an isotropic BRDF, the half vector is taken with a
zero azimuth, ``spherical_to_cartesian(theta_h, 0)``, and its geodesic
distance is ``|delta theta_h|``.

The factors are in different units, angles or lengths of chords, and
wavelengths, so their distances can be divided by scales before being
combined. A natural choice of scales is the diameters of the factors,
the largest distances on them, given by :func:`diameters`: the scaled
distances are then all in ``[0, 1]``. The diameters are those of the
whole factors, ignoring the symmetries of the BRDF and the horizon.

Geodesic and chordal distances are computed from ``|u - v|`` and
``|u + v|``, which stay accurate for close vectors, where ``arccos`` of
the dot product loses half of the significant digits.

The kernels of :mod:`rivoli.warp.distances` compute the same quantities
in parallel, on the CPU or a GPU.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "chordal_distance",
    "diameters",
    "geodesic_distance",
    "nearest_neighbour_spacings",
    "pairwise_distances",
]

# Number of pairs of sites whose distances are held in memory at once by
# nearest_neighbour_spacings and pairwise_distances, about 24 MB for the
# differences of vectors
_PAIRS_PER_CHUNK = 2**20


def geodesic_distance(u: "ArrayLike", v: "ArrayLike") -> "NDArray[np.float64]":
    """Compute the geodesic distance between unit vectors.

    Parameters
    ----------
    u, v : array_like, shape (..., 3)
        Unit vectors, broadcast together.

    Returns
    -------
    ndarray, shape (...)
        Angles between `u` and `v`, in ``[0, pi]``.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    # The half angle is the angle of the right triangle whose legs are the
    # half chord |u - v| / 2 and the distance |u + v| / 2 to the midpoint
    return 2.0 * np.arctan2(np.linalg.norm(u - v, axis=-1), np.linalg.norm(u + v, axis=-1))


def chordal_distance(u: "ArrayLike", v: "ArrayLike") -> "NDArray[np.float64]":
    """Compute the chordal distance between unit vectors.

    Parameters
    ----------
    u, v : array_like, shape (..., 3)
        Unit vectors, broadcast together.

    Returns
    -------
    ndarray, shape (...)
        Lengths of the chords between `u` and `v`, in ``[0, 2]``.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return np.linalg.norm(u - v, axis=-1)


def diameters(
    *,
    isotropic: bool,
    metric: Literal["geodesic", "chordal"] = "geodesic",
    wavelength_range: tuple[float, float] | None = None,
) -> tuple[float, ...]:
    """Give the diameters of the factors of a Rusinkiewicz domain.

    The diameter of a factor is the largest distance between two of its
    points. Two opposite points of the equator are the farthest apart on
    a hemisphere, at a geodesic distance of ``pi``, or a chordal one of
    2. For an isotropic BRDF, the half vector only spans a quarter of a
    meridian, from the normal to the equator, whose ends are at a
    geodesic distance of ``pi / 2``, or a chordal one of ``sqrt(2)``.

    Parameters
    ----------
    isotropic : bool
        Whether the half vectors have a zero azimuth.
    metric : {"geodesic", "chordal"}, default "geodesic"
        Distance between the half vectors and between the difference
        vectors.
    wavelength_range : tuple of float, optional
        Lower and upper bounds of the wavelengths, for a spectral BRDF.

    Returns
    -------
    tuple of float
        Diameters of the factors of the half vectors, of the difference
        vectors and, when `wavelength_range` is given, of the wavelengths,
        to be passed as the `scales` of :func:`nearest_neighbour_spacings`.

    Raises
    ------
    ValueError
        If `metric` is unknown, or if `wavelength_range` is not increasing.
    """
    _check_metric(metric)
    if metric == "geodesic":
        hemisphere, quarter = np.pi, np.pi / 2
    else:
        hemisphere, quarter = 2.0, np.sqrt(2.0)
    factors = (quarter if isotropic else hemisphere, hemisphere)
    if wavelength_range is None:
        return factors
    low, high = wavelength_range
    if not low < high:
        raise ValueError(f"the wavelength range must be increasing, got {wavelength_range}")
    return *factors, high - low


def _check_metric(metric: str) -> None:
    if metric not in ("geodesic", "chordal"):
        raise ValueError(f"metric must be 'geodesic' or 'chordal', got {metric!r}")


def _check_scales(scales: "tuple[float, ...] | None", count: int) -> tuple[float, ...]:
    """Check the scales of `count` factors, which are 1 when not given."""
    if scales is None:
        return (1.0,) * count
    if len(scales) != count:
        raise ValueError(f"expected {count} scales, one per factor, got {len(scales)}")
    if not all(scale > 0 for scale in scales):
        raise ValueError(f"the scales must be positive, got {scales}")
    return tuple(float(scale) for scale in scales)


_DISTANCES: dict[str, "Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]"] = {
    "geodesic": geodesic_distance,
    "chordal": chordal_distance,
}


def nearest_neighbour_spacings(
    h: "ArrayLike",
    d: "ArrayLike",
    wavelength: "ArrayLike | None" = None,
    *,
    metric: Literal["geodesic", "chordal"] = "geodesic",
    scales: tuple[float, ...] | None = None,
) -> tuple["NDArray[np.float64]", ...]:
    """Measure how far each site of a sampling is from its nearest neighbour.

    The nearest neighbour of a site minimizes the product distance, the
    Euclidean norm of the distances between the half vectors, between
    the difference vectors and between the wavelengths, each divided by
    its scale. The spacings are
    the components of the distance to that neighbour, so that each
    factor of the domain can get its own scale, such as the width of a
    kernel. The computation is exhaustive, in O(n^2) time.

    Parameters
    ----------
    h : array_like, shape (n, 3)
        Half vectors of the sites. For an isotropic BRDF, give them a
        zero azimuth.
    d : array_like, shape (n, 3)
        Difference vectors of the sites.
    wavelength : array_like, shape (n,), optional
        Wavelengths of the sites, for a spectral BRDF. They are compared
        to angles in the product distance, so they must either be
        expressed in a commensurate unit, or be given a scale.
    metric : {"geodesic", "chordal"}, default "geodesic"
        Distance between the half vectors and between the difference
        vectors.
    scales : tuple of float, optional
        Positive scales of the distances between the half vectors, the
        difference vectors and, with `wavelength`, the wavelengths, such
        as the :func:`diameters` of the domain. Default: 1 for each.

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
    distance = _DISTANCES[metric]
    h = np.asarray(h, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    factors = [(h, distance), (d, distance)]
    if wavelength is not None:
        wavelength = np.asarray(wavelength, dtype=np.float64)
        factors.append((wavelength, lambda a, b: np.abs(np.subtract(a, b))))
    n = len(h)
    if n < 2:
        raise ValueError(f"the number of sites must be at least 2, got {n}")
    scale = np.array(_check_scales(scales, len(factors)))[:, np.newaxis, np.newaxis]

    spacings = np.empty((len(factors), n))
    chunk = max(1, _PAIRS_PER_CHUNK // n)
    for low in range(0, n, chunk):
        rows = np.arange(low, min(low + chunk, n))
        components = np.stack([dist(x[rows, np.newaxis], x[np.newaxis]) for x, dist in factors])
        components /= scale
        squared = np.sum(components**2, axis=0)
        # A site is not its own neighbour
        squared[np.arange(len(rows)), rows] = np.inf
        nearest = np.argmin(squared, axis=1)
        spacings[:, rows] = components[:, np.arange(len(rows)), nearest]
    return tuple(spacings)


def _factors(sites: "Sequence[ArrayLike]", name: str) -> list["NDArray[np.float64]"]:
    """Check and convert the factors of sites: (h, d) or (h, d, wavelength)."""
    if len(sites) not in (2, 3):
        raise ValueError(f"{name} must be (h, d) or (h, d, wavelength), got {len(sites)} arrays")
    return [np.asarray(x, dtype=np.float64) for x in sites]


def pairwise_distances(
    sites: "Sequence[ArrayLike]",
    centres: "Sequence[ArrayLike]",
    *,
    metric: Literal["geodesic", "chordal"] = "geodesic",
    scales: tuple[float, ...] | None = None,
) -> "NDArray[np.float64]":
    """Compute the product distances between sites and centres.

    The product distance is the Euclidean norm of the distances between
    the half vectors, between the difference vectors and between the
    wavelengths, each divided by its scale.

    Parameters
    ----------
    sites : sequence of array_like
        ``(h, d)`` or, for a spectral BRDF, ``(h, d, wavelength)``: the
        half vectors, of shape ``(m, 3)``, the difference vectors, of
        shape ``(m, 3)``, and the wavelengths, of shape ``(m,)``, of the
        sites.
    centres : sequence of array_like
        The same for the centres, with ``n`` instead of ``m``.
    metric : {"geodesic", "chordal"}, default "geodesic"
        Distance between the half vectors and between the difference
        vectors.
    scales : tuple of float, optional
        Positive scales of the distances on each factor, such as the
        :func:`diameters` of the domain. Default: 1 for each.

    Returns
    -------
    ndarray, shape (m, n)
        Product distances between each site and each centre.

    Raises
    ------
    ValueError
        If `sites` and `centres` do not both have 2 or 3 factors, if
        `metric` is unknown, or if `scales` are not one positive number
        per factor.
    """
    _check_metric(metric)
    distance = _DISTANCES[metric]
    sites = _factors(sites, "sites")
    centres = _factors(centres, "centres")
    if len(sites) != len(centres):
        raise ValueError(
            f"sites and centres must have the same factors, got {len(sites)} and {len(centres)}"
        )
    scale = _check_scales(scales, len(sites))
    distances = (distance, distance, lambda a, b: np.abs(np.subtract(a, b)))

    m, n = len(sites[0]), len(centres[0])
    result = np.empty((m, n))
    chunk = max(1, _PAIRS_PER_CHUNK // max(n, 1))
    for low in range(0, m, chunk):
        rows = slice(low, min(low + chunk, m))
        squared = sum(
            (dist(x[rows, np.newaxis], c[np.newaxis]) / s) ** 2
            for x, c, dist, s in zip(sites, centres, distances, scale, strict=False)
        )
        result[rows] = np.sqrt(squared)
    return result
