# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Low-discrepancy sampling of the Rusinkiewicz domain.

The samplers place sites uniformly with respect to the volume form of the
domain, so that they are well distributed for a metric built from
great-circle distances on its spherical factors:

-   For an isotropic BRDF, the domain is ``theta_h`` in ``[0, pi / 2]``
    times the hemisphere of the difference vector ``d``. Its volume form
    is ``dtheta_h sin(theta_d) dtheta_d dphi_d``: ``theta_h`` is flat and
    ``d`` is uniform in area.
-   For an anisotropic BRDF, the domain is the hemisphere of the half
    vector ``h`` times the hemisphere of ``d``, and both are uniform in
    area.
-   For a spectral BRDF, either domain is multiplied by an interval of
    wavelengths, which is flat. The spectral dimension is not assumed
    separable from the angular ones: the sites are spread jointly over
    the whole product.

The construction follows the Super-Fibonacci spirals of Alexa [1]_: the
unit cube is mapped onto the domain by measure-preserving maps, one of
its coordinates is stratified, ``(i + 1/2) / n`` for the site ``i``, and
the others are filled with Kronecker sequences ``frac((i + 1/2) alpha)``.
The increments ``alpha`` are the powers of the inverse of the generalized
golden ratio of Roberts [2]_, the root of ``x^(k + 1) = x + 1`` for ``k``
Kronecker coordinates, which makes them jointly well distributed. The
more coordinates, the less regular their projections on pairs of
coordinates, while staying more regular than a random sampling.

Each site depends only on its index and on ``n``. The samplings are not
nested: the sites for ``n`` are not a subset of those for ``n + 1``.
Adding the spectral dimension changes the increments, so the angles of a
spectral sampling differ from those of the same sampling without it.

Part of the domain is below the horizon: ``wi`` or ``wo`` points under
the surface there. With ``above_horizon=True``, the samplers return only
sites above it, as :func:`rivoli.parametrization.is_above_horizon` tells.
They sample the whole domain with the smallest number of sites, found by
bisection, of which at least ``n`` are above the horizon. As the
samplings are not nested, that number can exceed ``n`` by a few sites,
which are then removed evenly along the stratified coordinate.

A reciprocal BRDF, invariant by the half turn ``phi_d -> phi_d + pi``
of the difference vector, is determined by its values on the fundamental
domain ``phi_d`` in ``(0, pi]``. With ``symmetry="reciprocity"``, the
samplers spread their sites over that half of the domain only, with the
same density for half as many sites as over the whole domain: the right
choice for the centres of reciprocal kernels, which have the same
orbits. The horizon is invariant by the half turn, so the two options
combine.

The angles follow the conventions of :mod:`rivoli.geometry`, with
azimuths in ``(-pi, pi]``, and come in the order of
:func:`rivoli.parametrization.rusinkiewicz_to_spherical`.

References
----------
.. [1] M. Alexa, "Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling
    of SO(3)", IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR), 2022, pp. 8281-8290.
.. [2] M. Roberts, "The Unreasonable Effectiveness of Quasirandom
    Sequences", 2018.
"""

from functools import cache
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from rivoli.kernels import _check_symmetry
from rivoli.parametrization import is_above_horizon

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = [
    "super_fibonacci_anisotropic",
    "super_fibonacci_isotropic",
]


@cache
def _generalized_golden_ratio(k: int) -> float:
    # Root of x^(k + 1) = x + 1, by a fixed-point iteration that contracts
    # by a factor below 1 / (k + 1) at each step
    x = 1.0
    for _ in range(100):
        x = (1.0 + x) ** (1.0 / (k + 1))
    return x


def _super_fibonacci_cube(n: int, k: int) -> "NDArray[np.float64]":
    """Sample the unit cube of dimension ``k + 1``.

    Returns an array of shape ``(k + 1, n)``: the stratified coordinate
    first, then the ``k`` Kronecker coordinates, by increasing power of
    their increment.
    """
    if n < 1:
        raise ValueError(f"the number of sites must be at least 1, got {n}")
    s = np.arange(n, dtype=np.float64) + 0.5
    alpha = _generalized_golden_ratio(k) ** -np.arange(1.0, k + 1)
    return np.vstack([s / n, np.mod(np.outer(alpha, s), 1.0)])


def _azimuth(u: "NDArray[np.float64]", period: float = 2.0 * np.pi) -> "NDArray[np.float64]":
    # Maps [0, 1) onto (pi - period, pi], uniformly
    return np.pi - period * u


def _difference_period(symmetry: str) -> float:
    # The azimuths of d of the fundamental domain of the symmetries
    _check_symmetry(symmetry)
    return np.pi if symmetry == "reciprocity" else 2.0 * np.pi


def _hemisphere_polar_angle(u: "NDArray[np.float64]") -> "NDArray[np.float64]":
    # cos(theta) uniform in (0, 1] makes the hemisphere uniform in area,
    # as the area of the polar cap up to theta is 2 pi (1 - cos(theta))
    return np.arccos(1.0 - u)


def _wavelength(
    u: "NDArray[np.float64]", wavelength_range: tuple[float, float]
) -> "NDArray[np.float64]":
    low, high = wavelength_range
    if not low < high:
        raise ValueError(f"the wavelength range must be increasing, got {wavelength_range}")
    return low + (high - low) * u


def _isotropic(
    n: int, wavelength_range: tuple[float, float] | None, period: float
) -> tuple["NDArray[np.float64]", ...]:
    cube = _super_fibonacci_cube(n, 2 if wavelength_range is None else 3)
    angles = (
        0.5 * np.pi * cube[0],
        _hemisphere_polar_angle(cube[2]),
        _azimuth(cube[1], period),
    )
    if wavelength_range is None:
        return angles
    return *angles, _wavelength(cube[3], wavelength_range)


def _anisotropic(
    n: int, wavelength_range: tuple[float, float] | None, period: float
) -> tuple["NDArray[np.float64]", ...]:
    cube = _super_fibonacci_cube(n, 3 if wavelength_range is None else 4)
    angles = (
        _hemisphere_polar_angle(cube[0]),
        _azimuth(cube[1]),
        _hemisphere_polar_angle(cube[2]),
        _azimuth(cube[3], period),
    )
    if wavelength_range is None:
        return angles
    return *angles, _wavelength(cube[4], wavelength_range)


def _above_horizon(
    sample: "Callable[[int], tuple[NDArray[np.float64], ...]]", n: int, angle_count: int
) -> tuple["NDArray[np.float64]", ...]:
    """Sample exactly ``n`` sites above the horizon.

    `sample` returns the sites of the whole domain, whose first
    `angle_count` coordinates are the Rusinkiewicz angles.
    """

    def survivors(size: int) -> tuple["NDArray[np.float64]", ...]:
        sites = sample(size)
        above = is_above_horizon(*sites[:angle_count])
        return tuple(x[above] for x in sites)

    # The number of survivors only increases on average with the size, as
    # the samplings are not nested: the bisection finds a size for which
    # there are enough of them, not necessarily the smallest one
    low = high = n
    while len(survivors(high)[0]) < n:
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if len(survivors(middle)[0]) >= n:
            high = middle
        else:
            low = middle

    sites = survivors(high)
    # Thin out the surplus evenly in index, hence along the stratified
    # coordinate. The step between the kept indices is at least 1, so
    # they are distinct after rounding
    kept = np.round(np.linspace(0, len(sites[0]) - 1, n)).astype(np.intp)
    return tuple(x[kept] for x in sites)


@overload
def super_fibonacci_isotropic(
    n: int,
    *,
    wavelength_range: None = None,
    above_horizon: bool = False,
    symmetry: Literal["none", "reciprocity"] = "none",
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]: ...
@overload
def super_fibonacci_isotropic(
    n: int,
    *,
    wavelength_range: tuple[float, float],
    above_horizon: bool = False,
    symmetry: Literal["none", "reciprocity"] = "none",
) -> tuple[
    "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"
]: ...
def super_fibonacci_isotropic(
    n: int,
    *,
    wavelength_range: tuple[float, float] | None = None,
    above_horizon: bool = False,
    symmetry: Literal["none", "reciprocity"] = "none",
) -> tuple["NDArray[np.float64]", ...]:
    """Sample the isotropic Rusinkiewicz domain.

    ``theta_h`` is stratified, so that the sites close to the pole of the
    difference vector, where ``phi_d`` degenerates, still have their
    ``theta_h`` evenly spread.

    Parameters
    ----------
    n : int
        Number of sites, at least 1.
    wavelength_range : tuple of float, optional
        Lower and upper bounds of the wavelengths, in any unit. When
        given, the domain gets a spectral dimension, sampled uniformly.
    above_horizon : bool, default False
        Whether to keep only sites whose incident and outgoing directions
        are both above the horizon.
    symmetry : {"none", "reciprocity"}, default "none"
        Symmetry of the BRDF, whose fundamental domain is sampled: with
        ``"reciprocity"``, ``phi_d`` is in ``(0, pi]``.

    Returns
    -------
    theta_h, theta_d, phi_d : ndarray, shape (n,)
        Angles of the sites, with ``theta_h`` and ``theta_d`` in
        ``[0, pi / 2]`` and ``phi_d`` in ``(-pi, pi]``, or ``(0, pi]``
        with reciprocity.
    wavelength : ndarray, shape (n,)
        Wavelengths of the sites, only when `wavelength_range` is given.

    Raises
    ------
    ValueError
        If `n` is less than 1, if `wavelength_range` is not increasing,
        or if `symmetry` is unknown.
    """
    period = _difference_period(symmetry)
    if above_horizon:
        return _above_horizon(lambda size: _isotropic(size, wavelength_range, period), n, 3)
    return _isotropic(n, wavelength_range, period)


@overload
def super_fibonacci_anisotropic(
    n: int,
    *,
    wavelength_range: None = None,
    above_horizon: bool = False,
    symmetry: Literal["none", "reciprocity"] = "none",
) -> tuple[
    "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"
]: ...
@overload
def super_fibonacci_anisotropic(
    n: int,
    *,
    wavelength_range: tuple[float, float],
    above_horizon: bool = False,
    symmetry: Literal["none", "reciprocity"] = "none",
) -> tuple[
    "NDArray[np.float64]",
    "NDArray[np.float64]",
    "NDArray[np.float64]",
    "NDArray[np.float64]",
    "NDArray[np.float64]",
]: ...
def super_fibonacci_anisotropic(
    n: int,
    *,
    wavelength_range: tuple[float, float] | None = None,
    above_horizon: bool = False,
    symmetry: Literal["none", "reciprocity"] = "none",
) -> tuple["NDArray[np.float64]", ...]:
    """Sample the anisotropic Rusinkiewicz domain.

    ``cos(theta_h)`` is stratified, and the other coordinates are
    Kronecker sequences.

    Parameters
    ----------
    n : int
        Number of sites, at least 1.
    wavelength_range : tuple of float, optional
        Lower and upper bounds of the wavelengths, in any unit. When
        given, the domain gets a spectral dimension, sampled uniformly.
    above_horizon : bool, default False
        Whether to keep only sites whose incident and outgoing directions
        are both above the horizon.
    symmetry : {"none", "reciprocity"}, default "none"
        Symmetry of the BRDF, whose fundamental domain is sampled: with
        ``"reciprocity"``, ``phi_d`` is in ``(0, pi]``.

    Returns
    -------
    theta_h, phi_h, theta_d, phi_d : ndarray, shape (n,)
        Angles of the sites, with ``theta_h`` and ``theta_d`` in
        ``[0, pi / 2]``, ``phi_h`` and ``phi_d`` in ``(-pi, pi]``, ``phi_d``
        in ``(0, pi]`` with reciprocity.
    wavelength : ndarray, shape (n,)
        Wavelengths of the sites, only when `wavelength_range` is given.

    Raises
    ------
    ValueError
        If `n` is less than 1, if `wavelength_range` is not increasing,
        or if `symmetry` is unknown.
    """
    period = _difference_period(symmetry)
    if above_horizon:
        return _above_horizon(lambda size: _anisotropic(size, wavelength_range, period), n, 4)
    return _anisotropic(n, wavelength_range, period)
