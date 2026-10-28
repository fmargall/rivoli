# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Parametrizations of pairs of directions.

A BRDF is a function of an incident direction ``wi`` and an outgoing
direction ``wo``. The functions of this module convert such pairs to the
half and difference parametrization of Rusinkiewicz [1]_, and back.

The half vector ``h`` is the normalized bisector of ``wi`` and ``wo``.
The difference vector ``d`` is ``wi`` expressed in the frame where ``h``
is the normal: ``wi`` is rotated around the normal by ``-phi_h``, then
around the binormal by ``-theta_h``. In that frame ``wo`` is the mirror
image of ``d`` across the normal, so ``(h, d)`` determines ``(wi, wo)``.

The angles follow the conventions of :mod:`rivoli.geometry`:
``theta_h`` and ``theta_d`` are in ``[0, pi]``, ``phi_h`` and ``phi_d``
in ``(-pi, pi]``. When both directions lie in the upper hemisphere,
``theta_h`` and ``theta_d`` are in ``[0, pi / 2]``. Swapping ``wi`` and
``wo`` leaves ``h`` unchanged and turns ``phi_d`` into ``phi_d + pi``
(modulo ``2 pi``). When ``theta_h`` is exactly 0, ``phi_h`` is 0, as
the azimuth of the normal is. Near the specular configuration, however,
rounding errors leave a ``theta_h`` of the order of the machine epsilon
and an arbitrary ``phi_h``, hence an arbitrary ``phi_d`` since it is
measured from ``phi_h``. This singularity of the parametrization is kept
as is: an isotropic BRDF does not depend on ``phi_d`` when ``theta_h``
is 0.

The conversions are undefined when ``wi`` and ``wo`` are opposite, as
their bisector is then zero.

The directions can also be given by their spherical angles
``(theta_i, phi_i, theta_o, phi_o)``. For an isotropic BRDF, which is
invariant by rotation around the normal, three angles are enough on each
side: ``(theta_i, theta_o, delta_phi)``, where ``delta_phi`` is
``phi_o - phi_i`` in ``(-pi, pi]``, and ``(theta_h, theta_d, phi_d)``.

References
----------
.. [1] S. Rusinkiewicz, "A New Change of Variables for Efficient BRDF
    Representation", Eurographics Workshop on Rendering, 1998.
"""

from typing import TYPE_CHECKING, overload

import numpy as np

from rivoli.geometry import cartesian_to_spherical, normalize, rotate, spherical_to_cartesian

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "directions_to_half_difference",
    "directions_to_rusinkiewicz",
    "half_difference_to_directions",
    "is_above_horizon",
    "rusinkiewicz_to_directions",
    "rusinkiewicz_to_spherical",
    "spherical_to_rusinkiewicz",
]

_NORMAL = np.array([0.0, 0.0, 1.0])
_BINORMAL = np.array([0.0, 1.0, 0.0])


def directions_to_half_difference(
    wi: "ArrayLike", wo: "ArrayLike"
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Convert pairs of directions to half and difference vectors.

    Parameters
    ----------
    wi : array_like, shape (..., 3)
        Incident unit directions.
    wo : array_like, shape (..., 3)
        Outgoing unit directions, not opposite to `wi`. Broadcast against
        `wi`.

    Returns
    -------
    h : ndarray, shape (..., 3)
        Half vectors, in the local frame of the surface.
    d : ndarray, shape (..., 3)
        Difference vectors, in the frame of the half vectors.
    """
    wi = np.asarray(wi, dtype=np.float64)
    h = normalize(wi + np.asarray(wo, dtype=np.float64))
    theta_h, phi_h = cartesian_to_spherical(h)
    d = rotate(rotate(wi, _NORMAL, -phi_h), _BINORMAL, -theta_h)
    return h, d


def half_difference_to_directions(
    h: "ArrayLike", d: "ArrayLike"
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Convert half and difference vectors to pairs of directions.

    This is the inverse of :func:`directions_to_half_difference`.

    Parameters
    ----------
    h : array_like, shape (..., 3)
        Half vectors, in the local frame of the surface. They need not be
        normalized.
    d : array_like, shape (..., 3)
        Difference unit vectors, in the frame of the half vectors.
        Broadcast against `h`.

    Returns
    -------
    wi : ndarray, shape (..., 3)
        Incident directions.
    wo : ndarray, shape (..., 3)
        Outgoing directions.
    """
    d = np.asarray(d, dtype=np.float64)
    theta_h, phi_h = cartesian_to_spherical(h)
    # wo is the mirror image of d across the normal of the half vector frame
    mirrored = d * np.array([-1.0, -1.0, 1.0])
    wi = rotate(rotate(d, _BINORMAL, theta_h), _NORMAL, phi_h)
    wo = rotate(rotate(mirrored, _BINORMAL, theta_h), _NORMAL, phi_h)
    return wi, wo


def directions_to_rusinkiewicz(
    wi: "ArrayLike", wo: "ArrayLike"
) -> tuple[
    "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"
]:
    """Convert pairs of directions to Rusinkiewicz angles.

    Parameters
    ----------
    wi : array_like, shape (..., 3)
        Incident unit directions.
    wo : array_like, shape (..., 3)
        Outgoing unit directions, not opposite to `wi`. Broadcast against
        `wi`.

    Returns
    -------
    theta_h, phi_h : ndarray, shape (...)
        Polar and azimuthal angles of the half vectors.
    theta_d, phi_d : ndarray, shape (...)
        Polar and azimuthal angles of the difference vectors.
    """
    h, d = directions_to_half_difference(wi, wo)
    return *cartesian_to_spherical(h), *cartesian_to_spherical(d)


def rusinkiewicz_to_directions(
    theta_h: "ArrayLike", phi_h: "ArrayLike", theta_d: "ArrayLike", phi_d: "ArrayLike"
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Convert Rusinkiewicz angles to pairs of directions.

    This is the inverse of :func:`directions_to_rusinkiewicz`. For an
    isotropic BRDF, which does not depend on `phi_h`, pass 0 for it.

    Parameters
    ----------
    theta_h, phi_h : array_like
        Polar and azimuthal angles of the half vectors.
    theta_d, phi_d : array_like
        Polar and azimuthal angles of the difference vectors. All four
        angles are broadcast together.

    Returns
    -------
    wi : ndarray, shape (..., 3)
        Incident unit directions.
    wo : ndarray, shape (..., 3)
        Outgoing unit directions.
    """
    return half_difference_to_directions(
        spherical_to_cartesian(theta_h, phi_h), spherical_to_cartesian(theta_d, phi_d)
    )


@overload
def spherical_to_rusinkiewicz(
    theta_i: "ArrayLike", theta_o: "ArrayLike", delta_phi: "ArrayLike", /
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]: ...
@overload
def spherical_to_rusinkiewicz(
    theta_i: "ArrayLike", phi_i: "ArrayLike", theta_o: "ArrayLike", phi_o: "ArrayLike", /
) -> tuple[
    "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"
]: ...
def spherical_to_rusinkiewicz(*angles: "ArrayLike") -> tuple["NDArray[np.float64]", ...]:
    """Convert spherical angles of pairs of directions to Rusinkiewicz angles.

    Parameters
    ----------
    *angles : array_like
        Either ``(theta_i, phi_i, theta_o, phi_o)``, the polar and
        azimuthal angles of the incident and outgoing directions, or
        ``(theta_i, theta_o, delta_phi)`` for an isotropic BRDF, where
        ``delta_phi`` is ``phi_o - phi_i``. All angles are broadcast
        together.

    Returns
    -------
    tuple of ndarray
        ``(theta_h, phi_h, theta_d, phi_d)`` for four angles, or
        ``(theta_h, theta_d, phi_d)`` for three.

    Raises
    ------
    TypeError
        If the number of angles is neither 3 nor 4.
    """
    match angles:
        case (theta_i, phi_i, theta_o, phi_o):
            return directions_to_rusinkiewicz(
                spherical_to_cartesian(theta_i, phi_i), spherical_to_cartesian(theta_o, phi_o)
            )
        case (theta_i, theta_o, delta_phi):
            theta_h, _, theta_d, phi_d = directions_to_rusinkiewicz(
                spherical_to_cartesian(theta_i, 0.0), spherical_to_cartesian(theta_o, delta_phi)
            )
            return theta_h, theta_d, phi_d
        case _:
            raise TypeError(f"expected 3 or 4 angles, got {len(angles)}")


@overload
def rusinkiewicz_to_spherical(
    theta_h: "ArrayLike", theta_d: "ArrayLike", phi_d: "ArrayLike", /
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]: ...
@overload
def rusinkiewicz_to_spherical(
    theta_h: "ArrayLike", phi_h: "ArrayLike", theta_d: "ArrayLike", phi_d: "ArrayLike", /
) -> tuple[
    "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"
]: ...
def rusinkiewicz_to_spherical(*angles: "ArrayLike") -> tuple["NDArray[np.float64]", ...]:
    """Convert Rusinkiewicz angles to spherical angles of pairs of directions.

    This is the inverse of :func:`spherical_to_rusinkiewicz`.

    Parameters
    ----------
    *angles : array_like
        Either ``(theta_h, phi_h, theta_d, phi_d)``, or
        ``(theta_h, theta_d, phi_d)`` for an isotropic BRDF. All angles
        are broadcast together.

    Returns
    -------
    tuple of ndarray
        ``(theta_i, phi_i, theta_o, phi_o)`` for four angles, or
        ``(theta_i, theta_o, delta_phi)`` for three, where ``delta_phi``
        is ``phi_o - phi_i`` in ``(-pi, pi]``.

    Raises
    ------
    TypeError
        If the number of angles is neither 3 nor 4.
    """
    match angles:
        case (theta_h, phi_h, theta_d, phi_d):
            wi, wo = rusinkiewicz_to_directions(theta_h, phi_h, theta_d, phi_d)
            return *cartesian_to_spherical(wi), *cartesian_to_spherical(wo)
        case (theta_h, theta_d, phi_d):
            wi, wo = rusinkiewicz_to_directions(theta_h, 0.0, theta_d, phi_d)
            theta_i, phi_i = cartesian_to_spherical(wi)
            # Measuring the azimuth of wo in a frame where wi has a zero
            # azimuth wraps delta_phi into (-pi, pi]
            theta_o, delta_phi = cartesian_to_spherical(rotate(wo, _NORMAL, -phi_i))
            return theta_i, theta_o, delta_phi
        case _:
            raise TypeError(f"expected 3 or 4 angles, got {len(angles)}")


def is_above_horizon(*angles: "ArrayLike") -> "NDArray[np.bool_]":
    """Tell whether Rusinkiewicz angles describe directions above the horizon.

    Rotating ``d`` around the binormal by ``theta_h`` gives the heights
    ``cos(theta_h) cos(theta_d) -+ sin(theta_h) sin(theta_d) cos(phi_d)``
    of ``wi`` and ``wo``, which do not depend on ``phi_h``. Both are
    positive when the first term exceeds the absolute value of the second.

    Parameters
    ----------
    *angles : array_like
        Either ``(theta_h, phi_h, theta_d, phi_d)``, or
        ``(theta_h, theta_d, phi_d)`` for an isotropic BRDF. All angles
        are broadcast together.

    Returns
    -------
    ndarray of bool
        True where both ``wi`` and ``wo`` are strictly above the horizon.
        Grazing directions are below it, up to rounding errors: for
        instance ``cos(pi / 2)`` is not exactly 0 in floating point.

    Raises
    ------
    TypeError
        If the number of angles is neither 3 nor 4.
    """
    match angles:
        case (theta_h, _, theta_d, phi_d) | (theta_h, theta_d, phi_d):
            theta_h = np.asarray(theta_h, dtype=np.float64)
            theta_d = np.asarray(theta_d, dtype=np.float64)
            axial = np.cos(theta_h) * np.cos(theta_d)
            lateral = np.sin(theta_h) * np.sin(theta_d) * np.cos(phi_d)
            return axial > np.abs(lateral)
        case _:
            raise TypeError(f"expected 3 or 4 angles, got {len(angles)}")
