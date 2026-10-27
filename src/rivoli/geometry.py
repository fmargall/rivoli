# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Geometric primitives on the unit sphere.

Conventions shared by the whole package:

-   Directions are expressed in the local frame of the surface, whose
    normal is the z axis, the tangent the x axis and the binormal the
    y axis. The frame is right-handed.
-   Vectors are stored along the last axis, so an array of shape
    ``(..., 3)`` holds a batch of vectors. All functions broadcast over
    the leading dimensions.
-   ``theta`` is the polar angle measured from the normal, in
    ``[0, pi]``, and ``phi`` the azimuthal angle measured from the
    tangent towards the binormal, in ``(-pi, pi]``.
-   Rotations follow the right-hand rule: a positive angle turns
    counterclockwise when looking down the axis towards the origin.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "cartesian_to_spherical",
    "normalize",
    "rotate",
    "spherical_to_cartesian",
]


def normalize(v: "ArrayLike") -> "NDArray[np.float64]":
    """Scale vectors to unit length.

    Parameters
    ----------
    v : array_like, shape (..., 3)
        Vectors to normalize. They must not be zero.

    Returns
    -------
    ndarray, shape (..., 3)
        Unit vectors with the directions of `v`.
    """
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def spherical_to_cartesian(theta: "ArrayLike", phi: "ArrayLike") -> "NDArray[np.float64]":
    """Convert spherical angles to unit vectors.

    Parameters
    ----------
    theta : array_like
        Polar angles from the normal, in radians.
    phi : array_like
        Azimuthal angles from the tangent, in radians. Broadcast
        against `theta`.

    Returns
    -------
    ndarray, shape (..., 3)
        Unit vectors, the leading shape being the broadcast shape of
        `theta` and `phi`.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    sin_theta = np.sin(theta)
    return np.stack(
        np.broadcast_arrays(sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)),
        axis=-1,
    )


def cartesian_to_spherical(
    v: "ArrayLike",
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Convert vectors to spherical angles.

    The vectors need not be normalized: only their direction is used.
    Both angles are computed with ``arctan2``, which stays accurate
    near the poles, where ``arccos(z)`` loses precision.

    Parameters
    ----------
    v : array_like, shape (..., 3)
        Vectors to convert. They must not be zero.

    Returns
    -------
    theta : ndarray, shape (...)
        Polar angles from the normal, in ``[0, pi]``.
    phi : ndarray, shape (...)
        Azimuthal angles from the tangent, in ``(-pi, pi]``. At the
        poles, where the azimuth is undefined, it is 0.
    """
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    # Adding 0.0 turns -0.0 into +0.0, for which arctan2 would return
    # -pi instead of pi on the negative x axis, and -pi or pi at the poles
    return np.arctan2(np.hypot(x, y), z), np.arctan2(y + 0.0, x + 0.0)


def rotate(v: "ArrayLike", axis: "ArrayLike", angle: "ArrayLike") -> "NDArray[np.float64]":
    """Rotate vectors around an axis with Rodrigues' formula.

    Parameters
    ----------
    v : array_like, shape (..., 3)
        Vectors to rotate.
    axis : array_like, shape (..., 3)
        Rotation axes. They need not be normalized, but must not be
        zero. Broadcast against `v`.
    angle : array_like
        Rotation angles in radians, following the right-hand rule.
        Broadcast against the leading dimensions of `v` and `axis`.

    Returns
    -------
    ndarray, shape (..., 3)
        Rotated vectors, the leading shape being the broadcast shape of
        the inputs.
    """
    v = np.asarray(v, dtype=np.float64)
    k = normalize(axis)
    angle = np.asarray(angle, dtype=np.float64)[..., np.newaxis]
    cos_angle = np.cos(angle)
    k_dot_v = np.sum(k * v, axis=-1, keepdims=True)
    return v * cos_angle + np.cross(k, v) * np.sin(angle) + k * k_dot_v * (1.0 - cos_angle)
