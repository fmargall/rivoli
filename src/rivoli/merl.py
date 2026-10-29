# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Files of the MERL BRDF database.

The MERL database [1]_ tabulates isotropic RGB BRDFs on a regular grid of
the Rusinkiewicz angles of :mod:`rivoli.parametrization`, of shape
``(90, 90, 180)``:

-   ``theta_h`` in ``[0, pi / 2]``, on a square-root scale that
    concentrates the bins near the specular peak: the bin ``i`` starts at
    ``(i / 90)^2 pi / 2``;
-   ``theta_d`` in ``[0, pi / 2]``, by steps of 1 degree;
-   ``phi_d`` in ``[0, pi)``, by steps of 1 degree. The values at
    ``phi_d`` and ``phi_d + pi`` are equal by reciprocity, which swaps
    the incident and outgoing directions, so only half of the azimuths
    are stored.

A file is a header of three little-endian 32-bit integers, the shape of
the grid, followed by the red, green and blue tables of little-endian
64-bit floats, in C order. The stored values are multiplied by the
factors of :data:`SCALES` on reading, to give values in ``sr^-1``, and
divided by them on writing. Negative values mark bins without
measurement, such as those below the horizon.

A model of :mod:`rivoli.model`, of RGB values or spectral, is written as
a MERL file by :func:`write_model`.

References
----------
.. [1] W. Matusik, H. Pfister, M. Brand, L. McMillan, "A Data-Driven
    Reflectance Model", ACM Transactions on Graphics, vol. 22, no. 3,
    2003, pp. 759-769.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from rivoli.geometry import spherical_to_cartesian
from rivoli.parametrization import is_above_horizon

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import ArrayLike, NDArray

    from rivoli.model import RBFModel

__all__ = [
    "SCALES",
    "SHAPE",
    "bin_angles",
    "bin_indices",
    "read_merl",
    "write_merl",
    "write_model",
]

SHAPE = (90, 90, 180)
"""Numbers of bins in ``theta_h``, ``theta_d`` and ``phi_d``."""

SCALES = (1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0)
"""Factors of the red, green and blue values stored in the files."""

_HEADER = np.dtype("<i4")
_VALUE = np.dtype("<f8")
_COUNT = SHAPE[0] * SHAPE[1] * SHAPE[2]
# Angles exactly at the start of a bin can fall short of it by a rounding
# error: this fraction of a bin brings them back in
_EDGE_TOLERANCE = 1e-9


def bin_angles() -> tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]:
    """Give the angles of the bins of the MERL grid.

    Each bin is represented by its start, the angles at which the
    measurements of the database are usually taken.

    Returns
    -------
    theta_h, theta_d, phi_d : ndarray, shape (90, 90, 180)
        Rusinkiewicz angles of the bins, in radians.
    """
    i, j, k = (np.arange(n, dtype=np.float64) for n in SHAPE)
    theta_h = (i / SHAPE[0]) ** 2 * (np.pi / 2)
    theta_d = j / SHAPE[1] * (np.pi / 2)
    phi_d = k / SHAPE[2] * np.pi
    theta_h, theta_d, phi_d = np.meshgrid(theta_h, theta_d, phi_d, indexing="ij")
    return theta_h, theta_d, phi_d


def bin_indices(
    theta_h: "ArrayLike", theta_d: "ArrayLike", phi_d: "ArrayLike"
) -> tuple["NDArray[np.intp]", "NDArray[np.intp]", "NDArray[np.intp]"]:
    """Find the bins of the MERL grid that contain angles.

    This is the lookup of the reference implementation of the database:
    ``phi_d`` is taken modulo ``pi``, by reciprocity, and angles beyond
    the grid are clamped to its first or last bins. The bins of the
    angles of :func:`bin_angles` are their own.

    Parameters
    ----------
    theta_h, theta_d, phi_d : array_like
        Rusinkiewicz angles of an isotropic BRDF, in radians. They are
        broadcast together.

    Returns
    -------
    tuple of ndarray of int
        Indices of the bins along each axis of the grid, of the
        broadcast shape of the angles. They index the tables of
        :func:`read_merl`, as in ``values[bin_indices(...)]``.
    """
    theta_h, theta_d, phi_d = np.broadcast_arrays(
        *(np.asarray(angle, dtype=np.float64) for angle in (theta_h, theta_d, phi_d))
    )
    positions = (
        np.sqrt(np.maximum(theta_h, 0.0) / (np.pi / 2)) * SHAPE[0],
        theta_d / (np.pi / 2) * SHAPE[1],
        np.mod(phi_d, np.pi) / np.pi * SHAPE[2],
    )
    i, j, k = (
        np.clip(np.floor(position + _EDGE_TOLERANCE), 0, n - 1).astype(np.intp)
        for position, n in zip(positions, SHAPE, strict=True)
    )
    return i, j, k


def read_merl(path: "str | PathLike[str]") -> "NDArray[np.float64]":
    """Read a MERL file.

    Parameters
    ----------
    path : str or path-like
        Path of the file, usually with the ``.binary`` extension.

    Returns
    -------
    ndarray, shape (90, 90, 180, 3)
        Red, green and blue values of the BRDF in ``sr^-1``, indexed by
        the bins of ``theta_h``, ``theta_d`` and ``phi_d``. Negative
        values mark bins without measurement.

    Raises
    ------
    ValueError
        If the file is not a MERL file of the expected shape and size.
    """
    content = Path(path).read_bytes()
    size = _HEADER.itemsize * len(SHAPE) + _VALUE.itemsize * len(SCALES) * _COUNT
    if len(content) < _HEADER.itemsize * len(SHAPE):
        raise ValueError(f"{path}: the header of a MERL file is truncated")
    shape = tuple(int(n) for n in np.frombuffer(content, dtype=_HEADER, count=len(SHAPE)))
    if shape != SHAPE:
        raise ValueError(f"{path}: expected a MERL grid of shape {SHAPE}, got {shape}")
    if len(content) != size:
        raise ValueError(f"{path}: expected {size} bytes, got {len(content)}")
    tables = np.frombuffer(content, dtype=_VALUE, offset=_HEADER.itemsize * len(SHAPE))
    values = np.moveaxis(tables.reshape(len(SCALES), *SHAPE), 0, -1).astype(np.float64)
    return values * np.array(SCALES)


def write_merl(path: "str | PathLike[str]", values: "ArrayLike") -> None:
    """Write a MERL file.

    Parameters
    ----------
    path : str or path-like
        Path of the file, usually with the ``.binary`` extension. An
        existing file is overwritten.
    values : array_like, shape (90, 90, 180, 3)
        Red, green and blue values of the BRDF in ``sr^-1``, as returned
        by :func:`read_merl`. Negative values mark bins without
        measurement.

    Raises
    ------
    ValueError
        If `values` do not have the shape of the grid, or are not finite.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (*SHAPE, len(SCALES)):
        raise ValueError(f"expected values of shape {(*SHAPE, len(SCALES))}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    tables = np.moveaxis(values / np.array(SCALES), -1, 0)
    header = np.array(SHAPE, dtype=_HEADER).tobytes()
    Path(path).write_bytes(header + tables.astype(_VALUE).tobytes())


def write_model(
    path: "str | PathLike[str]",
    model: "RBFModel",
    *,
    wavelengths: "ArrayLike | None" = None,
    illuminant: Literal["D65", "E"] = "D65",
    device: str | None = None,
) -> None:
    """Write a model of a BRDF as a MERL file.

    The model is evaluated at the angles of the bins of :func:`bin_angles`
    that are above the horizon, with half vectors of zero azimuth, as an
    isotropic BRDF. Its values are clamped to 0 from below, as a BRDF is
    not negative, and the bins below the horizon are marked as without
    measurement, by the value -1.

    Parameters
    ----------
    path : str or path-like
        Path of the file, usually with the ``.binary`` extension. An
        existing file is overwritten.
    model : RBFModel
        A model of 3 channels, the red, green and blue values, or a
        spectral model of a single channel.
    wavelengths : array_like, shape (k,), optional
        Increasing wavelengths at which a spectral model is evaluated,
        to be converted to RGB by :meth:`rivoli.model.RBFModel.rgb`.
        Required for a spectral model, and only for it.
    illuminant : {"D65", "E"}, default "D65"
        Illuminant under which a spectral model is seen.
    device : str, optional
        Warp device, such as ``"cuda:0"``, on which the model is
        evaluated, which takes seconds on a GPU instead of minutes with
        NumPy for a model of a thousand centres. Default: NumPy, without
        Warp.

    Raises
    ------
    ValueError
        If `wavelengths` are given for a model that is not spectral, or
        missing for a spectral one, if a model that is not spectral does
        not have 3 channels, or for an invalid spectral model, as in
        :meth:`rivoli.model.RBFModel.rgb`.
    """
    spectral = len(model.centres) == 3
    if spectral != (wavelengths is not None):
        raise ValueError("wavelengths must be given if and only if the model is spectral")
    if not spectral and model.weights.shape[1:] != (len(SCALES),):
        raise ValueError(
            f"expected a model of {len(SCALES)} channels, red, green and blue, "
            f"got weights of shape {model.weights.shape}"
        )
    theta_h, theta_d, phi_d = bin_angles()
    above = is_above_horizon(theta_h, theta_d, phi_d)
    sites = (
        spherical_to_cartesian(theta_h[above], 0.0),
        spherical_to_cartesian(theta_d[above], phi_d[above]),
    )
    values = np.full((*SHAPE, len(SCALES)), -1.0)
    if wavelengths is None:
        values[above] = np.maximum(model(sites, device=device), 0.0)
    else:
        rgb = model.rgb(sites, wavelengths, illuminant=illuminant, device=device)
        values[above] = np.maximum(rgb, 0.0)
    write_merl(path, values)
