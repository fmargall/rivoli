# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Conversions of spectra to colours.

A spectral BRDF, or any spectral quantity such as a reflectance, is seen
under an illuminant ``S`` by the CIE 1931 standard observer, whose colour
matching functions ``x``, ``y`` and ``z`` give its tristimulus values::

    X = sum f(lambda) S(lambda) x(lambda) / sum S(lambda) y(lambda)

and likewise for ``Y`` and ``Z``, the sums running over the wavelengths
of the CIE tables, from 360 to 830 nm by steps of 1 nm. The
normalization by the ``Y`` of the illuminant keeps the units of the
spectrum: a constant spectrum of value 1, a perfect white diffuser for a
reflectance, has a ``Y`` of 1.

Spectra are given at increasing wavelengths, in nm, and interpolated
linearly between them. Beyond the first and the last wavelengths, they
are extended by their first and last values, so that a spectrum measured
on a part of the visible range is not darkened by the rest.

The tristimulus values are then converted to the linear RGB of sRGB [1]_,
without the gamma encoding of images. Its white is D65: under that
illuminant, the white has the RGB ``(1, 1, 1)``. Under the illuminant E,
of equal energy, the colours are first adapted to D65 by the Bradford
transform [2]_, so that the white is still ``(1, 1, 1)``. The matrix of
sRGB is derived from its primaries and from the white given by the CIE
tables, so that this holds up to rounding, and differs from the rounded
matrix of the standard by less than ``5e-4``.

All conversions are linear in the spectrum, so they are also given as a
matrix by :func:`spectral_matrix`, to be applied to the weights of a
spectral model as well as to its values.

The colour matching functions [3]_ and the illuminant D65 [4]_ are the
tables of the CIE, distributed under the Creative Commons
Attribution-ShareAlike 4.0 International license, in the ``data``
directory of the package.

References
----------
.. [1] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
    measurement and management - Part 2-1: Default RGB colour space -
    sRGB".
.. [2] K. M. Lam, "Metamerism and Colour Constancy", PhD thesis,
    University of Bradford, 1985.
.. [3] CIE, "CIE 1931 colour-matching functions, 2 degree observer",
    2019, doi:10.25039/CIE.DS.xvudnb9b.
.. [4] CIE, "CIE standard illuminant D65", 2022,
    doi:10.25039/CIE.DS.hjfjmt59.
"""

from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "spectral_matrix",
    "spectrum_to_rgb",
    "spectrum_to_xyz",
    "xyz_to_rgb",
]

# Chromaticities of the red, green and blue primaries of sRGB
_PRIMARIES = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])
# Cone responses of the Bradford transform
_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)


@cache
def _tables() -> tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]:
    """Wavelengths, colour matching functions and D65 of the CIE tables."""
    data = files("rivoli") / "data"
    with (data / "CIE_xyz_1931_2deg.csv").open() as stream:
        observer = np.loadtxt(stream, delimiter=",")
    with (data / "CIE_std_illum_D65.csv").open() as stream:
        d65 = np.loadtxt(stream, delimiter=",")
    wavelengths = observer[:, 0]
    d65 = np.interp(wavelengths, d65[:, 0], d65[:, 1])
    return wavelengths, observer[:, 1:], d65


def _check_illuminant(illuminant: str) -> None:
    if illuminant not in ("D65", "E"):
        raise ValueError(f"illuminant must be 'D65' or 'E', got {illuminant!r}")


@cache
def _xyz_weights(illuminant: str) -> "NDArray[np.float64]":
    """Weights of the tristimulus values at the wavelengths of the CIE tables."""
    wavelengths, observer, d65 = _tables()
    power = d65 if illuminant == "D65" else np.ones_like(wavelengths)
    weights = power[:, np.newaxis] * observer
    return weights / weights[:, 1].sum()


def _white(illuminant: str) -> "NDArray[np.float64]":
    return _xyz_weights(illuminant).sum(axis=0)


@cache
def _rgb_from_xyz() -> "NDArray[np.float64]":
    """Matrix of the linear RGB of sRGB, whose white is D65."""
    x, y = _PRIMARIES.T
    primaries = np.stack([x / y, np.ones(3), (1 - x - y) / y])
    xyz_from_rgb = primaries * np.linalg.solve(primaries, _white("D65"))
    return np.linalg.inv(xyz_from_rgb)


@cache
def _adaptation(illuminant: str) -> "NDArray[np.float64]":
    """Bradford transform of the colours under an illuminant to D65."""
    gains = (_BRADFORD @ _white("D65")) / (_BRADFORD @ _white(illuminant))
    return np.linalg.solve(_BRADFORD, gains[:, np.newaxis] * _BRADFORD)


def _interpolation(wavelengths: "ArrayLike") -> "NDArray[np.float64]":
    """Matrix of the linear interpolation of spectra at the CIE wavelengths."""
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if wavelengths.ndim != 1 or len(wavelengths) == 0:
        raise ValueError(f"expected wavelengths of shape (k,), got {wavelengths.shape}")
    if not np.all(np.isfinite(wavelengths)) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("the wavelengths must be finite and increasing")
    grid = _tables()[0]
    # Interpolating each spectrum that is 1 at a single wavelength
    return np.array([np.interp(grid, wavelengths, row) for row in np.eye(len(wavelengths))])


def _apply(spectra: "ArrayLike", matrix: "NDArray[np.float64]") -> "NDArray[np.float64]":
    spectra = np.asarray(spectra, dtype=np.float64)
    if spectra.ndim == 0 or spectra.shape[-1] != len(matrix):
        raise ValueError(
            f"expected spectra of shape (..., {len(matrix)}), one value per wavelength, "
            f"got {spectra.shape}"
        )
    return spectra @ matrix


def spectral_matrix(
    wavelengths: "ArrayLike",
    *,
    space: Literal["rgb", "xyz"] = "rgb",
    illuminant: Literal["D65", "E"] = "D65",
) -> "NDArray[np.float64]":
    """Give the linear map from spectra to colours.

    Parameters
    ----------
    wavelengths : array_like, shape (k,)
        Increasing wavelengths of the spectra, in nm.
    space : {"rgb", "xyz"}, default "rgb"
        Colours in the linear RGB of sRGB, or tristimulus values.
    illuminant : {"D65", "E"}, default "D65"
        Illuminant under which the spectra are seen.

    Returns
    -------
    ndarray, shape (k, 3)
        Matrix ``M`` such that the colours of spectra ``f`` of shape
        ``(..., k)`` are ``f @ M``.

    Raises
    ------
    ValueError
        If the wavelengths are not finite and increasing, or if `space`
        or `illuminant` is unknown.
    """
    if space not in ("rgb", "xyz"):
        raise ValueError(f"space must be 'rgb' or 'xyz', got {space!r}")
    _check_illuminant(illuminant)
    matrix = _interpolation(wavelengths) @ _xyz_weights(illuminant)
    if space == "rgb":
        matrix = matrix @ (_rgb_from_xyz() @ _adaptation(illuminant)).T
    return matrix


def spectrum_to_xyz(
    wavelengths: "ArrayLike",
    spectra: "ArrayLike",
    *,
    illuminant: Literal["D65", "E"] = "D65",
) -> "NDArray[np.float64]":
    """Convert spectra to CIE 1931 tristimulus values.

    Parameters
    ----------
    wavelengths : array_like, shape (k,)
        Increasing wavelengths of the spectra, in nm.
    spectra : array_like, shape (..., k)
        Values of the spectra at the wavelengths, such as those of a
        spectral BRDF, in any unit.
    illuminant : {"D65", "E"}, default "D65"
        Illuminant under which the spectra are seen.

    Returns
    -------
    ndarray, shape (..., 3)
        Tristimulus values ``X``, ``Y`` and ``Z``, in the unit of the
        spectra.

    Raises
    ------
    ValueError
        If the wavelengths are not finite and increasing, if the spectra
        do not have one value per wavelength, or if `illuminant` is
        unknown.
    """
    matrix = spectral_matrix(wavelengths, space="xyz", illuminant=illuminant)
    return _apply(spectra, matrix)


def xyz_to_rgb(xyz: "ArrayLike") -> "NDArray[np.float64]":
    """Convert tristimulus values to the linear RGB of sRGB.

    Parameters
    ----------
    xyz : array_like, shape (..., 3)
        Tristimulus values, whose white is D65.

    Returns
    -------
    ndarray, shape (..., 3)
        Linear red, green and blue values, without gamma encoding. They
        are negative for colours outside the gamut of sRGB.

    Raises
    ------
    ValueError
        If `xyz` does not have 3 values along its last axis.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim == 0 or xyz.shape[-1] != 3:
        raise ValueError(f"expected tristimulus values of shape (..., 3), got {xyz.shape}")
    return xyz @ _rgb_from_xyz().T


def spectrum_to_rgb(
    wavelengths: "ArrayLike",
    spectra: "ArrayLike",
    *,
    illuminant: Literal["D65", "E"] = "D65",
) -> "NDArray[np.float64]":
    """Convert spectra to the linear RGB of sRGB.

    Parameters
    ----------
    wavelengths : array_like, shape (k,)
        Increasing wavelengths of the spectra, in nm.
    spectra : array_like, shape (..., k)
        Values of the spectra at the wavelengths, such as those of a
        spectral BRDF, in any unit.
    illuminant : {"D65", "E"}, default "D65"
        Illuminant under which the spectra are seen. Under both, a
        constant spectrum of value 1 has the RGB ``(1, 1, 1)``.

    Returns
    -------
    ndarray, shape (..., 3)
        Linear red, green and blue values, in the unit of the spectra,
        such as those of :func:`rivoli.merl.write_merl` for a BRDF.

    Raises
    ------
    ValueError
        If the wavelengths are not finite and increasing, if the spectra
        do not have one value per wavelength, or if `illuminant` is
        unknown.
    """
    matrix = spectral_matrix(wavelengths, illuminant=illuminant)
    return _apply(spectra, matrix)
