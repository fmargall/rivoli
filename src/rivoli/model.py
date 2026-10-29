# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Radial basis function models of BRDFs.

A model is an expansion on kernels of :mod:`rivoli.kernels`, whose
weights are fitted to measurements by :func:`fit`, with the Dirichlet
energy of :mod:`rivoli.dirichlet` as regularization, through the solver
of :mod:`rivoli.tikhonov`.

BRDFs span several orders of magnitude, from their diffuse part to their
specular peaks, so they are fitted in log space by default: the expansion
models ``log(f + offset)``, and the model returns ``exp(expansion) -
offset``. The small positive offset keeps the logarithm finite for zero
values, and the model always returns values above ``-offset``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from rivoli.colour import spectral_matrix
from rivoli.dirichlet import dirichlet_matrix
from rivoli.kernels import collocation_matrix, gaussian
from rivoli.tikhonov import Tikhonov, _check_regularization, _truncated_solve

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "RBFModel",
    "fit",
]

# Offset of the values in log space, as in earlier versions of RIVOLI
_LOG_OFFSET = 1e-6
# Entries of the collocation matrix computed at once by a model, 128 MiB
_ENTRIES_PER_CHUNK = 2**24


@dataclass(frozen=True)
class RBFModel:
    """Expansion of a BRDF on radial basis function kernels.

    Attributes
    ----------
    centres : tuple of ndarray
        ``(h, d)`` or ``(h, d, wavelength)`` of the ``n`` centres.
    weights : ndarray, shape (n,) or (n, c)
        Weights of the kernels, with a column per channel.
    widths : tuple of float
        Widths of the kernels on each factor.
    metric : {"geodesic", "chordal"}
        Distance between the half vectors and between the difference
        vectors in the kernels.
    kernel : {"gaussian"}
        Profile of the kernels.
    log_offset : float or None
        Offset of the values in log space, or None if the expansion
        models the values themselves.
    symmetry : {"reciprocity", "none"}, default "reciprocity"
        Symmetry of the BRDF, over whose orbit the kernels are summed.
    """

    centres: tuple["NDArray[np.float64]", ...]
    weights: "NDArray[np.float64]"
    widths: tuple[float, ...]
    metric: Literal["geodesic", "chordal"]
    kernel: Literal["gaussian"]
    log_offset: float | None
    symmetry: Literal["none", "reciprocity"] = "reciprocity"

    def __call__(self, sites: "Sequence[ArrayLike]") -> "NDArray[np.float64]":
        """Evaluate the model.

        Parameters
        ----------
        sites : sequence of array_like
            ``(h, d)`` or ``(h, d, wavelength)`` of ``m`` sites, as the
            centres.

        Returns
        -------
        ndarray, shape (m,) or (m, c)
            Values of the model at the sites.
        """
        return self._values(self._expand(sites, self.centres, self.weights, self.widths))

    def spectra(
        self, sites: "Sequence[ArrayLike]", wavelengths: "ArrayLike"
    ) -> "NDArray[np.float64]":
        """Evaluate a spectral model at every wavelength of each site.

        The Gaussian of the product distance is the product of the
        Gaussians of the distances on each factor, so the kernels of the
        half and difference vectors are evaluated once for all the
        wavelengths: the cost is that of a single evaluation per site.

        Parameters
        ----------
        sites : sequence of array_like
            ``(h, d)`` of ``m`` sites, without wavelengths.
        wavelengths : array_like, shape (k,)
            Wavelengths at which the model is evaluated at each site.

        Returns
        -------
        ndarray, shape (m, k) or (m, k, c)
            Spectra of the model at the sites.

        Raises
        ------
        ValueError
            If the model is not spectral, if `sites` do not have 2
            factors, or if `wavelengths` is not one-dimensional.
        """
        if len(self.centres) != 3:
            raise ValueError("the model is not spectral: its centres have no wavelengths")
        if len(sites) != 2:
            raise ValueError(f"expected sites (h, d) without wavelengths, got {len(sites)} factors")
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        if wavelengths.ndim != 1:
            raise ValueError(f"expected wavelengths of shape (k,), got {wavelengths.shape}")
        # Kernels of the wavelengths, of shape (k, n), whose products with the
        # weights are the weights of the kernels of (h, d) at each wavelength
        spectral = gaussian((wavelengths[:, np.newaxis] - self.centres[2]) / self.widths[2])
        channels = self.weights.shape[1:]
        weights = spectral.T.reshape(*spectral.T.shape, *(1,) * len(channels)) * (
            self.weights.reshape(len(self.weights), 1, *channels)
        )
        expansion = self._expand(
            sites, self.centres[:2], weights.reshape(len(weights), -1), self.widths[:2]
        )
        return self._values(expansion.reshape(len(expansion), len(wavelengths), *channels))

    def rgb(
        self,
        sites: "Sequence[ArrayLike]",
        wavelengths: "ArrayLike",
        *,
        illuminant: Literal["D65", "E"] = "D65",
    ) -> "NDArray[np.float64]":
        """Evaluate the colour of a spectral model.

        The spectra of the model at the wavelengths are converted to the
        linear RGB of sRGB by :func:`rivoli.colour.spectrum_to_rgb`, and
        are extended by their values at the first and last wavelengths
        beyond them.

        Parameters
        ----------
        sites : sequence of array_like
            ``(h, d)`` of ``m`` sites, without wavelengths.
        wavelengths : array_like, shape (k,)
            Increasing wavelengths at which the spectra are evaluated,
            such as every 5 nm over the range of the model.
        illuminant : {"D65", "E"}, default "D65"
            Illuminant under which the BRDF is seen.

        Returns
        -------
        ndarray, shape (m, 3)
            Linear red, green and blue values of the model at the sites.

        Raises
        ------
        ValueError
            If the model is not spectral, if it has several channels, or
            for invalid sites or wavelengths, as in :meth:`spectra`.
        """
        if self.weights.ndim != 1:
            raise ValueError("the colour of a model with several channels is undefined")
        matrix = spectral_matrix(wavelengths, illuminant=illuminant)
        return self.spectra(sites, wavelengths) @ matrix

    def _expand(
        self,
        sites: "Sequence[ArrayLike]",
        centres: "Sequence[NDArray[np.float64]]",
        weights: "NDArray[np.float64]",
        widths: tuple[float, ...],
    ) -> "NDArray[np.float64]":
        """Expansion of weights on the kernels of centres at sites."""
        sites = [np.asarray(factor, dtype=np.float64) for factor in sites]
        m = len(sites[0])
        # The collocation matrix is computed by chunks of sites, as for a
        # whole MERL grid it would not fit in memory
        step = max(1, _ENTRIES_PER_CHUNK // len(weights))
        return np.concatenate(
            [
                collocation_matrix(
                    [factor[start : start + step] for factor in sites],
                    centres,
                    widths=widths,
                    metric=self.metric,
                    kernel=self.kernel,
                    symmetry=self.symmetry,
                )
                @ weights
                for start in range(0, m, step)
            ]
        )

    def _values(self, expansion: "NDArray[np.float64]") -> "NDArray[np.float64]":
        """Values of the model of an expansion, out of log space."""
        if self.log_offset is None:
            return expansion
        return np.exp(expansion) - self.log_offset


def fit(
    sites: "Sequence[ArrayLike]",
    values: "ArrayLike",
    centres: "Sequence[ArrayLike]",
    *,
    widths: tuple[float, ...],
    metric: Literal["geodesic", "chordal"],
    isotropic: bool,
    regularization: float,
    wavelength_range: tuple[float, float] | None = None,
    scales: tuple[float, ...] | None = None,
    log_space: bool = True,
    kernel: Literal["gaussian"] = "gaussian",
    symmetry: Literal["none", "reciprocity"] = "reciprocity",
    energy: "ArrayLike | None" = None,
) -> RBFModel:
    """Fit a model to measurements of a BRDF.

    The weights minimize ``(1 - lambda) |K w - f|^2 / m + lambda w^T G
    w``, where ``K`` is the collocation matrix of the centres at the
    sites, ``f`` the values, in log space by default, and ``G`` the
    Dirichlet matrix of the kernels: see :mod:`rivoli.tikhonov`.

    Parameters
    ----------
    sites : sequence of array_like
        ``(h, d)`` or, for a spectral BRDF, ``(h, d, wavelength)`` of
        the ``m`` measurements, as in
        :func:`rivoli.kernels.collocation_matrix`.
    values : array_like, shape (m,) or (m, c)
        Measured values, possibly with several channels, such as RGB.
    centres : sequence of array_like
        ``(h, d)`` or ``(h, d, wavelength)`` of the ``n`` centres of the
        kernels.
    widths : tuple of float
        Positive widths of the kernels on each factor.
    metric : {"geodesic", "chordal"}
        Distance between the half vectors and between the difference
        vectors in the kernels.
    isotropic : bool
        Whether the BRDF is isotropic, with half vectors of zero azimuth.
    regularization : float
        Weight ``lambda`` of the Dirichlet energy, in ``[0, 1)``. With
        0, the fit is a plain least-squares one, and an interpolation if
        the sites are the centres.
    wavelength_range : tuple of float, optional
        Lower and upper bounds of the wavelengths of a spectral BRDF.
    scales : tuple of float, optional
        Scales of the factors in the Dirichlet energy, such as their
        :func:`rivoli.distances.diameters`. Default: 1 for each.
    log_space : bool, default True
        Whether to fit ``log(f + 1e-6)`` instead of ``f``.
    kernel : {"gaussian"}, default "gaussian"
        Profile of the kernels.
    symmetry : {"reciprocity", "none"}, default "reciprocity"
        Symmetry of the BRDF, over whose orbit the kernels are summed, so
        that the model has it exactly. With reciprocity, the centres are
        best sampled over the fundamental domain, as by
        :func:`rivoli.sampling.super_fibonacci_isotropic` with the same
        option: centres that are images of each other have the same
        kernel.
    energy : array_like, shape (n, n), optional
        Dirichlet matrix of the kernels, if already computed, such as by
        :func:`rivoli.warp.dirichlet.dirichlet_matrix`. It is otherwise
        computed when `regularization` is not 0.

    Returns
    -------
    RBFModel
        The fitted model.

    Raises
    ------
    ValueError
        If the values are not one per site, if a value is not above
        ``-1e-6`` in log space, if `regularization` is not in ``[0, 1)``,
        or for invalid kernels, as in
        :func:`rivoli.kernels.collocation_matrix` and
        :func:`rivoli.dirichlet.dirichlet_matrix`.
    """
    _check_regularization(regularization)
    values = np.asarray(values, dtype=np.float64)
    log_offset = _LOG_OFFSET if log_space else None
    if log_offset is not None:
        if np.any(values <= -log_offset):
            raise ValueError(f"values must be above {-log_offset} to be fitted in log space")
        values = np.log(values + log_offset)
    collocation = collocation_matrix(
        sites, centres, widths=widths, metric=metric, kernel=kernel, symmetry=symmetry
    )
    if values.shape[:1] != collocation.shape[:1] or values.ndim > 2:
        raise ValueError(
            f"expected values of shape ({len(collocation)},) or ({len(collocation)}, c), "
            f"got {values.shape}"
        )
    if regularization == 0.0:
        # No energy to compute
        weights = _truncated_solve(np.linalg.svd(collocation, full_matrices=False), values)
    else:
        if energy is None:
            energy = dirichlet_matrix(
                centres,
                widths=widths,
                metric=metric,
                isotropic=isotropic,
                wavelength_range=wavelength_range,
                scales=scales,
                kernel=kernel,
                symmetry=symmetry,
            )
        weights = Tikhonov(collocation, energy).solve(values, regularization)
    return RBFModel(
        centres=tuple(np.asarray(c, dtype=np.float64) for c in centres),
        weights=weights,
        widths=tuple(float(width) for width in widths),
        metric=metric,
        kernel=kernel,
        log_offset=log_offset,
        symmetry=symmetry,
    )
