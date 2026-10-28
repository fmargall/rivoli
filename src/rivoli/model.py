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

from rivoli.dirichlet import dirichlet_matrix
from rivoli.kernels import collocation_matrix
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
    """

    centres: tuple["NDArray[np.float64]", ...]
    weights: "NDArray[np.float64]"
    widths: tuple[float, ...]
    metric: Literal["geodesic", "chordal"]
    kernel: Literal["gaussian"]
    log_offset: float | None

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
        sites = [np.asarray(factor, dtype=np.float64) for factor in sites]
        m = len(sites[0])
        # The collocation matrix is computed by chunks of sites, as for a
        # whole MERL grid it would not fit in memory
        step = max(1, _ENTRIES_PER_CHUNK // len(self.weights))
        expansion = np.concatenate(
            [
                collocation_matrix(
                    [factor[start : start + step] for factor in sites],
                    self.centres,
                    widths=self.widths,
                    metric=self.metric,
                    kernel=self.kernel,
                )
                @ self.weights
                for start in range(0, m, step)
            ]
        )
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
    collocation = collocation_matrix(sites, centres, widths=widths, metric=metric, kernel=kernel)
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
            )
        weights = Tikhonov(collocation, energy).solve(values, regularization)
    return RBFModel(
        centres=tuple(np.asarray(c, dtype=np.float64) for c in centres),
        weights=weights,
        widths=tuple(float(width) for width in widths),
        metric=metric,
        kernel=kernel,
        log_offset=log_offset,
    )
