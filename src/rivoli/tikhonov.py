# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Least-squares fits regularized by a quadratic energy.

The weights ``w`` of an expansion on ``n`` kernels are fitted to ``m``
values ``f`` at sites through the collocation matrix ``K`` of the
kernels at the sites, and regularized by an energy ``w^T G w``, such as
the Dirichlet energy of :func:`rivoli.dirichlet.dirichlet_matrix`. For a
regularization ``lambda`` in ``[0, 1)``, they minimize the convex
combination of the mean squared residual and of the energy::

    (1 - lambda) |K w - f|^2 / m + lambda w^T G w

``lambda = 0`` is the plain least-squares fit, an interpolation when the
sites are the centres, and ``lambda`` close to 1 gives the smoothest
fits, of the least energy. Dividing by ``1 - lambda``, it is the
Tikhonov problem of multiplier ``mu = lambda / (1 - lambda)``.

The energy is positive semidefinite, and its matrix has eigenvalues
close to 0 for redundant kernels, such as kernels close to each other,
or narrow along a factor and wide along another one. Its eigenvalues are
floored at a small fraction of the largest one, as in earlier versions
of RIVOLI, so that every direction of the weights is penalized: where
the values do not constrain the weights, as away from measurements made
in a single plane, the regularization still bounds them, instead of
leaving them to least squares, which make them diverge. The energy has
no exact null space to preserve: the constant functions, of zero
Dirichlet energy, are not expansions on Gaussian kernels.

With the floored energy ``G = U diag(s) U^T``, the problem is solved in
the standard form of the Tikhonov problem [1]_: the weights ``w = T z``,
with ``T = U diag(s)^(-1/2)``, have the energy ``|z|^2``, and ``z`` is
computed from the singular value decomposition of ``K T``.

References
----------
.. [1] P. C. Hansen, "Rank-Deficient and Discrete Ill-Posed Problems",
    SIAM, 1998, chapter 2.3.
"""

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Tikhonov",
]

# Eigenvalues of the energy are floored at this fraction of the largest one
_ENERGY_FLOOR = 1e-9


class Tikhonov:
    """Least-squares fit regularized by a quadratic energy.

    The decompositions of the collocation matrix and of the energy are
    computed once, when the problem is built, so that its solutions for
    several values, or several regularizations, are cheap.

    Parameters
    ----------
    collocation : array_like, shape (m, n)
        Values of the ``n`` kernels at the ``m`` sites, such as given by
        :func:`rivoli.kernels.collocation_matrix`.
    energy : array_like, shape (n, n)
        Symmetric positive semidefinite matrix of the energy of the
        weights, such as given by
        :func:`rivoli.dirichlet.dirichlet_matrix`.

    Raises
    ------
    ValueError
        If the shapes of `collocation` and `energy` do not match.
    """

    def __init__(self, collocation: "ArrayLike", energy: "ArrayLike") -> None:
        collocation = np.asarray(collocation, dtype=np.float64)
        energy = np.asarray(energy, dtype=np.float64)
        if collocation.ndim != 2 or energy.shape != (collocation.shape[1],) * 2:
            raise ValueError(
                "expected a collocation matrix of shape (m, n) and an energy of shape (n, n), "
                f"got {collocation.shape} and {energy.shape}"
            )
        self._collocation = collocation

        # Floored energy G = U diag(s) U^T
        s, u = np.linalg.eigh(energy)
        largest = s.max(initial=0.0)
        if largest <= 0.0:
            raise ValueError("the energy must not be zero")
        s = np.maximum(s, _ENERGY_FLOOR * largest)
        # Weights w = T z, where z has the energy |z|^2
        self._transform = u / np.sqrt(s)
        self._standard = np.linalg.svd(collocation @ self._transform, full_matrices=False)

    @cached_property
    def _least_squares(
        self,
    ) -> tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"]:
        """Decomposition of the plain least-squares fit, for a zero regularization."""
        return np.linalg.svd(self._collocation, full_matrices=False)

    def solve(self, values: "ArrayLike", regularization: float) -> "NDArray[np.float64]":
        """Fit the weights of the kernels to values at the sites.

        Parameters
        ----------
        values : array_like, shape (m,) or (m, c)
            Values at the sites, possibly with several channels, which
            are fitted independently.
        regularization : float
            Weight ``lambda`` of the energy, in ``[0, 1)``.

        Returns
        -------
        ndarray, shape (n,) or (n, c)
            Weights of the kernels.

        Raises
        ------
        ValueError
            If `values` do not have one row per site, or if
            `regularization` is not in ``[0, 1)``.
        """
        values = np.asarray(values, dtype=np.float64)
        m = self._collocation.shape[0]
        if values.shape[:1] != (m,) or values.ndim > 2:
            raise ValueError(f"expected values of shape ({m},) or ({m}, c), got {values.shape}")
        _check_regularization(regularization)
        if regularization == 0.0:
            return _truncated_solve(self._least_squares, values)

        # Multiplier of |z|^2 for the sum of the squared residuals
        mu = m * regularization / (1.0 - regularization)
        left, singular_values, right = self._standard
        filters = singular_values / (singular_values**2 + mu)
        return self._transform @ (right.T @ _scale_rows(filters, left.T @ values))


def _check_regularization(regularization: float) -> None:
    if not 0.0 <= regularization < 1.0:
        raise ValueError(f"the regularization must be in [0, 1), got {regularization}")


def _scale_rows(
    factors: "NDArray[np.float64]", matrix: "NDArray[np.float64]"
) -> "NDArray[np.float64]":
    """Multiply the rows of a vector or a matrix by factors."""
    return factors.reshape(-1, *(1,) * (matrix.ndim - 1)) * matrix


def _truncated_solve(
    svd: tuple["NDArray[np.float64]", "NDArray[np.float64]", "NDArray[np.float64]"],
    values: "NDArray[np.float64]",
) -> "NDArray[np.float64]":
    """Minimum-norm least-squares solution, without the negligible singular values."""
    left, singular_values, right = svd
    cutoff = np.finfo(np.float64).eps * max(left.shape[0], right.shape[0]) * singular_values[0]
    inverses = np.divide(
        1.0, singular_values, out=np.zeros_like(singular_values), where=singular_values > cutoff
    )
    return right.T @ _scale_rows(inverses, left.T @ values)
