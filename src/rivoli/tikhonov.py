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

The energy is positive semidefinite. Its null space, such as the
constant functions for a Dirichlet energy, is not penalized: the problem
is solved in the standard form of the generalized Tikhonov problem [1]_,
in which the penalized part of the weights has the identity as energy,
and the unpenalized part is fitted by least squares. Both are computed
from singular value decompositions, which give a minimum-norm solution
when several weights synthesize the same function, as happens with
close or wide kernels.

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

# Eigenvalues of the energy below this fraction of the largest one are
# taken as 0, as numerical noise on its null space
_NULL_ENERGY = 1e-9
# Singular values of the fit of the unpenalized part below this fraction of
# the largest one are dropped: their weights synthesize no function
_NULL_SINGULAR_VALUE = 1e-10


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

        # Energy G = U diag(s) U^T, whose small eigenvalues are its null space
        s, u = np.linalg.eigh(energy)
        s = np.maximum(s, 0.0)
        penalized = s > _NULL_ENERGY * s.max(initial=0.0)
        if not np.any(penalized):
            raise ValueError("the energy must not be zero")
        # Weights w = T z + Z a, where z has the energy |z|^2, and a none
        self._transform = u[:, penalized] / np.sqrt(s[penalized])
        self._null_basis = u[:, ~penalized]

        # The unpenalized part a fits what it can of the values, by least
        # squares, so z only fits the rest, orthogonal to the range of K Z
        fitted = collocation @ self._null_basis
        left, singular_values, right = np.linalg.svd(fitted, full_matrices=False)
        kept = singular_values > _NULL_SINGULAR_VALUE * singular_values.max(initial=0.0)
        self._null_range = left[:, kept]
        self._null_pseudoinverse = (right[kept].T / singular_values[kept]) @ left[:, kept].T
        standard = collocation @ self._transform
        standard -= self._null_range @ (self._null_range.T @ standard)
        self._standard = np.linalg.svd(standard, full_matrices=False)

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
        remainder = values - self._null_range @ (self._null_range.T @ values)
        filters = singular_values / (singular_values**2 + mu)
        z = right.T @ _scale_rows(filters, left.T @ remainder)
        weights = self._transform @ z
        residual = values - self._collocation @ weights
        return weights + self._null_basis @ (self._null_pseudoinverse @ residual)


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
