# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.tikhonov, the regularized least-squares solver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli.tikhonov import Tikhonov


@pytest.fixture
def rng():
    return np.random.default_rng(20261029)


def random_problem(rng, m=50, n=20, null=3):
    """Collocation matrix and an energy with a null space of the given dimension."""
    collocation = rng.normal(size=(m, n))
    root = rng.normal(size=(n, n - null))
    return collocation, root @ root.T


@pytest.mark.parametrize("regularization", [1e-6, 0.1, 0.5, 0.99])
@pytest.mark.parametrize("channels", [(), (3,)])
def test_solution_solves_the_normal_equations(rng, regularization, channels):
    collocation, energy = random_problem(rng)
    values = rng.normal(size=(50, *channels))
    weights = Tikhonov(collocation, energy).solve(values, regularization)
    assert weights.shape == (20, *channels)
    # Minimizer of (1 - lambda) |K w - f|^2 / m + lambda w^T G w
    mu = 50 * regularization / (1 - regularization)
    expected = np.linalg.solve(collocation.T @ collocation + mu * energy, collocation.T @ values)
    assert_allclose(weights, expected, rtol=0, atol=1e-12 * np.abs(expected).max())


def test_zero_regularization_is_least_squares(rng):
    collocation, energy = random_problem(rng)
    values = rng.normal(size=50)
    weights = Tikhonov(collocation, energy).solve(values, 0.0)
    assert_allclose(weights, np.linalg.lstsq(collocation, values)[0], atol=1e-14)


def test_zero_regularization_interpolates(rng):
    collocation, energy = random_problem(rng, m=20)
    values = rng.normal(size=20)
    weights = Tikhonov(collocation, energy).solve(values, 0.0)
    assert_allclose(collocation @ weights, values, atol=1e-12)


def test_regularization_trades_the_residual_for_the_energy(rng):
    collocation, energy = random_problem(rng)
    values = rng.normal(size=50)
    problem = Tikhonov(collocation, energy)
    solutions = [problem.solve(values, r) for r in (0.0, 0.01, 0.1, 0.5, 0.9, 0.999)]
    residuals = [np.linalg.norm(collocation @ w - values) for w in solutions]
    energies = [w @ energy @ w for w in solutions]
    assert np.all(np.diff(residuals) > 0)
    assert np.all(np.diff(energies) < 0)


def test_the_null_space_of_the_energy_is_not_penalized(rng):
    collocation, energy = random_problem(rng)
    # Values synthesized by weights of zero energy are fitted exactly
    null_basis = np.linalg.eigh(energy)[1][:, :3]
    values = collocation @ null_basis @ rng.normal(size=3)
    weights = Tikhonov(collocation, energy).solve(values, 0.999)
    assert_allclose(collocation @ weights, values, atol=1e-12)
    assert abs(weights @ energy @ weights) < 1e-12 * np.abs(energy).max()


def test_duplicate_kernels_give_the_same_fit(rng):
    # A duplicate kernel adds weights that synthesize the zero function, of
    # zero energy, which must not disturb the fit
    collocation, energy = random_problem(rng, null=1)
    merge = np.hstack([np.eye(20), np.eye(20)[:, :1]])
    values = rng.normal(size=50)
    for regularization in (0.0, 0.3):
        expected = Tikhonov(collocation, energy).solve(values, regularization)
        duplicated = Tikhonov(collocation @ merge, merge.T @ energy @ merge)
        weights = duplicated.solve(values, regularization)
        assert np.all(np.isfinite(weights))
        assert_allclose(merge @ weights, expected, atol=1e-10 * np.abs(expected).max())


def test_positive_definite_energy(rng):
    # Without a null space, nothing is left to the unpenalized part
    collocation, energy = random_problem(rng, null=0)
    values = rng.normal(size=50)
    weights = Tikhonov(collocation, energy).solve(values, 0.2)
    mu = 50 * 0.2 / 0.8
    expected = np.linalg.solve(collocation.T @ collocation + mu * energy, collocation.T @ values)
    assert_allclose(weights, expected, rtol=1e-12)


def test_invalid_problems(rng):
    collocation, energy = random_problem(rng)
    with pytest.raises(ValueError, match="got \\(50, 20\\) and \\(19, 19\\)"):
        Tikhonov(collocation, energy[1:, 1:])
    with pytest.raises(ValueError, match="got \\(50,\\) and"):
        Tikhonov(collocation[:, 0], energy)
    with pytest.raises(ValueError, match="must not be zero"):
        Tikhonov(collocation, np.zeros_like(energy))


@pytest.mark.parametrize("regularization", [-0.1, 1.0, 2.0])
def test_invalid_regularization(rng, regularization):
    problem = Tikhonov(*random_problem(rng))
    with pytest.raises(ValueError, match="must be in \\[0, 1\\)"):
        problem.solve(np.zeros(50), regularization)


@pytest.mark.parametrize("shape", [(49,), (50, 2, 2)])
def test_invalid_values(rng, shape):
    problem = Tikhonov(*random_problem(rng))
    with pytest.raises(ValueError, match="expected values of shape"):
        problem.solve(np.zeros(shape), 0.5)
