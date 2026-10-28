# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Dirichlet energy of radial basis function expansions.

The Dirichlet energy of a function ``f`` on a domain ``Omega`` measures
how much it varies: it is the mean of the squared norm of its gradient,
``E(f) = (1 / |Omega|) int_Omega |grad f|^2``. For an expansion
``f = sum_j w_j phi_j`` on kernels ``phi_j``, it is the quadratic form
``E(f) = w^T G w`` of the Dirichlet matrix
``G_ij = (1 / |Omega|) int_Omega grad phi_i . grad phi_j``, which
regularizes the fit of the weights ``w``.

The domain is a product of factors, as in :mod:`rivoli.distances`:

-   the half vectors span a quarter of a meridian, ``theta_h`` in
    ``[0, pi / 2]`` measured by its length, for an isotropic BRDF, and
    the hemisphere measured by its area for an anisotropic one;
-   the difference vectors span the hemisphere;
-   the wavelengths, for a spectral BRDF, span an interval.

As in earlier versions of RIVOLI, this is the whole domain, below the
horizon too, without the symmetries of the BRDF. The squared norm of a
gradient on the product is the sum of those of its components on each
factor, where the gradient on a hemisphere is that of the round sphere.
They can be weighted by the squares of scales, which is the energy for
the distances of each factor divided by its scale: with the diameters
given by :func:`rivoli.distances.diameters`, each factor is measured
relative to its extent, and the wavelengths are made commensurate with
the angles.

The Gaussian kernel is the product of a Gaussian on each factor, so the
Dirichlet matrix is the sum, over the factors ``k``, of the Hadamard
product of the stiffness matrix of ``k``, made of the integrals of the
products of the gradients of the Gaussians on ``k``, with the mass
matrices of the other factors, made of the integrals of the products of
their Gaussians. These integrals are computed by quadrature: composite
Gauss-Legendre rules, with panels no longer than the width of the
Gaussians, along the polar angle and the wavelength, and the
trapezoidal rule, exact up to rounding for smooth periodic functions,
along the azimuth. Their error is close to the rounding error, except
with the geodesic metric and widths beyond about 1 radian: the geodesic
distance is not smooth at the antipode of a centre, which reaches the
hemisphere for a centre on the equator, and the error grows as
``exp(-pi^2 / width^2)``.

The kernels of :mod:`rivoli.warp.dirichlet` compute the same matrix in
parallel, on the CPU or a GPU.
"""

import math
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from rivoli.distances import _PAIRS_PER_CHUNK, _check_metric, _check_scales, _factors
from rivoli.geometry import spherical_to_cartesian
from rivoli.kernels import _check_kernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "dirichlet_matrix",
]

# Gauss-Legendre nodes per panel, whose length is at most the width of the
# Gaussians: the integrals of their products are then exact up to rounding
_NODES_PER_PANEL = 10
# Trapezoidal nodes along the azimuth per radian of width, and at least, for
# the same accuracy
_AZIMUTHAL_NODES_PER_WIDTH = 24
_MIN_AZIMUTHAL_NODES = 16
# Largest azimuthal component of an isotropic half vector, which should be 0
_AZIMUTH_TOLERANCE = 1e-12


class _Rule(NamedTuple):
    """Quadrature rule on a factor, whose weights sum to 1.

    The nodes are unit vectors of shape (q, 3) on a hemisphere, with an
    orthonormal basis of the tangent space of the factor at each node, of
    shape (q, k, 3), or wavelengths of shape (q,), without tangents.
    """

    nodes: "NDArray[np.float64]"
    tangents: "NDArray[np.float64] | None"
    weights: "NDArray[np.float64]"


def _gauss_legendre(
    low: float, high: float, width: float, refinement: int
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Composite Gauss-Legendre rule on [low, high], with panels at most `width` long."""
    panels = refinement * max(1, math.ceil((high - low) / width))
    edges = np.linspace(low, high, panels + 1)
    x, w = np.polynomial.legendre.leggauss(_NODES_PER_PANEL)
    half = 0.5 * np.diff(edges)[:, np.newaxis]
    middle = 0.5 * (edges[1:] + edges[:-1])[:, np.newaxis]
    return (middle + half * x).ravel(), (half * w).ravel()


def _meridian_rule(width: float, refinement: int) -> _Rule:
    """Rule on the quarter of meridian of isotropic half vectors."""
    theta, weights = _gauss_legendre(0.0, np.pi / 2, width, refinement)
    nodes = spherical_to_cartesian(theta, 0.0)
    tangents = spherical_to_cartesian(theta + np.pi / 2, 0.0)[:, np.newaxis]
    return _Rule(nodes, tangents, weights / (np.pi / 2))


def _hemisphere_rule(width: float, refinement: int) -> _Rule:
    """Rule on the hemisphere, as a product of rules along the polar angle and the azimuth."""
    theta, polar_weights = _gauss_legendre(0.0, np.pi / 2, width, refinement)
    count = refinement * max(_MIN_AZIMUTHAL_NODES, math.ceil(_AZIMUTHAL_NODES_PER_WIDTH / width))
    phi = -np.pi + 2 * np.pi * (np.arange(count) + 0.5) / count
    theta, phi = (x.ravel() for x in np.meshgrid(theta, phi, indexing="ij"))
    nodes = spherical_to_cartesian(theta, phi)
    # Unit vectors of increasing polar angle and of increasing azimuth
    tangents = np.stack(
        [
            spherical_to_cartesian(theta + np.pi / 2, phi),
            spherical_to_cartesian(np.pi / 2, phi + np.pi / 2),
        ],
        axis=1,
    )
    # The area element sin(theta) dtheta dphi, over the area 2 pi
    weights = np.repeat(polar_weights, count) * np.sin(theta) / count
    return _Rule(nodes, tangents, weights)


def _interval_rule(low: float, high: float, width: float, refinement: int) -> _Rule:
    """Rule on the interval of wavelengths."""
    nodes, weights = _gauss_legendre(low, high, width, refinement)
    return _Rule(nodes, None, weights / (high - low))


def _sphere_kernels(
    nodes: "NDArray[np.float64]",
    tangents: "NDArray[np.float64]",
    centres: "NDArray[np.float64]",
    width: float,
    metric: str,
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Values, of shape (q, n), and tangent gradients, of shape (k, q, n), of Gaussians."""
    x = nodes[:, np.newaxis]
    difference = np.linalg.norm(x - centres, axis=-1)
    if metric == "geodesic":
        distance = 2.0 * np.arctan2(difference, np.linalg.norm(x + centres, axis=-1))
        # The gradient of the geodesic distance is that of the chordal one
        # times gamma / sin(gamma), which tends to 1 at the centre
        sine = 0.5 * difference * np.linalg.norm(x + centres, axis=-1)
        ratio = np.divide(distance, sine, out=np.ones_like(distance), where=sine > 0)
    else:
        distance, ratio = difference, 1.0
    values = np.exp(-((distance / width) ** 2))
    # On the unit sphere, the gradient of -|x - c|^2 is 2 (c - (x . c) x),
    # whose tangent components are 2 c . t
    gradients = (2.0 / width**2) * ratio * values * np.einsum("qkj,nj->kqn", tangents, centres)
    return values, gradients


def _interval_kernels(
    nodes: "NDArray[np.float64]", centres: "NDArray[np.float64]", width: float
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Values, of shape (q, n), and derivatives, of shape (1, q, n), of Gaussians."""
    offset = (nodes[:, np.newaxis] - centres) / width
    values = np.exp(-(offset**2))
    return values, (-2.0 / width * offset * values)[np.newaxis]


def _blocks(
    rule: _Rule, centres: "NDArray[np.float64]", width: float, metric: str
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Mass and stiffness matrices of the Gaussians of the centres on a factor."""
    n = len(centres)
    mass = np.zeros((n, n))
    stiffness = np.zeros((n, n))
    chunk = max(1, _PAIRS_PER_CHUNK // n)
    for low in range(0, len(rule.weights), chunk):
        rows = slice(low, low + chunk)
        if rule.tangents is None:
            values, gradients = _interval_kernels(rule.nodes[rows], centres, width)
        else:
            values, gradients = _sphere_kernels(
                rule.nodes[rows], rule.tangents[rows], centres, width, metric
            )
        root = np.sqrt(rule.weights[rows])[:, np.newaxis]
        values *= root
        gradients *= root
        mass += values.T @ values
        for component in gradients:
            stiffness += component.T @ component
    return mass, stiffness


class _Problem(NamedTuple):
    """Checked arguments of dirichlet_matrix, with a rule per factor."""

    rules: list[_Rule]
    centres: list["NDArray[np.float64]"]
    widths: tuple[float, ...]
    scales: tuple[float, ...]


def _problem(
    centres: "Sequence[ArrayLike]",
    widths: tuple[float, ...],
    metric: str,
    isotropic: bool,
    wavelength_range: tuple[float, float] | None,
    scales: tuple[float, ...] | None,
    kernel: str,
    refinement: int,
) -> _Problem:
    """Check the arguments of dirichlet_matrix, and build the rules of its factors."""
    _check_kernel(kernel)
    _check_metric(metric)
    factors = _factors(centres, "centres")
    if (len(factors) == 3) != (wavelength_range is not None):
        raise ValueError("wavelength_range must be given if and only if centres have wavelengths")
    widths = _check_scales(widths, len(factors))
    scales = _check_scales(scales, len(factors))
    if not (isinstance(refinement, int) and refinement >= 1):
        raise ValueError(f"refinement must be a positive integer, got {refinement!r}")
    h = factors[0]
    if isotropic and (
        np.any(np.abs(h[:, 1]) > _AZIMUTH_TOLERANCE) or np.any(h[:, 0] < -_AZIMUTH_TOLERANCE)
    ):
        raise ValueError("the half vectors of an isotropic BRDF must have a zero azimuth")

    half_rule = _meridian_rule if isotropic else _hemisphere_rule
    rules = [half_rule(widths[0], refinement), _hemisphere_rule(widths[1], refinement)]
    if wavelength_range is not None:
        low, high = wavelength_range
        if not low < high:
            raise ValueError(f"the wavelength range must be increasing, got {wavelength_range}")
        rules.append(_interval_rule(low, high, widths[2], refinement))
    return _Problem(rules, factors, widths, scales)


def _assemble(
    blocks: list[tuple["NDArray[np.float64]", "NDArray[np.float64]"]], scales: tuple[float, ...]
) -> "NDArray[np.float64]":
    """Sum the stiffness of each factor times the masses of the others."""
    matrix = np.zeros_like(blocks[0][0])
    for k, (_, stiffness) in enumerate(blocks):
        term = scales[k] ** 2 * stiffness
        for mass, _ in blocks[:k] + blocks[k + 1 :]:
            term *= mass
        matrix += term
    # The products of the quadrature are symmetric only up to rounding
    return 0.5 * (matrix + matrix.T)


def dirichlet_matrix(
    centres: "Sequence[ArrayLike]",
    *,
    widths: tuple[float, ...],
    metric: Literal["geodesic", "chordal"],
    isotropic: bool,
    wavelength_range: tuple[float, float] | None = None,
    scales: tuple[float, ...] | None = None,
    kernel: Literal["gaussian"] = "gaussian",
    refinement: int = 1,
) -> "NDArray[np.float64]":
    """Compute the Dirichlet matrix of kernels on the Rusinkiewicz domain.

    The entry ``(i, j)`` is the mean over the domain of the scalar
    product of the gradients of the kernels of the centres ``i`` and
    ``j``, so that the Dirichlet energy of ``sum_j w_j phi_j`` is
    ``w^T G w``. The matrix is symmetric positive semidefinite.

    Parameters
    ----------
    centres : sequence of array_like
        ``(h, d)`` or, for a spectral BRDF, ``(h, d, wavelength)``: the
        half vectors, of shape ``(n, 3)``, the difference vectors, of
        shape ``(n, 3)``, and the wavelengths, of shape ``(n,)``, of the
        centres of the kernels.
    widths : tuple of float
        Positive widths of the kernels on each factor, as given to
        :func:`rivoli.kernels.collocation_matrix`.
    metric : {"geodesic", "chordal"}
        Distance between the half vectors and between the difference
        vectors in the kernels.
    isotropic : bool
        Whether the BRDF is isotropic: its half vectors then have a zero
        azimuth, and span a quarter of a meridian instead of the
        hemisphere.
    wavelength_range : tuple of float, optional
        Lower and upper bounds of the wavelengths, required if and only
        if the centres have wavelengths.
    scales : tuple of float, optional
        Positive scales of the distances on each factor, such as the
        :func:`rivoli.distances.diameters` of the domain: the squared
        gradient on each factor is multiplied by the square of its
        scale. Default: 1 for each, which weighs angles and wavelengths
        in their own units.
    kernel : {"gaussian"}, default "gaussian"
        Profile of the kernels.
    refinement : int, default 1
        Multiplier of the numbers of quadrature nodes along each
        dimension, to check the convergence of the quadrature.

    Returns
    -------
    ndarray, shape (n, n)
        Dirichlet matrix of the kernels.

    Raises
    ------
    ValueError
        If `kernel` or `metric` is unknown, if `centres` do not have 2
        or 3 factors, if `widths` or `scales` are not one positive
        number per factor, if `wavelength_range` is missing, unexpected
        or not increasing, if `refinement` is not a positive integer, or
        if the half vectors of an isotropic BRDF have a nonzero azimuth.
    """
    problem = _problem(
        centres, widths, metric, isotropic, wavelength_range, scales, kernel, refinement
    )
    blocks = [
        _blocks(rule, factor, width, metric)
        for rule, factor, width in zip(problem.rules, problem.centres, problem.widths, strict=True)
    ]
    return _assemble(blocks, problem.scales)
