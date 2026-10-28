# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

# The bodies of Warp functions and kernels are in the kernel language of Warp,
# which Warp type checks when compiling them: its stubs give no operators nor
# indexing to its vector types, and its array annotations are calls
# pyright: reportOperatorIssue=false, reportIndexIssue=false, reportInvalidTypeForm=false

"""Dirichlet energy of radial basis function expansions, with Warp.

This is the parallel counterpart of :mod:`rivoli.dirichlet`, which
documents the matrix and its quadrature. The values and gradients of the
kernels at the quadrature nodes, and their products, are computed on the
device, in double precision, as in NumPy. Their products are plain
loops, which are faster than the matrix products of NumPy on a GPU, but
not on the CPU.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from rivoli.dirichlet import _assemble, _problem

try:
    import warp as wp
except ModuleNotFoundError as error:  # pragma: no cover
    # Tests always run with Warp installed
    raise ModuleNotFoundError(
        "rivoli.warp requires Warp, installed by `pip install rivoli[warp]`"
    ) from error

from rivoli.warp.distances import _GEODESIC, _METRICS, chordal, geodesic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from rivoli.dirichlet import _Rule

__all__ = [
    "dirichlet_matrix",
]

# Number of products of a node and a centre whose values and gradients are
# held in device memory at once, about 100 MB
_PAIRS_PER_CHUNK = 2**22


@wp.kernel
def _sphere_kernels(
    nodes: wp.array(dtype=wp.vec3d),
    tangents: wp.array2d(dtype=wp.vec3d),
    roots: wp.array(dtype=wp.float64),
    centres: wp.array(dtype=wp.vec3d),
    width: wp.float64,
    metric: int,
    values: wp.array2d(dtype=wp.float64),
    gradients: wp.array3d(dtype=wp.float64),
) -> None:
    # Warp's stubs do not tell that the thread index of a 2D launch is a pair
    q, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    x = nodes[q]
    c = centres[j]
    ratio = wp.float64(1.0)
    if metric == _GEODESIC:
        distance = geodesic(x, c)
        # The gradient of the geodesic distance is that of the chordal one
        # times gamma / sin(gamma), which tends to 1 at the centre
        sine = wp.float64(0.5) * wp.length(x - c) * wp.length(x + c)
        if sine > wp.float64(0.0):
            ratio = distance / sine
    else:
        distance = chordal(x, c)
    scaled = distance / width
    value = roots[q] * wp.exp(-scaled * scaled)
    values[q, j] = value
    factor = wp.float64(2.0) / (width * width) * ratio * value
    for k in range(tangents.shape[1]):
        gradients[k, q, j] = factor * wp.dot(c, tangents[q, k])


@wp.kernel
def _interval_kernels(
    nodes: wp.array(dtype=wp.float64),
    roots: wp.array(dtype=wp.float64),
    centres: wp.array(dtype=wp.float64),
    width: wp.float64,
    values: wp.array2d(dtype=wp.float64),
    gradients: wp.array3d(dtype=wp.float64),
) -> None:
    q, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    offset = (nodes[q] - centres[j]) / width
    value = roots[q] * wp.exp(-offset * offset)
    values[q, j] = value
    gradients[0, q, j] = wp.float64(-2.0) / width * offset * value


@wp.kernel
def _accumulate(
    values: wp.array2d(dtype=wp.float64),
    gradients: wp.array3d(dtype=wp.float64),
    mass: wp.array2d(dtype=wp.float64),
    stiffness: wp.array2d(dtype=wp.float64),
) -> None:
    i, j = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    # The matrices are symmetric: only their upper triangles are computed
    if j < i:
        return
    product = wp.float64(0.0)
    gradient_product = wp.float64(0.0)
    for q in range(values.shape[0]):
        product += values[q, i] * values[q, j]
        for k in range(gradients.shape[0]):
            gradient_product += gradients[k, q, i] * gradients[k, q, j]
    mass[i, j] += product
    stiffness[i, j] += gradient_product


def _symmetric(upper: wp.array) -> "NDArray[np.float64]":
    """Fill the lower triangle of a matrix from its upper one."""
    matrix = np.asarray(upper.numpy(), dtype=np.float64)
    return np.triu(matrix) + np.triu(matrix, 1).T


def _blocks(
    rule: "_Rule",
    centres: "NDArray[np.float64]",
    width: float,
    metric: str,
    device: str | None,
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Mass and stiffness matrices of the Gaussians of the centres on a factor."""
    n = len(centres)
    dtype = wp.float64 if rule.tangents is None else wp.vec3d
    centres_array = wp.array(np.ascontiguousarray(centres), dtype=dtype, device=device)
    mass = wp.zeros((n, n), dtype=wp.float64, device=device)
    stiffness = wp.zeros((n, n), dtype=wp.float64, device=device)
    components = 1 if rule.tangents is None else rule.tangents.shape[1]
    chunk = max(1, _PAIRS_PER_CHUNK // n)
    for low in range(0, len(rule.weights), chunk):
        rows = slice(low, low + chunk)
        count = len(rule.weights[rows])
        nodes = wp.array(np.ascontiguousarray(rule.nodes[rows]), dtype=dtype, device=device)
        roots = wp.array(np.sqrt(rule.weights[rows]), dtype=wp.float64, device=device)
        # Warp's stubs type dtype as a Python type, not as one of its scalar types
        values = wp.empty(
            (count, n),
            dtype=wp.float64,  # pyright: ignore[reportArgumentType]
            device=device,
        )
        gradients = wp.empty(
            (components, count, n),
            dtype=wp.float64,  # pyright: ignore[reportArgumentType]
            device=device,
        )
        if rule.tangents is None:
            inputs = [nodes, roots, centres_array, width]
            wp.launch(
                _interval_kernels,
                dim=(count, n),
                inputs=inputs,
                outputs=[values, gradients],
                device=device,
            )
        else:
            tangents = wp.array(
                np.ascontiguousarray(rule.tangents[rows]), dtype=wp.vec3d, device=device
            )
            inputs = [nodes, tangents, roots, centres_array, width, _METRICS[metric]]
            wp.launch(
                _sphere_kernels,
                dim=(count, n),
                inputs=inputs,
                outputs=[values, gradients],
                device=device,
            )
        wp.launch(
            _accumulate,
            dim=(n, n),
            inputs=[values, gradients],
            outputs=[mass, stiffness],
            device=device,
        )
    return _symmetric(mass), _symmetric(stiffness)


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
    device: str | None = None,
) -> "NDArray[np.float64]":
    """Compute the Dirichlet matrix of kernels on the Rusinkiewicz domain.

    See :func:`rivoli.dirichlet.dirichlet_matrix`, which this function
    computes with one thread per pair of a quadrature node and a centre,
    then one thread per pair of centres.

    Parameters
    ----------
    centres : sequence of array_like
        ``(h, d)`` or ``(h, d, wavelength)`` of the centres.
    widths : tuple of float
        Positive widths of the kernels on each factor.
    metric : {"geodesic", "chordal"}
        Distance between the half vectors and between the difference
        vectors in the kernels.
    isotropic : bool
        Whether the half vectors have a zero azimuth.
    wavelength_range : tuple of float, optional
        Lower and upper bounds of the wavelengths, required if and only
        if the centres have wavelengths.
    scales : tuple of float, optional
        Positive scales of the distances on each factor. Default: 1 for
        each.
    kernel : {"gaussian"}, default "gaussian"
        Profile of the kernels.
    refinement : int, default 1
        Multiplier of the numbers of quadrature nodes along each
        dimension.
    device : str, optional
        Warp device, such as ``"cpu"`` or ``"cuda:0"``. Default: the
        default device of Warp.

    Returns
    -------
    ndarray, shape (n, n)
        Dirichlet matrix of the kernels.

    Raises
    ------
    ValueError
        As :func:`rivoli.dirichlet.dirichlet_matrix`.
    """
    problem = _problem(
        centres, widths, metric, isotropic, wavelength_range, scales, kernel, refinement
    )
    blocks = [
        _blocks(rule, factor, width, metric, device)
        for rule, factor, width in zip(problem.rules, problem.centres, problem.widths, strict=True)
    ]
    return _assemble(blocks, problem.scales)
