# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Measurements of BRDFs read from HDF5 files described by a layout.

HDF5 files of measurements have as many structures as instruments. Rather
than a reader per format, :func:`read_hdf5` reads any file whose
measurements are tables on grids of angles and wavelengths, given a
layout that tells where each of them is. The layout is a mapping, or a
JSON file holding it, such as::

    {
        "angle_unit": "degree",
        "axes": {
            "theta_i": "/@incidences",
            "phi_i": 180.0,
            "theta_o": "/incidence_{theta_i}@zeniths",
            "phi_o": "/@azimuths",
            "wavelength": "/spectrum",
        },
        "values": {"path": "/incidence_{theta_i}/band_{wavelength}", "axes": ["phi_o", "theta_o"]},
        "cosine_weighted": "/@cosine",
    }

-   ``axes`` gives the grid of each coordinate of the measurements: the
    polar and azimuthal angles of the incident and outgoing directions,
    ``theta_i``, ``phi_i``, ``theta_o`` and ``phi_o``, all required,
    and the optional ``wavelength``. A grid is either a number, a
    coordinate shared by all the measurements, or a reference to an
    array of the file: ``"path"`` for a dataset, and ``"path@name"`` for
    the attribute ``name`` of the group or dataset at ``path``.
-   ``values`` gives the tables of values: ``path`` is the template of
    their paths, and ``axes`` the coordinates along the dimensions of
    each table. The other coordinates read from the file are indices in
    the templates: ``{theta_i}`` is replaced by the index of the incident
    polar angle in its grid, and the tables are read for all the indices.
    References of grids can be templates too, such as a grid of outgoing
    polar angles that depends on the incident one.
-   ``angle_unit``, ``"degree"`` or ``"radian"``, is the unit of the
    angles, including the numbers of ``axes``. Default: ``"radian"``.
-   ``cosine_weighted``, optional, tells whether the values are the BRDF
    times ``cos(theta_o)``, as measured by some instruments: either a
    boolean, or the reference of a value of the file, such as an
    attribute, read as a boolean, the strings ``"true"`` and ``"1"``
    meaning true whatever their case. The values are then divided by
    ``cos(theta_o)``. Default: false.

The measurements are returned as they are, converted to the Rusinkiewicz
parametrization of :mod:`rivoli.parametrization`, except for the missing
values, NaN in the file, which are dropped. Wavelengths are returned in
the unit of the file.
"""

import json
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from rivoli.geometry import spherical_to_cartesian
from rivoli.parametrization import spherical_to_rusinkiewicz

try:
    import h5py
except ModuleNotFoundError as error:  # pragma: no cover
    # Tests always run with h5py installed
    raise ModuleNotFoundError(
        "rivoli.hdf5 requires h5py, installed by `pip install rivoli[hdf5]`"
    ) from error

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import NDArray

__all__ = [
    "Measurements",
    "read_hdf5",
]

_ANGLES = ("theta_i", "phi_i", "theta_o", "phi_o")
_AXES = (*_ANGLES, "wavelength")
_ANGLE_UNITS = {"radian": 1.0, "degree": np.pi / 180}
_TRUE = ("true", "1")


class Measurements(NamedTuple):
    """Measurements of a BRDF, as given to :func:`rivoli.model.fit`.

    It unpacks as ``sites, values = measurements``.

    Attributes
    ----------
    sites : tuple of ndarray
        ``(h, d)``, or ``(h, d, wavelength)`` for a spectral BRDF: the
        half vectors, of shape ``(m, 3)``, the difference vectors, of
        shape ``(m, 3)``, and the wavelengths, of shape ``(m,)``, of the
        ``m`` measurements.
    values : ndarray, shape (m,)
        Measured values of the BRDF.
    """

    sites: tuple["NDArray[np.float64]", ...]
    values: "NDArray[np.float64]"


def _layout(layout: "Mapping[str, Any] | str | PathLike[str]") -> Mapping[str, Any]:
    """Load and check a layout."""
    if not isinstance(layout, Mapping):
        with Path(layout).open(encoding="utf-8") as stream:
            layout = json.load(stream)
        if not isinstance(layout, Mapping):
            raise ValueError("a layout must be a JSON object")
    unknown = set(layout) - {"angle_unit", "axes", "values", "cosine_weighted"}
    if unknown:
        raise ValueError(f"unknown keys in the layout: {sorted(unknown)}")
    if layout.get("angle_unit", "radian") not in _ANGLE_UNITS:
        raise ValueError(
            f"angle_unit must be one of {list(_ANGLE_UNITS)}, got {layout['angle_unit']!r}"
        )
    axes = layout.get("axes", {})
    if set(axes) - set(_AXES) or set(_ANGLES) - set(axes):
        raise ValueError(
            f"the axes must be {list(_ANGLES)}, and optionally 'wavelength', got {list(axes)}"
        )
    values = layout.get("values", {})
    if set(values) != {"path", "axes"}:
        raise ValueError(f"values must have a path and axes, got {list(values)}")
    table_axes = list(values["axes"])
    if set(table_axes) - set(axes) or len(set(table_axes)) != len(table_axes):
        raise ValueError(f"the axes of the values must be distinct axes, got {table_axes}")
    for name in table_axes:
        if not isinstance(axes[name], str):
            raise ValueError(f"the axis {name!r} of the values must be read from the file")
    return layout


def _read(file: h5py.File, reference: str, indices: Mapping[str, int]) -> "NDArray[Any]":
    """Read the dataset or attribute of a reference, whose template is filled with indices."""
    try:
        filled = reference.format_map(indices)
    except KeyError as error:
        raise ValueError(f"the reference {reference!r} depends on {error}, not an index") from None
    path, _, attribute = filled.partition("@")
    item = file.get(path or "/")
    if isinstance(item, h5py.Group | h5py.Dataset):
        if attribute and attribute in item.attrs:
            return np.asarray(item.attrs[attribute])
        if not attribute and isinstance(item, h5py.Dataset):
            return np.asarray(item[()])
    raise ValueError(f"{filled!r} is not an attribute or a dataset of the file")


def _grid(
    file: h5py.File, source: "float | str", indices: Mapping[str, int], scale: float
) -> "NDArray[np.float64]":
    """Coordinates of an axis, as a one-dimensional array."""
    values = source if not isinstance(source, str) else _read(file, source, indices)
    grid = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if grid.ndim != 1:
        raise ValueError(f"the grid {source!r} must be one-dimensional, got {grid.shape}")
    return grid * scale


def _flag(file: h5py.File, source: "bool | str") -> bool:
    """Read a boolean of the layout, or of the file."""
    if isinstance(source, bool):
        return source
    value = np.asarray(_read(file, source, {})).ravel()[0]
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip().lower() in _TRUE


def read_hdf5(
    path: "str | PathLike[str]",
    layout: "Mapping[str, Any] | str | PathLike[str]",
    *,
    isotropic: bool = True,
) -> Measurements:
    """Read measurements of a BRDF from an HDF5 file.

    Parameters
    ----------
    path : str or path-like
        Path of the HDF5 file.
    layout : mapping, str or path-like
        Layout of the file, as described in :mod:`rivoli.hdf5`, or the
        path of a JSON file holding it.
    isotropic : bool, default True
        Whether the BRDF is isotropic: its half vectors are then given a
        zero azimuth, which only keeps the difference of the azimuths of
        the incident and outgoing directions.

    Returns
    -------
    Measurements
        Sites and values of the measurements, without the missing ones.
        The sites have wavelengths if the layout has a wavelength axis.

    Raises
    ------
    ValueError
        If the layout is invalid, or does not match the file: missing
        references, or tables whose shape is not that of the grids of
        their axes.
    """
    layout = _layout(layout)
    axes, values = layout["axes"], layout["values"]
    scale = _ANGLE_UNITS[layout.get("angle_unit", "radian")]
    table_axes = list(values["axes"])
    # Axes along which tables are read one by one, the indices of the template
    indexed = [name for name in axes if "{" + name + "}" in values["path"]]
    if set(indexed) & set(table_axes):
        raise ValueError("an axis of the values cannot be an index of their path")

    columns: dict[str, list[NDArray[np.float64]]] = {name: [] for name in (*axes, "values")}
    with h5py.File(path, "r") as file:
        cosine_weighted = _flag(file, layout.get("cosine_weighted", False))
        # The grids of the indexed axes cannot depend on indices
        sizes = [len(_grid(file, axes[name], {}, 1.0)) for name in indexed]
        for combination in product(*(range(size) for size in sizes)):
            indices = dict(zip(indexed, combination, strict=True))
            grids = {
                name: _grid(file, source, indices, 1.0 if name == "wavelength" else scale)
                for name, source in axes.items()
            }
            table = np.asarray(_read(file, values["path"], indices), dtype=np.float64)
            shape = tuple(len(grids[name]) for name in table_axes)
            if table.shape != shape:
                raise ValueError(
                    f"the table {values['path'].format_map(indices)!r} has the shape "
                    f"{table.shape}, expected {shape} from the grids of {table_axes}"
                )
            # Coordinates of each entry of the table
            coordinates = {name: grids[name][indices[name]] for name in indexed}
            coordinates |= dict(
                zip(
                    table_axes,
                    np.meshgrid(*(grids[n] for n in table_axes), indexing="ij"),
                    strict=True,
                )
            )
            # The other axes are a single coordinate shared by the whole table
            for name in axes.keys() - coordinates.keys():
                if len(grids[name]) != 1:
                    raise ValueError(
                        f"the axis {name!r} has {len(grids[name])} coordinates, but is neither "
                        "an axis of the values nor an index of their path"
                    )
                coordinates[name] = grids[name][0]
            for name in axes:
                columns[name].append(np.broadcast_to(coordinates[name], shape).ravel())
            columns["values"].append(table.ravel())

    data = {name: np.concatenate(column) for name, column in columns.items()}
    measured = ~np.isnan(data["values"])
    data = {name: column[measured] for name, column in data.items()}
    if cosine_weighted:
        data["values"] = data["values"] / np.cos(data["theta_o"])
    angles = [data[name] for name in _ANGLES]
    if isotropic:
        theta_h, theta_d, phi_d = spherical_to_rusinkiewicz(
            angles[0], angles[2], angles[3] - angles[1]
        )
        phi_h = 0.0
    else:
        theta_h, phi_h, theta_d, phi_d = spherical_to_rusinkiewicz(*angles)
    sites = (spherical_to_cartesian(theta_h, phi_h), spherical_to_cartesian(theta_d, phi_d))
    if "wavelength" in data:
        sites = (*sites, data["wavelength"])
    return Measurements(sites, data["values"])
