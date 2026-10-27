# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Physically-based interpolation by radial basis functions.

Dedicated to BRDF, spherical or bispherical modeling functions
and data reconstruction.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rivoli")
except PackageNotFoundError:  # pragma: no cover
    # Fallback when the source tree is imported without install
    # Tests always run on an installed package, so this line is
    # excluded from coverage measurement thanks to the pragma.
    __version__ = "0+unknown"
