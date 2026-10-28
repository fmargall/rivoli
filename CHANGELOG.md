# Changelog

All notable changes to RIVOLI are documented in this file, for its
users: changes to the repository tooling (CI, linters, templates) are not listed.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `rivoli.geometry`: conversions between spherical angles and unit vectors,
  normalization and rotations around an axis, vectorized over batches of vectors.
- `rivoli.parametrization`: conversions between pairs of directions and the
  Rusinkiewicz parametrization, as half and difference vectors or as angles.
  The spherical angles of the directions are converted by
  `spherical_to_rusinkiewicz` and `rusinkiewicz_to_spherical`, with four angles,
  or three for isotropic BRDFs.
- `rivoli.sampling`: Super-Fibonacci low-discrepancy sampling of the
  Rusinkiewicz domain, for isotropic and anisotropic BRDFs, with an optional
  spectral dimension sampled jointly with the angles. With `above_horizon=True`,
  the samplers return exactly the requested number of sites whose incident and
  outgoing directions are both above the horizon, as told by
  `rivoli.parametrization.is_above_horizon`.
- `rivoli.distances`: geodesic and chordal distances between unit vectors, and
  the spacings between each site of a sampling and its nearest neighbour. The
  distances on each factor of the domain can be divided by scales, such as the
  diameters of the factors given by `diameters`, to make them comparable.
- `rivoli.warp.distances`: the same distances and spacings computed in parallel
  on a GPU or the CPU with NVIDIA Warp, installed by the `warp` extra.

[unreleased]: https://github.com/fmargall/rivoli/commits/HEAD
