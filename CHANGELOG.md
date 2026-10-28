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
  spectral dimension sampled jointly with the angles.

[unreleased]: https://github.com/fmargall/rivoli/commits/HEAD
