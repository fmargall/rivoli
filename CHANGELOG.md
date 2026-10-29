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
- `rivoli.distances.pairwise_distances`: product distances between sites and
  centres, with a scale per factor.
- `rivoli.kernels`: Gaussian radial basis function kernel with a width per
  factor of the domain, and the collocation matrix of centres at sites. The
  Gaussian of the chordal distance is positive definite, unlike that of the
  geodesic one, which is kept for compatibility.
- `rivoli.warp.kernels`: the same collocation matrix computed with Warp.
- `rivoli.dirichlet`: Dirichlet matrix of the Gaussian kernels, whose quadratic
  form is the mean squared gradient of an expansion on the kernels over the
  whole Rusinkiewicz domain, with an optional weight per factor such as the
  squared diameters. It is computed by quadrature, accurate up to rounding.
- `rivoli.warp.dirichlet`: the same Dirichlet matrix computed with Warp.
- `rivoli.tikhonov`: least-squares fit of the weights of kernels regularized by
  a quadratic energy, with a regularization in [0, 1) that weighs the energy
  against the mean squared residual, solved in standard form. The eigenvalues of
  the energy are floored at `1e-9` times the largest one, so that every direction
  of the weights is penalized and the fit stays bounded away from measurements,
  such as out of the plane of measurements made in a single plane.
- `rivoli.model`: fit of a radial basis function model to measurements of a
  BRDF, regularized by its Dirichlet energy, in log space by default, and its
  evaluation at new sites, by chunks of sites so that it fits in memory for a
  whole MERL grid.
- `rivoli.merl`: reading and writing of the RGB files of the MERL BRDF database,
  the angles of the bins of its grid, and the lookup of the bins that contain
  Rusinkiewicz angles, as in its reference implementation.
- `rivoli.colour`: conversions of spectra, such as those of a spectral BRDF, to
  CIE 1931 tristimulus values and to the linear RGB of sRGB, under the
  illuminant D65 or E, with the colour matching functions and the illuminant
  tabulated by the CIE. The conversions are linear, and also given as a matrix.
- `rivoli.model.RBFModel.spectra` and `RBFModel.rgb`: evaluation of a spectral
  model at a set of wavelengths at each site, at the cost of a single
  evaluation thanks to the separability of the Gaussian kernels, and its colour
  in the linear RGB of sRGB.
- `rivoli.warp.kernels.expansion`: evaluation of expansions on the kernels with
  Warp, one thread per site, without the collocation matrix. Models are evaluated
  with it by passing a Warp `device` to `RBFModel`, `RBFModel.spectra`,
  `RBFModel.rgb` and `rivoli.merl.write_model`: writing a MERL file of a model of
  a thousand centres takes seconds on a GPU instead of minutes with NumPy.
- `rivoli.merl.write_model`: writing of an RGB or spectral model as a MERL file,
  evaluated above the horizon and clamped to non-negative values.
- Reciprocity of the kernels, `symmetry="reciprocity"` by default in
  `rivoli.kernels`, `rivoli.dirichlet`, their Warp counterparts and
  `rivoli.model`: the kernels are summed over the orbit of their centre by the
  swap of the incident and outgoing directions, `phi_d -> phi_d + pi`, so that
  the models are exactly reciprocal and smooth, and their Dirichlet matrix is
  that of the fundamental domain. `symmetry="none"` gives the plain kernels.
- `symmetry="reciprocity"` in the samplers of `rivoli.sampling`, which then
  sample the fundamental domain, `phi_d` in `(0, pi]`, for the centres of
  reciprocal kernels.
- `rivoli.hdf5`: reading of measurements of BRDFs from HDF5 files, described
  by a layout given as a mapping or a JSON file: the grids of the angles and
  wavelengths, the template of the paths of the tables of values, the unit of
  the angles and whether the values are weighted by the cosine of the outgoing
  polar angle. The measurements are returned as `Measurements`, which unpacks
  into the sites and values given to `rivoli.model.fit`. h5py is installed by
  the `hdf5` extra.
- `examples/merl_reconstruction.py`: example script that decimates a BRDF of the
  MERL database, reconstructs it and compares the reconstruction with it, with
  an optional MERL output and figure. The examples are run by the tests, and
  their dependencies, such as matplotlib, are in the `examples` dependency group.
- `rivoli.warp.distances`: the same distances and spacings computed in parallel
  on a GPU or the CPU with NVIDIA Warp, installed by the `warp` extra.

[unreleased]: https://github.com/fmargall/rivoli/commits/HEAD
