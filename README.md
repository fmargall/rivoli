<h1 align="center">
  <img src="https://raw.githubusercontent.com/fmargall/rivoli/main/docs/assets/header.png" alt="RIVOLI, a BRDF toolbox: interpolation, reconstruction">
</h1>

[![CI][ci-badge]][ci-workflow]
[![Coverage][coverage-badge]][coverage]
[![pre-commit][pre-commit-badge]][pre-commit]
[![DOI][doi-badge]][doi]
[![PyPI][pypi-badge]][pypi]

RIVOLI reconstructs continuous BRDFs from sparse measurements. It fits expansions on
radial basis functions over the product of the hemispheres of the half and difference
vectors, optionally times an interval of wavelengths, regularized by their Dirichlet
energy so that they stay smooth between the measurements. The models are reciprocal by
construction, can be evaluated anywhere, on a GPU with NVIDIA Warp, and written as MERL
files for renderers.

## Features

- **Kernels**: Gaussian radial basis functions on the Rusinkiewicz domain, with a width
  per factor, for isotropic or anisotropic BRDFs, made reciprocal by summation over
  orbits, and spectral with a wavelength factor.
- **Fitting**: least squares in log space, regularized by the Dirichlet energy of the
  expansion, from interpolation to the smoothest fits.
- **Sampling**: Super-Fibonacci low-discrepancy samplings of the isotropic and
  anisotropic domains, above the horizon or over the fundamental domain of reciprocity.
- **Geometry**: half and difference vectors, Rusinkiewicz angles, geodesic and chordal
  distances, and spacings of samplings, from which the widths of the kernels follow.
- **Colour**: spectra converted to CIE XYZ and linear sRGB, under the D65 or E
  illuminants.
- **Input and output**: MERL files read and written, and measurements read from HDF5
  files of any structure, described by a JSON layout.
- **GPU**: models evaluated, MERL files written, and Dirichlet energies computed in
  parallel with [NVIDIA Warp](https://github.com/NVIDIA/warp), on GPUs or CPUs.

## Installation

```bash
pip install rivoli
```

RIVOLI requires Python 3.13 or later, and depends only on NumPy. Optional features are
installed with extras:

| Extra   | Installs                                        | Enables                                                                               |
| ------- | ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| `warp`  | [NVIDIA Warp](https://github.com/NVIDIA/warp)   | `rivoli.warp` and the `device` arguments. Enables GPU acceleration with NVIDIA GPUs.  |
| `hdf5`  | [h5py](https://www.h5py.org/)                   | `rivoli.hdf5`. Required for reading/writing HDF5 files.                               |

```bash
pip install "rivoli[warp,hdf5]"
```

## Quick start

Fit a model to measurements of an isotropic BRDF, here an analytic one, then evaluate it
at new sites:

```python
import numpy as np

from rivoli.distances import nearest_neighbour_spacings
from rivoli.geometry import spherical_to_cartesian
from rivoli.model import fit
from rivoli.sampling import super_fibonacci_isotropic


def sites_of(theta_h, theta_d, phi_d):
    """Half and difference vectors of an isotropic BRDF, from Rusinkiewicz angles."""
    return spherical_to_cartesian(theta_h, 0.0), spherical_to_cartesian(theta_d, phi_d)


def brdf(theta_h, theta_d, phi_d):
    """A diffuse and a glossy lobe, standing for measurements."""
    return 0.2 / np.pi + 5.0 * np.cos(theta_h) ** 200


# Measurements at 5,000 well-spread sites above the horizon
angles = super_fibonacci_isotropic(5000, above_horizon=True)
sites, values = sites_of(*angles), brdf(*angles)

# 500 kernels, centred over the fundamental domain of reciprocity, narrower
# along the half vector, around the specular peak
centres = sites_of(*super_fibonacci_isotropic(500, above_horizon=True, symmetry="reciprocity"))
spacing_h, spacing_d = nearest_neighbour_spacings(*centres, metric="chordal")
widths = (0.7 * spacing_h.mean(), 12.0 * spacing_d.mean())

model = fit(
    sites, values, centres, widths=widths, metric="chordal", isotropic=True, regularization=1e-5
)

# The model is continuous: evaluate it anywhere, here on new sites
test = super_fibonacci_isotropic(20000, above_horizon=True)
reconstruction = model(sites_of(*test))
print(np.linalg.norm(reconstruction - brdf(*test)) / np.linalg.norm(brdf(*test)))  # 0.005
```

A model of three channels, such as RGB, is written as a MERL file by
`rivoli.merl.write_model`, and evaluated on a GPU with `device="cuda:0"` once the `warp`
extra is installed.

## Examples

The [`examples`](https://github.com/fmargall/rivoli/tree/main/examples) directory holds complete scripts, run by the tests so that
they follow the API:

- [`merl_reconstruction.py`](https://github.com/fmargall/rivoli/blob/main/examples/merl_reconstruction.py) reconstructs a BRDF of the
  MERL database from a few percent of its bins, and measures the error on the others.
- [`hdf5_reconstruction.py`](https://github.com/fmargall/rivoli/blob/main/examples/hdf5_reconstruction.py) reconstructs the colour of
  spectral measurements read from an HDF5 file, per wavelength, per RGB channel, or with
  a spectral model.

See [their guide](https://github.com/fmargall/rivoli/blob/main/examples/README.md) to install their dependencies and run them.

## Contributing

Bug reports, questions and contributions are welcome! Whether you fix a bug, improve documentation
or add features, feel free to open an issue or submit a pull request. You will find all the information
you need in the [contributing guide](https://github.com/fmargall/rivoli/blob/main/CONTRIBUTING.md) to set up a development environment. Everyone taking
part in the project is expected to follow the [Code of Conduct](https://github.com/fmargall/rivoli/blob/main/CODE_OF_CONDUCT.md).

### Contact

François Margall - fr.margall@proton.me

## Citation

If you use RIVOLI in your research, please cite it with the metadata of
[`CITATION.cff`](https://github.com/fmargall/rivoli/blob/main/CITATION.cff), also available through the "Cite this repository" button
of GitHub. Each release is archived on Zenodo: the DOI
[10.5281/zenodo.23804096][doi] cites the latest one, and the Zenodo page lists the DOI of
each version, to cite the exact one you used.

## Publications

- **La Coupole: an SVBRDF measurement device for large and non-planar objects**.
  *Antoine Lucat, Pierre Mézières, François Margall, Louis De Oliveira, Marjorie Paillet,
  Arnaud Tizon, Pierre Bénard, Romain Pacanowski*. Optics Express Vol. 34, Issue 7,
  pp. 11695-11709 (March 2026). DOI: [10.1364/OE.587877](https://doi.org/10.1364/OE.587877).


## License

RIVOLI is distributed under the [AGPL-3.0-or-later](https://github.com/fmargall/rivoli/blob/main/LICENSE.txt) license.

---

<div align="center">

<small>*This repository was* [<img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-black.json" alt="made with Copier" align="absmiddle">][copier] *using the [Scientific Python Template][template].*</small>

</div>

[ci-badge]: https://github.com/fmargall/rivoli/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/fmargall/rivoli/actions/workflows/ci.yml
[coverage-badge]: https://codecov.io/gh/fmargall/rivoli/graph/badge.svg
[coverage]: https://codecov.io/gh/fmargall/rivoli
[pre-commit-badge]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
[pre-commit]: https://github.com/pre-commit/pre-commit
[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.23804096.svg
[doi]: https://doi.org/10.5281/zenodo.23804096
[pypi-badge]: https://img.shields.io/pypi/v/rivoli?label=PyPI&logo=pypi&logoColor=white
[pypi]: https://pypi.org/project/rivoli/
[copier]: https://github.com/copier-org/copier
[template]: https://github.com/fmargall/scientific-python-template
