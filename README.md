<h1 align="center">
  <img src="docs/assets/header.png" alt="RIVOLI, a BRDF toolbox: interpolation, reconstruction">
</h1>

[![CI][ci-badge]][ci-workflow]

Physically-based interpolation code by radial basis function dedicated to BRDF, spherical or bispherical modeling functions and data reconstruction.

## Installation

```bash
pip install rivoli
```

RIVOLI requires Python 3.13 or later.

## Usage

```python
import rivoli

print(rivoli.__version__)
```

## Contributing

Bug reports, questions and contributions are welcome: see the
[contributing guide](CONTRIBUTING.md) to set up a development environment. Everyone taking
part in the project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

If you use RIVOLI in your research, please cite it with the metadata of
[`CITATION.cff`](CITATION.cff), also available through the "Cite this repository" button
of GitHub.

## License

RIVOLI is distributed under the [AGPL-3.0-or-later](LICENSE.txt) license.

---

<div align="center">

<small>*This repository was* [<img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-black.json" alt="made with Copier" align="absmiddle">][copier] *using the [Scientific Python Template][template].*</small>

</div>

[ci-badge]: https://github.com/fmargall/rivoli/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/fmargall/rivoli/actions/workflows/ci.yml
[copier]: https://github.com/copier-org/copier
[template]: https://github.com/fmargall/scientific-python-template
