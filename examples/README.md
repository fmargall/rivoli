# Examples

Small, independent scripts showing typical uses of RIVOLI. They follow the
latest version of the package, and are run by its tests.

Install RIVOLI with the dependencies of the examples, such as matplotlib for
the figures, from a clone of the repository:

```bash
uv sync --group examples
```

## Scripts

- [`merl_reconstruction.py`](merl_reconstruction.py): decimates a BRDF of the
  MERL database, reconstructs it from a few percent of its bins, compares the
  reconstruction with it, and optionally writes it as a MERL file and plots
  slices of both.

  ```bash
  uv run python examples/merl_reconstruction.py blue-acrylic.binary --figure comparison.png
  ```

  With [Warp](https://github.com/NVIDIA/warp), installed by the `warp` extra, add
  `--device cuda:0` to evaluate the model on a GPU: the MERL file is then written
  in seconds instead of minutes.

## Data

The MERL BRDF database is not distributed with RIVOLI. It is free for research
or academic use, under the terms of its copyright notice, and can be downloaded
from <https://cdfg.mit.edu/wojciech/brdfdatabase/>.
