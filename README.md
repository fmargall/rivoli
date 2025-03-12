![Header](./docs/rivoli-header-readme.png)

# RIVOLI &middot; <a href="./LICENSE"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"> </a> <a href="https://github.com/qoomon/starlines"> <img src="https://starlines.qoo.monster/assets/fmargall/rivoli" align="right" alt="Starline counter"/> </a>
<sup> *Radial-based Interpolation for Visualisation, Optics and Light-matter Interaction* </sup>

## Documentation

### Using a `.ini` file

#### `[INPUT]`

| Key | Description | Value | Required? |
| --- | ----------- | ----- | --------- |
| `inputAnglesUnit` | Unit in which angles are given in the input file | `degrees` or `radians` | Yes |
| `inputParameterisation` | Parameterisation of the input coordinates | `spherical` or `rusinkiewicz` | Yes for > 2D dimensions |
| `inputFilePath` | Relative or absolute path to the input data file | `type:string` | Yes |

> [!Note]
> The input file should have a `.txt` format with no header, each row containing the coordinates of each value and the value itself. Colums should be separated by a tab `\t` and rows by a new line `\n`.

#### `[SOFTWARE]`

| Key | Description | Value | Required? |
| --- | ----------- | ----- | --------- |
| `floatingPointFormat` |
| `logLevel` |
| `parallelComputing` |

###### `[INTERPOLATION.RBF.LOCATION]`

| Key | Description | Value | Required? |
| --- | ----------- | ----- | --------- |
| `locationRBFFilePath` |
| `locationRBFParameterisation` |
| `locationRBFAngleUnit` |

> [!Warning]
> Chosing the right locations for the RBF is crucial for the interpolation to work properly and may be complicated, a good positioning depending strongly of the chosen topology. Be aware that if you do not define RBF locations, the software will automatically locate the RBF on the data coordinates. This can lead to a much longer interpolator calculation time, as well as a higher number of output coefficients, with no guarantee of a better quality result.

Documentation in progress...

### Using a `.RBFCoeffs` file



## Dependencies

> [!Note]
> The following dependencies are only required for 
> - the developers that would like to use or build the source files themselves,
> - the users that want to use the library as header-only.
>
> If this is your case, when cloning the repository, simply execute
>
> ```bash
> $ git clone --recursive https://github.com/fmargall/rivoli.git
> ```
> If you have already cloned the repository, you can update the submodules with
>
> ```bash
> $ git submodule update --init --recursive
> ```
>
>For other use, for instance with the executable, there is no need to take this into account.

The following dependencies are used in this project:
 - [Eigen](https://gitlab.com/libeigen/eigen) - A C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
 - [glm](https://github.com/g-truc/glm) - OpenGL Mathematics (GLM) is a header only C++ mathematics library for graphics software based on the OpenGL Shading Language (GLSL) specifications.
