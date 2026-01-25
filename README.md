# EMpy - ElectroMagnetic Python

[![CI](https://github.com/lbolla/EMpy/actions/workflows/python-app.yml/badge.svg)](https://github.com/lbolla/EMpy/actions/workflows/python-app.yml) [![PyPI](https://badge.fury.io/py/ElectroMagneticPython.svg)](https://badge.fury.io/py/ElectroMagneticPython)

`EMpy - ElectroMagnetic Python` is a suite of algorithms widely known and used in electromagnetic problems and optics: the transfer matrix algorithm, the rigorous coupled wave analysis algorithm and more.

Run the examples in `examples/*` to have an idea how EMpy works.

Visit http://lbolla.github.io/EMpy/ for more information.

## Installation

```bash
pip install ElectromagneticPython
```

Optionally, install `bvp`:

```bash
pip install scikits.bvp1lg
```

## Development

First, download the source code from https://github.com/lbolla/EMpy.

EMpy is managed with `uv` (https://docs.astral.sh/uv/):

Run tests with:

```bash
make test
```

Install dev environment with:

```bash
make develop
```

Upgrade dependencies with `uv`:

```bash
# add or update a dependency, then refresh lockfile and sync environment
make upgrade
```

## Release process

Package version is handled by `setuptools_scm`/`uv`:

```bash
make release PART=...
```

GitHub actions will build and upload the package to PyPI.

## Citation

If you find EMpy useful in your job, please consider adding a citation.

As reference:

```
Bolla, L. (2017). EMpy [Computer software]. https://github.com/lbolla/EMpy/
```

As text:

```
We used EMpy (version x.y.z) to complete our work.
```
