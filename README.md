# `ropt`
`ropt` is a Python module for running robust optimization workflows.

`ropt` is developed by the Netherlands Organisation for Applied Scientific
Research (TNO). All files in this repository are released under the GNU General
Public License v3.0 (a copy is provided in the LICENSE file).

Detailed documentation and examples can be found in the online
[manual](https://tno-ropt.github.io/ropt/).


## Dependencies
`ropt` has been tested with Python versions 3.8-3.12.

`ropt` requires one or more optimization backends to function. A backend based
on optimizers included with [SciPy](https://scipy.org/) is installed by default.


## Installation
From PyPI:
```bash
pip install ropt
```

The following optional-dependencies can be installed to enable extra functionality:

- `pandas` : Enables support for `pandas` export and tabular output.
- `xarray` : Enables support for xarray and writing netCDF files.
- `parsl`  : Enables the parsl-based evaluator functionality.

Install with:
```bash
pip install ropt[<dep>]
```
where \<dep\> is one of the optional dependencies listed above. To install all:
```bash
pip install ropt[pandas,xarray,parsl]
```


## Development
The `ropt` source distribution can be found on
[GitHub](https://github.com/tno-ropt/ropt). To install from source, enter the
distribution directory and execute:

```bash
pip install .
```

Documentation is written using [`MkDocs`](https://www.mkdocs.org/) and
[`mkdocstrings`](https://mkdocstrings.github.io/). To view it, install the
necessary dependencies and start the `MkDocs` built-in server within the `ropt`
distribution directory:

```bash
pip install .[docs]
mkdocs serve
```

## Running the tests
To run the test suite, install the necessary dependencies and execute `pytest`:

```bash
pip install .[test]
pytest
```
