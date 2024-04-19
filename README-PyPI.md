# `ropt`
`ropt` is a Python module for running robust optimization workflows.

`ropt` is developed by the Netherlands Organisation for Applied Scientific
Research (TNO). All files in this repository are released under the GNU General
Public License v3.0 (a copy is provided in the LICENSE file).

## Dependencies
`ropt` has been tested with Python versions 3.8-3.12.

`ropt` requires one or more optimization backends to function. A backend based
on optimizers included in [SciPy](https://scipy.org/) is installed by default.

## Installation
```bash
pip install ropt
```

To enable some options, the following optional-dependencies can be installed:

- `storage` : Enables `pandas`, `xarrays` and netCDF support.
- `parsl`   : Enables the parsl-based evaluator functionality.

To install both:
```bash
pip install ropt[storage,parsl]
```

## Documentation
Detailed documentation and examples can be found in the online manual (on
[GitHubPages](https://tno-ropt.github.io/ropt/) or on [Read the
Docs](https://ropt.readthedocs.io/)).
