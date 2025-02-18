# `ropt`
`ropt` is a Python module for running robust optimization workflows.

`ropt` is developed by the Netherlands Organisation for Applied Scientific
Research (TNO). All files in this repository are released under the GNU General
Public License v3.0 (a copy is provided in the LICENSE file).

Detailed documentation and examples can be found in the online
[manual](https://tno-ropt.github.io/ropt/).


## Dependencies
`ropt` has been tested with Python versions 3.11-3.13.

`ropt` requires one or more optimization plugins to function. By default, a
plugin based on optimizers included with [SciPy](https://scipy.org/) is
installed.


## Installation
From PyPI:
```bash
pip install ropt
```

To enable support for `pandas` export, you can install the `pandas` dependency:
```bash
pip install ropt[pandas]
```


## Development
The `ropt` source distribution can be found on
[GitHub](https://github.com/tno-ropt/ropt). It uses a standard `pyproject.toml`
file, which contains build information and configuration settings for various
tools. A development environment can be set up with compatible tools of your
choice.

The [uv](https://docs.astral.sh/uv/) package manager offers an easy way to
install `ropt` in its own virtual environment:

```bash
uv sync               # or:
uv sync --all-extras  # To add all optional dependencies
```

The `ropt` package uses [ruff](https://docs.astral.sh/ruff/) (for formatting and
linting), [mypy](https://www.mypy-lang.org/) (for static typing), and
[pytest](https://docs.pytest.org/en/stable/) (for running the test suite).

The documentation is written using [MkDocs](https://www.mkdocs.org/) and
[mkdocstrings](https://mkdocstrings.github.io/). To view the documentation
locally, start the built-in server within the `ropt` distribution directory:

```bash
mkdocs serve
```

All development and documentation tools are declared in `pyproject.toml` as `uv`
development dependencies and can be installed using the `--dev` flag:

```bash
uv sync --dev
```
