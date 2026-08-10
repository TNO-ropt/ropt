# Installation

`ropt` is distributed on [PyPI](https://pypi.org/project/ropt/) and can be
installed with any standard Python package manager. It requires Python 3.11 or
newer.

## Install the core package

Using `pip`:

```bash
pip install ropt
```

The core install includes the SciPy-based optimizer and sampler backends, which
are sufficient for most basic optimization tasks.

## Optional extras

`ropt` exposes a few optional dependency groups that enable additional
functionality:

| Extra          | Pulls in                | Enables                                                    |
| -------------- | ----------------------- | ---------------------------------------------------------- |
| `pandas`       | `pandas`                | Exporting results to data frames .                         |
| `cloudpickle`  | `cloudpickle`           | Serializing Python code to run them in external processes. |
| `hpc`          | `pysqa`, `cloudpickle`  | Running evaluations on HPC clusters.                       |

Without the `cloudpickle` limits the Python code that can be run in external
processes; installing it adds support for lambdas, closures, and
notebook-defined functions. The `hpc` extra already includes `cloudpickle`.

Install with:

```bash
pip install "ropt[pandas]"
pip install "ropt[pandas,hpc,cloudpickle]"
```

## Plugin packages

Additional optimization backends are provided as standalone packages that
register themselves through Python entry points. Once installed they become
available to `ropt` automatically:

| Package                                                                  | Adds                                                                           |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| [`ropt-dakota`](https://tno-ropt.github.io/ropt-dakota/)                 | Algorithms from the [Dakota](https://dakota.sandia.gov/) toolkit.              |
| [`ropt-nomad`](https://tno-ropt.github.io/ropt-nomad/)                   | The MADS algorithm via [NOMAD](https://www.gerad.ca/en/software/nomad/).       |
| [`ropt-pymoo`](https://tno-ropt.github.io/ropt-pymoo/)                   | Algorithms from [`pymoo`](https://pymoo.org/).                                 |

Install any of them alongside `ropt`:

```bash
pip install ropt ropt-pymoo
```

After installation, the plugin's methods become available through the
`backend.method` field in your configuration. This field accepts two forms:

- **Explicit** — `"plugin/method"` names both the plugin and the method, for
  example `"pymoo/nelder-mead"`.
- **Implicit** — a bare `"method"` (for example `"nelder-mead"`) lets `ropt`
  search the installed plugins for one that provides it. This is unambiguous
  only when a single plugin exposes that method name.

Both the plugin name and the method name are case-insensitive. See the
[method strings](../optimizer_configuration/configuration.md#method-strings) section of the
configuration guide for the full details.

## Verifying the installation

A quick sanity check:

```python

# Print the current version:
from ropt.version import __version__
print(__version__)

# Verify the SciPy backend is available:
from ropt.workflow import find_backend_plugin
print(find_backend_plugin("slsqp"))  # should print "scipy"
```

If `scipy` is printed, the default backend plugin is working. Any additional
plugin packages you installed can be verified by checking their methods in the
same way.

## Where to next

- Run your first optimization: [Quickstart](quickstart.md).
- Read the conceptual introduction: [Background](background.md).
