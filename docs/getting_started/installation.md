# Installation

`ropt` is distributed on [PyPI](https://pypi.org/project/ropt/) and can be
installed with any standard Python package manager. It requires Python 3.11 or
newer.

## Install the core package

Using `pip`:

```bash
pip install ropt
```

The core install includes the built-in SciPy-based optimizers and samplers,
which are enough for most basic optimization tasks.

## Optional extras

`ropt` offers a few optional dependency groups that add extra functionality:

| Extra          | Pulls in                | Enables                                                    |
| -------------- | ----------------------- | ---------------------------------------------------------- |
| `pandas`       | `pandas`                | Exporting results to pandas data frames.                   |
| `polars`       | `polars`                | Exporting results to polars data frames.                   |
| `cloudpickle`  | `cloudpickle`           | Running lambdas, closures, and notebook-defined functions in external processes. |
| `hpc`          | `pysqa`                 | Running evaluations on HPC clusters.                       |

Without `cloudpickle`, code that runs in external processes is transferred using
Python's standard `pickle` module, which only supports plain, named functions
imported from a module. Installing `cloudpickle` lifts that restriction, so
functions defined inline (lambdas), functions defined inside other functions
(closures), and functions defined in a Jupyter notebook can be used too. The
same rule applies wherever the work leaves your process: process pools, local
jobs and cluster jobs alike.

`cloudpickle` is also required to [run the optimizer
itself in a separate process](../optimizer_setup/configuration.md#external-backend).

Install with:

```bash
pip install "ropt[pandas]"
pip install "ropt[polars]"
pip install "ropt[pandas,hpc,cloudpickle]"
```

## Plugin packages

Additional optimization backends are provided as standalone packages. Once
installed alongside `ropt`, they are picked up automatically:

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
`backend.method` field in your configuration, either as `"plugin/method"`
(for example `"pymoo/nelder-mead"`) or, if the method name is unique among
your installed plugins, just `"method"` (for example `"nelder-mead"`). See the
[method strings](../optimizer_setup/configuration.md#method-strings) section of the
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
