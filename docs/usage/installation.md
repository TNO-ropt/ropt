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

| Extra        | Pulls in                | Enables                                                                                              |
| ------------ | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `pandas`     | `pandas`                | Exporting [`Results`][ropt.results.Results] to data frames via [`results_to_dataframe`][ropt.results.results_to_dataframe]. |
| `hpc`        | `pysqa`, `cloudpickle`  | Running evaluations on HPC clusters via [`HPCExecutor`][ropt.workflow.executors.HPCExecutor].        |
| `external`   | `cloudpickle`           | Running evaluations in an external Python process via the `external` backend.                        |

Install with:

```bash
pip install "ropt[pandas]"
pip install "ropt[pandas,hpc,external]"
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

After installation, select a method from the plugin by setting the
`backend.method` field in your configuration (see
[Configuration](configuration.md)) to a `"plugin/method"` string such as
`"pymoo/nelder-mead"`.

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
