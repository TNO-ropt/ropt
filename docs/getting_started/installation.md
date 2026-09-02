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
| `cloudpickle`  | `cloudpickle`           | Optional everywhere: lets lambdas, closures, and notebook-defined code cross a process boundary. |
| `hpc`          | `pysqa`                 | Running evaluations on HPC clusters.                       |

Without `cloudpickle`, anything that crosses a process boundary is transferred
using Python's standard `pickle` module, which only handles functions and
classes it can look up by name — those defined at the top level of an importable
module. Installing `cloudpickle` lifts that restriction, so code defined inline
(lambdas), inside another function (closures), or in a Jupyter notebook can
cross as well.

`cloudpickle` is optional in every case. It never changes what `ropt` can do,
only where you are free to define the code it carries, and each place it applies
works without it:

| Where | Works without `cloudpickle` | What `cloudpickle` adds |
| ----- | --------------------------- | ----------------------- |
| [Process pools](../getting_started/execution.md#process-pool) | Evaluation functions at the top level of a module *or of the script you ran* | Lambdas, closures, and notebook-defined evaluation functions |
| [Local and cluster jobs](../getting_started/execution.md#local-pool) | Evaluation functions at the top level of a module the worker can **import** | The same, plus functions defined in the script you ran, and results built from locally defined classes |
| [The external backend](../optimizer_setup/configuration.md#external-backend) | The built-in plugins, and any plugin class in an importable module | Plugin instances of classes defined in a function or a notebook |

The two pool rows differ, and the difference bites in practice. A process pool
starts its workers with `spawn`, which re-imports the script you launched, so a
function defined there can be looked up again. A local or cluster job is a fresh
command whose `__main__` is `ropt`'s own worker module, so a function defined in
your script cannot be found by name — it has to live in a module the worker can
import, which on a cluster means installed on the compute nodes. **Installing
`cloudpickle` is the recommended way to use local and cluster jobs**, unless
your evaluation function already lives in such a module.

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
from ropt.utils import get_plugin_name
print(get_plugin_name("backend", "slsqp"))  # should print "scipy"
```

If `scipy` is printed, the default backend plugin is working. Any additional
plugin packages you installed can be verified by checking their methods in the
same way.

## Where to next

- Run your first optimization: [Quickstart](quickstart.md).
- Read the conceptual introduction: [Background](background.md).
