# Installation

`ropt` is distributed on [PyPI](https://pypi.org/project/ropt/) and can be
installed with any standard Python package manager. It requires Python 3.12 or
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
| `cloudpickle`  | `cloudpickle`           | Copying lambdas, closures, and notebook-defined code into separate processes. |
| `hpc`          | `pysqa`                 | Running evaluations on HPC clusters.                       |

??? info "Why `cloudpickle`?"

    Some ways of running evaluations do not call your function inside your own
    program: they start separate processes — on your own machine, or as jobs on
    an HPC cluster — and run it there. Your function and its data must be copied
    into those processes, which Python does with its standard `pickle` module.
    `pickle` copies a function by storing its name, so the other process can
    only rebuild functions and classes that are defined at the top level of a
    module it can import. With `cloudpickle` installed, the code itself is
    copied, which adds lambdas, functions defined inside another function
    (closures), and functions written in a Jupyter notebook.

    `cloudpickle` is optional in every case: it never changes what `ropt` can
    do, only where the copied code may be defined. Both
    [evaluating in parallel](../running/parallel.md) and the
    [external backend](../optimizer_setup/optimizer.md#external-backend) state
    what they accept without it.

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
from ropt.plugins import get_plugin_name
print(get_plugin_name("backend", "slsqp"))  # should print "scipy"
```

If `scipy` is printed, the default backend plugin is working. Any additional
plugin packages you installed can be verified by checking their methods in the
same way.
