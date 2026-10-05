# `ropt`: A Python module for robust optimization

`ropt` is developed by the Netherlands Organisation for Applied Scientific
Research (TNO) and released under the GNU General Public License v3.0.

## Overview

`ropt` is a module for implementing and executing robust optimization
workflows. In classical optimization problems, a deterministic function is
optimized. In robust optimization, the function is stochastic and is
represented by an ensemble of functions
(realizations) for different values of some (possibly unknown) random
parameters. The optimal solution is then determined by optimizing the value of a
statistic, such as the mean, over the ensemble.

`ropt` provides the following features for robust optimization problems:

- Robust optimization over an ensemble of models, i.e., optimizing the average
  of a set of objective functions. Alternative objectives can be implemented
  using plugins, for instance, to implement risk-aware optimization, such as
  Conditional Value at Risk (CVaR) or standard-deviation-based functions.
- Support for black-box optimization of arbitrary functions.
- Running several optimizations at the same time, each with its own
  configuration, start point and evaluation function.
- Support for nested optimization: an evaluation function can run an
  optimization of its own — for example to optimize a sub-set of the
  variables as part of a black-box function.
- Evaluation of the functions in parallel: on background threads, in worker
  processes, as separate processes on the local machine, or as jobs on an HPC
  cluster. One set of workers serves every optimization started on it, and work
  of your own can be sent to it as well.
- Stopping a run on a criterion of your own, or aborting it from another
  thread; both return the best result reached so far.
- Restarting an optimization from the point an earlier run reached, passing in
  the function and gradient values already computed there so that they are not
  evaluated again.
- An interface for running various continuous and discrete optimization methods.
  By default, optimizers from the
  [`scipy.optimize`](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
  package are included, but additional optimizers can be added via a plugin
  mechanism. The most common options of these optimizers can be configured in a
  uniform manner, although algorithm- or package-specific options can still be
  passed.
- Estimation of gradients using a Stochastic Simplex Approximate
  Gradient (StoSAG) approach. Additional samplers for generating perturbed
  values for gradient estimation can be added via a plugin mechanism.
- Support for linear and non-linear constraints, if supported by the chosen
  optimizer.
- Configuration of the optimization process using
  [`pydantic`](https://docs.pydantic.dev/).
- Support for tracking and processing optimization results generated during the
  optimization process, with a callback invoked for each evaluation, or with
  handler objects that can be shared between runs.
- Optional support for exporting results as
  [`pandas`](https://pandas.pydata.org/) or [`polars`](https://pola.rs/) data
  frames.

`ropt` can be used to construct optimization workflows directly in Python
scripts or as a building block in optimization applications. At a minimum, the
user needs to provide additional code to calculate the values for each function
realization in the ensemble. This can range from calling a Python
function that returns the objective values to initiating a long-running
simulation on an HPC cluster and reading the results. Furthermore, `ropt`
exposes all intermediate results of the optimization, such as objective and
gradient values, but functionality to report or store any of these values must
be added by the user. Optional functionality to assist with this is included
with `ropt`.

`ropt` separates two concerns. The **optimizer setup** describes *what* to
solve: the variables, objectives, constraints, and the components that drive the
optimization. It is covered in the
[Optimizer Setup](optimizer_setup/key_concepts.md) section.

*How* to run a configured optimization is covered in
[Running Optimizations](running/running.md), which describes the `ropt.simple`
API. It starts a run with a single function call and covers
[parallel evaluation](running/parallel.md),
[many runs at once](running/many_runs.md),
[nested optimization](running/nested.md), and custom result handling.

## Related packages

### Plugins
Additional backend optimizers can be installed separately and used via the plugin system:

- The [`ropt-dakota`](https://tno-ropt.github.io/ropt-dakota/) plugin provides
  access to algorithms from the [Dakota](https://dakota.sandia.gov/) package.
- The [`ropt-nomad`](https://tno-ropt.github.io/ropt-nomad/) plugin implements
  the MADS algorithm based on the
  [NOMAD](https://www.gerad.ca/en/software/nomad/) package.
- The [`ropt-pymoo`](https://tno-ropt.github.io/ropt-pymoo/) plugin makes the
  algorithms from the [`pymoo`](https://pymoo.org/) package available to `ropt`.


### Applications
The `ropt` package is used by the
[Everest](https://everest.readthedocs.io/en/latest/) decision-making tool as its
core optimization engine.
