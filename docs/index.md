# `ropt`: A Python module for robust optimization

`ropt` is a module designed for implementing and executing robust optimization
workflows. In classical optimization problems, a deterministic function is
optimized. However, in robust optimization, the function is expected to exhibit
a stochastic nature and is represented by an ensemble of functions
(realizations) for different values of some (possibly unknown) random
parameters. The optimal solution is then determined by optimizing the value of a
statistic, such as the mean, over the ensemble.

`ropt` can be employed to construct optimization workflows directly in Python or
as a building block in optimization applications. At a minimum, the user needs
to provide additional code to calculate the values for each function realization
in the ensemble. This can range from simply calling a Python function that
returns the objective values to initiating a long-running simulation on an HPC
cluster and reading the results. Furthermore, `ropt` exposes all intermediate
results of the optimization, such as objective and gradient values, but
functionality to report or store any of these values must be added by the user.
Optional functionality to assist with this is included with `ropt`.

`ropt` provides several features for efficiently solving complex robust
optimization problems:

- Robust optimization over an ensemble of models, i.e., optimizing the average
  of a set of objective functions. Alternative objectives can be implemented
  using plugins, for instance, to implement risk-aware optimization, such as
  Conditional Value at Risk (CVaR) or standard-deviation-based functions.
- Support for black-box optimization of arbitrary functions.
- Support for running complex optimization workflows, such as multiple runs with
  different optimization settings or even different optimization methods.
- Support for nested optimization, allowing sub-sets of the variables to be
  optimized by optimization workflows that run as part of the black-box function
  to be optimized.
- An interface for running various continuous and discrete optimization methods.
  By default, optimizers from the
  [`scipy.optimize`](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
  package are included, but additional optimizers can be added via a plugin
  mechanism. The most common options of these optimizers can be configured in a
  uniform manner, although algorithm- or package-specific options can still be
  passed.
- Efficient estimation of gradients using a Stochastic Simplex Approximate
  Gradient (StoSAG) approach. Additional samplers for generating perturbed
  values for gradient estimation can be added via a plugin mechanism.
- Support for linear and non-linear constraints, if supported by the chosen
  optimizer.
- Flexible configuration of the optimization process using
  [`pydantic`](https://pydantic-docs.helpmanual.io/).
- Support for tracking and processing optimization results generated during the
  optimization process.
- Optional support for exporting results as
  [`pandas`](https://pandas.pydata.org/) data frames.
