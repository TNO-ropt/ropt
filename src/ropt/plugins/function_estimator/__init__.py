"""Plugin support for function estimators.

A function estimator combines the objective and gradient values of a set of
realizations into a single value. A
[`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.FunctionEstimatorPlugin]
is a factory that creates the
[`FunctionEstimator`][ropt.function_estimator.FunctionEstimator] objects doing
the actual work, which the
[`PluginManager`][ropt.plugins.manager.PluginManager] discovers through the
`ropt.plugins.function_estimator` entry point group.

`ropt` ships
[`DefaultFunctionEstimator`][ropt.function_estimator.default.DefaultFunctionEstimator],
which calculates the weighted mean or standard deviation of the realizations.

See [Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.
"""

from ._base import FunctionEstimatorPlugin

__all__ = [
    "FunctionEstimatorPlugin",
]
