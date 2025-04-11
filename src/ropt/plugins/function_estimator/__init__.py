"""Provides plugin functionality for adding function estimators.

Function estimators are used by the optimization process to combine the results
(objective function values and gradients) from a set of realizations into a
single representative value. This module allows for the extension of `ropt` with
custom strategies for aggregating ensemble results.

**Core Concepts:**

* **Plugin Interface:** Function estimator plugins must inherit from the
  [`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.base.FunctionEstimatorPlugin]
  base class. This class acts as a factory, defining a `create` method to
  instantiate estimator objects.
* **Estimator Implementation:** The actual aggregation logic resides in classes
  that inherit from the
  [`FunctionEstimator`][ropt.plugins.function_estimator.base.FunctionEstimator]
  abstract base class. These classes are initialized with the optimization
  configuration ([`EnOptConfig`][ropt.config.enopt.EnOptConfig]) and the index
  of the specific estimator configuration to use (`estimator_index`). The core
  functionality is provided by the `calculate_function` and
  `calculate_gradient` methods, which combine the function values and gradients
  from multiple realizations, respectively, using realization weights.
* **Discovery:** The [`PluginManager`][ropt.plugins.PluginManager] discovers
  available `FunctionEstimatorPlugin` implementations (typically via entry
  points) and uses them to create `FunctionEstimator` instances as needed
  during plan execution.

**Built-in Function Estimator Plugins:**

The default
[`DefaultFunctionEstimator`][ropt.plugins.function_estimator.default.DefaultFunctionEstimator]
class provides methods for calculating the weighted mean (`mean`) and standard
deviation (`stddev`) of the realization results.
"""
