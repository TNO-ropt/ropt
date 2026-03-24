"""Framework and Implementations for Optimizer Plugins.

This module provides the necessary components for integrating optimization
algorithms into `ropt` via its plugin system. Optimizer plugins allow `ropt` to
utilize various optimization techniques, either built-in or provided by
third-party packages.

**Built-in Optimizers:**

`ropt` includes the following optimizers by default:

* [`SciPyOptimizer`][ropt.plugins.optimizer.scipy.SciPyOptimizer]: Provides
  access to various algorithms from the `scipy.optimize` library.
* [`ExternalOptimizer`][ropt.plugins.optimizer.external.ExternalOptimizer]:
  Enables running other optimizer plugins in a separate external process, useful
  for isolation or specific execution environments.

**Core Concepts:**

* **Plugin Interface:** Optimizer plugins must inherit from the
  [`OptimizerPlugin`][ropt.plugins.optimizer.OptimizerPlugin] base class.
  This class acts as a factory, defining a `create` method to instantiate
  optimizer objects.
* **Optimizer Implementation:** The actual optimization logic resides in classes
  that inherit from the [`Optimizer`][ropt.optimizer.Optimizer] abstract base
  class. These classes are initialized with the optimization configuration
  ([`EnOptConfig`][ropt.config.EnOptConfig]) and an
  [`OptimizerCallback`][ropt.core.OptimizerCallback]. The callback is used by
  the optimizer to request function and gradient evaluations from `ropt`. The
  optimization process is initiated by calling the optimizer's `start` method.
* **Discovery:** The [`PluginManager`][ropt.plugins.manager.PluginManager]
  discovers available `OptimizerPlugin` implementations (typically via entry
  points) and uses them to create `Optimizer` instances as needed during
  workflow execution.
"""

from ._base import OptimizerPlugin

__all__ = [
    "OptimizerPlugin",
]
