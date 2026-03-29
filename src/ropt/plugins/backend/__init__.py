"""Framework and Implementations for Backend Plugins.

This module provides the necessary components for integrating optimization
algorithms into `ropt` via its plugin system. Backend plugins allow `ropt` to
utilize various optimization techniques, either built-in or provided by
third-party packages.

**Built-in Backends:**

`ropt` includes the following backends by default:

* [`SciPyBackend`][ropt.plugins.backend.scipy.SciPyBackend]: Provides
  access to various algorithms from the `scipy.optimize` library.
* [`ExternalBackend`][ropt.plugins.backend.external.ExternalBackend]:
  Enables running other backend plugins in a separate external process, useful
  for isolation or specific execution environments.

**Core Concepts:**

* **Plugin Interface:** Backend plugins must inherit from the
  [`BackendPlugin`][ropt.plugins.backend.BackendPlugin] base class.
  This class acts as a factory, defining a `create` method to instantiate
  backend objects.
* **Backend Implementation:** The actual optimization logic resides in classes
  that inherit from the [`Backend`][ropt.backend.Backend] abstract base
  class. These classes are initialized with the optimization configuration
  ([`EnOptContext`][ropt.context.EnOptContext]) and an
  [`OptimizerCallback`][ropt.core.OptimizerCallback]. The callback is used by
  the backend to request function and gradient evaluations from `ropt`. The
  optimization process is initiated by calling the backend's `start` method.
* **Discovery:** The [`PluginManager`][ropt.plugins.manager.PluginManager]
  discovers available `BackendPlugin` implementations (typically via entry
  points) and uses them to create `Backend` instances as needed during
  workflow execution.
"""

from ._base import BackendPlugin

__all__ = [
    "BackendPlugin",
]
