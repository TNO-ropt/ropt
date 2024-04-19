"""The plugin manager."""

from __future__ import annotations

import contextlib
import sys
from functools import lru_cache
from typing import Any, Dict, Literal

from ropt.exceptions import ConfigError

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

BackendType = Literal[
    "optimizer",
    "sampler",
    "realization_filter",
    "function_transform",
    "optimization_step",
]
""" Plugin Types Supported by `ropt`

`ropt` supports various types of plugins for different functionalities:

`optimizer`:
: These plugins implement optimizer backends providing optimization algorithms.
  The default backend is `scipy`, utilizing the
  [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
  module.

`sampler`:
: These plugins implement sampler backends generating perturbations for
  estimating gradients. By default, a backend based on
  [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) is
  installed:

`realization_filter`:
: These plugins implement filters for selecting a sub-set of realizations used
  in calculating objective or constraint functions and their gradients. The
  default backend provides filters based on ranking and for CVaR optimization.

`function_transform`:
: These plugins implement the final objective and gradient from sets of
  objectives or constraints and their gradients for individual realizations. The
  default built-in plugin supports objectives defined by the mean or standard
  deviation of these values.

`optimization_step`:
: Optimization step plugins implement the steps evaluated during the execution
  plan. The built-in backend offers a full set of steps for executing complex
  plans.
"""


class PluginManager:
    """The plugin manager."""

    def __init__(self) -> None:
        """Initialize the plugin manager.

        The plugin manager object is initialized with the built-in plugins and
        all plugins installed via the entry points mechanism (see
        [`importlib.metadata`](https://docs.python.org/3/library/importlib.metadata.html)).

        For instance, to install an additional optimizer backend, implemented in
        an independent package, and assuming installation via a `pyproject.toml`
        file, add the following:

        ```toml
        [project.entry-points."ropt.plugins.optimizer"]
        my_optimizer = "my_optimizer_pkg.basic_backend:MyOptimizer"
        ```

        This will make the `MyOptimizer` class from the `my_optimizer_pkg`
        package available under the name `my_optimizer`.

        Plugins can also be added dynamically using the add_backends method.
        """
        # Not done globally to avoid circular imports:
        import ropt.plugins.function_transform.default
        import ropt.plugins.optimization_steps.default
        import ropt.plugins.optimizer.scipy
        import ropt.plugins.realization_filter.default
        import ropt.plugins.sampler.scipy

        # Built-in plugins, listed for all possible backend types:
        self._backend_classes: Dict[BackendType, Dict[str, Any]] = {
            "optimizer": {
                "scipy": ropt.plugins.optimizer.scipy.SciPyOptimizer,
            },
            "sampler": {
                "scipy": ropt.plugins.sampler.scipy.SciPySampler,
            },
            "realization_filter": {
                "default": ropt.plugins.realization_filter.default.DefaultRealizationFilter,
            },
            "function_transform": {
                "default": ropt.plugins.function_transform.default.DefaultFunctionTransform,
            },
            "optimization_step": {
                "default": ropt.plugins.optimization_steps.default.get_step,
            },
        }

        for backend_type in self._backend_classes:
            self.add_backends(backend_type, _from_entry_points(backend_type))

    def add_backends(self, backend_type: BackendType, backends: Dict[str, Any]) -> None:
        """Add a backend at runtime.

        This method adds one or more backends of a specific type to the plugin
        manager. The `backends` argument maps the names of the new plugins to a
        callable that creates the plugin. The callable can be any function or
        class that creates a plugin object.

        The plugin names are case-insensitive.

        Args:
            backend_type: Type of the backend.
            backends:     Dictionary of plugins.
        """
        for name, backend in backends.items():
            name_lower = name.lower()
            if name_lower in self._backend_classes[backend_type]:
                msg = f"Duplicate backend name: {name_lower}"
                raise ConfigError(msg)
            self._backend_classes[backend_type][name_lower] = backend

    def get_backend(self, backend_type: BackendType, plugin_name: str) -> Any:  # noqa: ANN401
        """Retrieve a backend by type and plugin name.

        Given a type and plugin name, this method returns a callable that can be
        used to create a plugin object.

        Args:
            backend_type: The type of the backend to retrieve.
            plugin_name:  The name of the plugin that implements the backend.
        """
        plugin = self._backend_classes[backend_type].get(plugin_name.lower())
        if plugin is None:
            backend_name = backend_type.replace("_", " ").capitalize()
            msg = f"{backend_name} backend not supported: {plugin_name}"
            raise ConfigError(msg)
        return plugin


@lru_cache  # Without the cache, repeated calls are very slow
def _from_entry_points(plugin_type: str) -> Dict[str, Any]:
    plugins: Dict[str, Any] = {}
    for entry_point in entry_points().select(group=f"ropt.plugins.{plugin_type}"):
        with contextlib.suppress(ModuleNotFoundError):
            plugins[entry_point.name] = entry_point.load()
    return plugins
