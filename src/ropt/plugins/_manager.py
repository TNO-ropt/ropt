"""The plugin manager."""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Literal

from ropt.exceptions import ConfigError
from ropt.plugins.function_transform.base import FunctionTransformPlugin
from ropt.plugins.optimization_steps.base import OptimizationStepsPlugin
from ropt.plugins.optimizer.base import OptimizerPlugin
from ropt.plugins.realization_filter.base import RealizationFilterPlugin
from ropt.plugins.sampler.base import SamplerPlugin
from ropt.plugins.workflow.base import WorkflowPlugin

if TYPE_CHECKING:
    from ropt.plugins.base import Plugin

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

PluginType = Literal[
    "optimizer",
    "sampler",
    "realization_filter",
    "function_transform",
    "optimization_step",
    "workflow",
]
""" Plugin Types Supported by `ropt`

`ropt` supports various types of plugins for different functionalities:

`optimizer`:
: These plugins implement optimizer plugins providing optimization methods.
  The default plugin is `scipy`, utilizing the
  [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
  module.

`sampler`:
: These plugins implement sampler plugins generating perturbations for
  estimating gradients. By default, a plugin based on
  [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) is
  installed:

`realization_filter`:
: These plugins implement filters for selecting a sub-set of realizations used
  in calculating objective or constraint functions and their gradients. The
  default plugin provides filters based on ranking and for CVaR optimization.

`function_transform`:
: These plugins implement the final objective and gradient from sets of
  objectives or constraints and their gradients for individual realizations. The
  default built-in plugin supports objectives defined by the mean or standard
  deviation of these values.

`optimization_step`:
: Optimization step plugins implement the steps evaluated during the execution
  plan. The built-in plugin offers a full set of steps for executing complex
  plans.

`workflow`:
: Workflow plugins implement the objects that execute an optimization workflow.
  The built-in plugin offers a full set of workflow objects for executing
  complex workflows.
"""


_PLUGIN_TYPES = {
    "function_transform": FunctionTransformPlugin,
    "optimizer": OptimizerPlugin,
    "optimization_step": OptimizationStepsPlugin,
    "sampler": SamplerPlugin,
    "realization_filter": RealizationFilterPlugin,
    "workflow": WorkflowPlugin,
}


class PluginManager:
    """The plugin manager."""

    def __init__(self) -> None:
        """Initialize the plugin manager.

        The plugin manager object is initialized via the entry points mechanism
        (see
        [`importlib.metadata`](https://docs.python.org/3/library/importlib.metadata.html)).

        For instance, to install an additional optimizer plugin, implemented in
        an independent package, and assuming installation via a `pyproject.toml`
        file, add the following:

        ```toml
        [project.entry-points."ropt.plugins.optimizer"]
        my_optimizer = "my_optimizer_pkg.my_plugin:MyOptimizer"
        ```

        This will make the `MyOptimizer` class from the `my_optimizer_pkg`
        package available under the name `my_optimizer`.

        Plugins can also be added dynamically using the add_plugins method.
        """
        # Built-in plugins, listed for all possible plugin types:
        self._plugins: Dict[PluginType, Dict[str, Plugin]] = {
            "optimizer": {},
            "sampler": {},
            "realization_filter": {},
            "function_transform": {},
            "optimization_step": {},
            "workflow": {},
        }

        for plugin_type in self._plugins:
            self.add_plugins(plugin_type, _from_entry_points(plugin_type))

    def add_plugins(self, plugin_type: PluginType, plugins: Dict[str, Plugin]) -> None:
        """Add a plugin at runtime.

        This method adds one or more plugins of a specific type to the plugin
        manager. The `plugins` argument maps the names of the new plugins to a
        callable that creates the plugin. The callable can be any function or
        class that creates a plugin object.

        The plugin names are case-insensitive.

        Args:
            plugin_type: Type of the plugin.
            plugins:     Dictionary of plugins.
        """
        for name, plugin in plugins.items():
            name_lower = name.lower()
            if name_lower in self._plugins[plugin_type]:
                msg = f"Duplicate plugin name: {name_lower}"
                raise ConfigError(msg)
            self._plugins[plugin_type][name_lower] = plugin

    def get_plugin(self, plugin_type: PluginType, method: str) -> Any:  # noqa: ANN401
        """Retrieve a plugin by type and method name.

        Args:
            plugin_type: The type of the plugin to retrieve.
            method:      The name of the method the plugin should provide.
        """
        plugin_name, sep, method_name = method.rpartition("/")

        if sep == "/":
            assert method_name != ""
            plugin = self._plugins[plugin_type].get(plugin_name.lower())
            if plugin is None:
                msg = f"Plugin not found: {method}"
                raise ConfigError(msg)
            return plugin

        for plugin in self._plugins[plugin_type].values():
            if plugin.is_supported(method_name):
                return plugin

        msg = f"Method not found: {method}"
        raise ConfigError(msg)

    def is_supported(self, plugin_type: PluginType, method: str) -> bool:
        """Check if a method is supported.

        Args:
            plugin_type: The type of the plugin to retrieve.
            method:      The name of the method the plugin should provide.
        """
        _, _, method_name = method.rpartition("/")
        try:
            plugin = self.get_plugin(plugin_type, method)
        except ConfigError:
            return False
        return bool(plugin.is_supported(method_name))


@lru_cache  # Without the cache, repeated calls are very slow
def _from_entry_points(plugin_type: str) -> Dict[str, Plugin]:
    plugins: Dict[str, Plugin] = {}
    for entry_point in entry_points().select(group=f"ropt.plugins.{plugin_type}"):
        plugin = entry_point.load()
        plugins[entry_point.name] = plugin()
        if not isinstance(plugins[entry_point.name], _PLUGIN_TYPES[plugin_type]):
            msg = (
                f"Incorrect type for {plugin_type} plugin `{entry_point.name}`"
                f": {type(plugins[entry_point.name])}"
            )
            raise TypeError(msg)
    return plugins
