"""The plugin manager."""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Final, Literal

from ropt.exceptions import ConfigError

from .function_transform.base import FunctionTransformPlugin
from .optimizer.base import OptimizerPlugin
from .plan.base import PlanPlugin
from .realization_filter.base import RealizationFilterPlugin
from .sampler.base import SamplerPlugin

if TYPE_CHECKING:
    from ropt.plugins.base import Plugin

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points


_PLUGIN_TYPES: Final = {
    "function_transform": FunctionTransformPlugin,
    "optimizer": OptimizerPlugin,
    "sampler": SamplerPlugin,
    "realization_filter": RealizationFilterPlugin,
    "plan": PlanPlugin,
}

PluginType = Literal[
    "optimizer",
    "sampler",
    "realization_filter",
    "function_transform",
    "plan",
]
""" Plugin Types Supported by `ropt`"""


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
        package available under the name `my_optimizer`. The `MyOptimizer` class
        will be used to create
        [`Optimizer`][ropt.plugins.optimizer.base.Optimizer] objects and to
        facilitate this, it should derive from the
        [`OptimizerPlugin`][ropt.plugins.optimizer.base.OptimizerPlugin] class.

        Plugins can also be added dynamically using the `add_plugins` method.
        """
        # Built-in plugins, listed for all possible plugin types:
        self._plugins: Dict[PluginType, Dict[str, Plugin]] = {
            "optimizer": {},
            "sampler": {},
            "realization_filter": {},
            "function_transform": {},
            "plan": {},
        }

        for plugin_type in self._plugins:
            self.add_plugins(plugin_type, _from_entry_points(plugin_type))

    def add_plugins(self, plugin_type: PluginType, plugins: Dict[str, Plugin]) -> None:
        """Add a plugin at runtime.

        This method adds one or more plugins of a specific type to the plugin
        manager. The `plugins` argument maps the names of the new plugins to
        [`Plugin`][ropt.plugins.Plugin] object.

        The plugin names are case-insensitive.

        Note: Plugin types


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
        split_method = method.split("/", maxsplit=1)
        if len(split_method) > 1:
            plugin = self._plugins[plugin_type].get(split_method[0].lower())
            if plugin and plugin.is_supported(split_method[1]):
                return plugin
        elif split_method[0] != "default":
            for plugin in self._plugins[plugin_type].values():
                if plugin.is_supported(split_method[0]):
                    return plugin
        msg = f"Method not found: {method}"
        raise ConfigError(msg)

    def is_supported(self, plugin_type: PluginType, method: str) -> bool:
        """Check if a method is supported.

        Args:
            plugin_type: The type of the plugin to retrieve.
            method:      The name of the method the plugin should provide.
        """
        try:
            self.get_plugin(plugin_type, method)
        except ConfigError:
            return False
        return True


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
