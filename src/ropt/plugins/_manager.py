"""The plugin manager."""

from __future__ import annotations

from functools import cache
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Final, Literal

from ropt.exceptions import ConfigError

from .function_estimator.base import FunctionEstimatorPlugin
from .optimizer.base import OptimizerPlugin
from .plan.base import PlanHandlerPlugin, PlanStepPlugin
from .realization_filter.base import RealizationFilterPlugin
from .sampler.base import SamplerPlugin

if TYPE_CHECKING:
    from ropt.plugins.base import Plugin


_PLUGIN_TYPES: Final = {
    "function_estimator": FunctionEstimatorPlugin,
    "optimizer": OptimizerPlugin,
    "sampler": SamplerPlugin,
    "realization_filter": RealizationFilterPlugin,
    "plan_handler": PlanHandlerPlugin,
    "plan_step": PlanStepPlugin,
}

PluginType = Literal[
    "optimizer",
    "sampler",
    "realization_filter",
    "function_estimator",
    "plan_handler",
    "plan_step",
]
"""Represents the valid types of plugins supported by `ropt`.

This type alias defines the string identifiers used to categorize different
plugins within the `ropt` framework. Each identifier corresponds to a specific
role in the optimization process:

* `"optimizer"`: Plugins implementing optimization algorithms
  ([`OptimizerPlugin`][ropt.plugins.optimizer.base.OptimizerPlugin]).
* `"sampler"`: Plugins for generating parameter samples
  ([`SamplerPlugin`][ropt.plugins.sampler.base.SamplerPlugin]).
* `"realization_filter"`: Plugins for filtering ensemble realizations
  ([`RealizationFilterPlugin`][ropt.plugins.realization_filter.base.RealizationFilterPlugin]).
* `"function_estimator"`: Plugins for estimating objective functions and gradients
  ([`FunctionEstimatorPlugin`][ropt.plugins.function_estimator.base.FunctionEstimatorPlugin]).
* `"plan_handler"`: Plugins that create handlers for processing plan results
  ([`PlanHandlerPlugin`][ropt.plugins.plan.base.PlanHandlerPlugin]).
* `"plan_step"`: Plugins that define executable steps within an optimization plan
  ([`PlanStepPlugin`][ropt.plugins.plan.base.PlanStepPlugin]).
"""


class PluginManager:
    """Manages the discovery and retrieval of `ropt` plugins.

    The `PluginManager` is responsible for finding available plugins based on
    Python's entry points mechanism and providing access to them. It serves as
    a central registry for different types of plugins used within `ropt`, such
    as optimizers, samplers, and plan components.

    Upon initialization, the manager scans for entry points defined under the
    `ropt.plugins.*` groups (e.g., `ropt.plugins.optimizer`). Plugins found
    this way are loaded and stored internally, categorized by their type.

    The primary way to interact with the manager is through the
    [`get_plugin`][ropt.plugins.PluginManager.get_plugin] method, which retrieves
    a specific plugin class based on its type and a method name it supports.
    The [`is_supported`][ropt.plugins.PluginManager.is_supported] method can be
    used to check for the availability of a plugin method without retrieving it.

    **Example: Registering a Custom Optimizer Plugin**

    To make a custom optimizer plugin available to `ropt`, you would typically
    define an entry point in your package's `pyproject.toml`:

    ```toml
    [project.entry-points."ropt.plugins.optimizer"]
    my_optimizer = "my_package.my_module:MyOptimizer"
    ```

    When `ropt` initializes the `PluginManager`, it will discover and load
    `MyOptimizer` from `my_package.my_module`, making it accessible via
    `plugin_manager.get_plugin("optimizer", "my_optimizer/some_method")` or
    potentially `plugin_manager.get_plugin("optimizer", "some_method")` if
    discovery is allowed and the method is unique.
    """

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        # Built-in plugins, listed for all possible plugin types:
        self._plugins: dict[PluginType, dict[str, type[Plugin]]] = {
            "optimizer": {},
            "sampler": {},
            "realization_filter": {},
            "function_estimator": {},
            "plan_handler": {},
            "plan_step": {},
        }

        for plugin_type in self._plugins:
            for name, plugin in _from_entry_points(plugin_type).items():
                self._add_plugin(plugin_type, name, plugin)

    def _add_plugin(
        self,
        plugin_type: PluginType,
        name: str,
        plugin: type[Plugin],
        *,
        prioritize: bool = False,
    ) -> None:
        name_lower = name.lower()
        if name_lower in self._plugins[plugin_type]:
            msg = f"Duplicate plugin name: {name_lower}"
            raise ConfigError(msg)
        if prioritize:
            plugins = self._plugins[plugin_type]
            self._plugins[plugin_type] = {name_lower: plugin}
            self._plugins[plugin_type].update(dict(plugins))
        else:
            self._plugins[plugin_type][name_lower] = plugin

    def _get_plugin(self, plugin_type: PluginType, method: str) -> Any | None:  # noqa: ANN401
        split_method = method.split("/", maxsplit=1)
        if len(split_method) > 1:
            plugin = self._plugins[plugin_type].get(split_method[0].lower())
            if plugin and plugin.is_supported(split_method[1]):
                return plugin
        else:
            if split_method[0] == "default":
                msg = "Cannot specify 'default' method without a plugin name"
                raise ConfigError(msg)
            for plugin in self._plugins[plugin_type].values():
                if plugin.allows_discovery() and plugin.is_supported(split_method[0]):
                    return plugin
        return None

    def get_plugin(self, plugin_type: PluginType, method: str) -> Any:  # noqa: ANN401
        """Retrieve a plugin class by its type and a supported method name.

        This method finds and returns the class of a plugin that matches the
        specified `plugin_type` and supports the given `method`.

        The `method` argument can be specified in two ways:

        1.  **Explicit Plugin:** Use the format `"plugin-name/method-name"`.
            This directly requests the `method-name` from the plugin named
            `plugin-name`.
        2.  **Implicit Plugin:** Provide only the `method-name`. The manager
            will search through all registered plugins of the specified
            `plugin_type` that allow discovery (see
            [`Plugin.allows_discovery`][ropt.plugins.base.Plugin.allows_discovery]).
            It returns the first plugin found that supports the `method-name`.

        Args:
            plugin_type: The category of the plugin (e.g., "optimizer", "sampler").
            method:      The name of the method the plugin must support, potentially
                         prefixed with the plugin name and a slash (`/`).

        Returns:
            The plugin class that matches the criteria.

        Raises:
            ConfigError: If no matching plugin is found for the given type and
                         method, or if "default" is used as a method name without
                         specifying a plugin name.
        """
        if (plugin := self._get_plugin(plugin_type, method)) is not None:
            return plugin
        msg = f"Method not found: {method}"
        raise ConfigError(msg)

    def is_supported(self, plugin_type: PluginType, method: str) -> bool:
        """Check if a specific plugin method is available.

        Verifies whether a plugin of the specified `plugin_type` supports the
        given `method`. This is useful for checking availability before attempting
        to retrieve a plugin with [`get_plugin`][ropt.plugins.PluginManager.get_plugin].

        The `method` argument can be specified in two ways:

        1.  **Explicit Plugin:** `"plugin-name/method-name"` checks if the specific
            plugin named `plugin-name` supports `method-name`.
        2.  **Implicit Plugin:** `"method-name"` searches through all discoverable
            plugins of the given `plugin_type` to see if any support `method-name`.

        Args:
            plugin_type: The category of the plugin (e.g., "optimizer", "sampler").
            method:      The name of the method to check, potentially prefixed
                         with the plugin name and a slash (`/`).

        Returns:
            `True` if a matching plugin supports the specified method.
        """
        return self._get_plugin(plugin_type, method) is not None


@cache  # Without the cache, repeated calls are very slow
def _from_entry_points(plugin_type: str) -> dict[str, type[Plugin]]:
    plugins: dict[str, type[Plugin]] = {}
    for entry_point in entry_points().select(group=f"ropt.plugins.{plugin_type}"):
        plugin = entry_point.load()
        plugins[entry_point.name] = plugin
        if not issubclass(plugins[entry_point.name], _PLUGIN_TYPES[plugin_type]):
            msg = (
                f"Incorrect type for {plugin_type} plugin `{entry_point.name}`"
                f": {type(plugins[entry_point.name])}"
            )
            raise TypeError(msg)
    return plugins
