"""The plugin manager."""

from __future__ import annotations

from functools import cache
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from .compute_step.base import ComputeStepPlugin
from .evaluator.base import EvaluatorPlugin
from .event_handler.base import EventHandlerPlugin
from .function_estimator.base import FunctionEstimatorPlugin
from .optimizer.base import OptimizerPlugin
from .realization_filter.base import RealizationFilterPlugin
from .sampler.base import SamplerPlugin
from .server.base import ServerPlugin

if TYPE_CHECKING:
    from ropt.plugins.base import Plugin


PluginType = Literal[
    "optimizer",
    "sampler",
    "realization_filter",
    "function_estimator",
    "event_handler",
    "compute_step",
    "evaluator",
    "server",
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
* `"event_handler"`: Plugins that create event handlers for processing optimization
  results ([`EventHandlerPlugin`][ropt.plugins.event_handler.base.EventHandlerPlugin]).
* `"compute_step"`: Plugins that define executable steps within an optimization workflow
  ([`ComputeStepPlugin`][ropt.plugins.compute_step.base.ComputeStepPlugin]).
* `"evaluator"`: Plugins that define evaluators within an optimization workflow
  ([`EvaluatorPlugin`][ropt.plugins.evaluator.base.EvaluatorPlugin]).
* `"server"`: Plugins that define servers within an optimization workflow
  ([`ServerPlugin`][ropt.plugins.server.base.ServerPlugin]).
"""


_PLUGIN_TYPES: Final = {
    "function_estimator": FunctionEstimatorPlugin,
    "optimizer": OptimizerPlugin,
    "sampler": SamplerPlugin,
    "realization_filter": RealizationFilterPlugin,
    "event_handler": EventHandlerPlugin,
    "compute_step": ComputeStepPlugin,
    "evaluator": EvaluatorPlugin,
    "server": ServerPlugin,
}

_DEFAULT_PLUGINS: Final = {
    "function_estimator": "default",
    "optimizer": "scipy",
    "sampler": "scipy",
    "realization_filter": "default",
    "event_handler": "default",
    "compute_step": "default",
    "evaluator": "default",
    "server": "default",
}


class PluginManager:
    """Manages the discovery and retrieval of `ropt` plugins.

    The `PluginManager` is responsible for finding available plugins based on
    Python's entry points mechanism and providing access to them. It serves as
    a central registry for different types of plugins used within `ropt`, such
    as optimizers, samplers, and workflow components.

    Upon initialization, the manager scans for entry points defined under the
    `ropt.plugins.*` groups (e.g., `ropt.plugins.optimizer`). Plugins found
    this way are loaded and stored internally, categorized by their type.

    The primary way to interact with the manager is through the
    [`get_plugin`][ropt.plugins.PluginManager.get_plugin] method, which
    retrieves a specific plugin class based on its type and a method name it
    supports. The
    [`get_plugin_name`][ropt.plugins.PluginManager.get_plugin_name] method can
    be used to find the name of a plugin that supports a given method.

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
        self._plugins: dict[PluginType, dict[str, type[Plugin]]] = {}

    def _init(self) -> None:
        # ruff: disable[I001,PLC0415]
        if self._plugins:
            return

        from .optimizer.scipy import SciPyOptimizerPlugin
        from .optimizer.external import ExternalOptimizerPlugin
        from .sampler.scipy import SciPySamplerPlugin
        from .realization_filter.default import DefaultRealizationFilterPlugin
        from .function_estimator.default import DefaultFunctionEstimatorPlugin
        from .compute_step.default import DefaultComputeStepPlugin
        from .event_handler.default import DefaultEventHandlerPlugin
        from .evaluator.default import DefaultEvaluatorPlugin
        from .server.default import DefaultServerPlugin

        self._add_plugin("optimizer", "scipy", SciPyOptimizerPlugin)
        self._add_plugin("optimizer", "external", ExternalOptimizerPlugin)
        self._add_plugin("sampler", "scipy", SciPySamplerPlugin)
        self._add_plugin(
            "realization_filter", "default", DefaultRealizationFilterPlugin
        )
        self._add_plugin(
            "function_estimator", "default", DefaultFunctionEstimatorPlugin
        )
        self._add_plugin("compute_step", "default", DefaultComputeStepPlugin)
        self._add_plugin("event_handler", "default", DefaultEventHandlerPlugin)
        self._add_plugin("evaluator", "default", DefaultEvaluatorPlugin)
        self._add_plugin("server", "default", DefaultServerPlugin)

        for plugin_type in _PLUGIN_TYPES:
            for name, plugin in _from_entry_points(plugin_type).items():
                assert plugin_type in _PLUGIN_TYPES
                self._add_plugin(cast("PluginType", plugin_type), name, plugin)
        # ruff: enable[I001,PLC0415]

    def _add_plugin(
        self,
        plugin_type: PluginType,
        name: str,
        plugin: type[Plugin],
    ) -> None:
        if plugin_type not in self._plugins:
            self._plugins[plugin_type] = {}
        name_lower = name.lower()
        if name_lower in self._plugins[plugin_type]:
            msg = f"Duplicate plugin name: {name_lower}"
            raise ValueError(msg)
        self._plugins[plugin_type][name_lower] = plugin

    def _get_plugin(
        self, plugin_type: PluginType, method: str
    ) -> tuple[str, Any] | None:
        self._init()
        split_method = method.split("/", maxsplit=1)
        if len(split_method) > 1:
            plugin_name, method = split_method
            plugin = self._plugins[plugin_type].get(plugin_name)
            if plugin and plugin.is_supported(method):
                return plugin_name, plugin
        else:
            method = split_method[0]
            if method == "default":
                msg = "Cannot specify 'default' method without a plugin name"
                raise ValueError(msg)
            plugins = {
                plugin_name: plugin
                for plugin_name, plugin in self._plugins[plugin_type].items()
                if plugin.allows_discovery() and plugin.is_supported(method)
            }
            default_plugin = _DEFAULT_PLUGINS[plugin_type]
            if default_plugin in plugins:
                return default_plugin, plugins[default_plugin]
            if len(plugins) > 1:
                msg = f"Ambiguous method: '{method}' is available in multiple plugins: {set(plugins.keys())}"
                raise ValueError(msg)
            if plugins:
                return plugins.popitem()
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
            If the method is found in the default plugin of `ropt`, that plugin
            is used. Otherwise it returns the first plugin found that supports
            the `method-name`.

        Args:
            plugin_type: The category of the plugin (e.g., "optimizer", "sampler").
            method:      The name of the method the plugin must support, potentially
                         prefixed with the plugin name and a slash (`/`).

        Returns:
            The plugin class that matches the criteria.

        Raises:
            ValueError: If no matching plugin is found for the given type and
                        method, or if "default" is used as a method name without
                        specifying a plugin name.
        """
        plugin = self._get_plugin(plugin_type, method)
        if plugin is not None:
            return plugin[1]
        msg = f"Method not found: {method}"
        raise ValueError(msg)

    def get_plugin_name(self, plugin_type: PluginType, method: str) -> str | None:
        """Return the name of the plugin that supports a given method.

        Verifies whether a plugin of the specified `plugin_type` supports the
        given `method`. This is useful for checking availability before attempting
        to retrieve a plugin with [`get_plugin`][ropt.plugins.PluginManager.get_plugin].

        The `method` argument can be specified in two ways:

        1.  **Explicit Plugin:** `"plugin-name/method-name"` checks if the specific
            plugin named `plugin-name` supports `method-name`.
        2.  **Implicit Plugin:** `"method-name"` searches through all discoverable
            plugins of the given `plugin_type` to see if any support `method-name`.
            If the method is found in the default plugin of `ropt`, that plugin
            is used. Otherwise it returns the first plugin found that supports
            the `method-name`.

        Args:
            plugin_type: The category of the plugin (e.g., "optimizer", "sampler").
            method:      The name of the method to check, potentially prefixed
                         with the plugin name and a slash (`/`).

        Returns:
            The name of a matching plugin supporting the specified method, or `None`.
        """
        plugin = self._get_plugin(plugin_type, method)
        if plugin is None:
            return None
        return plugin[0]


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
