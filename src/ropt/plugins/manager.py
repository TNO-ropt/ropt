"""The plugin manager."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from ropt._logging import get_logger

if TYPE_CHECKING:
    from ropt.plugins.base import Plugin


PluginType = Literal[
    "backend",
    "sampler",
    "realization_filter",
    "function_estimator",
    "variable_transform",
    "objective_transform",
    "nonlinear_constraint_transform",
]
"""Represents the valid types of plugins supported by `ropt`.

This type alias defines the string identifiers used to categorize different
plugins within the `ropt` framework.
"""


_DEFAULT_PLUGINS: Final = {
    "function_estimator": "default",
    "backend": "scipy",
    "sampler": "scipy",
    "realization_filter": "default",
    "variable_transform": "default",
    "objective_transform": "default",
    "nonlinear_constraint_transform": "default",
}

_logger = get_logger(__name__)


class PluginManager:
    """Manages the discovery and retrieval of `ropt` plugins.

    On initialization, scans the `ropt.plugins.*` entry-point groups (for
    example `ropt.plugins.backend`) and registers what it finds, alongside the
    plugins built into `ropt`. Retrieve a plugin class with
    [`get_plugin`][ropt.plugins.manager.PluginManager.get_plugin], or just its
    name with
    [`get_plugin_name`][ropt.plugins.manager.PluginManager.get_plugin_name].

    A third-party plugin registers itself under the relevant group in its own
    `pyproject.toml`, for example:

    ```toml
    [project.entry-points."ropt.plugins.backend"]
    my_backend = "my_package.my_module:MyBackendPlugin"
    ```
    """

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        # ruff: disable[import-outside-top-level]
        from .backend import BackendPlugin
        from .function_estimator import FunctionEstimatorPlugin
        from .realization_filter import RealizationFilterPlugin
        from .sampler import SamplerPlugin
        from .transforms import (
            NonlinearConstraintTransformPlugin,
            ObjectiveTransformPlugin,
            VariableTransformPlugin,
        )
        # ruff: enable[import-outside-top-level]

        self._PLUGIN_TYPES: Final = {
            "function_estimator": FunctionEstimatorPlugin,
            "backend": BackendPlugin,
            "sampler": SamplerPlugin,
            "realization_filter": RealizationFilterPlugin,
            "variable_transform": VariableTransformPlugin,
            "objective_transform": ObjectiveTransformPlugin,
            "nonlinear_constraint_transform": NonlinearConstraintTransformPlugin,
        }

        self._plugins: dict[PluginType, dict[str, type[Plugin]]] = {}
        self._init()

    def _init(self) -> None:
        # ruff: disable[unsorted-imports,import-outside-top-level]
        if self._plugins:
            return

        from ropt.sampler.scipy import SciPySamplerPlugin
        from ropt.realization_filter.default import DefaultRealizationFilterPlugin
        from ropt.function_estimator.default import DefaultFunctionEstimatorPlugin
        from ropt.backend.external import ExternalBackendPlugin
        from ropt.backend.scipy import SciPyBackendPlugin
        from ropt.transforms.default import (
            DefaultVariableTransformPlugin,
            DefaultObjectiveTransformPlugin,
            DefaultNonlinearConstraintTransformPlugin,
        )

        self._add_plugin("backend", "scipy", SciPyBackendPlugin)
        self._add_plugin("backend", "external", ExternalBackendPlugin)
        self._add_plugin("sampler", "scipy", SciPySamplerPlugin)
        self._add_plugin(
            "realization_filter", "default", DefaultRealizationFilterPlugin
        )
        self._add_plugin(
            "function_estimator", "default", DefaultFunctionEstimatorPlugin
        )
        self._add_plugin(
            "variable_transform", "default", DefaultVariableTransformPlugin
        )
        self._add_plugin(
            "objective_transform", "default", DefaultObjectiveTransformPlugin
        )
        self._add_plugin(
            "nonlinear_constraint_transform",
            "default",
            DefaultNonlinearConstraintTransformPlugin,
        )

        for plugin_type in self._PLUGIN_TYPES:
            for name, plugin in self._from_entry_points(plugin_type).items():
                assert plugin_type in self._PLUGIN_TYPES
                self._add_plugin(cast("PluginType", plugin_type), name, plugin)
        # ruff: enable[unsorted-imports,import-outside-top-level]

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
        _logger.debug("Registering plugin: %s/%s", plugin_type, name_lower)
        self._plugins[plugin_type][name_lower] = plugin

    def _get_plugin(
        self, plugin_type: PluginType, method: str
    ) -> tuple[str, Any] | None:
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
                msg = f"Method '{method}' is ambiguous across plugins: {set(plugins.keys())}"
                raise ValueError(msg)
            if plugins:
                return plugins.popitem()
        return None

    def get_plugin(self, plugin_type: PluginType, method: str) -> Any:  # ruff: ignore[any-type]
        """Retrieve a plugin class by its type and a supported method name.

        `method` is either `"plugin-name/method-name"` to request a specific
        plugin, or just `"method-name"` to search discoverable plugins of
        `plugin_type` for one that supports it (preferring the default plugin).

        Args:
            plugin_type: The category of the plugin (for example "backend", "sampler").
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

        Useful for checking availability before calling
        [`get_plugin`][ropt.plugins.manager.PluginManager.get_plugin], which
        takes `method` in the same two forms (`"plugin-name/method-name"` or
        just `"method-name"`).

        Args:
            plugin_type: The category of the plugin (for example "backend", "sampler").
            method:      The name of the method to check, potentially prefixed
                         with the plugin name and a slash (`/`).

        Returns:
            The name of a matching plugin supporting the specified method, or `None`.
        """
        plugin = self._get_plugin(plugin_type, method)
        if plugin is None:
            return None
        return plugin[0]

    def _from_entry_points(self, plugin_type: str) -> dict[str, type[Plugin]]:
        plugins: dict[str, type[Plugin]] = {}
        for entry_point in entry_points().select(group=f"ropt.plugins.{plugin_type}"):
            plugin = entry_point.load()
            plugins[entry_point.name] = plugin
            if not issubclass(
                plugins[entry_point.name], self._PLUGIN_TYPES[plugin_type]
            ):
                msg = (
                    f"Wrong type for {plugin_type} plugin `{entry_point.name}`"
                    f": {type(plugins[entry_point.name])}"
                )
                raise TypeError(msg)
        return plugins


_plugin_manager = None


def get_plugin(plugin_type: PluginType, method: str) -> Any:  # ruff: ignore[any-type]
    """Retrieve a plugin class by its type and a supported method name.

    Uses a lazily created, module-level [`PluginManager`][ropt.plugins.manager.PluginManager];
    see [`PluginManager.get_plugin`][ropt.plugins.manager.PluginManager.get_plugin]
    for the argument format.

    Args:
        plugin_type: The category of the plugin (for example "backend", "sampler").
        method:      The name of the method the plugin must support, potentially
                        prefixed with the plugin name and a slash (`/`).

    Returns:
        The plugin class that matches the criteria.
    """
    global _plugin_manager  # ruff: ignore[global-statement]
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager.get_plugin(plugin_type, method)


def get_plugin_name(plugin_type: PluginType, method: str) -> str | None:
    """Return the name of the plugin that supports a given method.

    Uses a lazily created, module-level [`PluginManager`][ropt.plugins.manager.PluginManager];
    see [`PluginManager.get_plugin_name`][ropt.plugins.manager.PluginManager.get_plugin_name]
    for the argument format.

    Args:
        plugin_type: The category of the plugin (for example "backend", "sampler").
        method:      The name of the method to check, potentially prefixed
                        with the plugin name and a slash (`/`).

    Returns:
        The name of a matching plugin supporting the specified method, or `None`.
    """
    global _plugin_manager  # ruff: ignore[global-statement]

    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager.get_plugin_name(plugin_type, method)
