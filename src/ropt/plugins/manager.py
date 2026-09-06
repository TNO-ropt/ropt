"""The plugin manager."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Set as AbstractSet
from importlib.metadata import entry_points
from typing import Any, Final, Literal, TypeAlias, cast

from ropt._logging import get_logger

MethodSpec: TypeAlias = AbstractSet[str] | Callable[[str], bool]
"""How a plugin declares the methods it provides.

Either a set of method names, which the registry matches case-insensitively, or
a predicate for the plugins that cannot enumerate what they support and must
decide per name. A predicate receives the name verbatim, so it owns its own
casing, and it cannot be listed. A plugin that has a default method lists
`"default"` in its set; nothing is added on its behalf, since not every plugin
has one.
"""


PluginType = Literal[
    "backend",
    "sampler",
    "realization_filter",
    "function_estimator",
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
}

_logger = get_logger(__name__)


def _supports(plugin: type[Any], method: str) -> bool:
    # A predicate is passed the name verbatim, since a plugin that needs one
    # owns its own casing. A set is matched case-insensitively.
    methods = plugin.methods
    if callable(methods):
        return bool(methods(method))
    return method.lower() in {name.lower() for name in methods}


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
    my_backend = "my_package.my_module:MyBackend"
    ```

    A plugin that is not installed, for instance one defined in a script or a
    notebook, is added with
    [`register_plugin`][ropt.plugins.manager.PluginManager.register_plugin].
    """

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        # ruff: disable[import-outside-top-level]
        from ropt.backend import Backend
        from ropt.function_estimator import FunctionEstimator
        from ropt.realization_filter import RealizationFilter
        from ropt.sampler import Sampler
        # ruff: enable[import-outside-top-level]

        self._PLUGIN_TYPES: Final = {
            "function_estimator": FunctionEstimator,
            "backend": Backend,
            "sampler": Sampler,
            "realization_filter": RealizationFilter,
        }

        self._plugins: dict[PluginType, dict[str, type[Any]]] = {}
        self._installed: dict[PluginType, frozenset[str]] = {}
        self._init()
        self._installed = {
            plugin_type: frozenset(plugins)
            for plugin_type, plugins in self._plugins.items()
        }

    def _init(self) -> None:
        # ruff: disable[unsorted-imports,import-outside-top-level]
        if self._plugins:
            return

        from ropt.sampler.scipy import SciPySampler
        from ropt.realization_filter.default import DefaultRealizationFilter
        from ropt.function_estimator.default import DefaultFunctionEstimator
        from ropt.backend.external import ExternalBackend
        from ropt.backend.scipy import SciPyBackend

        self._add_plugin("backend", "scipy", SciPyBackend)
        self._add_plugin("backend", "external", ExternalBackend)
        self._add_plugin("sampler", "scipy", SciPySampler)
        self._add_plugin("realization_filter", "default", DefaultRealizationFilter)
        self._add_plugin("function_estimator", "default", DefaultFunctionEstimator)

        for plugin_type in self._PLUGIN_TYPES:
            assert plugin_type in self._PLUGIN_TYPES
            area = cast("PluginType", plugin_type)
            for name, plugin in self._from_entry_points(plugin_type).items():
                # Installed plugins share one namespace, so an entry point may
                # not take a name that a built-in or another package has.
                if name.lower() in self._plugins.get(area, {}):
                    msg = f"Duplicate plugin name: {name.lower()}"
                    raise ValueError(msg)
                self._add_plugin(area, name, plugin)
        # ruff: enable[unsorted-imports,import-outside-top-level]

    def _add_plugin(
        self,
        plugin_type: PluginType,
        name: str,
        plugin: type[Any],
    ) -> None:
        if not issubclass(plugin, self._PLUGIN_TYPES[plugin_type]):
            msg = f"Wrong type for {plugin_type} plugin `{name}`: {plugin}"
            raise TypeError(msg)
        name_lower = name.lower()
        if getattr(plugin, "methods", None) is None:
            msg = (
                f"The {plugin_type} plugin `{name_lower}` does not declare "
                "the methods it provides"
            )
            raise TypeError(msg)
        _logger.debug("Registering plugin: %s/%s", plugin_type, name_lower)
        # Replaced rather than updated: a lookup may be iterating this mapping
        # on another thread while a plugin is registered, and a new mapping
        # leaves that iteration on the one it started with.
        self._plugins[plugin_type] = {
            **self._plugins.get(plugin_type, {}),
            name_lower: plugin,
        }

    def register_plugin(
        self,
        plugin_type: PluginType,
        name: str,
        plugin: type[Any],
    ) -> None:
        """Register a plugin that is not installed.

        For a plugin defined where an entry point cannot reach it, such as a
        script or a notebook. Registering is otherwise equivalent to installing:
        the plugin is found by the same lookups, under the same rules, and
        `plugin` must subclass the base class of its area just the same. It must
        declare the methods it provides, as an installed plugin does.

        Registering the name of an installed plugin is an error; installed
        plugins cannot be shadowed. Registering a name that was registered
        before replaces it, and takes effect for every lookup that follows,
        including one made while an optimization is running.

        Args:
            plugin_type: The category of the plugin (for example "backend").
            name:        The name to register the plugin under.
            plugin:      The class to register.

        Raises:
            ValueError: If `name` is the name of an installed plugin.
        """
        name_lower = name.lower()
        if name_lower in self._installed.get(plugin_type, frozenset()):
            msg = (
                f"An installed {plugin_type} plugin is named `{name_lower}`, "
                "it cannot be replaced"
            )
            raise ValueError(msg)
        self._add_plugin(plugin_type, name, plugin)

    def _get_plugin(
        self, plugin_type: PluginType, method: str
    ) -> tuple[str, type[Any]] | None:
        split_method = method.split("/", maxsplit=1)
        if len(split_method) > 1:
            plugin_name, method = split_method
            # Plugin names are the registry's own namespace, and _add_plugin
            # stores them lowercased, so the lookup must lower them too.
            name_lower = plugin_name.lower()
            plugin = self._plugins[plugin_type].get(name_lower)
            if plugin is not None and _supports(plugin, method):
                return name_lower, plugin
        else:
            method = split_method[0]
            if method == "default":
                msg = "Cannot specify 'default' method without a plugin name"
                raise ValueError(msg)
            plugins = {
                name: plugin
                for name, plugin in self._plugins[plugin_type].items()
                if getattr(plugin, "discoverable", True) and _supports(plugin, method)
            }
            default_plugin = _DEFAULT_PLUGINS[plugin_type]
            if default_plugin in plugins:
                return default_plugin, plugins[default_plugin]
            if len(plugins) > 1:
                msg = f"Method '{method}' is ambiguous across plugins: {set(plugins)}"
                raise ValueError(msg)
            if plugins:
                return plugins.popitem()
        return None

    def get_plugin(self, plugin_type: PluginType, method: str) -> type[Any]:
        """Retrieve the class of a plugin by its type and a supported method name.

        `method` is either `"plugin-name/method-name"` to request a specific
        plugin, or just `"method-name"` to search discoverable plugins of
        `plugin_type` for one that supports it (preferring the default plugin).

        Args:
            plugin_type: The category of the plugin (for example "backend", "sampler").
            method:      The name of the method the plugin must support, potentially
                         prefixed with the plugin name and a slash (`/`).

        Returns:
            The class of the plugin that matches the criteria.

        Raises:
            ValueError: If no matching plugin is found for the given type and
                        method, or if "default" is used as a method name without
                        specifying a plugin name.
        """
        found = self._get_plugin(plugin_type, method)
        if found is not None:
            return found[1]
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
        found = self._get_plugin(plugin_type, method)
        if found is None:
            return None
        return found[0]

    @staticmethod
    def _from_entry_points(plugin_type: str) -> dict[str, type[Any]]:
        plugins: dict[str, type[Any]] = {}
        for entry_point in entry_points().select(group=f"ropt.plugins.{plugin_type}"):
            plugins[entry_point.name] = entry_point.load()
        return plugins


_plugin_manager = None


def get_plugin(plugin_type: PluginType, method: str) -> type[Any]:
    """Retrieve the class of a plugin by its type and a supported method name.

    Uses a lazily created, module-level [`PluginManager`][ropt.plugins.manager.PluginManager];
    see [`PluginManager.get_plugin`][ropt.plugins.manager.PluginManager.get_plugin]
    for the argument format.

    Args:
        plugin_type: The category of the plugin (for example "backend", "sampler").
        method:      The name of the method the plugin must support, potentially
                        prefixed with the plugin name and a slash (`/`).

    Returns:
        The class of the plugin that matches the criteria.
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


def register_plugin(
    plugin_type: PluginType,
    name: str,
    plugin: type[Any],
) -> None:
    """Register a plugin that is not installed.

    Adds the plugin to the lazily created, module-level
    [`PluginManager`][ropt.plugins.manager.PluginManager] that `optimize()` and
    `evaluate()` resolve against, so a plugin registered here is available to
    every workflow started afterwards. See
    [`PluginManager.register_plugin`][ropt.plugins.manager.PluginManager.register_plugin]
    for the rules.

    Args:
        plugin_type: The category of the plugin (for example "backend").
        name:        The name to register the plugin under.
        plugin:      The class to register.
    """
    global _plugin_manager  # ruff: ignore[global-statement]

    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    _plugin_manager.register_plugin(plugin_type, name, plugin)
