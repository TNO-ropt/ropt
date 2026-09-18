"""Plugin discovery and registration.

A [`PluginManager`][ropt.plugins.PluginManager] finds plugins through their
entry points and looks them up by method name, returning the class, which is
called with the configuration object of its area to build the component.
[`get_plugin`][ropt.plugins.get_plugin] and
[`get_plugin_name`][ropt.plugins.get_plugin_name] do the same lookups against
one shared manager, created on first use; every optimization resolves against
that manager, so it is the one
[`register_plugin`][ropt.plugins.register_plugin] has to add to for a plugin
that cannot be installed. See
[Writing a Plugin](../advanced/writing_plugins.md).
"""

from ._manager import (
    MethodSpec,
    PluginManager,
    PluginType,
    get_plugin,
    get_plugin_name,
    register_plugin,
)

__all__ = [
    "MethodSpec",
    "PluginManager",
    "PluginType",
    "get_plugin",
    "get_plugin_name",
    "register_plugin",
]
