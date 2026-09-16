"""Extending `ropt` with plugins."""

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
