"""This module defines the abstract base class for plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Plugin(ABC):
    """Abstract base class for plugins.

    `ropt`  plugins should derive from this base class, which specifies an
    `is_supported` method.
    """

    @abstractmethod
    def is_supported(self, method: str, *, explicit: bool) -> bool:
        """Check wether a given method is supported.

        This method is used by the
        [`is_supported`][ropt.plugins.PluginManager.is_supported] method
        of [`PluginManager`][ropt.plugins.PluginManager] objects, to
        check whether a given method is supported.

        If the `explicit` flag is set, then the method was explicitly requested
        for this plugin. If not, the caller does not know which plugin contains
        the requested method and is checking whether this plugin supports a
        method with this name. This can be used to return `False` if the plugin
        only wants to be specified explicitly using its name. In other cases,
        this argument can be ignored.

        Args:
            method:   The method name.
            explicit: Whether the plugin was requested explicitly.

        Returns:
            True if the method is supported.
        """
