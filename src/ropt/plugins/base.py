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
        """Check whether a given method is supported.

        This method is called by the
        [`is_supported`][ropt.plugins.PluginManager.is_supported] method of
        [`PluginManager`][ropt.plugins.PluginManager] objects to verify if a
        specific method is supported by this plugin.

        If the `explicit` flag is set to `True`, the plugin has been explicitly
        requested. In this case, `True` should be returned if the specified
        optimization method is supported.

        If `explicit` is `False`, the plugin manager is performing a general
        search across all available plugins for the requested method. In this
        scenario, `True` should only be returned if the plugin's method is
        compatible with this search. If the search compatibility is not
        supported, then `False` should be returned, even if the plugin
        technically supports the requested method.

        For example, the
        [`external`][ropt.plugins.optimizer.external.ExternalOptimizer]
        optimizer plugin does not define its own methods but launches methods
        from other plugins as an external process. Therefore, the `external`
        optimizer plugin must always be specified explicitly. As a result, its
        `is_supported` method returns `False` if `explicit` is `False`.

        Args:
            method:   The name of the method to check.
            explicit: Indicates whether the plugin was explicitly requested.

        Returns:
            True if the method is supported; otherwise, False.
        """
