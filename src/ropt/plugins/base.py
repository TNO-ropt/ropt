"""This module defines the abstract base class for plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Plugin(ABC):
    """Abstract base class for plugins.

    `ropt`  plugins should derive from this base class, which specifies an
    `is_supported` method.
    """

    @abstractmethod
    def is_supported(self, method: str) -> bool:
        """Check whether a given method is supported.

        This method is called by the
        [`is_supported`][ropt.plugins.PluginManager.is_supported] method of
        [`PluginManager`][ropt.plugins.PluginManager] objects to verify if a
        specific method is supported by this plugin.

        Args:
            method:   The name of the method to check.

        Returns:
            True if the method is supported; otherwise, False.
        """

    @property
    def allows_discovery(self) -> bool:
        """Check if the plugin can be discovered automatically.

        Normally, plugins may be discovered automatically by the plugin manager
        by checking if they support a specific method. However, some plugins may
        not support this behavior and should be explicitly requested. These
        should override this property to return `False`.

        For example, the
        [`external`][ropt.plugins.optimizer.external.ExternalOptimizer]
        optimizer plugin does not define its own methods but launches methods
        from other plugins as an external process. Therefore, the `external`
        optimizer plugin must always be specified explicitly, and this method
        is overloaded to return False.
        """
        return True
