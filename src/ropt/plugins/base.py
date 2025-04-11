"""This module defines the abstract base class for plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Plugin(ABC):
    """Abstract base class for all `ropt` plugins.

    This class serves as the fundamental building block for all plugins within
    the `ropt` framework. Any class intended to function as a plugin (e.g.,
    an optimizer, sampler, plan step, or plan handler) must inherit from this
    base class.

    It defines the core interface that all plugins must adhere to, ensuring
    consistency and enabling the [`PluginManager`][ropt.plugins.PluginManager]
    to discover and manage them effectively.

    Subclasses must implement the `is_supported` class method to indicate which
    named methods (functionalities) they provide. They can optionally override
    the `allows_discovery` class method if they should not be automatically
    selected by the plugin manager when a method name is provided without an
    explicit plugin name.
    """

    @classmethod
    @abstractmethod
    def is_supported(cls, method: str) -> bool:
        """Verify if this plugin supports a specific named method.

        This class method is used by the
        [`PluginManager`][ropt.plugins.PluginManager] (specifically its
        [`is_supported`][ropt.plugins.PluginManager.is_supported] method) to
        determine if this plugin class provides the functionality associated
        with the given `method` name.

        Args:
            method: The string identifier of the method to check for support.

        Returns:
            `True` if the plugin supports the specified method, `False` otherwise.
        """

    @classmethod
    def allows_discovery(cls) -> bool:
        """Determine if the plugin allows implicit discovery by method name.

        By default (`True`), plugins can be found by the
        [`PluginManager`][ropt.plugins.PluginManager] when a user provides only
        a method name (without specifying the plugin, e.g., `"method-name"`).

        If a plugin should *only* be used when explicitly named (e.g.,
        `"plugin-name/method-name"`), it must override this class method to
        return `False`.

        For instance, the
        [`external`][ropt.plugins.optimizer.external.ExternalOptimizer] optimizer
        plugin acts as a wrapper for other optimizers run in separate processes.
        It doesn't provide methods directly and must always be explicitly
        requested, so it overrides this method to return `False`.

        Returns:
            `True` if the plugin can be discovered implicitly by method name.
        """
        return True
