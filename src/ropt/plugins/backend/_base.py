"""Base class for optimzers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.backend import Backend
    from ropt.config import BackendConfig


class BackendPlugin(Plugin):
    """Abstract Base Class for Backend Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`Backend`][ropt.backend.Backend] instances. These plugins
    act as factories for specific optimization algorithms or backends.
    """

    @classmethod
    @abstractmethod
    def create(cls, backend_config: BackendConfig) -> Backend:
        """Create an Backend instance.

        This abstract class method serves as a factory for creating concrete
        [`Backend`][ropt.backend.Backend] objects. Plugin
        implementations must override this method to return an instance of  their
        specific `Backend` subclass.

        The [`PluginManager`][ropt.plugins.manager.PluginManager] calls this
        method when an optimization workflow requires an optimizer provided by
        this plugin.

        Args:
            backend_config: The configuration object containing the
                            backend settings.

        Returns:
            An initialized instance of an `Backend` subclass.
        """
