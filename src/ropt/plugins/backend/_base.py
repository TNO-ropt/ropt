"""Base class for optimizers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.backend import Backend
    from ropt.config import BackendConfig


class BackendPlugin(Plugin):
    """Abstract base class for backend plugins (factories).

    Creates [`Backend`][ropt.backend.Backend] instances; concrete plugins
    implement `create` as a factory for their own `Backend` subclass.
    """

    @classmethod
    @abstractmethod
    def create(cls, backend_config: BackendConfig) -> Backend:
        """Create a Backend instance.

        Called by the [`PluginManager`][ropt.plugins.manager.PluginManager]
        when an optimization workflow requires an optimizer from this plugin.

        Args:
            backend_config: The configuration object containing the
                            backend settings.

        Returns:
            An initialized instance of a `Backend` subclass.
        """
