"""This module defines the abstract base class for realization filters."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config import RealizationFilterConfig
    from ropt.realization_filter import RealizationFilter


class RealizationFilterPlugin(Plugin):
    """Abstract Base Class for Realization Filter Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`RealizationFilter`][ropt.realization_filter.RealizationFilter]
    instances. These plugins act as factories for specific realization filtering
    strategies.
    """

    @classmethod
    @abstractmethod
    def create(cls, filter_config: RealizationFilterConfig) -> RealizationFilter:
        """Factory method to create a concrete RealizationFilter instance.

        This abstract class method serves as a factory for creating concrete
        [`RealizationFilter`][ropt.realization_filter.RealizationFilter]
        objects. Plugin implementations must override this method to return an
        instance of their specific `RealizationFilter` subclass.

        The [`PluginManager`][ropt.plugins.manager.PluginManager] calls this
        method when an optimization requires realization weights calculated by
        this plugin.

        Args:
            filter_config: The configuration object for this realization filter.

        Returns:
            An initialized RealizationFilter object ready for use.
        """
