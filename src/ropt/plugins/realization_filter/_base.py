"""This module defines the abstract base class for realization filters."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config import RealizationFilterConfig
    from ropt.realization_filter import RealizationFilter


class RealizationFilterPlugin(Plugin):
    """Abstract base class for realization filter plugins (factories).

    Creates [`RealizationFilter`][ropt.realization_filter.RealizationFilter]
    instances; concrete plugins implement `create` as a factory for their own
    `RealizationFilter` subclass.
    """

    @classmethod
    @abstractmethod
    def create(cls, filter_config: RealizationFilterConfig) -> RealizationFilter:
        """Create a RealizationFilter instance.

        Called by the [`PluginManager`][ropt.plugins.manager.PluginManager]
        when an optimization requires realization weights from this plugin.

        Args:
            filter_config: The configuration object for this realization filter.

        Returns:
            An initialized RealizationFilter object ready for use.
        """
