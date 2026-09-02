"""Plugin support for realization filters.

A realization filter sets the weight of each realization whenever objective and
constraint functions are evaluated. A
[`RealizationFilterPlugin`][ropt.plugins.realization_filter.RealizationFilterPlugin]
is a factory that creates the
[`RealizationFilter`][ropt.realization_filter.RealizationFilter] objects doing
the actual work, which the
[`PluginManager`][ropt.plugins.manager.PluginManager] discovers through the
`ropt.plugins.realization_filter` entry point group.

`ropt` ships
[`DefaultRealizationFilter`][ropt.realization_filter.default.DefaultRealizationFilter],
which provides sorting and CVaR based methods.

See [Realization Filters](../optimizer_setup/realization_filters.md) for usage,
and [Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.
"""

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
