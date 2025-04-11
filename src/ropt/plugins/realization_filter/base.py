"""This module defines the abstract base class for realization filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config.enopt import EnOptConfig


class RealizationFilter(ABC):
    """Abstract base class for realization filter classes."""

    def __init__(self, enopt_config: EnOptConfig, filter_index: int) -> None:  # noqa: B027
        """Initialize the realization filter plugin.

        Args:
            enopt_config:    The configuration of the optimizer.
            filter_index: The index of the filter to use.
        """

    @abstractmethod
    def get_realization_weights(
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return the updated weights of the realizations.

        This method is called by the optimizer with the current values of the
        objectives and constraints. Based on these values it must decide how
        much weight each realization should be given, and return those as a
        vector.

        The objectives and the constraints are passed as matrices, where the
        columns contain the values of the objectives or constraints. The index
        along the row axis corresponds to the number of the realization.

        Tip: Normalization
            The weights will be normalized to a sum of one by the optimizer
            before use, hence any non-negative weight value is permissable.

        Args:
            objectives:  The objectives of all realizations.
            constraints: The constraints for all realizations.

        Returns:
            A vector of weights of the realizations.
        """


class RealizationFilterPlugin(Plugin):
    """Abstract Base Class for Realization Filter Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`RealizationFilter`][ropt.plugins.realization_filter.base.RealizationFilter]
    instances. These plugins act as factories for specific realization filtering
    strategies.

    During plan execution, the [`PluginManager`][ropt.plugins.PluginManager]
    identifies the appropriate realization filter plugin based on the
    configuration and uses its `create` class method to instantiate the actual
    `RealizationFilter` object that will calculate the realization weights.
    """

    @classmethod
    @abstractmethod
    def create(cls, enopt_config: EnOptConfig, filter_index: int) -> RealizationFilter:
        """Factory method to create a concrete RealizationFilter instance.

        This abstract class method serves as a factory for creating concrete
        [`RealizationFilter`][ropt.plugins.realization_filter.base.RealizationFilter]
        objects. Plugin implementations must override this method to return an
        instance of their specific `RealizationFilter` subclass.

        The [`PluginManager`][ropt.plugins.PluginManager] calls this method when
        an optimization step requires realization weights calculated by this
        plugin.

        Args:
            enopt_config: The main EnOpt configuration object.
            filter_index: Index into `enopt_config.realization_filters` for
                          this filter.

        Returns:
            An initialized RealizationFilter object ready for use.
        """
