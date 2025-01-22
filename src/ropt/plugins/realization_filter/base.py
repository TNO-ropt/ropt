"""This module defines the abstract base class for realization filters.

Realization filters can be added via the plugin mechanism to implement
additional ways to filter the realizations that are used to calculate functions
and gradients. Any object that derives from the
[`RealizationFilter`][ropt.plugins.realization_filter.base.RealizationFilter]
abstract base class may be installed as a plugin.
"""

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

    @abstractmethod
    def __init__(self, enopt_config: EnOptConfig, filter_index: int) -> None:  # D107
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

        The objectives and the constraints are passed as matrix, where the
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
    """Abstract base class for realizationFilter plugins."""

    @abstractmethod
    def create(self, enopt_config: EnOptConfig, filter_index: int) -> RealizationFilter:
        """Initialize the realization filter plugin.

        Args:
            enopt_config: The configuration of the optimizer.
            filter_index: The index of the filter to use.
        """
