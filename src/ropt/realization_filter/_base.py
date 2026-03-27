"""This module defines the abstract base class for realization filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import EnOptConfig


class RealizationFilter(ABC):
    """Abstract base class for realization filter classes."""

    @abstractmethod
    def init(self, enopt_config: EnOptConfig) -> None:
        """Initialize the realization filter.

        Args:
            enopt_config: The main EnOpt configuration object.
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
            before use, hence any non-negative weight value is permissible.

        Args:
            objectives:   The objectives of all realizations.
            constraints:  The constraints for all realizations.

        Returns:
            A vector of weights of the realizations.
        """
