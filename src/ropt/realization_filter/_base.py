"""Abstract base class for realization filter implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config._realization_filter_config import RealizationFilterConfig
    from ropt.context import EnOptContext


class RealizationFilter(ABC):
    """Abstract base class for realization filter implementations.

    Subclasses must implement three methods:

    1. `__init__` — store configuration; defer heavy work to `init`.
    2. `init` — called once with the full optimization context; validate
       settings and pre-compute any method-specific state here.
    3. `get_realization_weights` — called at each evaluation; return a
       non-negative weight per realization.

    See [Realization Filters](../usage/realization_filters.md) for examples
    and further guidance.
    """

    @abstractmethod
    def __init__(self, filter_config: RealizationFilterConfig) -> None:  # D107
        """Create a new realization filter instance.

        Store the configuration; keep initialization lightweight.
        Context-dependent setup belongs in `init`.

        Args:
            filter_config: The realization filter configuration.
        """

    @abstractmethod
    def init(self, context: EnOptContext) -> None:
        """Finalize initialization with the optimization context.

        Called once after configuration is finalized. Use for validation,
        internal state setup, or precomputation.

        Args:
            context: The optimization context.
        """

    @abstractmethod
    def get_realization_weights(
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Compute one weight per realization from current evaluation results.

        Return a non-negative weight for each realization. The optimizer
        normalizes weights to sum to one before use.

        Args:
            objectives:  Shape `(n_realizations, n_objectives)`.
            constraints: Shape `(n_realizations, n_constraints)`, or `None`.

        Returns:
            1-D array of shape `(n_realizations,)`.
        """
