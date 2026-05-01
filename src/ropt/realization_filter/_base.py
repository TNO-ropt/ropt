"""Abstract base class for realization filter implementations.

Realization filters are responsible for deciding how strongly each
realization contributes to the aggregated objective and constraint values used
by the optimizer. This module defines the interface that all concrete filter
implementations must follow.
"""

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

    All concrete realization filter implementations must inherit from this
    class and implement the required weighting method. Filters are responsible
    for examining per-realization objective and constraint values and returning
    a non-negative weight for each realization.

    The weights produced by filters are used by the optimizer to control how
    strongly each realization contributes to aggregated objective and
    constraint values.

    **Lifecycle**

    1. Instantiation via `__init__`: Called by the plugin system with a
       configuration object.
    2. Setup via `init`: Called once per optimization workflow with the
       [`EnOptContext`][ropt.context.EnOptContext], allowing final
       initialization based on the full optimization configuration.
    3. Evaluation via `get_realization_weights`: Called repeatedly during
       optimization as new objective and constraint values become available.

    Subclasses must implement:

    - `__init__`: Stores filter configuration and performs lightweight setup.
    - `init`: Receives the optimization context for context-dependent initialization.
    - `get_realization_weights`: Returns one weight per realization based on
      the current objective and constraint values.
    """

    @abstractmethod
    def __init__(self, filter_config: RealizationFilterConfig) -> None:  # D107
        """Create a new realization filter instance.

        Called during instantiation. Subclasses should store the configuration
        and perform any lightweight initialization. Validation and
        context-dependent setup should be deferred to the `init` method.

        Args:
            filter_config: The realization filter configuration.
        """

    @abstractmethod
    def init(self, context: EnOptContext) -> None:
        """Finalize initialization after the optimization context is known.

        Called once at the start of each optimization workflow, after all
        configuration is finalized. Use this method to perform context-dependent
        initialization such as validation of filter settings, setup of internal
        state, or precomputation of method-specific data.

        Args:
            context: The main EnOpt context object.
        """

    @abstractmethod
    def get_realization_weights(
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Compute one weight per realization from current evaluation results.

        Examines the current per-realization objective and constraint values
        and returns a non-negative weight vector that determines how strongly
        each realization influences subsequent aggregation steps. The exact
        weighting rule is defined by the concrete filter implementation.

        The `objectives` and `constraints` inputs are organized with
        realizations along the first axis and objective or constraint indices
        along the second axis.

        Tip: Normalization
            The weights will be normalized to a sum of one by the optimizer
            before use, hence any non-negative weight value is permissible.

        Args:
            objectives: Array of shape `(n_realizations, n_objectives)`
                containing objective values for each realization.
            constraints: Array of shape `(n_realizations, n_constraints)`
                containing constraint values for each realization, or `None`
                when the workflow has no constraints.

        Returns:
            A 1D array of shape `(n_realizations,)` containing the weight of
            each realization.
        """
