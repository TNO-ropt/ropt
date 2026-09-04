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

    See [Realization Filters](../optimizer_setup/realization_filters.md) for examples
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

        Called once at the start of a run, for every configured filter, also
        for those that no objective or constraint refers to. Use for
        validation, internal state setup, or precomputation. The number of
        realizations is available as `context.realizations.weights.size`.

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

        Called once per function evaluation, and only if at least one objective
        or nonlinear constraint refers to this filter. The weights returned by
        a single call are applied to all of them, and are reused for the
        gradients derived from that evaluation.

        Both arguments are two-dimensional arrays with one row per realization
        and one column per objective or per nonlinear constraint, in the order
        in which they are configured: `objectives[i, j]` is the value of
        objective `j` for realization `i`. The values are scaled, but not
        negated for maximization: direction applies to aggregates, and these
        are per-realization.

        A realization that failed to evaluate carries `nan` values. The filter
        should check for these and handle them, for instance by assigning such
        realizations a weight of zero.

        The returned weights replace the weights configured in the
        `realizations` section, and are normalized to sum to one before use. If
        no realization can be given a positive weight, raise
        [`TooFewRealizations`][ropt.exceptions.TooFewRealizations] to record
        the evaluation as failed.

        Args:
            objectives:  Objectives, shape `(n_realizations, n_objectives)`.
            constraints: Nonlinear constraints, shape
                         `(n_realizations, n_constraints)`, or `None` if no
                         nonlinear constraints are configured.

        Returns:
            The non-negative weights, shape `(n_realizations,)`.
        """
