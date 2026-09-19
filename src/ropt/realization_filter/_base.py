"""Abstract base class for realization filter implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config._realization_filter_config import RealizationFilterConfig
    from ropt.plugins import MethodSpec


class RealizationFilter(ABC):
    """Abstract base class for realization filter implementations.

    See [Realization Filters](../optimizer_setup/realization_filters.md) for examples
    and further guidance.
    """

    methods: ClassVar[MethodSpec]
    """The filter methods this class provides.

    Either a set of names, which the registry matches case-insensitively, or a
    predicate for classes that cannot enumerate them. Include `"default"` in the
    set if this class has one. See [`MethodSpec`][ropt.plugins.MethodSpec].
    """

    @abstractmethod
    def __init__(self, filter_config: RealizationFilterConfig) -> None:  # D107
        """Create a new realization filter instance.

        Store the configuration and pre-compute any method-specific state.

        Args:
            filter_config: The realization filter configuration.
        """

    @abstractmethod
    def get_realization_weights(
        self,
        objectives: NDArray[np.float64],
        constraints: NDArray[np.float64] | None,
        *,
        objective_scales: NDArray[np.float64],
        maximize: NDArray[np.bool_],
        objective_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute one weight per realization from current evaluation results.

        Called once per function evaluation, and only if at least one objective
        or nonlinear constraint refers to this filter. The weights returned by
        a single call are applied to all of them, and are reused for the
        gradients derived from that evaluation.

        `objectives` and `constraints` are two-dimensional arrays with one row
        per realization and one column per objective or per nonlinear
        constraint, in the order in which they are configured:
        `objectives[i, j]` is the value of objective `j` for realization `i`.
        The number of realizations is therefore `objectives.shape[0]`. The
        values are as the evaluator returned them: neither scaled nor negated
        for maximization, since both apply to aggregates and these are
        per-realization. A filter that ranks by what the optimizer minimizes
        should apply `objective_scales` and `maximize` itself.

        A realization that failed to evaluate carries `nan` values. The filter
        should check for these and handle them, for instance by assigning such
        realizations a weight of zero.

        The returned weights replace the weights configured in the
        `realizations` section, and are normalized to sum to one before use. If
        no realization can be given a positive weight, raise
        [`TooFewRealizations`][ropt.exceptions.TooFewRealizations] to record
        the evaluation as failed.

        `objective_scales` is passed on every call because auto-scaling only
        fixes the scales after the first batch.

        Args:
            objectives:        Objectives, shape `(n_realizations, n_objectives)`.
            constraints:       Constraints, shape `(n_realizations, n_constraints)`.
            objective_scales:  The scale applied to each objective.
            maximize:          Which objectives are maximized.
            objective_weights: The configured weight of each objective.

        Returns:
            The non-negative weights, shape `(n_realizations,)`.
        """
