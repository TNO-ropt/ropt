"""The data exchanged between `ropt` and a user evaluation function.

An [`EvaluationBatchContext`][ropt._evaluation.EvaluationBatchContext] describes
which rows must be evaluated, an
[`EvaluationBatchResult`][ropt._evaluation.EvaluationBatchResult] carries the
objective and constraint values back, and
[`EvaluationBatchCallback`][ropt._evaluation.EvaluationBatchCallback] is the
protocol such a callable must follow. The
[`Evaluator`][ropt._components.evaluators.Evaluator] classes that wrap them are
in `ropt._components.evaluators`.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ropt.context import EnOptContext


@dataclass(slots=True)
class EvaluationBatchContext:
    """Per-batch metadata passed to evaluator functions.

    Attributes:
        context:       The [`EnOptContext`][ropt.context.EnOptContext] for the run.
        active:        Boolean array indicating which rows require evaluation.
        realizations:  Realization index for each row.
        perturbations: Perturbation index for each row (< 0 means unperturbed).
        function_ids:  Function evaluation index for each row (< 0 means the row
                       is not a function evaluation).
        batch_id:      Integer identifying the current evaluation batch.
        metadata:      The metadata the run was started with, if any.
    """

    context: EnOptContext
    active: NDArray[np.bool_]
    realizations: NDArray[np.intc]
    perturbations: NDArray[np.intc]
    function_ids: NDArray[np.intc]
    batch_id: int = 0
    metadata: dict[str, Any] | None = None

    def get_active_evaluations[T: np.generic](self, array: NDArray[T]) -> NDArray[T]:
        """Return only the rows of `array` where `active` is `True`.

        Args:
            array: A 1-D or 2-D array with one entry/row per variable vector.

        Returns:
            The subset of rows corresponding to active evaluations.
        """
        return array[self.active, ...]

    def insert_inactive_results[T: np.generic](
        self, array: NDArray[T], *, fill_value: float = 0.0
    ) -> NDArray[T]:
        """Expand a filtered array back to full size, filling inactive rows.

        Inserts `fill_value` at positions where `active` is `False`, restoring
        the array to its original number of rows.

        Args:
            array:      The filtered array (output of `get_active_evaluations`).
            fill_value: The value to insert for inactive entries.

        Returns:
            An expanded array matching the original number of rows.
        """
        expanded_array = np.full(
            (self.realizations.shape[0], *array.shape[1:]),
            fill_value=fill_value,
            dtype=array.dtype,
        )
        expanded_array[self.active, ...] = array
        return expanded_array


@dataclass
class EvaluationBatchResult:
    """Results of a function evaluation batch.

    Stores objective values (and optional constraint values) for a batch of
    variable vectors. Inactive rows should be set to zero; failed active rows
    should be set to `numpy.nan`.

    Args:
        objectives:  Objective values, shape `(n_rows, n_objectives)`.
        constraints: Optional constraint values, shape `(n_rows, n_constraints)`.
        batch_id:    Integer identifying this evaluation batch.
        metadata:    Optional dict of per-row metadata (not used internally by `ropt`).
    """

    objectives: NDArray[np.float64]
    constraints: NDArray[np.float64] | None = None
    batch_id: int = 0
    metadata: dict[str, NDArray[Any]] = field(default_factory=dict)


class EvaluationBatchCallback(Protocol):
    """Defines the call signature for batch evaluation callbacks."""

    def __call__(
        self, variables: NDArray[np.float64], context: EvaluationBatchContext, /
    ) -> EvaluationBatchResult:
        """Evaluate the given variables within the provided context.

        Args:
            variables: The variables to pass to the evaluation function.
            context:   The evaluator context to pass.

        Returns:
            The results of the evaluation.
        """
