"""Internal batch and row contracts, and the public per-row context and result."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BatchContext:
    """Row identities and metadata for an internal evaluation batch.

    Attributes:
        active:        Boolean array indicating which rows require evaluation.
        realizations:  Realization index for each row.
        perturbations: Perturbation index for each row (< 0 means unperturbed).
        function_ids:  Function evaluation index for each row (< 0 means the row
                       is not a function evaluation).
        metadata:      Metadata for this batch, including any event-handler updates.
    """

    active: NDArray[np.bool_]
    realizations: NDArray[np.intc]
    perturbations: NDArray[np.intc]
    function_ids: NDArray[np.intc]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Results of an internal evaluation batch.

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


class BatchEvaluator(Protocol):
    """The internal batch evaluation interface."""

    def __call__(
        self, variables: NDArray[np.float64], batch: BatchContext, /
    ) -> BatchResult:
        """Evaluate the rows of a batch.

        Args:
            variables: The variables to pass to the evaluation function.
            batch:     Row identities and metadata of the batch.

        Returns:
            The results of the evaluation.
        """


@dataclass(slots=True)
class EvaluationFunctionContext:
    """Context for a single function evaluation.

    Attributes:
        realization:  The realization index.
        perturbation: The perturbation index (`-1` when unperturbed).
        function_id:  The function evaluation index within the batch (`-1` when
                      the row is not a function evaluation).
        batch_id:     Integer identifying the current evaluation batch.
        row_idx:      Row index within the batch.
        metadata:     Metadata for this batch, including any event-handler updates.
    """

    realization: int
    perturbation: int
    function_id: int
    batch_id: int
    row_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationFunctionResult:
    """Result of a single function evaluation.

    Attributes:
        objectives:  The objective values as an array.
        constraints: Optional constraint values as an array.
        metadata:    Extra information for this evaluation (default: empty).
    """

    objectives: NDArray[np.float64] | float
    constraints: NDArray[np.float64] | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RowEvaluator(Protocol):
    """The internal per-row evaluation interface.

    A function following this protocol is called once per active row of the
    evaluation batch, receiving the variable vector for that row together with
    an
    [`EvaluationFunctionContext`][ropt.EvaluationFunctionContext]
    object that identifies the evaluation, and returns an
    [`EvaluationFunctionResult`][ropt.EvaluationFunctionResult].
    """

    def __call__(
        self,
        variables: NDArray[np.float64],
        context: EvaluationFunctionContext,
    ) -> EvaluationFunctionResult:
        """Evaluate objectives and constraints for a single variable vector.

        Args:
            variables:    1-D variable vector for this evaluation.
            context:      The context identifying the evaluation.

        Returns:
            The objective and constraint values for this variable vector.
        """
