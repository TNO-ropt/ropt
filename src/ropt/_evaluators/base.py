"""The per-row types exchanged with an evaluation function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


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
        metadata:     The metadata the run was started with, if any.
    """

    realization: int
    perturbation: int
    function_id: int
    batch_id: int
    row_idx: int
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class EvaluationFunctionResult:
    """Result of a single function evaluation.

    Attributes:
        objectives:  The objective values as an array.
        constraints: Optional constraint values as an array.
        metadata:    Optional dictionary with extra information.
    """

    objectives: NDArray[np.float64] | float
    constraints: NDArray[np.float64] | float | None = None
    metadata: dict[str, Any] | None = None


class EvaluationFunctionCallback(Protocol):
    """Defines the call signature for function callbacks.

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
