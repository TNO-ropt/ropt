"""This module implements the default function evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult

from ._common import _active_evaluations, _scatter_result
from ._counter import BatchIdCounter
from .base import (
    EvaluationFunctionCallback,
    Evaluator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class FunctionEvaluator(Evaluator):
    """An evaluator that calls a function.

    This Evaluator stores a single function that returns a value for each
    objective and constraint.
    """

    # NOTE: A single instance of this class may be used from different threads,
    # e.g. if it is shared by optimizers running in different threads. The
    # batch ID is protected by a lock.

    def __init__(
        self,
        *,
        function: EvaluationFunctionCallback,
        batch_id_callback: Callable[[], int] | None = None,
    ) -> None:
        """Initialize the FunctionEvaluator.

        Args:
            function:          The function used for objectives and constraints.
            batch_id_callback: Callable that returns the next batch ID each time it is called.
        """
        super().__init__()
        self._function = function
        self._batch_id_callback = (
            batch_id_callback if batch_id_callback is not None else BatchIdCounter()
        )

    def eval(
        self, variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate all objective and constraints.

        Args:
            variables:         The matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.
        """
        batch_id = self._batch_id_callback()
        no = evaluator_context.context.objectives.weights.size
        nc = (
            0
            if evaluator_context.context.nonlinear_constraints is None
            else evaluator_context.context.nonlinear_constraints.lower_bounds.size
        )
        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        metadata: dict[str, NDArray[Any]] = {}

        for eval_idx, function_context in _active_evaluations(
            evaluator_context, batch_id
        ):
            _scatter_result(
                eval_idx,
                self._function(variables[eval_idx, :], function_context),
                results,
                metadata,
                no,
                variables.shape[0],
            )
        return EvaluationBatchResult(
            batch_id=batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            metadata=metadata,
        )
