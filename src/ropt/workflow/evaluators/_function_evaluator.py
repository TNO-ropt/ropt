"""This module implements the default function evaluator."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult

from .base import (
    Evaluator,
    EvaluatorFunctionCallback,
    EvaluatorFunctionContext,
    EvaluatorFunctionResult,
)

if TYPE_CHECKING:
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
        function: EvaluatorFunctionCallback,
    ) -> None:
        """Initialize the FunctionEvaluator.

        Args:
            function: The function used for objectives and constraints.
        """
        super().__init__()
        self._function = function
        self._batch_id = 0
        self._batch_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        # threading.Lock is not picklable; drop it and recreate in __setstate__.
        state = self.__dict__.copy()
        state.pop("_batch_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._batch_lock = threading.Lock()

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
        with self._batch_lock:
            batch_id = self._batch_id
            self._batch_id += 1
        no = evaluator_context.context.objectives.weights.size
        nc = (
            0
            if evaluator_context.context.nonlinear_constraints is None
            else evaluator_context.context.nonlinear_constraints.lower_bounds.size
        )
        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        evaluation_info: dict[str, NDArray[Any]] = {}

        for eval_idx, realization in enumerate(evaluator_context.realizations):
            perturbation = (
                -1
                if evaluator_context.perturbations is None
                else int(evaluator_context.perturbations[eval_idx])
            )
            if evaluator_context.active is None or evaluator_context.active[eval_idx]:
                _handle_result(
                    eval_idx,
                    self._function(
                        variables[eval_idx, :],
                        EvaluatorFunctionContext(
                            realization=int(realization),
                            perturbation=perturbation,
                            batch_id=batch_id,
                            eval_idx=eval_idx,
                        ),
                    ),
                    results,
                    evaluation_info,
                    no,
                    variables.shape[0],
                )
        return EvaluationBatchResult(
            batch_id=batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            evaluation_info=evaluation_info,
        )


def _handle_result(  # noqa: PLR0913, PLR0917
    eval_idx: int,
    result: EvaluatorFunctionResult,
    results: NDArray[np.float64],
    evaluation_info: dict[str, NDArray[Any]],
    objective_count: int,
    eval_count: int,
) -> None:
    results[eval_idx, :objective_count] = result.objectives
    if result.constraints is not None:
        results[eval_idx, objective_count:] = result.constraints
    if result.evaluation_info is not None:
        for key, value in result.evaluation_info.items():
            if key not in evaluation_info:
                evaluation_info[key] = np.zeros(
                    eval_count,
                    dtype=(
                        np.array(value).dtype
                        if isinstance(value, (int, float, complex, np.number))
                        else object
                    ),
                )
            evaluation_info[key][eval_idx] = value
