"""This module implements the default function evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.evaluator import EvaluatorContext, EvaluatorResult

from .base import Evaluator

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class FunctionEvaluator(Evaluator):
    """An evaluator that calls a function.

    This Evaluator stores a single function that returns a value for each
    objective and constraint.
    """

    def __init__(
        self,
        *,
        function: Callable[..., NDArray[np.float64] | dict[str, Any]],
    ) -> None:
        """Initialize the FunctionEvaluator.

        Args:
            function: The function used for objectives and constraints.
        """
        super().__init__()
        self._function = function
        self._batch_id = -1

    def eval(
        self, variables: NDArray[np.float64], evaluator_context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate all objective and constraints.

        Args:
            variables:         The matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.
        """
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
                        realization=int(realization),
                        perturbation=perturbation,
                        batch_id=self._batch_id,
                        eval_idx=eval_idx,
                    ),
                    results,
                    evaluation_info,
                    variables.shape[0],
                )
        return EvaluatorResult(
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            evaluation_info=evaluation_info,
        )


def _handle_result(
    eval_idx: int,
    result: NDArray[np.float64] | dict[str, Any],
    results: NDArray[np.float64],
    evaluation_info: dict[str, NDArray[Any]],
    eval_count: int,
) -> None:
    if isinstance(result, np.ndarray):
        results[eval_idx, :] = result
    else:
        assert isinstance(result, Mapping)
        assert "result" in result
        for key, value in result.items():
            if key == "result":
                results[eval_idx, :] = value
            else:
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
