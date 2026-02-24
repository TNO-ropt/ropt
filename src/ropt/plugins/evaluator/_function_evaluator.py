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


class DefaultFunctionEvaluator(Evaluator):
    """An evaluator that calls a function.

    This Evaluator stores a single function that returns a value for each
    objective and constraint.
    """

    def __init__(
        self,
        *,
        function: Callable[..., NDArray[np.float64] | dict[str, Any]],
        evaluation_info: dict[str, np.dtype] | None = None,
    ) -> None:
        """Initialize the DefaultFunctionEvaluator.

        Args:
            function:        The function used for objectives and constraints.
            evaluation_info: Optional dictionary of evaluations info keys and data types.
        """
        super().__init__()
        self._function = function
        self._evaluation_info = {} if evaluation_info is None else evaluation_info
        self._batch_id = -1

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate all objective and constraints.

        Args:
            variables: The matrix of variables to evaluate.
            context:   The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.
        """
        self._batch_id += 1
        no = context.config.objectives.weights.size
        nc = (
            0
            if context.config.nonlinear_constraints is None
            else context.config.nonlinear_constraints.lower_bounds.size
        )
        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        evaluation_info: dict[str, NDArray[Any]] = {
            key: np.zeros(variables.shape[0], dtype=dtype)
            for key, dtype in self._evaluation_info.items()
        }

        for eval_idx, realization in enumerate(context.realizations):
            perturbation = (
                -1
                if context.perturbations is None
                else int(context.perturbations[eval_idx])
            )
            if context.active is None or context.active[eval_idx]:
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
) -> None:
    if isinstance(result, np.ndarray):
        results[eval_idx, :] = result
    else:
        assert isinstance(result, Mapping)
        assert "result" in result
        for key, value in result.items():
            if key == "result":
                results[eval_idx, :] = value
            elif key in evaluation_info:
                evaluation_info[key][eval_idx] = value
