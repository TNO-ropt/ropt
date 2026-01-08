"""This module implements the default function evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt.evaluator import EvaluatorContext, EvaluatorResult

from .base import Evaluator

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class DefaultFunctionEvaluator(Evaluator):
    """An evaluator that calls a function for each objective and constraint.

    This Evaluator stores a function for each objective and constraint and
    calls these sequentially.
    """

    def __init__(
        self,
        *,
        functions: list[Callable[[NDArray[np.float64], int], float]],
    ) -> None:
        """Initialize the DefaultFunctionEvaluator.

        Args:
            functions: The functions used for objectives and constraints.
        """
        super().__init__()
        self._functions = functions

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Evaluate all objective and constraint functions.

        Args:
            variables: The matrix of variables to evaluate.
            context:   The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.
        """
        objective_count = context.config.objectives.weights.size
        constraint_count = (
            0
            if context.config.nonlinear_constraints is None
            else context.config.nonlinear_constraints.lower_bounds.size
        )
        objective_results = np.zeros(
            (variables.shape[0], objective_count),
            dtype=np.float64,
        )
        constraint_results = (
            np.zeros((variables.shape[0], constraint_count), dtype=np.float64)
            if constraint_count > 0
            else None
        )
        for eval_idx, realization in enumerate(context.realizations):
            if context.active is None or context.active[eval_idx]:
                for idx in range(objective_count):
                    function = self._functions[idx]
                    objective_results[eval_idx, idx] = function(
                        variables[eval_idx, :], int(realization)
                    )
                for idx in range(constraint_count):
                    function = self._functions[idx + objective_count]
                    assert constraint_results is not None
                    constraint_results[eval_idx, idx] = function(
                        variables[eval_idx, :], int(realization)
                    )

        return EvaluatorResult(
            objectives=objective_results,
            constraints=constraint_results,
        )
