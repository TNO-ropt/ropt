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
    """An evaluator that calls a function.

    This Evaluator stores a single function that returns a value for each
    objective and constraint.
    """

    def __init__(
        self,
        *,
        function: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Initialize the DefaultFunctionEvaluator.

        Args:
            function: The function used for objectives and constraints.
        """
        super().__init__()
        self._function = function
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
        for eval_idx, realization in enumerate(context.realizations):
            perturbation = (
                -1
                if context.perturbations is None
                else int(context.perturbations[eval_idx])
            )
            if context.active is None or context.active[eval_idx]:
                results[eval_idx] = self._function(
                    variables[eval_idx, :],
                    realization=int(realization),
                    perturbation=perturbation,
                    batch_id=self._batch_id,
                    eval_idx=eval_idx,
                )
        return EvaluatorResult(
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
        )
