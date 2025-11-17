"""This module implements the default function evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Evaluator

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorCallback, EvaluatorContext, EvaluatorResult


class DefaultFunctionEvaluator(Evaluator):
    """An evaluator that forwards calls to an evaluator function.

    This class acts as an adapter, allowing a standard Python callable (which
    matches the signature of the `eval` method) to be used as an
    [`Evaluator`][ropt.plugins.evaluator.base.Evaluator] within an optimization
    workflow.

    It is initialized with an `evaluator` callable. When the `eval` method of
    this class is invoked, it simply delegates the call, along with all
    arguments, to the wrapped `evaluator` function.
    """

    def __init__(self, *, evaluator: EvaluatorCallback) -> None:
        """Initialize the DefaultFunctionEvaluator.

        Args:
            evaluator: The callable that will perform the actual evaluation.
        """
        super().__init__()
        self._evaluator_callback = evaluator

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Forward the evaluation call to the wrapped evaluator function.

        Args:
            variables: The matrix of variables to evaluate.
            context:   The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.
        """
        return self._evaluator_callback(variables, context)
