"""This module implements the default callback evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Evaluator

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluator import EvaluatorCallback, EvaluatorContext, EvaluatorResult


class BatchEvaluator(Evaluator):
    """An evaluator that defers to a callable callback."""

    def __init__(self, *, callback: EvaluatorCallback) -> None:
        """Initialize the BatchEvaluator.

        Forwards the evaluation to the provided callback, which should implement
        the [`EvaluatorCallback`][ropt.evaluator.EvaluatorCallback] protocol.

        Args:
            callback: The callback to defer evaluation to.
        """
        super().__init__()
        self._callback = callback

    def eval(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Call the stored callback with the given variables and context.

        Args:
            variables: Matrix of variables to evaluate (each row is a vector).
            context:   The evaluation context.

        Returns:
            An `EvaluatorResult` with the evaluation results.
        """
        return self._callback(variables, context)
