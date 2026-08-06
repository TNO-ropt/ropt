"""This module implements the default callback evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Evaluator

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.evaluation import (
        EvaluationBatchCallback,
        EvaluationBatchContext,
        EvaluationBatchResult,
    )


class BatchEvaluator(Evaluator):
    """An evaluator that defers to a callable callback."""

    # NOTE: This class is a thin pass-through to `callback`. A single instance
    # may be reused serially across threads, but must not be used concurrently
    # (the base class raises if two threads call `eval` at the same time). For
    # concurrent use the wrapped callback must itself be safe to call
    # concurrently.

    def __init__(self, *, callback: EvaluationBatchCallback) -> None:
        """Initialize the BatchEvaluator.

        Forwards the evaluation to the provided callback, which should implement
        the [`EvaluationBatchCallback`][ropt.evaluation.EvaluationBatchCallback] protocol.

        Args:
            callback: The callback to defer evaluation to.
        """
        super().__init__()
        self._callback = callback

    def eval(
        self, variables: NDArray[np.float64], context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Call the stored callback with the given variables and context.

        Args:
            variables: Matrix of variables to evaluate (each row is a vector).
            context:   The evaluation context.

        Returns:
            An `EvaluationBatchResult` with the evaluation results.
        """
        return self._callback(variables, context)
