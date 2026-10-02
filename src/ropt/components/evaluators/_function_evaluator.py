"""This module implements the default function evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.exceptions import OptimizerStop

from ._common import _active_evaluations, _build_metadata, _scatter_result
from ._counter import BatchIdCounter
from .base import (
    EvaluationFunctionCallback,
    Evaluator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt.components.concurrency import AbortSignal


class FunctionEvaluator(Evaluator):
    """An evaluator that calls a function once for each variable vector.

    The function returns the objective and constraint values for that vector.
    """

    # NOTE: A single instance may be reused serially across threads, for example by
    # optimizers that run one after another on different threads. It must not be
    # used concurrently: the base class raises if two threads call `eval` at the
    # same time. The batch ID is protected by a lock.

    def __init__(
        self,
        *,
        function: EvaluationFunctionCallback,
        batch_id_callback: Callable[[], int] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> None:
        """Initialize the FunctionEvaluator.

        Args:
            function:          The function used for objectives and constraints.
            batch_id_callback: Callable that returns the next batch ID each time it is called.
            abort_signal:      An optional signal that abandons a running batch.
        """
        super().__init__()
        self._function = function
        self._batch_id_callback = (
            batch_id_callback if batch_id_callback is not None else BatchIdCounter()
        )
        self._abort_signal = abort_signal

    def _eval(
        self, variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        """Evaluate all objective and constraints.

        Args:
            variables:         The matrix of variables to evaluate.
            evaluator_context: The evaluation context.

        Returns:
            The result of calling the wrapped evaluator function.

        Raises:
            OptimizerStop: If the abort signal abandoned the batch.
        """
        batch_id = self._batch_id_callback()
        no = evaluator_context.context.objectives.weights.size
        nc = (
            0
            if evaluator_context.context.nonlinear_constraints is None
            else evaluator_context.context.nonlinear_constraints.lower_bounds.size
        )
        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        metadata: dict[str, dict[int, Any]] = {}

        for eval_idx, function_context in _active_evaluations(
            evaluator_context, batch_id
        ):
            # The rows run on this thread, so this is the only place the batch
            # can be abandoned part way.
            if self._abort_signal is not None and self._abort_signal.aborting:
                raise OptimizerStop(self._abort_signal.exit_code)
            _scatter_result(
                eval_idx,
                self._function(variables[eval_idx, :], function_context),
                results,
                metadata,
                no,
            )
        return EvaluationBatchResult(
            batch_id=batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            metadata=_build_metadata(metadata, variables.shape[0]),
        )
