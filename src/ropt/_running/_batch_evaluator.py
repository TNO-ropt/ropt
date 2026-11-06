"""Build the batch callable that evaluates each active row of a batch."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt._evaluation import (
    EvaluationBatchContext,
    EvaluationBatchResult,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
)
from ropt._executors import ExecutorFailure, WorkItem, WorkNotRun
from ropt._logging import get_logger
from ropt.enums import EvaluationStatus
from ropt.exceptions import AbortedError, ExecutionError, WorkflowError

from ._batch_ids import BatchIdCounter
from ._evaluation_helpers import _active_evaluations, _build_metadata, _scatter_result
from ._status_callback import make_status_reporter

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ropt._concurrency import AbortScope
    from ropt._evaluation import EvaluationBatchCallback, EvaluationFunctionCallback
    from ropt._executors import Executor

    from ._status_callback import StatusCallback

_logger = get_logger(__name__)


def make_batch_evaluator(
    *,
    function: EvaluationFunctionCallback,
    executor: Executor | None = None,
    batch_id_callback: Callable[[], int] | None = None,
    abort_scope: AbortScope | None = None,
    status_callback: StatusCallback | None = None,
) -> EvaluationBatchCallback:
    """Build a batch evaluation callable that calls a function once per row.

    Without an executor the rows run one after another on the calling thread;
    an abort is checked before and after each function call.
    With an executor each active row is sent as its own work item, and the
    callable returns once every result is back.

    Args:
        function:          The function used for objectives and constraints.
        executor:          An optional executor to dispatch the rows to.
        batch_id_callback: Callable that returns the next batch ID each time it is called.
        abort_scope:       An optional scope that aborts a running batch.
        status_callback:   An optional callback for the status of each evaluation.

    Returns:
        A callable following the
        [`EvaluationBatchCallback`][ropt._evaluation.EvaluationBatchCallback]
        protocol.
    """
    next_batch_id = (
        batch_id_callback if batch_id_callback is not None else BatchIdCounter()
    )

    def _evaluate(
        variables: NDArray[np.float64], evaluator_context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        batch_id = next_batch_id()
        no = evaluator_context.context.objectives.weights.size
        nc = (
            0
            if evaluator_context.context.nonlinear_constraints is None
            else evaluator_context.context.nonlinear_constraints.lower_bounds.size
        )
        results = np.zeros((variables.shape[0], no + nc), dtype=np.float64)
        metadata: dict[str, dict[int, Any]] = {}

        active = list(_active_evaluations(evaluator_context, batch_id))
        if executor is None:
            handle = partial(
                _handle_result, results=results, metadata=metadata, objective_count=no
            )
            report = make_status_reporter(status_callback)
            _run_rows(
                function, variables, active, abort_scope, handle=handle, report=report
            )
        else:
            values = _dispatch_rows(function, variables, active, executor, abort_scope)
            for (row_idx, _), value in zip(active, values, strict=True):
                _handle_result(row_idx, value, results, metadata, no)

        return EvaluationBatchResult(
            batch_id=batch_id,
            objectives=results[:, :no],
            constraints=results[:, no:] if nc > 0 else None,
            metadata=_build_metadata(metadata, variables.shape[0]),
        )

    return _evaluate


def _run_rows(  # ruff: ignore[too-many-arguments]
    function: EvaluationFunctionCallback,
    variables: NDArray[np.float64],
    active: list[tuple[int, EvaluationFunctionContext]],
    abort_scope: AbortScope | None,
    *,
    handle: Callable[[int, object], None],
    report: StatusCallback,
) -> None:
    try:
        for row_idx, function_context in active:
            _raise_if_aborting(abort_scope)
            report(function_context, EvaluationStatus.RUNNING)
            try:
                result = function(variables[row_idx, :], function_context)
                _raise_if_aborting(abort_scope)
                handle(row_idx, result)
            except AbortedError:
                raise
            except Exception:
                report(function_context, EvaluationStatus.ERROR)
                raise
            report(function_context, EvaluationStatus.DONE)
    finally:
        # Rows that already finished are dropped by the reporter.
        for _, function_context in active:
            report(function_context, EvaluationStatus.ABORTED)


def _raise_if_aborting(abort_scope: AbortScope | None) -> None:
    if abort_scope is not None and abort_scope.aborting:
        raise AbortedError(abort_scope.exit_code)


def _dispatch_rows(
    function: EvaluationFunctionCallback,
    variables: NDArray[np.float64],
    active: list[tuple[int, EvaluationFunctionContext]],
    executor: Executor,
    abort_scope: AbortScope | None,
) -> list[object]:
    _logger.debug("Dispatching %d work item(s) to executor", len(active))
    # Only the function and one row's arguments cross to the worker.
    values = executor.run(
        [
            WorkItem(function=function, args=(variables[row_idx, :], function_context))
            for row_idx, function_context in active
        ],
        abort_scope=abort_scope,
    )
    # A `WorkNotRun` only appears when the scope aborted the batch, so the
    # rows that did come back describe a batch nobody is waiting for.
    if abort_scope is not None and any(
        isinstance(value, WorkNotRun) for value in values
    ):
        raise AbortedError(abort_scope.exit_code)
    return values


def _handle_result(
    row_idx: int,
    value: object,
    results: NDArray[np.float64],
    metadata: dict[str, dict[int, Any]],
    objective_count: int,
) -> None:
    if isinstance(value, (ExecutorFailure, WorkNotRun)):
        msg = f"An evaluation could not be run: {value.message}"
        raise ExecutionError(msg)
    if not isinstance(value, EvaluationFunctionResult):
        msg = (
            "The evaluation function must return EvaluationFunctionResult "
            f"objects, got {type(value).__name__}."
        )
        raise WorkflowError(msg)
    _scatter_result(row_idx, value, results, metadata, objective_count)
