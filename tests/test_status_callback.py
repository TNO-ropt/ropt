"""Tests for the status callback on the direct (no pool) path."""

# ruff: file-ignore[unused-function-argument]

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from ropt import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    EvaluationStatus,
    evaluate,
    evaluate_batch,
    optimize,
)
from ropt._concurrency import AbortScope
from ropt._evaluation import EvaluationBatchContext
from ropt._running._batch_evaluator import make_batch_evaluator
from ropt._running._status_callback import make_status_reporter
from ropt.context import EnOptContext
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError, WorkflowError

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CONFIG: dict[str, Any] = {
    "variables": {"variable_count": 2, "perturbation_magnitudes": 0.01},
    "realizations": {"weights": [1.0, 1.0]},
}

Seen = list[tuple[int, EvaluationStatus]]


def _record(seen: Seen) -> Any:
    def _callback(context: EvaluationFunctionContext, status: EvaluationStatus) -> None:
        seen.append((context.row_idx, status))

    return _callback


def _sum(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(objectives=np.array([variables.sum()]))


def _batch_context(active: list[bool]) -> EvaluationBatchContext:
    count = len(active)
    return EvaluationBatchContext(
        context=EnOptContext.model_validate(_CONFIG),
        active=np.array(active, dtype=np.bool_),
        realizations=np.zeros(count, dtype=np.intc),
        perturbations=np.zeros(count, dtype=np.intc),
        function_ids=np.zeros(count, dtype=np.intc),
    )


def test_status_callback_reports_running_then_done() -> None:
    seen: Seen = []
    evaluate_batch(_CONFIG, np.zeros((2, 2)), _sum, status_callback=_record(seen))
    running, done = EvaluationStatus.RUNNING, EvaluationStatus.DONE
    assert seen == [
        (0, running),
        (0, done),
        (1, running),
        (1, done),
        (2, running),
        (2, done),
        (3, running),
        (3, done),
    ]


def test_status_callback_receives_function_context() -> None:
    contexts: list[EvaluationFunctionContext] = []
    passed: list[EvaluationFunctionContext] = []

    def _function(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        passed.append(context)
        return _sum(variables, context)

    evaluate(
        _CONFIG,
        np.zeros(2),
        _function,
        metadata={"id": 7},
        status_callback=lambda context, _: contexts.append(context),
    )
    assert contexts[::2] == passed
    assert contexts[1::2] == passed
    assert all(context.metadata["id"] == 7 for context in contexts)


def test_status_callback_reports_error_and_aborts_remaining_rows() -> None:
    seen: Seen = []

    def _function(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        if context.row_idx == 1:
            msg = "failed"
            raise RuntimeError(msg)
        return _sum(variables, context)

    with pytest.raises(RuntimeError, match="failed"):
        evaluate_batch(
            _CONFIG, np.zeros((2, 2)), _function, status_callback=_record(seen)
        )
    assert seen == [
        (0, EvaluationStatus.RUNNING),
        (0, EvaluationStatus.DONE),
        (1, EvaluationStatus.RUNNING),
        (1, EvaluationStatus.ERROR),
        (2, EvaluationStatus.ABORTED),
        (3, EvaluationStatus.ABORTED),
    ]


def test_status_callback_reports_invalid_result_as_error() -> None:
    seen: Seen = []

    def _function(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        return cast("EvaluationFunctionResult", 1.0)

    evaluator = make_batch_evaluator(function=_function, status_callback=_record(seen))
    with pytest.raises(WorkflowError, match="must return EvaluationFunctionResult"):
        evaluator(np.zeros((2, 2)), _batch_context([True, True]))
    assert seen == [
        (0, EvaluationStatus.RUNNING),
        (0, EvaluationStatus.ERROR),
        (1, EvaluationStatus.ABORTED),
    ]


def test_status_callback_reports_abort_during_function_as_aborted() -> None:
    seen: Seen = []
    scope = AbortScope()

    def _function(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        scope.abort(ExitCode.USER_ABORT)
        return _sum(variables, context)

    evaluator = make_batch_evaluator(
        function=_function, abort_scope=scope, status_callback=_record(seen)
    )
    with pytest.raises(AbortedError):
        evaluator(np.zeros((2, 2)), _batch_context([True, True]))
    assert seen == [
        (0, EvaluationStatus.RUNNING),
        (0, EvaluationStatus.ABORTED),
        (1, EvaluationStatus.ABORTED),
    ]


def test_status_callback_skips_inactive_rows() -> None:
    seen: Seen = []
    evaluator = make_batch_evaluator(function=_sum, status_callback=_record(seen))
    evaluator(np.zeros((3, 2)), _batch_context([True, False, True]))
    assert [row for row, _ in seen] == [0, 0, 2, 2]


def test_status_callback_exception_propagates() -> None:
    calls = 0

    def _function(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> EvaluationFunctionResult:
        nonlocal calls
        calls += 1
        return _sum(variables, context)

    def _callback(_: EvaluationFunctionContext, status: EvaluationStatus) -> None:
        if status is EvaluationStatus.RUNNING:
            msg = "callback failed"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="callback failed"):
        evaluate(_CONFIG, np.zeros(2), _function, status_callback=_callback)
    assert calls == 0


def test_status_callback_is_called_by_optimize() -> None:
    seen: Seen = []
    config = {**_CONFIG, "optimizer": {"max_functions": 2}}
    optimize(config, np.zeros(2), _sum, status_callback=_record(seen))
    assert seen
    assert {status for _, status in seen} == {
        EvaluationStatus.RUNNING,
        EvaluationStatus.DONE,
    }


def test_status_reporter_drops_events_out_of_order() -> None:
    seen: Seen = []
    report = make_status_reporter(_record(seen))
    context = EvaluationFunctionContext(
        realization=0, perturbation=-1, function_id=0, batch_id=0, row_idx=0
    )
    for status in [
        EvaluationStatus.RUNNING,
        EvaluationStatus.SUBMITTED,
        EvaluationStatus.RUNNING,
        EvaluationStatus.DONE,
        EvaluationStatus.ERROR,
        EvaluationStatus.ABORTED,
    ]:
        report(context, status)
    assert seen == [(0, EvaluationStatus.RUNNING), (0, EvaluationStatus.DONE)]
