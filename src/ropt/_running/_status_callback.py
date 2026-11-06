"""Status callbacks for the evaluations of an optimization or evaluation."""

from __future__ import annotations

from collections.abc import Callable

from ropt._evaluation import EvaluationFunctionContext
from ropt.enums import EvaluationStatus

StatusCallback = Callable[[EvaluationFunctionContext, EvaluationStatus], None]
"""Called when the status of an evaluation changes.

Receives the context that is also passed to the evaluation function, and the
new status. It is called on the thread that started the optimization or
evaluation, so it does not need to be thread-safe.
"""


def make_status_reporter(callback: StatusCallback | None) -> StatusCallback:
    # Keeps the reported states of one batch in order: a state that is not
    # later than the last one reported for its row is dropped, so each row
    # receives at most one of the terminal states.
    reported: dict[int, int] = {}

    def _report(context: EvaluationFunctionContext, status: EvaluationStatus) -> None:
        if callback is None:
            return
        rank = min(status, EvaluationStatus.DONE)
        if rank <= reported.get(context.row_idx, 0):
            return
        reported[context.row_idx] = rank
        callback(context, status)

    return _report
