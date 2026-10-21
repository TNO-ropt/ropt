"""Run arbitrary callables with a pool, or one after another in-process.

The work is a single callable, or a sequence of callables that may be entirely
different functions. With a pool they run concurrently on its workers, and must be
picklable for a process or job pool. Without a pool they run in turn on the
calling thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ropt._executors import ExecutorFailure, WorkItem, WorkNotRun
from ropt.exceptions import AbortedError, ExecutionError

from ._linked_abort_scope import linked_abort_scope

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt._concurrency import AbortScope
    from ropt._executors import Executor

    from ._session import Session


def _run_work[T](
    session: Session | None,
    executor: Executor | None,
    work: Callable[[], T] | Sequence[Callable[[], T]],
) -> T | tuple[T, ...]:
    with linked_abort_scope(session) as scope:
        if callable(work):
            return cast("T", _run(executor, [work], scope)[0])
        functions = list(work)
        if not functions:
            return ()
        return tuple(_run(executor, functions, scope))


def _run(
    executor: Executor | None, functions: list[Callable[[], Any]], scope: AbortScope
) -> list[Any]:
    if executor is None:
        return _run_in_turn(functions, scope)
    values = executor.run(
        [WorkItem(function=function) for function in functions],
        abort_scope=scope,
    )
    not_run = any(isinstance(value, WorkNotRun) for value in values)
    # Both conditions: an abort that arrived after every call had finished cost
    # none of them its result, and is not reported as one.
    if not_run and scope.aborting:
        raise AbortedError(scope.exit_code)
    for value in values:
        if isinstance(value, (ExecutorFailure, WorkNotRun)):
            msg = f"A callable given to `run` could not be run: {value.message}"
            raise ExecutionError(msg)
    return values


def _run_in_turn(functions: list[Callable[[], Any]], scope: AbortScope) -> list[Any]:
    values: list[Any] = []
    with scope.as_current():
        for function in functions:
            _raise_if_aborting(scope)
            values.append(function())
    # A callable that saw the abort may have returned early, with a value that
    # does not stand for its result.
    _raise_if_aborting(scope)
    return values


def _raise_if_aborting(scope: AbortScope) -> None:
    if scope.aborting:
        raise AbortedError(scope.exit_code)
