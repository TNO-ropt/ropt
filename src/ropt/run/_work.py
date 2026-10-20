"""Run arbitrary callables on a pool, or on threads of their own.

The work is a single callable, or a sequence of callables that run concurrently
and may be entirely different functions. They must be picklable for a process or
job pool.

Which pool the work lands on is decided entirely by the pool it is called on:
calling `run` from inside an evaluation function dispatches to the pool that call
names, not to the one running the evaluation. Calling `run` on a session runs the
work on threads started for that call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ropt.components.executors import ExecutorFailure, WorkItem, WorkNotRun
from ropt.exceptions import AbortedError, ExecutionError

from ._linked_abort_scope import linked_abort_scope

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.concurrency import AbortScope
    from ropt.components.executors import Executor

    from ._session import Session


def _run_work[T](
    session: Session,
    executor: Executor,
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
    executor: Executor, functions: list[Callable[[], Any]], scope: AbortScope
) -> list[Any]:
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
