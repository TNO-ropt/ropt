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

from ropt.components.concurrency import AbortSignal, parent_signal
from ropt.components.executors import ExecutorFailure, WorkItem, WorkNotRun
from ropt.enums import ExitCode
from ropt.exceptions import AbortedError, ExecutionError, ExecutorStopped

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.executors import Executor

    from ._session import Session


def _run_work[T](
    session: Session,
    executor: Executor,
    work: Callable[[], T] | Sequence[Callable[[], T]],
) -> T | tuple[T, ...]:
    parent = parent_signal()
    failure_aborts_session = parent is None
    signal = AbortSignal()
    session._register(signal)  # ruff: ignore[private-member-access]
    try:
        with signal.aborts_with(parent):
            if callable(work):
                return cast("T", _run(executor, [work], signal)[0])
            functions = list(work)
            if not functions:
                return ()
            return tuple(_run(executor, functions, signal))
    except ExecutorStopped:
        # `run` has no result object to carry a reason, so a shutdown is
        # reported the way an abort is rather than escaping.
        raise AbortedError(ExitCode.EXECUTOR_SHUT_DOWN) from None
    except Exception:
        # `signal.aborting` means this call was aborted rather than failing.
        if not signal.aborting:
            # Aborts the tasks started from the functions.
            signal.abort(ExitCode.ABORTED_ON_ERROR)
            if failure_aborts_session:
                session._fail()  # ruff: ignore[private-member-access]
        raise
    except BaseException:
        # Not a failure, but the work started from this call must not outlive
        # it.
        signal.abort(ExitCode.ABORTED)
        raise
    finally:
        session._deregister(signal)  # ruff: ignore[private-member-access]


def _run(
    executor: Executor, functions: list[Callable[[], Any]], signal: AbortSignal
) -> list[Any]:
    values = executor.run(
        [WorkItem(function=function) for function in functions],
        abort_signal=signal,
    )
    not_run = any(isinstance(value, WorkNotRun) for value in values)
    # Both conditions: an abort that arrived after every call had finished cost
    # none of them its result, and is not reported as one.
    if not_run and signal.aborting:
        raise AbortedError(signal.exit_code)
    for value in values:
        if isinstance(value, (ExecutorFailure, WorkNotRun)):
            msg = f"A callable given to `run` could not be run: {value.message}"
            raise ExecutionError(msg)
    return values
