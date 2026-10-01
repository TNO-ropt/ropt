"""Offload arbitrary callables to a pool.

Offloading is defined by the relocation, so it needs workers and therefore a
pool: a single callable, or a sequence of callables that run concurrently and
may be entirely different functions. They must be picklable for a process or
job pool.

Which pool the work lands on is decided entirely by the pool it is called on:
offloading from inside an evaluation function dispatches to the pool that call
names, not to the one running the evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

from ropt.components.concurrency import AbortSignal
from ropt.components.executors import ExecutorFailure, WorkItem, WorkNotRun
from ropt.exceptions import AbortedError, ExecutionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.executors import Executor

    from ._session import Session

_T = TypeVar("_T")


def _offload(
    session: Session,
    executor: Executor,
    work: Callable[[], _T] | Sequence[Callable[[], _T]],
    *,
    keep_going: bool | None,
) -> _T | tuple[_T, ...]:
    signal = AbortSignal()
    session._register(  # ruff: ignore[private-member-access]
        signal,
        keep_going=session._resolve_keep_going(keep_going=keep_going),  # ruff: ignore[private-member-access]
    )
    try:
        if callable(work):
            return cast("_T", _run(executor, [work], signal)[0])
        functions = list(work)
        if not functions:
            return ()
        return tuple(_run(executor, functions, signal))
    except Exception:
        # Being cut off is not a failure, so it must not cut off anything else.
        if not signal.aborting:
            session._fail()  # ruff: ignore[private-member-access]
        raise
    finally:
        session._deregister(signal)  # ruff: ignore[private-member-access]


def _run(
    executor: Executor, functions: list[Callable[[], Any]], signal: AbortSignal
) -> list[Any]:
    # A sequence of offloaded callables is documented to run concurrently, so
    # they must not be bundled onto one worker.
    values = executor.run(
        [WorkItem(function=function) for function in functions],
        bundle_size=1,
        abort_signal=signal,
    )
    abandoned = any(isinstance(value, WorkNotRun) for value in values)
    # An abort that arrived too late to cost a call anything did not abort it.
    if abandoned and signal.aborting:
        raise AbortedError(signal.exit_reason)
    for value in values:
        if isinstance(value, (ExecutorFailure, WorkNotRun)):
            msg = f"An offloaded call could not be run: {value.message}"
            raise ExecutionError(msg)
    return values
