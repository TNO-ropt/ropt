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

from ropt.components.executors import ExecutorFailure, WorkItem, WorkNotRun
from ropt.exceptions import ExecutionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.executors import Executor

    from ._session import Session

_T = TypeVar("_T")


def _offload(
    session: Session,
    executor: Executor,
    work: Callable[[], _T] | Sequence[Callable[[], _T]],
) -> _T | tuple[_T, ...]:
    try:
        if callable(work):
            return cast("_T", _run(executor, [work])[0])
        functions = list(work)
        if not functions:
            return ()
        return tuple(_run(executor, functions))
    except Exception:
        session._fail()  # ruff: ignore[private-member-access]
        raise


def _run(executor: Executor, functions: list[Callable[[], Any]]) -> list[Any]:
    # A sequence of offloaded callables is documented to run concurrently, so
    # they must not be bundled onto one worker.
    values = executor.run(
        [WorkItem(function=function) for function in functions], bundle_size=1
    )
    for value in values:
        if isinstance(value, (ExecutorFailure, WorkNotRun)):
            msg = f"An offloaded call could not be run: {value.message}"
            raise ExecutionError(msg)
    return values
