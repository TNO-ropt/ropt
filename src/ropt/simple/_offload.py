"""Offload arbitrary callables to an executor.

`offload` runs a single callable, or a sequence of callables concurrently (which
may be entirely different functions), on the executor it is given. Without one,
it runs them inline on the calling thread — so a call site works the same
whether or not the caller has an executor to offer, and needs no guard. The
callables must be picklable for a process or job executor.

Which executor the work lands on is decided entirely by the argument: `offload`
called from inside an evaluation function dispatches to the executor that
evaluation was handed, not to the one running the evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from ropt.components.executors import ExecutorFailure, WorkItem
from ropt.exceptions import ExecutionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.executors import Executor

_T = TypeVar("_T")


@overload
def offload(work: Callable[[], _T], *, executor: Executor | None = ...) -> _T: ...


@overload
def offload(
    work: Sequence[Callable[[], _T]], *, executor: Executor | None = ...
) -> tuple[_T, ...]: ...


def offload(
    work: Callable[[], _T] | Sequence[Callable[[], _T]],
    *,
    executor: Executor | None = None,
) -> _T | tuple[_T, ...]:
    """Offload one or more callables to an executor.

    Pass a single zero-argument callable to run one call and return its result,
    or a sequence of callables to run them concurrently (they may be entirely
    different functions) and return a tuple of results in the order of `work`.
    Bind arguments with `functools.partial`.

    Without an executor the callables run inline on the calling thread, one
    after another. Code that may or may not have an executor to hand therefore
    needs no fallback: pass whatever it has, including `None`. The callables
    must be picklable for a process or job executor.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    A closed executor raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError]. A call that the machinery
    could not run, for instance because its worker process was killed, raises an
    [`ExecutionError`][ropt.exceptions.ExecutionError]. Work offloaded from
    inside an evaluation needs an executor with workers of its own: the one it
    is already running on refuses it.

    Args:
        work:     A single zero-argument callable, or a sequence of them.
        executor: The executor to dispatch to, or `None` to run inline.

    Returns:
        The single result, or a tuple of results in the order of `work`.
    """
    if callable(work):
        functions = [work]
        results = _run(executor, functions)
        return cast("_T", results[0])
    functions = list(work)
    if not functions:
        return ()
    return tuple(_run(executor, functions))


def _run(executor: Executor | None, functions: list[Callable[[], Any]]) -> list[Any]:
    if executor is None:
        return [function() for function in functions]
    # A sequence of offloaded callables is documented to run concurrently, so
    # they must not be bundled onto one worker.
    values = executor.run(
        [WorkItem(function=function) for function in functions], bundle_size=1
    )
    for value in values:
        if isinstance(value, ExecutorFailure):
            msg = f"An offloaded call could not be run: {value.message}"
            raise ExecutionError(msg)
    return values
