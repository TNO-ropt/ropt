"""Offload arbitrary callables to a pool.

`offload` runs a single callable, or a sequence of callables concurrently (which
may be entirely different functions), on the pool it is given. Without a pool,
or with a serial pool, it runs them inline on the calling thread — so a call
site works the same whether or not the caller has a pool to offer, and needs no
guard. The callables must be picklable for a process or HPC pool.

Which pool the work lands on is decided entirely by the argument: `offload`
called from inside an evaluation function dispatches to the pool that evaluation
was handed, not to the one running the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from ropt.components.executors import Submission, WorkItem
from ropt.exceptions import ExecutorFailure, WorkflowError

from ._guards import check_pool

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.executors import Executor

    from ._pool import WorkerPool

_T = TypeVar("_T")


@dataclass(kw_only=True)
class _IndexedWorkItem(WorkItem):
    # Results come back as they finish, so each item carries the position its
    # result belongs in.
    index: int


@overload
def offload(work: Callable[[], _T], *, pool: WorkerPool | None = ...) -> _T: ...


@overload
def offload(
    work: Sequence[Callable[[], _T]], *, pool: WorkerPool | None = ...
) -> tuple[_T, ...]: ...


def offload(
    work: Callable[[], _T] | Sequence[Callable[[], _T]],
    *,
    pool: WorkerPool | None = None,
) -> _T | tuple[_T, ...]:
    """Offload one or more callables to a pool.

    Pass a single zero-argument callable to run one call and return its result,
    or a sequence of callables to run them concurrently (they may be entirely
    different functions) and return a tuple of results in the order of `work`.
    Bind arguments with `functools.partial`.

    Without a pool, or with a [`serial_pool`][ropt.simple.serial_pool], the
    callables run inline on the calling thread, one after another. Code that may
    or may not have a pool to hand therefore needs no fallback: pass whatever it
    has, including `None`. The callables must be picklable for a process or HPC
    pool.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    A handler in a shared group runs on the pool's own event loop and cannot
    wait on it; offloading from there raises a
    [`WorkflowError`][ropt.exceptions.WorkflowError]. So does a pool that is
    closed, or one carried into a worker process.

    Args:
        work: A single zero-argument callable, or a sequence of them.
        pool: The pool to dispatch to, or `None` to run inline. Work offloaded
              from inside an evaluation needs a *different* pool: the pool it is
              already running on refuses it.

    Returns:
        The single result, or a tuple of results in the order of `work`.
    """
    check_pool(pool)
    executor = _dispatchable(pool)
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
    return _dispatch(executor, functions)


def _dispatchable(pool: WorkerPool | None) -> Executor | None:
    executor = None if pool is None else pool.executor
    if executor is not None and executor.on_worker_loop():
        msg = (
            "offload() cannot be called from a result handler "
            "(it runs on the session's event loop)."
        )
        raise WorkflowError(msg)
    return executor


def _dispatch(executor: Executor, functions: list[Callable[[], Any]]) -> list[Any]:
    submission = Submission(
        [
            _IndexedWorkItem(function=function, index=index)
            for index, function in enumerate(functions)
        ]
    )
    output: list[Any] = [None] * len(functions)
    executor.submit(submission)
    submission.collect(partial(_store, output))
    return output


def _store(output: list[Any], work_item: WorkItem) -> None:
    assert isinstance(work_item, _IndexedWorkItem)
    # Unlike an evaluation, a lost work item has no NaN to fall back on: there
    # is a result to return, or there is nothing.
    if isinstance(work_item.result, ExecutorFailure):
        raise work_item.result
    output[work_item.index] = work_item.result
