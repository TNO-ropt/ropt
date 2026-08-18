"""Offload arbitrary callables to the active executor.

`offload` runs a single callable, or a sequence of callables concurrently (which
may be entirely different functions), on the innermost open execution block
(`threads`/`processes`/`hpc`). It **requires** an executor: with no block open, or
from a handler in a `handlers` block (which runs on the session's event loop, or
on a dispatcher worker that has no block of its own), it raises. Use
`can_offload` to check first and call the callables directly when no executor is
available. The callables must be picklable for a process or HPC executor.

Because it targets the innermost open block, `offload` called from within an
evaluation function dispatches to a block that evaluation opens itself, not to
the enclosing optimizer's executor (an evaluation runs detached from it).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from ropt.components.executors import Submission, WorkItem
from ropt.exceptions import ExecutorFailure, WorkflowError

from ._session import current_executor

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.executors import Executor

_T = TypeVar("_T")


@dataclass(kw_only=True)
class _IndexedWorkItem(WorkItem):
    index: int


@overload
def offload(work: Callable[[], _T]) -> _T: ...


@overload
def offload(work: Sequence[Callable[[], _T]]) -> tuple[_T, ...]: ...


def offload(
    work: Callable[[], _T] | Sequence[Callable[[], _T]],
) -> _T | tuple[_T, ...]:
    """Offload one or more callables to the active executor.

    Runs `work` on the open execution block's executor
    (`threads`/`processes`/`hpc`). Pass a single zero-argument callable to run
    one call and return its result, or a sequence of callables to run them
    concurrently (they may be entirely different functions) and return a tuple
    of results in the order of `work`. Bind arguments with `functools.partial`.

    It raises a [`WorkflowError`][ropt.exceptions.WorkflowError] when no block is
    open, or when called from a handler in a `handlers` block; a handler passed
    to a single `optimize` call runs on the driving thread and can offload.
    Check [`can_offload`][ropt.simple.can_offload] first and call the callables
    directly when there is no executor. This holds for an empty sequence too, so
    that a call site is not silently accepted in a context where it cannot
    dispatch. The callables must be picklable for a process or HPC executor.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        work: A single zero-argument callable, or a sequence of them.

    Returns:
        The single result, or a tuple of results in the order of `work`.
    """
    executor = _require_executor()
    if callable(work):
        return cast("_T", _dispatch(executor, [work])[0])
    functions = list(work)
    if not functions:
        return ()
    return tuple(_dispatch(executor, functions))


def can_offload() -> bool:
    """Report whether `offload` would dispatch to an executor.

    Returns `True` when an execution block (`threads`/`processes`/`hpc`) is open
    and the caller can dispatch to it, and `False` otherwise (no block open, the
    block's executor has stopped, or called from a handler in a `handlers`
    block).

    Use it in code that may run with or without an execution block (for example
    a plugin, transform, or custom step) to fall back to a direct call instead
    of letting `offload` raise:

    ```python
    result = offload(work) if can_offload() else work()
    ```

    You do not need it when you know a block is open — call `offload` directly.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    Returns:
        `True` if `offload` would dispatch, `False` otherwise.
    """
    executor = current_executor()
    return (
        executor is not None and executor.is_running() and not executor.on_worker_loop()
    )


def _require_executor() -> Executor:
    executor = current_executor()
    if executor is None:
        # A threaded handler lands here even inside an open block: it runs on a
        # dispatcher worker, which carries no session to find the block through.
        msg = (
            "offload() found no executor to dispatch to here; open an execution "
            "block (threads/processes/hpc), or use can_offload() to run inline. "
            "A handler running in a thread always lands here, because the "
            "dispatcher worker it runs on carries no block of its own."
        )
        raise WorkflowError(msg)
    if executor.on_worker_loop():
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
    if isinstance(work_item.result, ExecutorFailure):
        raise work_item.result
    output[work_item.index] = work_item.result
