"""Offload arbitrary callables to the active executor.

`offload` runs a single callable, or a sequence of callables concurrently (which
may be entirely different functions), on whatever execution block
(`threads`/`processes`/`hpc`) is open — exactly as evaluations are offloaded. It
**requires** an executor: with no block open (or when called from a result handler,
which runs on the session's event loop) it raises. Use `can_offload` to check
first and call the callables directly when no executor is available. The
callables must be picklable for a process or HPC executor.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from ropt.components.executors import ResultsQueue, Task
from ropt.components.executors._collect import submit_and_collect
from ropt.exceptions import ExecutorFailure, WorkflowError

from ._session import current_executor

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ropt.components.executors import Executor

_T = TypeVar("_T")


@dataclass(kw_only=True)
class _IndexedTask(Task):
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
    open or when called from a result handler (which runs on the event loop);
    check [`can_offload`][ropt.simple.can_offload] first and call the callables
    directly when there is no executor. The callables must be picklable for a
    process or HPC executor.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        work: A single zero-argument callable, or a sequence of them.

    Returns:
        The single result, or a tuple of results in the order of `work`.
    """
    if callable(work):
        executor = _require_executor()
        return cast("_T", _dispatch(executor, [work])[0])
    functions = list(work)
    if not functions:
        return ()
    executor = _require_executor()
    return tuple(_dispatch(executor, functions))


def can_offload() -> bool:
    """Report whether `offload` would dispatch to an executor.

    Returns `True` when an execution block (`threads`/`processes`/`hpc`) is open
    and the caller can dispatch to it, and `False` otherwise (no block open, or
    called from a result handler).

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
    if _on_event_loop():
        return False
    executor = current_executor()
    return executor is not None and executor.is_running()


def _require_executor() -> Executor:
    if _on_event_loop():
        msg = (
            "offload() cannot be called from a result handler "
            "(it runs on the session's event loop)."
        )
        raise WorkflowError(msg)
    executor = current_executor()
    if executor is None or not executor.is_running():
        msg = (
            "offload() found no executor to dispatch to here; open an execution "
            "block (threads/processes/hpc), or use can_offload() to run inline."
        )
        raise WorkflowError(msg)
    return executor


def _dispatch(executor: Executor, functions: list[Callable[[], Any]]) -> list[Any]:
    results_queue = ResultsQueue()
    tasks = [
        _IndexedTask(function=function, results_queue=results_queue, index=index)
        for index, function in enumerate(functions)
    ]
    output: list[Any] = [None] * len(tasks)
    submit_and_collect(
        executor,
        _put(executor, tasks, results_queue),
        results_queue,
        len(tasks),
        partial(_store, output),
    )
    return output


async def _put(
    executor: Executor, tasks: list[_IndexedTask], results_queue: ResultsQueue
) -> None:
    try:
        for task in tasks:
            if not executor.is_running():
                break
            await executor.task_queue.put(task)
    except Exception:
        results_queue.put(None)
        results_queue.close()
        raise


def _store(output: list[Any], task: Task) -> None:
    assert isinstance(task, _IndexedTask)
    if isinstance(task.result, ExecutorFailure):
        raise task.result
    output[task.index] = task.result


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
