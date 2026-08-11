"""Submit tasks to an executor and collect their results."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from ropt.exceptions import ExecutorStopped

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any, NoReturn

    from .base import Executor, ResultsQueue, Task


def submit_and_collect(
    executor: Executor,
    producer: Coroutine[Any, Any, None],
    results_queue: ResultsQueue,
    expected: int,
    on_result: Callable[[Task], None],
) -> None:
    """Schedule a task producer on an executor and drain its results.

    The `producer` coroutine puts the tasks on the executor's task queue; it is
    scheduled on the executor's loop. Each finished task is drained from
    `results_queue` and passed to `on_result`, until `expected` tasks have been
    collected. The [`Task`][ropt.components.executors.Task] error contract is
    honored: a `None` sentinel or a stopped executor aborts with
    [`ExecutorStopped`][ropt.exceptions.ExecutorStopped], and a `BaseException`
    is re-raised unchanged.

    The executor must be running: its loop and task group must be set.

    Args:
        executor:      The running executor to dispatch to.
        producer:      The coroutine that submits the tasks.
        results_queue: The queue the tasks deliver their results on.
        expected:      The number of tasks to collect (one result each).
        on_result:     Callback invoked with each finished task.
    """
    assert executor.loop is not None
    assert executor.task_group is not None
    executor.loop.call_soon_threadsafe(executor.task_group.create_task, producer)
    received = 0
    while received < expected:
        while executor.is_running():
            try:
                item = results_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                _abort(results_queue)
            if isinstance(item, BaseException):
                raise item
            on_result(item)
            received += 1
            break
        if not executor.is_running():
            _abort(results_queue)


def _abort(results_queue: ResultsQueue) -> NoReturn:
    while True:
        try:
            item = results_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(item, BaseException):
            raise item
    raise ExecutorStopped
