import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ropt._logging import get_logger
from ropt.components.executors import (
    Executor,
    HPCExecutor,
    MultiprocessingExecutor,
    Submission,
    ThreadingExecutor,
    WorkItem,
)

_logger = get_logger(__name__)


@dataclass(kw_only=True)
class _IndexedWorkItem(WorkItem):
    id: int


def _collect_results(
    submission: Submission,
    count: int,
    report: Callable[[Any], None] | None = None,
) -> list[Any]:
    results: list[Any] = [None] * count

    def _store(work_item: WorkItem) -> None:
        assert isinstance(work_item, _IndexedWorkItem)
        results[work_item.id] = work_item.result
        if report is not None:
            report(work_item.result)

    submission.collect(_store)
    return results


async def dispatch_tasks(  # ruff: ignore[too-many-arguments]
    functions: Sequence[Callable[[], Any]] | Mapping[str, Callable[[], Any]],
    executor: Literal["threading", "multiprocessing", "hpc"],
    *,
    report: Callable[[Any], None] | None = None,
    workers: int = 4,
    workdir: str = "./",
    cluster: str | None = None,
    queue: str | None = None,
    cores: int = 1,
) -> list[Any]:
    """Dispatch a list of functions to run in parallel.

    The dispatched functions will run either in threads, in a multiprocessing pool,
    or on an HPC executor. See
    [Dispatching arbitrary tasks](../workflows/parallel.md#dispatching-arbitrary-tasks)
    for usage and the current-working-directory caveat.

    Args:
        functions: The functions to run.
        executor:  The type of executor to run the functions.
        report:    Optional report function.
        workers:   The number of workers to run in parallel.
        workdir:   Working directory used by the HPC executor; a relative path
                   is resolved against the current directory.
        cluster:   The name of the HPC cluster to use.
        queue:     Optional queue to use on the cluster.
        cores:     Optional number of cores per task.

    Returns:
        A list of function results.

    Raises:
        ValueError: If `executor` has an invalid value.
    """
    if isinstance(functions, Mapping):
        work_items = [
            _IndexedWorkItem(function=function, id=idx, name=name)
            for idx, (name, function) in enumerate(functions.items())
        ]
    else:
        work_items = [
            _IndexedWorkItem(function=function, id=idx)
            for idx, function in enumerate(functions)
        ]
    executor_instance: HPCExecutor | ThreadingExecutor | MultiprocessingExecutor
    match executor:
        case "hpc":
            executor_instance = HPCExecutor(
                workdir=Path(workdir).resolve(),  # ruff: ignore[blocking-path-method-in-async-function]
                workers=workers,
                cluster=cluster,
                queue=queue,
                cores=cores,
            )
        case "threading":
            executor_instance = ThreadingExecutor(workers=workers)
        case "multiprocessing":
            executor_instance = MultiprocessingExecutor(workers=workers)
        case _:
            msg = f"Invalid executor: {executor}"
            raise ValueError(msg)
    assert isinstance(executor_instance, Executor)
    _logger.debug(
        "Dispatching %d work item(s) via %s executor (%d worker(s))",
        len(work_items),
        executor,
        workers,
    )
    submission = Submission(work_items)
    async with asyncio.TaskGroup() as tg:
        await executor_instance.start(tg)
        executor_instance.submit(submission)
        results = await asyncio.to_thread(
            _collect_results, submission, len(work_items), report
        )
        executor_instance.cancel()
    return results
