import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ropt._logging import get_logger
from ropt.workflow.executors import (
    Executor,
    HPCExecutor,
    MultiprocessingExecutor,
    ResultsQueue,
    Task,
    ThreadingExecutor,
)

_logger = get_logger(__name__)


@dataclass(kw_only=True)
class _Task(Task):
    id: int


def _collect_results(
    results_queue: ResultsQueue,
    count: int,
    finished_event: asyncio.Event,
    results: dict[int, Any],
    *,
    report: Callable[[Any], None] | None = None,
) -> None:
    for _ in range(count):
        task = results_queue.get()
        if task is None:
            break
        assert isinstance(task, _Task)
        results[task.id] = task.result
        if report is not None and task.result is not None:
            report(task.result)
    finished_event.set()


async def dispatch_tasks(  # noqa: PLR0913
    functions: Sequence[Callable[[], None]] | Mapping[str, Callable[[], None]],
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
    or on an HPC executor.

    Args:
        functions: The functions to run.
        executor:  The type of executor to run the functions.
        report:    Optional report function.
        workers:   The number of workers to run in parallel.
        workdir:   Working directory used by the HPC executor.
        cluster:   The name of the HPC cluster to use.
        queue:     Optional queue to use on the cluster.
        cores:     Optional number of cores per task.

    Returns:
        A list of function results.

    Raises:
        ValueError: If `executor` has an invalid value.

    Note: Current working directory.
        The functions to run cannot rely on the current directory to be set
        consistently. Assume that the current directory is unknown and use
        absolute paths to read or write files.

        ALso, note that setting the current directory may not have the desired
        effect when using the `threading` executor, since changing it in one
        thread affects all threads. In case of the `multiprocessing` and `hpc`
        executors, the current directory can be changed safely if needed.
    """
    results: dict[int, Any] = {}
    results_queue = ResultsQueue()
    if isinstance(functions, Mapping):
        tasks = [
            _Task(function=function, results_queue=results_queue, id=idx, name=name)
            for idx, (name, function) in enumerate(functions.items())
        ]
    else:
        tasks = [
            _Task(function=function, results_queue=results_queue, id=idx)
            for idx, function in enumerate(functions)
        ]
    executor_instance: HPCExecutor | ThreadingExecutor | MultiprocessingExecutor
    match executor:
        case "hpc":
            executor_instance = HPCExecutor(
                workdir=workdir,
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
        "Dispatching %d task(s) via %s executor (%d worker(s))",
        len(tasks),
        executor,
        workers,
    )
    all_processed = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        await executor_instance.start(tg)
        tg.create_task(
            asyncio.to_thread(
                _collect_results,
                results_queue,
                len(tasks),
                all_processed,
                results,
                report=report,
            ),
        )
        for task in tasks:
            await executor_instance.task_queue.put(task)
        await all_processed.wait()
        executor_instance.cancel()
    return [results[idx] for idx in range(len(results))]
