# test_hpc_job_submitted_during_stop_is_cancelled is sensitive to what runs
# before it: it has failed under a mutation it does not detect, while passing
# 3/3 in isolation. Re-run it alone before believing a failure here.

from __future__ import annotations

import asyncio
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from ropt.components.compute_steps import OptimizationStep
from ropt.components.evaluators import (
    EvaluationFunctionCallback,
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    ParallelEvaluator,
)
from ropt.components.evaluators._parallel_evaluator import _handle_result
from ropt.components.event_handlers import ResultsHandler
from ropt.components.executors import (
    HPCExecutor,
    MultiprocessingExecutor,
    Submission,
    ThreadingExecutor,
    WorkItem,
)
from ropt.components.executors._multiprocessing_executor import (
    _HAVE_CLOUDPICKLE,
    _run_cloudpickled,
)
from ropt.context import EnOptContext
from ropt.evaluation import EvaluationBatchContext
from ropt.exceptions import (
    ExecutionError,
    ExecutorFailure,
    ExecutorStopped,
    TransferError,
    WorkflowError,
)
from ropt.workflow import dispatch_tasks

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from ropt.components.executors import Executor
    from ropt.components.executors.base import ExecutorBase
    from ropt.results import FunctionResults

try:
    import cloudpickle
    import pandas as pd
    import pysqa  # ruff: ignore[unused-import]

    from ropt.components.executors.__main__ import run_task

    _TEST_HPC = True
except ImportError:
    _TEST_HPC = False


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(5)]


def _collect(submission: Submission) -> list[Any]:
    """Drain a submission.

    Returns:
        The results, in delivery order.
    """
    collected: list[Any] = []
    submission.collect(lambda work_item: collected.append(work_item.result))
    return collected


def _collect_in_thread(
    submission: Submission, collected: list[Any], done: Callable[[], None]
) -> None:
    submission.collect(lambda work_item: collected.append(work_item.result))
    done()


def _finished_event() -> tuple[asyncio.Event, Callable[[], None]]:
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _done() -> None:
        loop.call_soon_threadsafe(finished.set)

    return finished, _done


def _function(input_value: int, *, raise_error: bool = False) -> int:
    if raise_error:
        msg = f"Test error in function {input_value}"
        raise ValueError(msg)
    return input_value + 1


def _raise_unpicklable_error(_input: int) -> int:
    raise ValueError(threading.Lock())  # a lock cannot be (cloud)pickled


def _call(function: Callable[[], Any]) -> Any:
    return function()


def _exit_task() -> int:
    sys.exit(3)


def _construct_handler_in_worker(_: int) -> str:
    return type(ResultsHandler()).__name__


def _worker_pid(_: int) -> int:
    return os.getpid()


def _wait_at_barrier(barrier: threading.Barrier, value: int) -> int:
    barrier.wait(timeout=4.0)
    return value


def _blocked_work(started: threading.Event, release: threading.Event) -> int:
    started.set()
    release.wait(timeout=5.0)
    return 0


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_ok(
    executor_name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    submission = Submission(
        [WorkItem(function=_function, args=(idx,)) for idx in range(2)]
    )
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert set(collected) == {1, 2}
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


async def test_threading_executor_exceeds_the_shared_default_pool() -> None:
    # Shrink asyncio's shared default executor (the pool the old code dispatched
    # work through) and configure more workers: all work items reach the barrier
    # at once only if the executor uses its own pool of that size.
    shared_pool_size = 2
    workers = shared_pool_size + 2
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=shared_pool_size)
    )
    barrier = threading.Barrier(workers)
    submission = Submission(
        [
            WorkItem(function=_wait_at_barrier, args=(barrier, idx))
            for idx in range(workers)
        ]
    )
    executor = ThreadingExecutor(workers=workers)
    collected: list[Any] = []
    finished, done = _finished_event()
    consumer = threading.Thread(
        target=_collect_in_thread, args=(submission, collected, done), daemon=True
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        consumer.start()
        executor.submit(submission)
        await finished.wait()
        executor.cancel()
    assert set(collected) == set(range(workers))


async def test_threading_executor_delivers_results_without_the_shared_default_pool() -> (
    None
):
    # Occupy asyncio's shared default executor completely. Delivering results
    # through it (the old behavior) could then hand over nothing at all.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    release = threading.Event()
    occupied = loop.run_in_executor(None, release.wait)

    count = 3
    submission = Submission(
        [WorkItem(function=_function, args=(idx,)) for idx in range(count)]
    )
    collected: list[Any] = []
    finished, done = _finished_event()
    # A raw thread: the shared pool asyncio.to_thread would use is taken.
    consumer = threading.Thread(
        target=_collect_in_thread, args=(submission, collected, done), daemon=True
    )
    executor = ThreadingExecutor(workers=2)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        consumer.start()
        executor.submit(submission)
        await finished.wait()
        executor.cancel()
    release.set()
    await occupied
    assert sorted(collected) == [1, 2, 3]


async def test_work_in_flight_aborted_on_stop() -> None:
    started = threading.Event()
    release = threading.Event()
    submission = Submission([WorkItem(function=_blocked_work, args=(started, release))])
    executor = ThreadingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        await asyncio.to_thread(started.wait)
        executor.cancel()
    release.set()
    with pytest.raises(ExecutorStopped):
        _collect(submission)


async def test_stopping_aborts_queued_submission() -> None:
    # One worker, blocked on the first work item, so the rest of the submission
    # is still sitting on the work queue when the executor stops.
    started = threading.Event()
    release = threading.Event()
    submission = Submission(
        [WorkItem(function=_blocked_work, args=(started, release)) for _ in range(5)]
    )
    executor = ThreadingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        await asyncio.to_thread(started.wait)
        executor.cancel()
    release.set()
    with pytest.raises(ExecutorStopped):
        _collect(submission)


async def test_submitting_to_stopped_executor_aborts() -> None:  # ruff: ignore[unused-async]
    # The caller is released by the executor rather than left waiting for
    # results that can never arrive.
    executor = ThreadingExecutor(workers=1)
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor.submit(submission)
    with pytest.raises(ExecutorStopped):
        _collect(submission)


@pytest.mark.slow
@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_executor_polls_without_the_shared_default_pool(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Occupy asyncio's shared default executor completely: a poll loop that
    # borrows a thread from it (the old behavior) never gets to run.
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    release = threading.Event()
    occupied = loop.run_in_executor(None, release.wait)

    count = 2
    submission = Submission(
        [WorkItem(function=_function, args=(idx,)) for idx in range(count)]
    )
    collected: list[Any] = []
    finished, done = _finished_event()
    consumer = threading.Thread(
        target=_collect_in_thread, args=(submission, collected, done), daemon=True
    )
    executor = HPCExecutor(workdir=tmp_path, workers=count, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        consumer.start()
        executor.submit(submission)
        await finished.wait()
        executor.cancel()
    release.set()
    await occupied
    assert sorted(collected) == [1, 2]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_scheduler_query_fails_after_retry_limit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # A scheduler that cannot be queried must not look like "nothing finished
    # yet": without a bound on the failures the caller waits forever.
    class _UnreachableScheduler(MockedHPCAdapter):
        def get_status_of_my_jobs(self) -> pd.DataFrame:  # ruff: ignore[no-self-use]
            msg = "squeue: error: Unable to contact slurm controller"
            raise RuntimeError(msg)

    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _UnreachableScheduler(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=2, template=""
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(collected) == 1
    assert isinstance(collected[0], ExecutorFailure)
    assert "could not be queried" in str(collected[0])


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_missing_output_file_fails_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # A job that dies before writing its result: the scheduler reports it gone,
    # but there is nothing to read, so waiting forever is not an option.
    class _VanishingJob(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

        def get_status_of_my_jobs(self) -> pd.DataFrame:  # ruff: ignore[no-self-use]
            return pd.DataFrame([], columns=["jobid"])

    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _VanishingJob(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=2, template=""
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(collected) == 1
    assert isinstance(collected[0], ExecutorFailure)
    assert "never appeared" in str(collected[0])


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_unreadable_output_file_fails_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # A result read while it is still being written: retrying is right, but it
    # has to give up eventually rather than retry for ever.
    class _CorruptResult(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            (self._path / f"{job_name}.out").write_bytes(b"half a pickle")
            return self._job_id

        def get_status_of_my_jobs(self) -> pd.DataFrame:  # ruff: ignore[no-self-use]
            return pd.DataFrame([], columns=["jobid"])

    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _CorruptResult(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=2, template=""
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(collected) == 1
    assert isinstance(collected[0], ExecutorFailure)
    assert "No valid result" in str(collected[0])


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_outstanding_work_aborted_on_stop(
    tmp_path: Path, monkeypatch: Any
) -> None:
    submitted = threading.Event()

    class _StuckAdapter(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            # Never run the work, so the job stays outstanding when we stop.
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            submitted.set()
            return self._job_id

    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _StuckAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        await asyncio.to_thread(submitted.wait)
        executor.cancel()
    with pytest.raises(ExecutorStopped):
        _collect(submission)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_stopping_hpc_executor_cancels_jobs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    submitted = threading.Event()
    cancelled = threading.Event()

    class _StuckAdapter(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            submitted.set()
            return self._job_id

        def delete_job(self, process_id: int) -> str:
            deleted = super().delete_job(process_id)
            cancelled.set()
            return deleted

    adapter = _StuckAdapter(tmp_path)
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: adapter,  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission([WorkItem(function=_function, args=(0,), name="job1")])
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        await asyncio.to_thread(submitted.wait)
        executor.cancel()
    assert await asyncio.to_thread(cancelled.wait, 5)
    assert adapter.deleted == [1]
    assert not await asyncio.to_thread(lambda: list(tmp_path.glob("job1.*")))


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_job_submitted_during_stop_is_cancelled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    submitting = threading.Event()
    stopped = threading.Event()
    cancelled = threading.Event()

    class _SlowAdapter(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            submitting.set()
            stopped.wait(5)
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

        def delete_job(self, process_id: int) -> str:
            deleted = super().delete_job(process_id)
            cancelled.set()
            return deleted

    adapter = _SlowAdapter(tmp_path)
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: adapter,  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission([WorkItem(function=_function, args=(0,), name="job1")])
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        await asyncio.to_thread(submitting.wait)
        executor.cancel()
    stopped.set()
    assert await asyncio.to_thread(cancelled.wait, 5)
    assert adapter.deleted == [1]
    assert not await asyncio.to_thread(lambda: list(tmp_path.glob("job1.*")))


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_idle_hpc_executor_does_not_query_scheduler(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class _CountingScheduler(MockedHPCAdapter):
        queries = 0

        def get_status_of_my_jobs(self) -> pd.DataFrame:
            type(self).queries += 1
            return super().get_status_of_my_jobs()

    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _CountingScheduler(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        for _ in range(20):  # give the poll loop every chance to spin
            await asyncio.sleep(0)
        assert _CountingScheduler.queries == 0
        submission = Submission([WorkItem(function=_function, args=(0,))])
        executor.submit(submission)
        assert await asyncio.to_thread(_collect, submission) == [1]
        assert _CountingScheduler.queries > 0
        executor.cancel()


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_poll_loop_honours_interval_when_busy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=3600, template="")
    work_item = WorkItem(function=_function, args=(0,))
    submission = Submission([work_item])
    executor._items["busy"] = (submission, work_item)  # ruff: ignore[private-member-access]
    executor._work_queue.put_nowait((submission, work_item))  # ruff: ignore[private-member-access]

    busy = asyncio.create_task(executor._wait_for_work())  # ruff: ignore[private-member-access]
    for _ in range(10):
        await asyncio.sleep(0)
    assert not busy.done()
    busy.cancel()

    executor._items.clear()  # ruff: ignore[private-member-access]
    ready = asyncio.create_task(executor._wait_for_work())  # ruff: ignore[private-member-access]
    for _ in range(10):
        await asyncio.sleep(0)
    assert ready.done()


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_queued_hpc_work_resumes_on_free_worker(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission(
        [WorkItem(function=_function, args=(idx,)) for idx in range(4)]
    )
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0.01, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert sorted(collected) == [1, 2, 3, 4]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_executor_refuses_to_overwrite_existing_work_item_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    (tmp_path / "job1.out").touch()  # a stale file, e.g. from another executor
    work_item = WorkItem(function=_function, args=(0,), name="job1")
    await asyncio.sleep(0)  # this module runs tests on the event loop
    with pytest.raises(ExecutionError, match="already exist"):
        executor._submit("job1", work_item)  # ruff: ignore[private-member-access]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_failing_submission_fails_own_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    (tmp_path / "job1.out").touch()  # a stale file blocks submission of job1
    blocked = Submission([WorkItem(function=_function, args=(0,), name="job1")])
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(blocked)
        with pytest.raises(ExecutionError, match="already exist"):
            await asyncio.to_thread(_collect, blocked)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        accepted = Submission([WorkItem(function=_function, args=(1,), name="job2")])
        executor.submit(accepted)
        assert await asyncio.to_thread(_collect, accepted) == [2]
        executor.cancel()


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_rejected_hpc_submission_leaves_no_input_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # The input file is written before the job is handed over, so a scheduler
    # that rejects it would otherwise block a retry under the same name.
    class _RejectingScheduler(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[no-self-use, unused-method-argument]
            msg = "sbatch: error: Batch job submission failed"
            raise RuntimeError(msg)

    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _RejectingScheduler(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission([WorkItem(function=_function, args=(0,), name="job1")])
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(RuntimeError, match="submission failed"):
            await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert not await asyncio.to_thread(lambda: list(tmp_path.glob("job1.*")))


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_duplicate_name_fails_own_submission(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # A name clash must abort the offending submission rather than tear down
    # the session, and must not leave its caller waiting.
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission(
        [WorkItem(function=_function, args=(idx,), name="same") for idx in range(2)]
    )
    executor = HPCExecutor(workdir=tmp_path, workers=2, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(Exception, match="already in use"):
            await asyncio.to_thread(_collect, submission)
        executor.cancel()


@pytest.mark.skipif(not _HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
async def test_cloudpickled_worker_records_the_worker_traceback_as_a_note() -> None:
    payload = cloudpickle.dumps((partial(_function, 0, raise_error=True), (), {}))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    ok, blob = _run_cloudpickled(payload)
    assert not ok
    exc = cloudpickle.loads(blob)
    assert isinstance(exc, ValueError)
    assert any("Traceback" in note for note in exc.__notes__)


@pytest.mark.skipif(not _HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
async def test_cloudpickled_worker_wraps_an_unpicklable_exception() -> None:
    payload = cloudpickle.dumps((_raise_unpicklable_error, (0,), {}))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    ok, blob = _run_cloudpickled(payload)
    assert not ok
    exc = cloudpickle.loads(blob)
    assert isinstance(exc, RuntimeError)
    assert any("Traceback" in note for note in exc.__notes__)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_job_exit_writes_output_file(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(cloudpickle.dumps((_exit_task, (), {})))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    assert run_task(str(input_file), str(output_file)) == 1
    assert isinstance(cloudpickle.loads(output_file.read_bytes()), SystemExit)


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_worker_may_construct_workflow_objects() -> None:
    submission = Submission(
        [WorkItem(function=_construct_handler_in_worker, args=(0,))]
    )
    executor = MultiprocessingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert collected == ["ResultsHandler"]


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_max_tasks_per_child_restarts_worker() -> None:
    submission = Submission(
        [WorkItem(function=_worker_pid, args=(index,)) for index in range(3)]
    )
    executor = MultiprocessingExecutor(workers=1, max_tasks_per_child=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(set(collected)) == 3


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_error(
    executor_name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    submission = Submission(
        [
            WorkItem(function=_function, args=(idx,), kwargs={"raise_error": True})
            for idx in range(2)
        ]
    )
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        executor.submit(submission)
        # A user-code exception is re-raised in the caller as the original
        # exception, and does not tear the executor down.
        with pytest.raises(ValueError, match="Test error in function") as excinfo:
            await asyncio.to_thread(_collect, submission)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        expects_notes = executor_name == "hpc" or (
            executor_name == "multiprocessing" and _HAVE_CLOUDPICKLE
        )
        if expects_notes:
            notes = getattr(excinfo.value, "__notes__", [])
            assert any("Test error in function" in note for note in notes)
            assert any("Traceback" in note for note in notes)
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "max_functions": 8,
        },
        "backend": {
            "convergence_tolerance": 1e-2,
        },
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.001,
        },
        "gradient": {
            "number_of_perturbations": 3,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def _opt_function(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    test_functions: Any,
    *,
    raise_error: bool = False,
) -> EvaluationFunctionResult:
    if raise_error:
        msg = "Test error in function"
        raise ValueError(msg)
    return EvaluationFunctionResult(
        objectives=np.fromiter(
            (func(variables, context) for func in test_functions), dtype=np.float64
        )
    )


def _opt_workflow(
    executor: Executor,
    config: dict[str, Any],
    test_function: EvaluationFunctionCallback,
) -> FunctionResults | None:
    evaluator = ParallelEvaluator(function=test_function, executor=executor)
    result_handler = ResultsHandler()
    step = OptimizationStep(evaluator=evaluator)
    step.add_event_handler(result_handler)
    step.run(variables=initial_values, context=EnOptContext.model_validate(config))
    return cast("FunctionResults | None", result_handler["results"])


if _TEST_HPC:

    class MockedHPCAdapter:
        def __init__(self, path: Path) -> None:
            self._path = path
            self._jobs: dict[int, str] = {}
            self._job_id = 0
            self.deleted: list[int] = []

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            *_, input_file, output_file = command.split()
            threading.Thread(
                target=run_task, args=(input_file, output_file), daemon=True
            ).start()
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

        def delete_job(self, process_id: int) -> str:
            self.deleted.append(process_id)
            self._jobs.pop(process_id, None)
            return ""

        def get_status_of_my_jobs(self) -> pd.DataFrame:
            running = [
                job_id
                for job_id, job_name in self._jobs.items()
                if not (self._path / f"{job_name}.out").exists()
            ]
            self._jobs = {job_id: self._jobs[job_id] for job_id in running}
            return pd.DataFrame(list(self._jobs.keys()), columns=["jobid"])


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_evaluator_ok(
    config: dict[str, Any],
    eval_func: Any,
    executor_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        results = await asyncio.to_thread(
            _opt_workflow,
            executor,
            config,
            eval_func(),
        )
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]

    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_evaluator_error(
    config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    executor_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]
    with pytest.raises(ExceptionGroup) as excinfo:  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            assert executor._running.is_set()  # ruff: ignore[private-member-access]
            await asyncio.to_thread(
                _opt_workflow,
                executor,
                config,
                partial(_opt_function, test_functions=test_functions, raise_error=True),
            )
            executor.cancel()
    # The user-code error surfaces as the original exception (not an exit code),
    # wrapped by the consumer's task group when it leaves the block.
    matched, _ = excinfo.value.split(ValueError)
    assert matched is not None
    assert all("Test error in function" in str(err) for err in matched.exceptions)
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
    ],
)
async def test_executor_survives_user_code_error_and_is_reusable(
    config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
    eval_func: Any,
    executor_name: str,
) -> None:
    match executor_name:
        case "threading":
            executor: ExecutorBase = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        # A user-code error aborts only its own evaluation; the executor keeps
        # running, so a caught error does not prevent a subsequent reuse.
        with pytest.raises(ValueError, match="Test error in function"):
            await asyncio.to_thread(
                _opt_workflow,
                executor,
                config,
                partial(_opt_function, test_functions=test_functions, raise_error=True),
            )
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        results = await asyncio.to_thread(_opt_workflow, executor, config, eval_func())
        assert results is not None
        assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


async def test_error_escaping_the_body_closes_the_executor(
    config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
) -> None:
    # No explicit executor.cancel(): an error escaping the block must still
    # close the executor through the task group's teardown.
    executor = ThreadingExecutor(workers=2)
    with pytest.raises(ExceptionGroup):  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            await asyncio.to_thread(
                _opt_workflow,
                executor,
                config,
                partial(_opt_function, test_functions=test_functions, raise_error=True),
            )
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_executor_evaluator_two_optimizations(
    config: dict[str, Any],
    eval_func: Any,
    executor_name: str,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        results_list = await asyncio.gather(
            *(
                asyncio.to_thread(
                    _opt_workflow,
                    executor,
                    config,
                    eval_func(),
                )
                for _ in range(2)
            )
        )
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]

    assert len(results_list) == 2
    for results in results_list:
        assert results is not None
        assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize("bundle_size", [1, 2, 4, 0])
@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
    ],
)
async def test_groups_work_items(
    config: dict[str, Any],
    eval_func: Any,
    executor_name: str,
    bundle_size: int,
) -> None:
    match executor_name:
        case "threading":
            executor: Executor = ThreadingExecutor(workers=2)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=2)

    bundle_sizes: list[int] = []
    original_submit = executor.submit

    def _counting_submit(submission: Submission) -> None:
        bundle_sizes.extend(
            len(work_item.args[1]) for work_item in submission.work_items
        )
        original_submit(submission)

    executor.submit = _counting_submit  # type: ignore[method-assign]

    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        evaluator = ParallelEvaluator(
            function=eval_func(),
            executor=executor,
            bundle_size=bundle_size,
        )
        result_handler = ResultsHandler()
        step = OptimizationStep(evaluator=evaluator)
        step.add_event_handler(result_handler)
        await asyncio.to_thread(
            step.run,
            variables=initial_values,
            context=EnOptContext.model_validate(config),
        )
        executor.cancel()

    results = result_handler["results"]
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert bundle_sizes, "No work items were submitted"
    expected_max = max(bundle_sizes) if bundle_size == 0 else bundle_size
    for size in bundle_sizes:
        assert 1 <= size <= expected_max


async def test_invalid_bundle_size() -> None:  # ruff: ignore[unused-async]
    executor = ThreadingExecutor(workers=1)
    with pytest.raises(ValueError, match="bundle_size"):
        ParallelEvaluator(
            function=lambda variables, context: EvaluationFunctionResult(  # ruff: ignore[unused-lambda-argument]
                objectives=0.0
            ),
            executor=executor,
            bundle_size=-1,
        )


async def test_failing_submission_reraises() -> None:  # ruff: ignore[unused-async]
    submission = Submission([WorkItem(function=_function, args=(0,))])
    error = ValueError("Test error in function")
    submission.fail(error)
    with pytest.raises(ValueError, match="Test error in function") as excinfo:
        _collect(submission)
    assert excinfo.value is error


async def test_submitting_to_closed_loop_aborts() -> None:  # ruff: ignore[unused-async]
    # The executor still looks running, but its loop is gone, so handing the
    # submission over cannot succeed and the caller must not be left waiting.
    executor = ThreadingExecutor(workers=1)
    loop = asyncio.new_event_loop()
    loop.close()
    executor._loop = loop  # ruff: ignore[private-member-access]
    executor._running.set()  # ruff: ignore[private-member-access]
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor.submit(submission)
    with pytest.raises(ExecutorStopped):
        _collect(submission)


async def test_cancelling_unstarted_executor() -> None:  # ruff: ignore[unused-async]
    executor = ThreadingExecutor(workers=1)
    executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_relative_workdir_rejected(tmp_path: Path) -> None:  # ruff: ignore[unused-async, unused-function-argument]
    # The workdir is shared with the cluster nodes, which do not necessarily
    # share this process's working directory.
    with pytest.raises(ExecutionError, match="must be an absolute path"):
        HPCExecutor(workdir="relative/path", template="")


async def test_broken_worker_pool_reported_at_startup(
    monkeypatch: Any,
) -> None:
    # The same failure the unguarded-main test triggers out of process, checked
    # here without paying for real subprocesses.
    class _BrokenPool:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

        @staticmethod
        def submit(*_args: Any, **_kwargs: Any) -> None:
            raise BrokenProcessPool

        def shutdown(self, *_args: Any, **_kwargs: Any) -> None: ...

    monkeypatch.setattr(
        "ropt.components.executors._multiprocessing_executor.ProcessPoolExecutor",
        _BrokenPool,
    )
    executor = MultiprocessingExecutor(workers=1)
    with pytest.raises(ExceptionGroup) as excinfo:
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
    matched, _ = excinfo.value.split(ExecutionError)
    assert matched is not None
    assert "guard the program entry point" in str(matched.exceptions[0])


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_missing_workdir_rejected(tmp_path: Path) -> None:  # ruff: ignore[unused-async]
    with pytest.raises(ExecutionError, match="does not exist"):
        HPCExecutor(workdir=tmp_path / "nowhere", template="")


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_unconfigured_hpc_executor_rejected(  # ruff: ignore[unused-async]
    tmp_path: Path, monkeypatch: Any
) -> None:
    # No template and no usable pysqa configuration: there is nothing to submit
    # jobs with, so this must fail at construction rather than at submit time.
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor._get_config_path",
        lambda config_path: None,  # ruff: ignore[unused-lambda-argument]
    )
    with pytest.raises(ExecutionError, match="not configured"):
        HPCExecutor(workdir=tmp_path)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"workers": 0}, "workers must be at least one"),
        ({"workers": -1}, "workers must be at least one"),
        ({"interval": -1.0}, "interval must not be negative"),
        ({"retries": -1}, "retries must not be negative"),
    ],
)
async def test_hpc_out_of_range_setting_rejected(  # ruff: ignore[unused-async]
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ExecutionError, match=match):
        HPCExecutor(workdir=tmp_path, template="", **kwargs)


async def test_aborting_submission_raises_executor_stopped() -> None:  # ruff: ignore[unused-async]
    submission = Submission([WorkItem(function=_function, args=(0,))])
    submission.abort()
    with pytest.raises(ExecutorStopped) as excinfo:
        _collect(submission)
    assert excinfo.value.__cause__ is None


async def test_queued_exception_preferred_over_abort() -> None:  # ruff: ignore[unused-async]
    submission = Submission(
        [WorkItem(function=_function, args=(idx,)) for idx in range(2)]
    )
    error = TransferError("Workflow objects cannot be used in a worker process: X.")
    submission._results.put(None)  # ruff: ignore[private-member-access]
    submission._results.put(error)  # ruff: ignore[private-member-access]
    with pytest.raises(TransferError):
        _collect(submission)


async def test_finished_submission_delivers_nothing_more() -> None:  # ruff: ignore[unused-async]
    work_item = WorkItem(function=_function, args=(0,))
    submission = Submission([work_item])
    submission.abort()
    submission.deliver(work_item, 1)
    with pytest.raises(ExecutorStopped):
        _collect(submission)


async def test_empty_submission_not_retained() -> None:  # ruff: ignore[unused-async]
    executor = ThreadingExecutor(workers=1)
    executor._running.set()  # ruff: ignore[private-member-access]
    for _ in range(100):
        executor._accept(Submission([]))  # ruff: ignore[private-member-access]
    assert executor._submissions == set()  # ruff: ignore[private-member-access]
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor._accept(submission)  # ruff: ignore[private-member-access]
    assert executor._submissions == {submission}  # ruff: ignore[private-member-access]


async def test_accepting_submission_twice_queues_work_once() -> None:  # ruff: ignore[unused-async]
    submission = Submission(
        [WorkItem(function=_function, args=(idx,)) for idx in range(3)]
    )
    executor = ThreadingExecutor(workers=1)
    executor._running.set()  # ruff: ignore[private-member-access]
    executor._accept(submission)  # ruff: ignore[private-member-access]
    executor._accept(submission)  # ruff: ignore[private-member-access]
    assert executor._work_queue.qsize() == 3  # ruff: ignore[private-member-access]


async def test_submission_after_stopping_is_aborted() -> None:
    # submit() hands over on the loop, where stopping is decided; this is the
    # guard that makes "every submission settles" hold when the two race.
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = ThreadingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.cancel()
        # Resumes after _wait_for_cancel has run, so stopping is fully decided.
        await executor._stop_event.wait()  # ruff: ignore[private-member-access]
        executor._accept(submission)  # ruff: ignore[private-member-access]
    assert executor._work_queue.empty()  # ruff: ignore[private-member-access]
    with pytest.raises(ExecutorStopped):
        _collect(submission)


async def test_cancelling_from_another_thread() -> None:
    executor = ThreadingExecutor(workers=1)
    loop_is_idle = threading.Event()

    def _cancel_once_idle() -> None:
        loop_is_idle.wait(timeout=5)
        executor.cancel()

    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        threading.Thread(target=_cancel_once_idle, daemon=True).start()
        asyncio.get_running_loop().call_soon(loop_is_idle.set)
        await executor._stop_event.wait()  # ruff: ignore[private-member-access]
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


async def test_raising_callback_ends_submission() -> None:  # ruff: ignore[unused-async]
    work_items = [WorkItem(function=_function, args=(idx,)) for idx in range(3)]
    submission = Submission(work_items)
    submission.deliver(work_items[0], 1)

    def _reject(work_item: WorkItem) -> None:  # ruff: ignore[unused-function-argument]
        msg = "caller gave up"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="caller gave up"):
        submission.collect(_reject)
    assert submission.is_finished


def _record(executed: list[int], value: int) -> int:
    executed.append(value)
    return value


def _give_up(work_item: WorkItem) -> None:  # ruff: ignore[unused-function-argument]
    msg = "caller gave up"
    raise ValueError(msg)


def _returns_none() -> None:
    return None


async def test_dispatch_tasks_results_in_order() -> None:
    functions = [partial(_function, idx) for idx in range(4)]
    assert await dispatch_tasks(functions, "threading", workers=2) == [1, 2, 3, 4]


async def test_dispatch_tasks_named_functions() -> None:
    functions = {f"job{idx}": partial(_function, idx) for idx in range(3)}
    assert await dispatch_tasks(functions, "threading", workers=2) == [1, 2, 3]


async def test_dispatch_tasks_reports_every_result() -> None:
    reported: list[Any] = []
    functions = [partial(_function, idx) for idx in range(3)]
    results = await dispatch_tasks(
        functions, "threading", workers=2, report=reported.append
    )
    assert results == [1, 2, 3]
    assert sorted(reported) == [1, 2, 3]


async def test_dispatch_tasks_keeps_none_result() -> None:
    # None is a legitimate result, not a marker for "nothing was delivered".
    results = await dispatch_tasks(
        [_returns_none, partial(_function, 0)], "threading", workers=2
    )
    assert results == [None, 1]


async def test_dispatch_tasks_unknown_executor() -> None:
    with pytest.raises(ValueError, match="Invalid executor"):
        await dispatch_tasks([partial(_function, 0)], "bogus")  # type: ignore[arg-type]


async def test_dispatch_tasks_default_hpc_workdir(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> ThreadingExecutor:
        captured.update(kwargs)
        return ThreadingExecutor(workers=kwargs["workers"])

    monkeypatch.setattr("ropt.workflow._dispatch_tasks.HPCExecutor", _capture)
    functions = [partial(_function, idx) for idx in range(2)]
    assert await dispatch_tasks(functions, "hpc", workers=2) == [1, 2]
    assert Path(captured["workdir"]).is_absolute()


@pytest.mark.slow
@pytest.mark.timeout(60)
async def test_dispatch_tasks_multiprocessing() -> None:
    functions = [partial(_function, idx) for idx in range(2)]
    assert await dispatch_tasks(functions, "multiprocessing", workers=1) == [1, 2]


async def test_dispatch_tasks_reraises_failing_function() -> None:
    functions = [partial(_function, 0, raise_error=True)]
    with pytest.raises(ExceptionGroup) as excinfo:
        await dispatch_tasks(functions, "threading", workers=2)
    matched, _ = excinfo.value.split(ValueError)
    assert matched is not None
    assert "Test error in function" in str(matched.exceptions[0])


@pytest.mark.parametrize(
    "executor_name",
    [
        "threading",
        pytest.param("multiprocessing", marks=pytest.mark.slow),
        pytest.param(
            "hpc",
            marks=[
                pytest.mark.slow,
                pytest.mark.timeout(30),
                pytest.mark.skipif(
                    not _TEST_HPC, reason="hpc requirements are not installed"
                ),
            ],
        ),
    ],
)
async def test_stopped_executor_restarts(
    executor_name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    match executor_name:
        case "hpc":
            monkeypatch.setattr(
                "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
                lambda *args, **kwargs: MockedHPCAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
            )
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=1, interval=0, template=""
            )
        case "threading":
            executor = ThreadingExecutor(workers=1)
        case "multiprocessing":
            executor = MultiprocessingExecutor(workers=1)

    async def _run_once() -> list[Any]:
        submission = Submission([WorkItem(function=_function, args=(0,))])
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.submit(submission)
            collected = await asyncio.to_thread(_collect, submission)
            executor.cancel()
        return collected

    assert await _run_once() == [1]

    results: list[Any] = []

    def _restart_on_a_new_loop() -> None:
        results.extend(asyncio.run(_run_once()))

    await asyncio.to_thread(_restart_on_a_new_loop)
    assert results == [1]


async def test_starting_running_executor_refused() -> None:
    executor = ThreadingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        with pytest.raises(WorkflowError, match="already running"):
            await executor.start(tg)
        executor.cancel()


async def test_queued_work_for_ended_submission_not_run() -> None:
    # One worker and a FIFO queue, so the sentinel completing proves everything
    # queued ahead of it was dealt with, one way or the other.
    executed: list[int] = []
    submission = Submission(
        [WorkItem(function=partial(_record, executed), args=(idx,)) for idx in range(6)]
    )
    executor = ThreadingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(ValueError, match="caller gave up"):
            await asyncio.to_thread(submission.collect, _give_up)
        sentinel = Submission(
            [WorkItem(function=partial(_record, executed), args=(99,))]
        )
        executor.submit(sentinel)
        assert await asyncio.to_thread(_collect, sentinel) == [99]
        executor.cancel()
    assert len([value for value in executed if value < 6]) < 6


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_no_hpc_jobs_for_ended_submission(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # On this executor each item queued behind the failure would become a real
    # cluster job, holding an allocation to produce a result nobody reads.
    class _NamingAdapter(MockedHPCAdapter):
        names: list[str] = []  # ruff: ignore[mutable-class-default]

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            type(self).names.append(job_name)
            return super().submit_job(job_name, command, **kwargs)

    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _NamingAdapter(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission(
        [
            WorkItem(function=_function, args=(idx,), name=f"item{idx}")
            for idx in range(6)
        ]
    )
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(ValueError, match="caller gave up"):
            await asyncio.to_thread(submission.collect, _give_up)
        sentinel = Submission([WorkItem(function=_function, args=(9,), name="last")])
        executor.submit(sentinel)
        assert await asyncio.to_thread(_collect, sentinel) == [10]
        executor.cancel()
    assert "last" in _NamingAdapter.names
    assert "item5" not in _NamingAdapter.names


async def test_starting_running_executor_leaves_it_untouched() -> None:
    executor = ThreadingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        pool = executor._pool  # ruff: ignore[private-member-access]
        tasks = executor._worker_tasks  # ruff: ignore[private-member-access]
        # A second start must fail before it builds a pool that nothing owns.
        with pytest.raises(WorkflowError, match="already running"):
            await executor.start(tg)
        assert executor._pool is pool  # ruff: ignore[private-member-access]
        assert executor._worker_tasks is tasks  # ruff: ignore[private-member-access]
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


async def test_evaluating_on_executor_loop_raises(
    config: dict[str, Any], eval_func: Any
) -> None:
    executor = ThreadingExecutor(workers=1)
    evaluator = ParallelEvaluator(function=eval_func(), executor=executor)
    batch_context = EvaluationBatchContext(
        context=EnOptContext.model_validate(config),
        active=np.array([True]),
        realizations=np.array([0], dtype=np.intc),
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        with pytest.raises(WorkflowError, match="must run in a thread"):
            evaluator.eval(np.zeros((1, 1)), batch_context)
        executor.cancel()


@pytest.mark.parametrize("start_first", [False, True])
async def test_eval_raises_executor_stopped_for_an_unusable_executor(
    config: dict[str, Any], eval_func: Any, start_first: Any
) -> None:
    executor = ThreadingExecutor(workers=1)
    if start_first:
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.cancel()
    evaluator = ParallelEvaluator(function=eval_func(), executor=executor)
    batch_context = EvaluationBatchContext(
        context=EnOptContext.model_validate(config),
        active=np.array([True]),
        realizations=np.array([0], dtype=np.intc),
    )
    with pytest.raises(ExecutorStopped):
        await asyncio.to_thread(evaluator.eval, np.zeros((1, 1)), batch_context)


class _FatalError(BaseException):
    pass


async def test_worker_base_exception_propagates_into_task_group() -> None:
    def _raise_fatal(input_value: int) -> int:  # ruff: ignore[unused-function-argument]
        msg = "fatal"
        raise _FatalError(msg)

    executor = ThreadingExecutor(workers=1)
    submission = Submission([WorkItem(function=_raise_fatal, args=(0,))])
    with pytest.raises(BaseExceptionGroup) as excinfo:  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.submit(submission)
    matched, _ = excinfo.value.split(_FatalError)
    assert matched is not None
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


async def test_fatal_work_item_error_reaches_caller() -> None:
    # The executor's teardown would release the caller anyway, but with the
    # generic abort; only the worker can hand over the real cause.
    def _raise_fatal(input_value: int) -> int:  # ruff: ignore[unused-function-argument]
        msg = "fatal"
        raise _FatalError(msg)

    executor = ThreadingExecutor(workers=1)
    submission = Submission([WorkItem(function=_raise_fatal, args=(0,))])
    outcome: list[BaseException] = []

    def _consume() -> None:
        try:
            submission.collect(lambda work_item: None)  # ruff: ignore[unused-lambda-argument]
        except BaseException as exc:  # ruff: ignore[blind-except]
            outcome.append(exc)

    consumer = threading.Thread(target=_consume, daemon=True)
    with pytest.raises(BaseExceptionGroup):  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            consumer.start()
            executor.submit(submission)
    await asyncio.to_thread(consumer.join, 5.0)
    assert not consumer.is_alive()
    assert isinstance(outcome[0], _FatalError)


async def test_handle_result_records_executor_failure_as_nan() -> None:  # ruff: ignore[unused-async]
    results = np.zeros((2, 1), dtype=np.float64)
    bundle = [
        (
            np.zeros(2, dtype=np.float64),
            EvaluationFunctionContext(
                realization=0, perturbation=-1, batch_id=0, eval_idx=idx
            ),
        )
        for idx in range(2)
    ]
    work_item = WorkItem(
        function=_function,
        args=(None, bundle),
        result=ExecutorFailure("Background process was killed"),
    )
    _handle_result(work_item, results, {}, objective_count=1, eval_count=2)
    assert np.all(np.isnan(results))


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param("not a list", id="not_a_list"),
        pytest.param([], id="too_short"),
    ],
)
async def test_evaluation_function_wrong_shape_rejected(  # ruff: ignore[unused-async]
    returned: Any,
) -> None:
    bundle = [
        (
            np.zeros(2, dtype=np.float64),
            EvaluationFunctionContext(
                realization=0, perturbation=-1, batch_id=0, eval_idx=0
            ),
        )
    ]
    work_item = WorkItem(function=_function, args=(None, bundle), result=returned)
    with pytest.raises(WorkflowError, match="must return a list of 1"):
        _handle_result(
            work_item,
            np.zeros((1, 1), dtype=np.float64),
            {},
            objective_count=1,
            eval_count=1,
        )


async def test_wrong_evaluation_result_type_rejected() -> None:  # ruff: ignore[unused-async]
    bundle = [
        (
            np.zeros(2, dtype=np.float64),
            EvaluationFunctionContext(
                realization=0, perturbation=-1, batch_id=0, eval_idx=0
            ),
        )
    ]
    work_item = WorkItem(function=_function, args=(None, bundle), result=["not one"])
    with pytest.raises(WorkflowError, match="got str"):
        _handle_result(
            work_item,
            np.zeros((1, 1), dtype=np.float64),
            {},
            objective_count=1,
            eval_count=1,
        )


@pytest.mark.slow
@pytest.mark.timeout(60)
async def test_multiprocessing_unguarded_main_reports_startup_error(
    tmp_path: Path,
) -> None:
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import asyncio\n\n"
        "from ropt.workflow import dispatch_tasks\n\n\n"
        "def work() -> int:\n"
        "    return 1\n\n\n"
        'asyncio.run(dispatch_tasks([work], executor="multiprocessing"))\n'
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    assert proc.returncode != 0
    assert b"Could not start worker processes" in stderr


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
async def test_multiprocessing_cloudpickles_functions_and_results() -> None:
    def make_adder(offset: int) -> Callable[[int], int]:
        def add(value: int) -> int:
            return value + offset

        return add

    def local_double(value: int) -> int:
        return value * 2

    def make_callable(value: int) -> Callable[[], int]:
        return lambda: value

    submission = Submission(
        [
            WorkItem(function=lambda value: value + 100, args=(1,)),
            WorkItem(function=make_adder(10), args=(2,)),
            WorkItem(function=local_double, args=(3,)),
            WorkItem(function=make_callable, args=(42,)),
        ]
    )
    executor = MultiprocessingExecutor(workers=2)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        results = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert sorted(value for value in results if isinstance(value, int)) == [6, 12, 101]
    returned = [value for value in results if callable(value)]
    assert len(returned) == 1
    assert returned[0]() == 42


@pytest.mark.slow
async def test_multiprocessing_unserializable_payload_reports_error() -> None:
    lock = threading.Lock()

    def use_lock() -> Any:
        return lock

    submission = Submission([WorkItem(function=use_lock)])
    executor = MultiprocessingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        # The serialization failure is delivered as the exception, not a teardown.
        with pytest.raises(Exception, match=r"(?i)pickle"):
            await asyncio.to_thread(_collect, submission)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.slow
async def test_multiprocessing_without_cloudpickle_rejects_lambda(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._multiprocessing_executor._HAVE_CLOUDPICKLE",
        False,
    )
    submission = Submission([WorkItem(function=lambda: 1)])
    executor = MultiprocessingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(ExecutionError, match=r"ropt\[cloudpickle\]"):
            await asyncio.to_thread(_collect, submission)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.slow
async def test_multiprocessing_without_cloudpickle_rejects_an_unpicklable_argument(
    monkeypatch: Any,
) -> None:
    # ParallelEvaluator submits a picklable module-level function and passes the
    # user callback as an argument, so the arguments must be checked too.
    monkeypatch.setattr(
        "ropt.components.executors._multiprocessing_executor._HAVE_CLOUDPICKLE",
        False,
    )
    submission = Submission([WorkItem(function=_call, args=(lambda: 1,))])
    executor = MultiprocessingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(ExecutionError, match=r"ropt\[cloudpickle\]"):
            await asyncio.to_thread(_collect, submission)
        executor.cancel()


def _return_captured(_handler: Any) -> int:
    return 0


def _opt_function_capturing_handler(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    test_functions: Any,
    handler: Any,  # ruff: ignore[unused-function-argument]
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(
        objectives=np.fromiter(
            (func(variables, context) for func in test_functions), dtype=np.float64
        )
    )


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_work_item_capturing_a_workflow_object_raises_transfer_error() -> None:
    submission = Submission(
        [WorkItem(function=_return_captured, args=(ResultsHandler(),))]
    )
    executor = MultiprocessingExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(TransferError):
            await asyncio.to_thread(_collect, submission)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_transfer_error_from_parallel_evaluation_bubbles_up(
    config: dict[str, Any],
    test_functions: Sequence[Callable[[NDArray[np.float64], int], float]],
) -> None:
    executor = MultiprocessingExecutor(workers=1)
    with pytest.raises(ExceptionGroup) as excinfo:  # ruff: ignore[pytest-raises-with-multiple-statements]
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            await asyncio.to_thread(
                _opt_workflow,
                executor,
                config,
                partial(
                    _opt_function_capturing_handler,
                    test_functions=test_functions,
                    handler=ResultsHandler(),
                ),
            )
            executor.cancel()
    assert any(isinstance(err, TransferError) for err in excinfo.value.exceptions)
