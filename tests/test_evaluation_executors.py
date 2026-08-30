from __future__ import annotations

import asyncio
import collections
import importlib
import inspect
import logging
import os
import pickle  # ruff: ignore[suspicious-pickle-import]
import pkgutil
import shutil
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from multiprocessing.connection import Client, Listener
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from ropt._serialize import HAVE_CLOUDPICKLE, dumps, loads
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
    LocalJobExecutor,
    ProcessExecutor,
    Submission,
    ThreadExecutor,
    WorkItem,
)
from ropt.components.executors._process_executor import (
    _run_payload,
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

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from ropt.components.executors import Executor
    from ropt.components.executors.base import ExecutorBase
    from ropt.results import FunctionResults

# The job entry point needs no extras of its own, so its tests run either way.
from ropt.components.executors.__main__ import run_task
from ropt.components.executors._picklable import picklable_exception

try:
    import pysqa
    from pysqa.wrapper.abstract import SchedulerCommands

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


def _return_unpicklable() -> Any:
    return threading.Lock()  # a lock cannot be (cloud)pickled


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


def _blocked_work_at_barrier(
    barrier: threading.Barrier, release: threading.Event
) -> int:
    barrier.wait(timeout=4.0)
    release.wait(timeout=5.0)
    return 0


def _raise_locally_defined_error() -> None:
    class _LocalError(Exception):
        pass

    msg = "raised by a class the standard library cannot name"
    raise _LocalError(msg)


# Run by `_spawn_child_and_block` as a process of its own. It is a grandchild of
# the executor, in the job's process group, so only killing the group reaches it.
_GRANDCHILD_SOURCE = """
import sys
from multiprocessing.connection import Client

Client(sys.argv[1]).recv()
"""


def _spawn_child_and_block(address: str) -> int:
    child = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", _GRANDCHILD_SOURCE, address]
    )
    child.wait()
    return 0


def _print_and_die(value: int) -> int:
    # Killed outright, so nothing is written back and the print is the only
    # account of what the job was doing.
    print("about to be killed", flush=True)  # ruff: ignore[print]
    os.kill(os.getpid(), signal.SIGKILL)
    return value


async def _wait_for_local_cleanup(executor: LocalJobExecutor) -> None:
    # The teardown thread waits for the jobs and then takes the directory away,
    # off the loop thread on purpose: nothing else can say when it has finished.
    thread = executor._teardown_thread  # ruff: ignore[private-member-access]
    assert thread is not None
    await asyncio.to_thread(thread.join, 10.0)


def _start_blocking_process() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )


def _explode() -> Any:
    msg = "this result cannot be rebuilt"
    raise ValueError(msg)


class _Unrebuildable:
    # Pickles into a call to `_explode`, so reading it back raises rather than
    # returning an object.
    def __reduce__(self) -> tuple[Any, ...]:
        return (_explode, ())


def _block_until_disconnected(address: str) -> int:
    # A synchronization primitive cannot be sent to a worker process, but the
    # address of a listening socket can. Connecting says the work item is
    # running, and the read that follows never returns on its own.
    connection = Client(address)
    connection.recv()
    return 0


def _kill_own_process() -> int:
    os.kill(os.getpid(), signal.SIGKILL)
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
            _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadExecutor(workers=2)
        case "multiprocessing":
            executor = ProcessExecutor(workers=2)
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert set(collected) == {1, 2}
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


async def test_thread_executor_exceeds_the_shared_default_pool() -> None:
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
    executor = ThreadExecutor(workers=workers)
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


async def test_submitting_from_a_worker_thread_is_refused() -> None:
    # Waiting here would occupy a worker while waiting for one, so the executor
    # refuses rather than deadlock once every worker is busy. Two workers for one
    # work item is deliberate: a regression is served by the spare worker and
    # fails on the assertion below, instead of deadlocking on the ceiling.
    executor = ThreadExecutor(workers=2)

    def _submit_back() -> tuple[bool, str]:
        try:
            executor.submit(Submission([WorkItem(function=_function, args=(0,))]))
        except WorkflowError as exc:
            return executor.on_worker_thread(), str(exc)
        return executor.on_worker_thread(), "accepted"

    submission = Submission([WorkItem(function=_submit_back)])
    collected: list[Any] = []
    finished, done = _finished_event()
    consumer = threading.Thread(
        target=_collect_in_thread, args=(submission, collected, done), daemon=True
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        assert not executor.on_worker_thread()
        consumer.start()
        executor.submit(submission)
        await finished.wait()
        executor.cancel()
    on_worker, message = collected[0]
    assert on_worker
    assert "already running on it" in message


async def test_thread_executor_delivers_results_without_the_shared_default_pool() -> (
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
    executor = ThreadExecutor(workers=2)
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
    executor = ThreadExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        await asyncio.to_thread(started.wait)
        executor.cancel()
    release.set()
    with pytest.raises(ExecutorStopped):
        _collect(submission)


async def test_stopping_thread_executor_reports_running_work(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Nothing can take a work item away from a thread, so the only help the user
    # gets is being told what the program is waiting for. Three work items and
    # two workers: the third is still queued, and waiting for it is not what
    # holds the program up.
    barrier = threading.Barrier(3)
    release = threading.Event()
    submission = Submission(
        [
            WorkItem(function=_blocked_work_at_barrier, args=(barrier, release))
            for _ in range(3)
        ]
    )
    executor = ThreadExecutor(workers=2)
    with caplog.at_level(logging.WARNING, logger="ropt"):
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.submit(submission)
            await asyncio.to_thread(barrier.wait, 4.0)
            executor.cancel()
        release.set()
    assert "Stopping with 2 evaluation(s) still running" in caplog.text


async def test_stopping_thread_executor_after_work_reports_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A work item that already returned is not something to wait for.
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = ThreadExecutor(workers=1)
    with caplog.at_level(logging.WARNING, logger="ropt"):
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.submit(submission)
            collected = await asyncio.to_thread(_collect, submission)
            executor.cancel()
    assert collected == [1]
    assert "still running" not in caplog.text


async def test_stopping_aborts_queued_submission() -> None:
    # One worker, blocked on the first work item, so the rest of the submission
    # is still sitting on the work queue when the executor stops.
    started = threading.Event()
    release = threading.Event()
    submission = Submission(
        [WorkItem(function=_blocked_work, args=(started, release)) for _ in range(5)]
    )
    executor = ThreadExecutor(workers=1)
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
    executor = ThreadExecutor(workers=1)
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
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
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
async def test_hpc_reads_job_ids_from_the_scheduler_table(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # `pysqa` answers with a table, and exactly one line in ropt unwraps it.
    # This is where that line is held to the shape, which is what lets every
    # other mock deal in job ids alone.
    class _Column:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def tolist(self) -> list[int]:
            return self._values

    class _Table:
        def __init__(self, job_ids: list[int]) -> None:
            self._job_ids = job_ids

        def __getitem__(self, column: str) -> _Column:
            assert column == "jobid"
            return _Column(self._job_ids)

    class _TableScheduler(MockedHPCAdapter):
        def get_status_of_my_jobs(self) -> _Table:
            return _Table(sorted(self.live_job_ids()))

    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
        lambda *args, **kwargs: _TableScheduler(tmp_path),  # ruff: ignore[unused-lambda-argument]
    )
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert collected == [1]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_scheduler_query_fails_after_retry_limit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # A scheduler that cannot be queried must not look like "nothing finished
    # yet": without a bound on the failures the caller waits forever.
    class _UnreachableScheduler(MockedHPCAdapter):
        queries = 0

        def live_job_ids(self) -> set[int]:
            type(self).queries += 1
            msg = "squeue: error: Unable to contact slurm controller"
            raise RuntimeError(msg)

    _mock_scheduler(monkeypatch, _UnreachableScheduler(tmp_path))
    submission = Submission([WorkItem(function=_function, args=(0,))])
    # `retries=0` gives up on a missing result at once, and has nothing to say
    # about a scheduler that will not answer: the two budgets are separate.
    executor = HPCExecutor(
        workdir=tmp_path,
        workers=1,
        interval=0,
        retries=0,
        query_retries=2,
        template="",
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(collected) == 1
    assert isinstance(collected[0], ExecutorFailure)
    assert "could not be queried" in str(collected[0])
    assert "after 3 attempts" in str(collected[0])
    assert _UnreachableScheduler.queries == 3


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_scheduler_query_budget_resets_after_an_answer(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # The budget is for a run of failures, not for failures in total: a cluster
    # that has a bad moment every so often would otherwise fail a long run for
    # no reason at all.
    class _FlakyScheduler(MockedHPCAdapter):
        calls = 0

        def live_job_ids(self) -> set[int]:
            type(self).calls += 1
            if type(self).calls in {1, 3}:
                msg = "squeue: error: Unable to contact slurm controller"
                raise RuntimeError(msg)
            if type(self).calls == 2:
                return set(self._jobs)  # answered, and the job is still out
            return super().live_job_ids()

    _mock_scheduler(monkeypatch, _FlakyScheduler(tmp_path))
    submission = Submission([WorkItem(function=_function, args=(0,))])
    # One failure is survivable; two in a row are not. The answer in between is
    # what makes the second failure a first one again.
    executor = HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, query_retries=1, template=""
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert collected == [1]
    assert _FlakyScheduler.calls >= 4


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

        def live_job_ids(self) -> set[int]:  # ruff: ignore[no-self-use]
            return set()

    _mock_scheduler(monkeypatch, _VanishingJob(tmp_path))
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

        def live_job_ids(self) -> set[int]:  # ruff: ignore[no-self-use]
            return set()

    _mock_scheduler(monkeypatch, _CorruptResult(tmp_path))
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
async def test_hpc_result_of_an_unknown_type_fails_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # A result naming something this process cannot import is whole and correct;
    # it is only unreadable here. Retrying would postpone the same failure and
    # then blame the shared filesystem, which is the one thing not at fault.
    class _AlienResult(MockedHPCAdapter):
        polls = 0

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            # A valid pickle of a name from a module that does not exist. The
            # replacement is the same length as the original, so the frame and
            # length prefixes in the pickle stay right.
            payload = pickle.dumps(collections.OrderedDict, protocol=4)
            (self._path / f"{job_name}.out").write_bytes(
                payload.replace(b"collections", b"collectionx")
            )
            return self._job_id

        def live_job_ids(self) -> set[int]:
            type(self).polls += 1
            return set()

    _mock_scheduler(monkeypatch, _AlienResult(tmp_path))
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = HPCExecutor(
        workdir=tmp_path, workers=1, interval=3600, retries=30, template=""
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(collected) == 1
    assert isinstance(collected[0], ExecutorFailure)
    assert "could not be reconstructed" in str(collected[0])
    assert "collectionx" in str(collected[0])
    # An hour between polls: had this spent a retry, the answer would have come
    # an hour later at best, and the timeout on this test long before that.
    assert _AlienResult.polls == 1


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_result_that_cannot_be_rebuilt_fails_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Reading a result runs the code that rebuilds it, and that code can raise
    # anything at all. It belongs to the work item; the executor has to survive.
    class _UnrebuildableResult(MockedHPCAdapter):
        polls = 0

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            (self._path / f"{job_name}.out").write_bytes(pickle.dumps(_Unrebuildable()))
            return self._job_id

        def live_job_ids(self) -> set[int]:
            type(self).polls += 1
            return set()

    _mock_scheduler(monkeypatch, _UnrebuildableResult(tmp_path))
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = HPCExecutor(
        workdir=tmp_path, workers=1, interval=3600, retries=30, template=""
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    # Collecting at all is half the point: had the error escaped `_poll`, the
    # executor would have gone down with it and nothing would arrive.
    assert len(collected) == 1
    assert isinstance(collected[0], ExecutorFailure)
    assert "could not be read" in str(collected[0])
    assert "this result cannot be rebuilt" in str(collected[0])
    assert _UnrebuildableResult.polls == 1


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_job_command_uses_submitting_interpreter(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # A bare `python` resolves through the job's PATH, which need not be the
    # environment ropt is installed in.
    commands: list[str] = []

    class _RecordingAdapter(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            commands.append(command)
            return super().submit_job(job_name, command, **kwargs)

    _mock_scheduler(monkeypatch, _RecordingAdapter(tmp_path))
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        assert await asyncio.to_thread(_collect, submission) == [1]
        executor.cancel()
    assert commands
    assert commands[0].startswith(f"{sys.executable} -m ropt.components.executors ")


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_failed_work_keeps_job_output(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # A job that died before writing a result left its reason in the captured
    # output alone, so cleanup must not take that away with the rest.
    class _CrashingJob(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            (self._path / f"{job_name}.txt").write_text(
                "ModuleNotFoundError: No module named 'ropt'\n"
            )
            return self._job_id

        def live_job_ids(self) -> set[int]:  # ruff: ignore[no-self-use]
            return set()

    _mock_scheduler(monkeypatch, _CrashingJob(tmp_path))
    submission = Submission([WorkItem(function=_function, args=(0,), name="item")])
    executor = HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=2, template=""
    )
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert isinstance(collected[0], ExecutorFailure)
    assert "No module named 'ropt'" in str(collected[0])
    assert (tmp_path / "item.txt").exists()
    assert not (tmp_path / "item.in").exists()


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

    _mock_scheduler(monkeypatch, _StuckAdapter(tmp_path))
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
    _mock_scheduler(monkeypatch, adapter)
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

    class _SlowAdapter(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:  # ruff: ignore[unused-method-argument]
            submitting.set()
            stopped.wait(5)
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

    adapter = _SlowAdapter(tmp_path)
    _mock_scheduler(monkeypatch, adapter)
    submission = Submission([WorkItem(function=_function, args=(0,), name="job1")])
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        pool = executor._pool  # ruff: ignore[private-member-access]
        executor.submit(submission)
        await asyncio.to_thread(submitting.wait)
        executor.cancel()
    stopped.set()
    # Submitting, cancelling and deleting the files all happen on the poll
    # thread, which outlives the executor: join it, or the cancellation is
    # observable before the cleanup that follows it.
    assert pool is not None
    await asyncio.to_thread(pool.shutdown, wait=True)
    assert adapter.deleted == [1]
    assert not await asyncio.to_thread(lambda: list(tmp_path.glob("job1.*")))


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_idle_hpc_executor_does_not_query_scheduler(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class _CountingScheduler(MockedHPCAdapter):
        queries = 0

        def live_job_ids(self) -> set[int]:
            type(self).queries += 1
            return super().live_job_ids()

    _mock_scheduler(monkeypatch, _CountingScheduler(tmp_path))
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
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
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
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
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
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
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
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
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

    _mock_scheduler(monkeypatch, _RejectingScheduler(tmp_path))
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
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
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


async def test_worker_records_the_worker_traceback_as_a_note() -> None:
    payload = dumps((partial(_function, 0, raise_error=True), (), {}))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    ok, blob = _run_payload(payload)
    assert not ok
    exc = loads(blob)
    assert isinstance(exc, ValueError)
    assert any("Traceback" in note for note in exc.__notes__)


async def test_worker_wraps_an_unpicklable_exception() -> None:
    payload = dumps((_raise_unpicklable_error, (0,), {}))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    ok, blob = _run_payload(payload)
    assert not ok
    exc = loads(blob)
    assert isinstance(exc, RuntimeError)
    assert any("Traceback" in note for note in exc.__notes__)


async def test_a_result_that_cannot_be_sent_is_not_blamed_on_the_function() -> None:
    # The call itself succeeded and only its result could not be serialized.
    # Reporting that as the function raising is the misdiagnosis the second
    # `try` exists to remove, so the note has to name the result, not the call.
    payload = dumps((_return_unpicklable, (), {}))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    ok, blob = _run_payload(payload)
    assert not ok
    exc = loads(blob)
    assert any("Could not send the result back" in note for note in exc.__notes__)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_job_exit_writes_output_file(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    # Serialized with the shim, which is what the job path itself uses: the
    # task is a module-level function, so this runs with or without the extra.
    input_file.write_bytes(dumps((_exit_task, (), {})))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    assert run_task(str(input_file), str(output_file)) == 1
    assert isinstance(loads(output_file.read_bytes()), SystemExit)


async def test_job_wraps_an_exception_the_standard_library_cannot_send(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # `cloudpickle` is optional on the job path, so whether an exception can
    # travel has to be decided with the serializer that will do the sending.
    # Standing in for a job running without it: the failure still has to come
    # back, wrapped rather than lost.
    monkeypatch.setattr("ropt.components.executors._picklable.dumps", pickle.dumps)
    monkeypatch.setattr("ropt.components.executors.__main__.dump", pickle.dump)
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(pickle.dumps((_raise_locally_defined_error, (), {})))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    assert run_task(str(input_file), str(output_file)) == 1
    result = pickle.loads(output_file.read_bytes())  # ruff: ignore[suspicious-pickle-usage]
    assert isinstance(result, RuntimeError)
    assert "_LocalError" in str(result)
    assert any("Traceback" in note for note in result.__notes__)


def _vanishing_work_item() -> int:
    return 1


def _vanishing_job_task() -> int:
    return 1


async def test_a_work_item_the_worker_cannot_rebuild_reports_why(
    monkeypatch: Any,
) -> None:
    # It pickles, because the name resolves here. The worker is where it does
    # not, which is why no send-side check can catch this and why the load has
    # to happen somewhere the failure can be reported.
    payload = pickle.dumps((_vanishing_work_item, (), {}))
    monkeypatch.delattr(sys.modules[__name__], "_vanishing_work_item")
    await asyncio.sleep(0)  # this module runs tests on the event loop
    ok, blob = _run_payload(payload)
    assert not ok
    exc = loads(blob)
    assert isinstance(exc, AttributeError)
    assert any("Could not rebuild the work item" in note for note in exc.__notes__)


async def test_a_task_the_job_cannot_rebuild_reports_why(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # The same failure on the job path, where it matters most: the job has no
    # channel back, so the note in the result file is the whole diagnosis.
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(pickle.dumps((_vanishing_job_task, (), {})))
    monkeypatch.delattr(sys.modules[__name__], "_vanishing_job_task")
    await asyncio.sleep(0)  # this module runs tests on the event loop
    assert run_task(str(input_file), str(output_file)) == 1
    result = loads(output_file.read_bytes())
    assert isinstance(result, AttributeError)
    assert any("Could not rebuild the task" in note for note in result.__notes__)


async def test_the_hint_stays_off_the_task_s_own_exception(
    tmp_path: Path,
) -> None:
    # The load and the call are separated precisely so this exception, which is
    # the task's own, carries no advice about rebuilding it.
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(dumps((partial(_function, 0, raise_error=True), (), {})))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    assert run_task(str(input_file), str(output_file)) == 1
    result = loads(output_file.read_bytes())
    assert isinstance(result, ValueError)
    assert not any("Could not rebuild" in note for note in result.__notes__)


async def test_a_result_that_cannot_be_written_still_reaches_the_executor(
    tmp_path: Path,
) -> None:
    # The task ran to completion; only its result could not be written. Letting
    # that escape leaves no file at all, and a missing file is reported as a job
    # that produced nothing -- the same misdiagnosis in a different disguise.
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(dumps((_return_unpicklable, (), {})))
    await asyncio.sleep(0)  # this module runs tests on the event loop
    assert run_task(str(input_file), str(output_file)) == 1
    result = loads(output_file.read_bytes())
    assert any("Could not send the result back" in note for note in result.__notes__)
    # The abandoned first attempt takes its temporary file with it.
    names = await asyncio.to_thread(lambda: sorted(p.name for p in tmp_path.iterdir()))
    assert names == ["job.in", "job.out"]


@pytest.mark.skipif(not HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
async def test_job_sends_an_exception_the_standard_library_cannot_send() -> None:
    # The same exception, with `cloudpickle` present: it survives as itself, so
    # the wrapping above is a fallback rather than what always happens.
    await asyncio.sleep(0)  # this module runs tests on the event loop
    try:
        _raise_locally_defined_error()
    except Exception as exc:  # ruff: ignore[blind-except]
        result = picklable_exception(exc)
    assert type(result).__name__ == "_LocalError"


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_unserializable_work_item_fails_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Serializing happens before the job exists, so this failure belongs to the
    # work item, and it says what to install rather than what broke inside.
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
    submission = Submission(
        [WorkItem(function=_function, args=(threading.Lock(),), name="job1")]
    )
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(ExecutionError, match="could not be sent to a job"):
            await asyncio.to_thread(_collect, submission)
        executor.cancel()
    # Nothing was left behind: no job was submitted, so nothing would ever come
    # along to cancel it or to clean up after it.
    assert not await asyncio.to_thread(lambda: list(tmp_path.iterdir()))


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_worker_may_construct_workflow_objects() -> None:
    submission = Submission(
        [WorkItem(function=_construct_handler_in_worker, args=(0,))]
    )
    executor = ProcessExecutor(workers=1)
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
    executor = ProcessExecutor(workers=1, max_tasks_per_child=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(set(collected)) == 3


@pytest.mark.slow
@pytest.mark.timeout(30)
@pytest.mark.parametrize("public_api", [True, False])
async def test_stopping_kills_a_busy_worker(
    tmp_path: Path, monkeypatch: Any, *, public_api: bool
) -> None:
    # An idle worker leaves on its own when the pool shuts down, so only a
    # worker busy with work that never ends shows whether stopping stops it.
    if not public_api:
        # Python 3.11 to 3.13 have no `terminate_workers`, and the hand-rolled
        # path has to be covered whichever version the tests run on.
        monkeypatch.delattr(ProcessPoolExecutor, "terminate_workers", raising=False)
    listener = Listener(str(tmp_path / "worker"))
    submission = Submission(
        [WorkItem(function=_block_until_disconnected, args=(listener.address,))]
    )
    executor = ProcessExecutor(workers=1)
    connection = None
    try:
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.submit(submission)
            connection = await asyncio.to_thread(listener.accept)
            executor.cancel()
        # The worker's end of this connection closes when it dies, and it can
        # only die by being killed: its work item is waiting for a message that
        # is never sent. Its own `Process` object cannot be asked, because the
        # pool's manager thread may reap it first and leave `exitcode` unset.
        # The timeout is a ceiling rather than a wait: without it, a worker that
        # survives would hang here and then hang the interpreter on the way out.
        assert await asyncio.to_thread(connection.poll, 10.0)
        with pytest.raises(EOFError):
            connection.recv()
    finally:
        if connection is not None:
            connection.close()
        listener.close()


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_stopped_process_executor_starts_a_fresh_pool() -> None:
    # Stopping breaks the pool for good, so starting again has to build a new
    # one instead of handing back the pool whose workers were just killed.
    executor = ProcessExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.cancel()
    submission = Submission([WorkItem(function=_function, args=(1,))])
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert collected == [2]


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_dying_worker_reported_as_infrastructure_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A worker lost while the executor is running is not something it asked for,
    # so it is reported, unlike the workers a stop kills on purpose.
    submission = Submission([WorkItem(function=_kill_own_process)])
    executor = ProcessExecutor(workers=1)
    with caplog.at_level(logging.WARNING, logger="ropt"):
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.submit(submission)
            collected = await asyncio.to_thread(_collect, submission)
            executor.cancel()
    assert len(collected) == 1
    assert isinstance(collected[0], ExecutorFailure)
    assert "Worker process pool broken" in caplog.text


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
            _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadExecutor(workers=2)
        case "multiprocessing":
            executor = ProcessExecutor(workers=2)
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
        expects_notes = executor_name in {"hpc", "multiprocessing"}
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

        def live_job_ids(self) -> set[int]:
            running = [
                job_id
                for job_id, job_name in self._jobs.items()
                if not (self._path / f"{job_name}.out").exists()
            ]
            self._jobs = {job_id: self._jobs[job_id] for job_id in running}
            return set(self._jobs)

    def _mock_scheduler(monkeypatch: Any, adapter: MockedHPCAdapter) -> None:
        # `ropt` asks the scheduler for the ids of the jobs that are still
        # there; that the real one answers with a table is `pysqa`'s business,
        # so the mocks never build one.
        monkeypatch.setattr(
            "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
            lambda *args, **kwargs: adapter,  # ruff: ignore[unused-lambda-argument]
        )
        monkeypatch.setattr(
            HPCExecutor, "_live_job_ids", lambda _self: adapter.live_job_ids()
        )


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
            _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadExecutor(workers=2)
        case "multiprocessing":
            executor = ProcessExecutor(workers=2)
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
            _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadExecutor(workers=2)
        case "multiprocessing":
            executor = ProcessExecutor(workers=2)
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
            executor: ExecutorBase = ThreadExecutor(workers=2)
        case "multiprocessing":
            executor = ProcessExecutor(workers=2)
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
    executor = ThreadExecutor(workers=2)
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
            _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=2, interval=0, template=""
            )
        case "threading":
            executor = ThreadExecutor(workers=2)
        case "multiprocessing":
            executor = ProcessExecutor(workers=2)
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
            executor: Executor = ThreadExecutor(workers=2)
        case "multiprocessing":
            executor = ProcessExecutor(workers=2)

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
    executor = ThreadExecutor(workers=1)
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
    executor = ThreadExecutor(workers=1)
    loop = asyncio.new_event_loop()
    loop.close()
    executor._loop = loop  # ruff: ignore[private-member-access]
    executor._running.set()  # ruff: ignore[private-member-access]
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor.submit(submission)
    with pytest.raises(ExecutorStopped):
        _collect(submission)


async def test_cancelling_unstarted_executor() -> None:  # ruff: ignore[unused-async]
    executor = ThreadExecutor(workers=1)
    executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_relative_workdir_rejected(tmp_path: Path) -> None:  # ruff: ignore[unused-async, unused-function-argument]
    # The workdir is shared with the cluster nodes, which do not necessarily
    # share this process's working directory.
    with pytest.raises(ValueError, match="must be an absolute path"):
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
        "ropt.components.executors._process_executor.ProcessPoolExecutor",
        _BrokenPool,
    )
    executor = ProcessExecutor(workers=1)
    with pytest.raises(ExceptionGroup) as excinfo:
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
    matched, _ = excinfo.value.split(ExecutionError)
    assert matched is not None
    assert "guard the program entry point" in str(matched.exceptions[0])


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_missing_workdir_rejected(tmp_path: Path) -> None:  # ruff: ignore[unused-async]
    with pytest.raises(ValueError, match="does not exist"):
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
    with pytest.raises(ValueError, match=match):
        HPCExecutor(workdir=tmp_path, template="", **kwargs)


@pytest.fixture
def pysqa_config(tmp_path: Path) -> Path:
    """A `pysqa` configuration with two clusters and no scheduler behind it.

    `HPCExecutor` looks for `<config_path>/<queue_type>`, so the tree is built
    to match. Nothing here needs a queueing system to exist: `pysqa` only reads
    these files, which is what makes the whole cluster-selection path testable
    on a machine that has no scheduler at all.

    Returns:
        The directory to pass as `config_path`, holding a `slurm` subdirectory.
    """
    root = tmp_path / "pysqa" / "slurm"
    root.mkdir(parents=True)
    (root / "job.sh").write_text(
        "#!/bin/bash\n#SBATCH --job-name={{job_name}}\n"
        "#SBATCH --output={{output}}\n{{command}}\n"
    )
    (root / "cluster_a.yaml").write_text(
        "queue_type: SLURM\nqueue_primary: fast\nqueues:\n"
        "  fast: {cores_max: 4, cores_min: 1, run_time_max: 3600, script: job.sh}\n"
        "  shared: {cores_max: 8, cores_min: 1, run_time_max: 3600, script: job.sh}\n"
    )
    (root / "cluster_b.yaml").write_text(
        "queue_type: SLURM\nqueue_primary: bulk\nqueues:\n"
        "  bulk: {cores_max: 16, cores_min: 1, run_time_max: 7200, script: job.sh}\n"
        "  shared: {cores_max: 8, cores_min: 1, run_time_max: 7200, script: job.sh}\n"
    )
    (root / "clusters.yaml").write_text(
        "cluster_primary: cluster_a\ncluster:\n"
        "  cluster_a: cluster_a.yaml\n  cluster_b: cluster_b.yaml\n"
    )
    return tmp_path / "pysqa"


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_executor_builds_from_a_configuration_directory(  # ruff: ignore[unused-async]
    tmp_path: Path, pysqa_config: Path
) -> None:
    # The other branch of the constructor: every other test passes a template,
    # so this one was never built by a test at all.
    executor = HPCExecutor(
        workdir=tmp_path, config_path=pysqa_config, cluster="cluster_b"
    )
    adapter = executor._queue_adapter  # ruff: ignore[private-member-access]
    assert adapter.list_clusters() == ["cluster_a", "cluster_b"]
    assert adapter.queue_list == ["bulk", "shared"]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_hpc_executor_selects_the_cluster_holding_the_queue(  # ruff: ignore[unused-async]
    tmp_path: Path, pysqa_config: Path
) -> None:
    executor = HPCExecutor(workdir=tmp_path, config_path=pysqa_config, queue="bulk")
    assert executor._queue_adapter.queue_list == ["bulk", "shared"]  # ruff: ignore[private-member-access]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cluster": "nowhere"}, "Unknown HPC cluster"),
        ({"queue": "nowhere"}, "not available on any HPC cluster"),
        ({"cluster": "cluster_a", "queue": "bulk"}, "not available on HPC cluster"),
        ({"queue": "shared"}, "available on multiple HPC clusters"),
    ],
)
async def test_hpc_cluster_selection_rejects_what_it_cannot_resolve(  # ruff: ignore[unused-async]
    tmp_path: Path, pysqa_config: Path, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ExecutionError, match=match):
        HPCExecutor(workdir=tmp_path, config_path=pysqa_config, **kwargs)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_pysqa_still_accepts_what_the_executor_passes_by_name() -> None:  # ruff: ignore[unused-async]
    # Hand-written rather than `create_autospec`, deliberately: a spec'd mock
    # accepts arbitrary keywords, so it would wave through exactly the change
    # this is here to catch.
    signature = inspect.signature(pysqa.QueueAdapter.submit_job)
    for name in (
        "job_name",
        "working_directory",
        "command",
        "submission_template",
        "queue",
        "cores",
    ):
        assert name in signature.parameters, name

    # `output` is not one of them: it is a jinja2 variable for the submission
    # template, which reaches it through `**kwargs`. Removing that catch-all
    # would turn the executor's `output=` into a `TypeError` at submit time,
    # so both halves have to be pinned.
    assert "output" not in signature.parameters
    assert any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _scheduler_wrappers() -> dict[str, Any]:
    package = importlib.import_module("pysqa.wrapper")
    wrappers: dict[str, Any] = {}
    for info in pkgutil.iter_modules(package.__path__):
        try:
            module = importlib.import_module(f"pysqa.wrapper.{info.name}")
        except ImportError:
            # Some wrappers need their scheduler's own Python package.
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, SchedulerCommands)
                and cls is not SchedulerCommands
                and cls.__module__ == module.__name__
            ):
                wrappers[info.name] = cls
    return wrappers


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
async def test_pysqa_names_the_job_id_column_jobid() -> None:  # ruff: ignore[unused-async]
    # `_live_job_ids` indexes the result with `"jobid"`, so the name is a
    # contract. It holds for every wrapper that implements the conversion at
    # all -- but only for those, which is why the claim is scoped rather than
    # made about "all wrappers".
    checked = []
    for name, cls in _scheduler_wrappers().items():
        if "convert_queue_status" not in vars(cls):
            continue
        try:
            frame = cls.convert_queue_status("")
        except Exception:  # ruff: ignore[blind-except, try-except-continue]
            # Its parser expects a header this test would have to fabricate
            # per scheduler, which is a coupling worse than the one it checks.
            continue
        assert "jobid" in frame.columns, name
        checked.append(name)

    # Without this the loop above passes by checking nothing at all.
    assert "slurm" in checked, checked


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
    executor = ThreadExecutor(workers=1)
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
    executor = ThreadExecutor(workers=1)
    executor._running.set()  # ruff: ignore[private-member-access]
    executor._accept(submission)  # ruff: ignore[private-member-access]
    executor._accept(submission)  # ruff: ignore[private-member-access]
    assert executor._work_queue.qsize() == 3  # ruff: ignore[private-member-access]


async def test_submission_after_stopping_is_aborted() -> None:
    # submit() hands over on the loop, where stopping is decided; this is the
    # guard that makes "every submission settles" hold when the two race.
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = ThreadExecutor(workers=1)
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
    executor = ThreadExecutor(workers=1)
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
            _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
            executor: ExecutorBase = HPCExecutor(
                workdir=tmp_path, workers=1, interval=0, template=""
            )
        case "threading":
            executor = ThreadExecutor(workers=1)
        case "multiprocessing":
            executor = ProcessExecutor(workers=1)

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
    executor = ThreadExecutor(workers=1)
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
    executor = ThreadExecutor(workers=1)
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

    _mock_scheduler(monkeypatch, _NamingAdapter(tmp_path))
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
    executor = ThreadExecutor(workers=1)
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
    executor = ThreadExecutor(workers=1)
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
    executor = ThreadExecutor(workers=1)
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

    executor = ThreadExecutor(workers=1)
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

    executor = ThreadExecutor(workers=1)
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


async def test_handle_result_logs_executor_failure_reason(  # ruff: ignore[unused-async]
    caplog: pytest.LogCaptureFixture,
) -> None:
    # NaN is all that reaches the optimizer, so the log is the only place the
    # reason for a failed realization is stated.
    results = np.zeros((1, 1), dtype=np.float64)
    bundle = [
        (
            np.zeros(2, dtype=np.float64),
            EvaluationFunctionContext(
                realization=0, perturbation=-1, batch_id=0, eval_idx=0
            ),
        )
    ]
    work_item = WorkItem(
        function=_function,
        args=(None, bundle),
        result=ExecutorFailure("the job wrote to item.txt"),
    )
    with caplog.at_level(logging.WARNING, logger="ropt"):
        _handle_result(work_item, results, {}, objective_count=1, eval_count=1)
    assert "the job wrote to item.txt" in caplog.text


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
        "from ropt.components.executors import ProcessExecutor\n\n\n"
        "async def main() -> None:\n"
        "    executor = ProcessExecutor(workers=1)\n"
        "    async with asyncio.TaskGroup() as tg:\n"
        "        await executor.start(tg)\n"
        "        executor.cancel()\n\n\n"
        "asyncio.run(main())\n"
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
@pytest.mark.skipif(not HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
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
    executor = ProcessExecutor(workers=2)
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
    executor = ProcessExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        # The serialization failure is delivered as the exception, not a
        # teardown, and as ours: this path used to let the serializer's own
        # error through untouched, whatever it happened to say.
        with pytest.raises(ExecutionError, match="could not be sent"):
            await asyncio.to_thread(_collect, submission)
        assert executor._running.is_set()  # ruff: ignore[private-member-access]
        executor.cancel()
    assert not executor._running.is_set()  # ruff: ignore[private-member-access]


@pytest.mark.slow
async def test_multiprocessing_without_cloudpickle_rejects_lambda(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._process_executor.dumps",
        pickle.dumps,
    )
    submission = Submission([WorkItem(function=lambda: 1)])
    executor = ProcessExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        # The stable half of the message. What follows the colon depends on
        # whether the extra is installed, and this test forces the standard
        # library rather than uninstalling it.
        with pytest.raises(ExecutionError, match="could not be sent"):
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
        "ropt.components.executors._process_executor.dumps",
        pickle.dumps,
    )
    submission = Submission([WorkItem(function=_call, args=(lambda: 1,))])
    executor = ProcessExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(ExecutionError, match="could not be sent"):
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
    executor = ProcessExecutor(workers=1)
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
    executor = ProcessExecutor(workers=1)
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


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_jobs_evaluate_work() -> None:
    submission = Submission([WorkItem(function=_function, args=(i,)) for i in range(4)])
    executor = LocalJobExecutor(workers=2)
    workdir = executor.workdir
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert sorted(collected) == [1, 2, 3, 4]
    await _wait_for_local_cleanup(executor)
    # Nothing failed and cleanup is on, so there is nothing in there to read:
    # a temporary directory with no reason to stay is taken away again.
    assert not workdir.exists()


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_job_error_carries_its_traceback(tmp_path: Path) -> None:
    # The job is the only place the traceback existed, and it left no channel
    # back: it travels as a note on the exception or not at all.
    submission = Submission(
        [WorkItem(function=_function, args=(0,), kwargs={"raise_error": True})]
    )
    executor = LocalJobExecutor(workdir=tmp_path, workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        with pytest.raises(ValueError, match="Test error") as info:
            await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert any("Traceback" in note for note in info.value.__notes__)


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_jobs_run_in_separate_processes() -> None:
    # The point of a job over a thread: its own interpreter, which is also what
    # makes it killable.
    submission = Submission([WorkItem(function=os.getpid, args=()) for _ in range(2)])
    executor = LocalJobExecutor(workers=2)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(set(collected)) == 2
    assert os.getpid() not in collected
    await _wait_for_local_cleanup(executor)


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_stopping_kills_a_local_job_and_its_children(tmp_path: Path) -> None:
    # A job that started a process of its own: stopping has to reach that too,
    # or it is orphaned and outlives the run that asked for it.
    listener = Listener(str(tmp_path / "job"))
    submission = Submission(
        [WorkItem(function=_spawn_child_and_block, args=(listener.address,))]
    )
    executor = LocalJobExecutor(workdir=tmp_path, workers=1)
    connection = None
    try:
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.submit(submission)
            # Accepting proves the grandchild is up and the job is waiting on it.
            connection = await asyncio.to_thread(listener.accept)
            executor.cancel()
        # This end belongs to the grandchild, which is only reachable by killing
        # the group: end of file here is that process being gone.
        assert await asyncio.to_thread(connection.poll, 10.0)
        with pytest.raises(EOFError):
            connection.recv()
    finally:
        if connection is not None:
            connection.close()
        listener.close()


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_job_that_dies_without_a_result_fails_work(tmp_path: Path) -> None:
    # Killed outright, so nothing was written: the only account of the job is
    # what it printed, and that has to reach the caller.
    submission = Submission([WorkItem(function=_print_and_die, args=(0,))])
    executor = LocalJobExecutor(workdir=tmp_path, workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    assert len(collected) == 1
    assert isinstance(collected[0], ExecutorFailure)
    assert "never appeared" in str(collected[0])
    assert "about to be killed" in str(collected[0])


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_job_ids_are_not_pids(caplog: pytest.LogCaptureFixture) -> None:
    # Pids come round again, and job ids that were pids would come round with
    # them, letting a finished job be mistaken for one that is still running.
    executor = LocalJobExecutor(workers=1)
    with caplog.at_level(logging.DEBUG, logger="ropt"):
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            for value in range(3):
                submission = Submission([WorkItem(function=_function, args=(value,))])
                executor.submit(submission)
                assert await asyncio.to_thread(_collect, submission) == [value + 1]
            executor.cancel()
    started = [line for line in caplog.messages if line.startswith("Started local job")]
    assert [line.rsplit(" ", 1)[-1] for line in started] == ["1)", "2)", "3)"]
    await _wait_for_local_cleanup(executor)


async def test_local_executor_keeps_a_directory_it_was_given(tmp_path: Path) -> None:
    executor = LocalJobExecutor(workdir=tmp_path, workers=1, cleanup=False)
    await asyncio.sleep(0)  # this module runs tests on the event loop
    executor.cancel()
    assert await asyncio.to_thread(tmp_path.exists)


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_executor_keeps_its_directory_when_a_job_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Removing the directory would take the failed job's captured output with
    # it, which is the only account of why it failed.
    submission = Submission([WorkItem(function=_print_and_die, args=(0,))])
    executor = LocalJobExecutor(workers=1)
    workdir = executor.workdir
    with caplog.at_level(logging.WARNING, logger="ropt"):
        async with asyncio.TaskGroup() as tg:
            await executor.start(tg)
            executor.submit(submission)
            assert isinstance(
                (await asyncio.to_thread(_collect, submission))[0], ExecutorFailure
            )
            executor.cancel()
        await _wait_for_local_cleanup(executor)
    assert await asyncio.to_thread(workdir.exists)
    output = workdir / "".join(str(path.name) for path in workdir.glob("*.txt"))
    assert "about to be killed" in await asyncio.to_thread(output.read_text)
    # A random name kept and never mentioned is a directory nobody can find.
    assert any(
        str(workdir) in message and "a work item failed" in message
        for message in caplog.messages
    )
    shutil.rmtree(workdir)


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_executor_keeps_its_directory_when_polling_gives_up() -> None:
    # Giving up on polling fails whatever was out, and those items never reach
    # the pass that removes their files, so this route has to keep the
    # directory too. Contrived for a local backend, and the code is shared.
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = LocalJobExecutor(workers=1)
    workdir = executor.workdir

    def _unreachable() -> set[int]:
        msg = "cannot tell whether the job is alive"
        raise RuntimeError(msg)

    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor._live_job_ids = _unreachable  # type: ignore[method-assign]  # ruff: ignore[private-member-access]
        executor.submit(submission)
        collected = await asyncio.to_thread(_collect, submission)
        executor.cancel()
    await _wait_for_local_cleanup(executor)
    assert isinstance(collected[0], ExecutorFailure)
    assert "could not be queried" in str(collected[0])
    assert await asyncio.to_thread(workdir.exists)
    # Dropped by the base along with the job, and still a process of ours.
    assert executor._processes == {}  # ruff: ignore[private-member-access]
    shutil.rmtree(workdir)


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_executor_keeps_its_directory_when_cleanup_is_off() -> None:
    # `cleanup=False` means nothing here is removed; a directory that removed
    # itself anyway would make the flag mean the opposite of what it says.
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = LocalJobExecutor(workers=1, cleanup=False)
    workdir = executor.workdir
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        assert await asyncio.to_thread(_collect, submission) == [1]
        executor.cancel()
    await _wait_for_local_cleanup(executor)
    assert await asyncio.to_thread(workdir.exists)
    assert await asyncio.to_thread(lambda: list(workdir.glob("*.out"))) != []
    shutil.rmtree(workdir)


@pytest.mark.slow
@pytest.mark.timeout(30)
async def test_local_executor_restarts_in_a_new_directory() -> None:
    # The kept directory was handed to the user to read; writing a second run's
    # files into it would undo that.
    submission = Submission([WorkItem(function=_function, args=(0,))])
    executor = LocalJobExecutor(workers=1, cleanup=False)
    first = executor.workdir
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        executor.submit(submission)
        assert await asyncio.to_thread(_collect, submission) == [1]
        executor.cancel()
    await _wait_for_local_cleanup(executor)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        second = executor.workdir
        executor.cancel()
    await _wait_for_local_cleanup(executor)
    assert second != first
    assert await asyncio.to_thread(first.exists)
    shutil.rmtree(first)
    shutil.rmtree(second, ignore_errors=True)


@pytest.mark.timeout(30)
async def test_local_executor_restarts_while_the_previous_teardown_waits() -> None:
    # A restart replaces the queue and the directory while the previous teardown
    # thread may still be waiting on a job that is slow to die. That thread must
    # finish its own run: taking either from the executor would leave it stuck on
    # a queue whose sentinel is gone, holding a directory that never goes, and
    # would hand the run that follows a directory that is about to be removed.
    executor = LocalJobExecutor(workers=1)
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        first = executor.workdir
        # Stands in for a job that outlives the signal that cancelled it: it
        # exits when its stdin is closed, so the test says when it dies.
        blocker = await asyncio.to_thread(_start_blocking_process)
        executor._teardown_queue.put(blocker)  # ruff: ignore[private-member-access]
        executor.cancel()
    waiting = executor._teardown_thread  # ruff: ignore[private-member-access]
    assert waiting is not None
    async with asyncio.TaskGroup() as tg:
        await executor.start(tg)
        second = executor.workdir
        executor.cancel()
    await _wait_for_local_cleanup(executor)
    assert second != first
    assert await asyncio.to_thread(first.exists)
    assert blocker.stdin is not None
    blocker.stdin.close()
    await asyncio.to_thread(waiting.join, 10.0)
    assert not waiting.is_alive()
    assert not await asyncio.to_thread(first.exists)


async def test_local_executor_refuses_a_missing_directory(tmp_path: Path) -> None:
    await asyncio.sleep(0)  # this module runs tests on the event loop
    with pytest.raises(ValueError, match="does not exist"):
        LocalJobExecutor(workdir=tmp_path / "nowhere")


async def test_local_executor_refuses_a_non_posix_system(monkeypatch: Any) -> None:
    monkeypatch.setattr(os, "name", "nt")
    await asyncio.sleep(0)  # this module runs tests on the event loop
    with pytest.raises(ExecutionError, match="POSIX"):
        LocalJobExecutor()
