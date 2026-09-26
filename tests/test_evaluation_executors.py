from __future__ import annotations

# ruff: file-ignore[unused-function-argument, unused-method-argument, unused-lambda-argument, no-self-use, mutable-class-default, multiple-with-statements, private-member-access, subprocess-without-shell-equals-true]
import collections
import gc
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
import tempfile
import threading
import warnings
import weakref
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

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
    ExecutorFailure,
    HPCExecutor,
    LocalJobExecutor,
    ProcessExecutor,
    ThreadExecutor,
    WorkItem,
)
from ropt.components.executors.__main__ import run_task
from ropt.components.executors._job_executor import (
    JobExecutorBase,
    _StateUpdate,
)
from ropt.components.executors._picklable import picklable_exception
from ropt.components.executors._process_executor import _run_payload
from ropt.context import EnOptContext
from ropt.exceptions import ExecutionError, ExecutorStopped, WorkflowError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from ropt.components.executors import Executor
    from ropt.components.executors.base import ExecutorBase
    from ropt.results import FunctionResults

try:
    import pysqa

    _TEST_HPC = True
except ImportError:
    _TEST_HPC = False


pytestmark = pytest.mark.timeout(5)


def _function(input_value: int, *, raise_error: bool = False) -> int:
    if raise_error:
        msg = f"Test error in function {input_value}"
        raise ValueError(msg)
    return input_value + 1


def _raise_unpicklable_error(_input: int) -> int:
    raise ValueError(threading.Lock())


def _return_unpicklable() -> Any:
    return threading.Lock()


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


_GRANDCHILD_SOURCE = """
import sys
from multiprocessing.connection import Client

Client(sys.argv[1]).recv()
"""


def _spawn_child_and_block(address: str) -> int:
    child = subprocess.Popen([sys.executable, "-c", _GRANDCHILD_SOURCE, address])
    child.wait()
    return 0


def _print_and_die(value: int) -> int:
    # Killed outright, so nothing is written back and the print is the only
    # account of what the job was doing.
    print("about to be killed", flush=True)  # ruff: ignore[print]
    os.kill(os.getpid(), signal.SIGKILL)
    return value


def _wait_for_local_cleanup(executor: LocalJobExecutor) -> None:
    thread = executor._teardown_thread
    thread.join(10.0)


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
def test_executor_run_returns_results_in_input_order(
    executor_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [WorkItem(function=_function, args=(idx,)) for idx in range(2)]
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
    with executor:
        assert executor.run(items) == [1, 2]


def test_submitting_from_a_worker_thread_is_refused() -> None:
    # Waiting here would occupy a worker while waiting for one, so the executor
    # refuses rather than deadlock once every worker is busy. With a free worker
    # left, dropping the refusal returns "accepted" instead of hanging.
    executor = ThreadExecutor(workers=2)

    def _submit_back() -> str:
        try:
            executor.run([WorkItem(function=_function, args=(0,))])
        except WorkflowError as exc:
            return str(exc)
        return "accepted"

    with executor:
        message = executor.run([WorkItem(function=_submit_back)])[0]
    assert "already running on it" in message


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_work_in_flight_aborted_on_close(tmp_path: Path) -> None:
    listener = Listener(str(tmp_path / "work"))
    executor = LocalJobExecutor(workdir=tmp_path, workers=1)
    connection = None
    outcome: list[BaseException] = []

    def _run() -> None:
        try:
            executor.run(
                [WorkItem(function=_block_until_disconnected, args=(listener.address,))]
            )
        except ExecutorStopped as exc:
            outcome.append(exc)

    try:
        with executor:
            runner = threading.Thread(target=_run)
            runner.start()
            connection = listener.accept()
            executor.close()
        runner.join(5.0)
        assert not runner.is_alive()
        assert isinstance(outcome[0], ExecutorStopped)
    finally:
        if connection is not None:
            connection.close()
        listener.close()


def test_stopping_thread_executor_reports_running_work(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Nothing can take a work item away from a thread, so the only help the user
    # gets is being told what the program is waiting for.
    barrier = threading.Barrier(3)
    release = threading.Event()
    items = [
        WorkItem(function=_blocked_work_at_barrier, args=(barrier, release))
        for _ in range(2)
    ]
    executor = ThreadExecutor(workers=2)
    runner = threading.Thread(target=lambda: executor.run(items), daemon=True)
    with caplog.at_level(logging.WARNING, logger="ropt"), executor:
        runner.start()
        barrier.wait(timeout=4.0)
        executor.close()
        release.set()
        runner.join(5.0)
    assert "Closing with 2 evaluation(s) still running" in caplog.text


def test_stopping_thread_executor_after_work_reports_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A work item that already returned is not something to wait for.
    executor = ThreadExecutor(workers=1)
    with caplog.at_level(logging.WARNING, logger="ropt"), executor:
        assert executor.run([WorkItem(function=_function, args=(0,))]) == [1]
    assert "still running" not in caplog.text


def _resource_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(entry.message)
        for entry in caught
        if issubclass(entry.category, ResourceWarning)
    ]


def _teardown_threads() -> set[threading.Thread]:
    return {
        thread
        for thread in threading.enumerate()
        if thread.name == "ropt-local-teardown"
    }


@pytest.mark.skipif(os.name != "posix", reason="local jobs are POSIX only")
def test_dropping_a_local_executor_removes_its_temporary_directory() -> None:
    # `__del__` must not be reachable from anything the executor keeps alive, so
    # what this checks is that dropping the last reference is enough to run it.
    executor = LocalJobExecutor(workers=1)
    workdir = executor.workdir
    thread = executor._teardown_thread
    assert workdir.is_dir()
    ref = weakref.ref(executor)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del executor
        gc.collect()
    assert ref() is None
    assert _resource_warnings(caught) == [
        "LocalJobExecutor was not closed; releasing its resources."
    ]
    thread.join(10.0)
    assert not thread.is_alive()
    assert not workdir.exists()


@pytest.mark.skipif(os.name != "posix", reason="local jobs are POSIX only")
def test_dropping_a_local_executor_keeps_a_directory_it_was_given(
    tmp_path: Path,
) -> None:
    executor = LocalJobExecutor(workdir=tmp_path, workers=1)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        del executor
        gc.collect()
    assert tmp_path.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="local jobs are POSIX only")
def test_local_executor_rejecting_an_argument_leaves_no_thread_or_directory() -> None:
    # The teardown thread starts only after the base has accepted every
    # argument, and the temporary directory goes if the base rejects one.
    temp_root = Path(tempfile.gettempdir())
    directories = set(temp_root.glob("ropt-local-*"))
    threads = _teardown_threads()
    with pytest.raises(ValueError, match="at least one"):
        LocalJobExecutor(workers=0)
    assert set(temp_root.glob("ropt-local-*")) == directories
    assert _teardown_threads() == threads


@pytest.mark.skipif(os.name != "posix", reason="local jobs are POSIX only")
def test_closing_a_local_executor_leaves_nothing_to_release() -> None:
    executor = LocalJobExecutor(workers=1)
    executor.close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del executor
        gc.collect()
    assert _resource_warnings(caught) == []


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_reads_job_ids_from_the_scheduler_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `pysqa` answers with a table, and exactly one line in ropt unwraps it.
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
        lambda *args, **kwargs: _TableScheduler(tmp_path),
    )
    with HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="") as executor:
        assert executor.run([WorkItem(function=_function, args=(0,))]) == [1]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_scheduler_query_fails_after_retry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    with HPCExecutor(
        workdir=tmp_path,
        workers=1,
        interval=0,
        retries=0,
        query_retries=2,
        template="",
    ) as executor:
        result = executor.run([WorkItem(function=_function, args=(0,))])[0]
    assert isinstance(result, ExecutorFailure)
    assert "could not be queried" in result.message
    assert "after 3 attempts" in result.message
    assert _UnreachableScheduler.queries == 3


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_scheduler_query_budget_resets_after_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The budget is for a run of failures, not for failures in total.
    class _FlakyScheduler(MockedHPCAdapter):
        calls = 0

        def live_job_ids(self) -> set[int]:
            type(self).calls += 1
            if type(self).calls in {1, 3}:
                msg = "squeue: error: Unable to contact slurm controller"
                raise RuntimeError(msg)
            if type(self).calls == 2:
                return set(self._jobs)
            return super().live_job_ids()

    _mock_scheduler(monkeypatch, _FlakyScheduler(tmp_path))
    with HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, query_retries=1, template=""
    ) as executor:
        assert executor.run([WorkItem(function=_function, args=(0,))]) == [1]
    assert _FlakyScheduler.calls >= 4


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_missing_output_file_fails_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A job that dies before writing its result: the scheduler reports it gone,
    # but there is nothing to read, so waiting forever is not an option.
    class _VanishingJob(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

        def live_job_ids(self) -> set[int]:
            return set()

    _mock_scheduler(monkeypatch, _VanishingJob(tmp_path))
    with HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=2, template=""
    ) as executor:
        result = executor.run([WorkItem(function=_function, args=(0,))])[0]
    assert isinstance(result, ExecutorFailure)
    assert "never appeared" in result.message


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_unreadable_output_file_fails_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A result read while it is still being written: retrying is right, but it
    # has to give up eventually rather than retry for ever.
    class _CorruptResult(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            (self._path / f"{job_name}.out").write_bytes(b"half a pickle")
            return self._job_id

        def live_job_ids(self) -> set[int]:
            return set()

    _mock_scheduler(monkeypatch, _CorruptResult(tmp_path))
    with HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=2, template=""
    ) as executor:
        result = executor.run([WorkItem(function=_function, args=(0,))])[0]
    assert isinstance(result, ExecutorFailure)
    assert "No valid result" in result.message


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_result_of_an_unknown_type_fails_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A result naming something this process cannot import is whole and correct;
    # retrying would postpone the same failure and then blame the filesystem.
    class _AlienResult(MockedHPCAdapter):
        polls = 0

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            payload = pickle.dumps(collections.OrderedDict, protocol=4)
            (self._path / f"{job_name}.out").write_bytes(
                payload.replace(b"collections", b"collectionx")
            )
            return self._job_id

        def live_job_ids(self) -> set[int]:
            type(self).polls += 1
            return set()

    _mock_scheduler(monkeypatch, _AlienResult(tmp_path))
    with HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=30, template=""
    ) as executor:
        result = executor.run([WorkItem(function=_function, args=(0,))])[0]
    assert isinstance(result, ExecutorFailure)
    assert "could not be reconstructed" in result.message
    assert "collectionx" in result.message


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_result_that_cannot_be_rebuilt_fails_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reading a result runs the code that rebuilds it, and that code can raise
    # anything at all. It belongs to the work item; the executor has to survive.
    class _UnrebuildableResult(MockedHPCAdapter):
        polls = 0

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            (self._path / f"{job_name}.out").write_bytes(pickle.dumps(_Unrebuildable()))
            return self._job_id

        def live_job_ids(self) -> set[int]:
            type(self).polls += 1
            return set()

    _mock_scheduler(monkeypatch, _UnrebuildableResult(tmp_path))
    with HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=30, template=""
    ) as executor:
        result = executor.run([WorkItem(function=_function, args=(0,))])[0]
    assert isinstance(result, ExecutorFailure)
    assert "could not be read" in result.message
    assert "this result cannot be rebuilt" in result.message


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_job_command_uses_submitting_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bare `python` resolves through the job's PATH, which need not be the
    # environment ropt is installed in.
    commands: list[str] = []

    class _RecordingAdapter(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            commands.append(command)
            return super().submit_job(job_name, command, **kwargs)

    _mock_scheduler(monkeypatch, _RecordingAdapter(tmp_path))
    with HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="") as executor:
        assert executor.run([WorkItem(function=_function, args=(0,))]) == [1]
    assert commands
    assert commands[0].startswith(f"{sys.executable} -m ropt.components.executors ")


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_failed_work_keeps_job_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A job that died before writing a result left its reason in the captured
    # output alone, so cleanup must not take that away with the rest.
    class _CrashingJob(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            (self._path / f"{job_name}.txt").write_text(
                "ModuleNotFoundError: No module named 'ropt'\n"
            )
            return self._job_id

        def live_job_ids(self) -> set[int]:
            return set()

    _mock_scheduler(monkeypatch, _CrashingJob(tmp_path))
    with HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=2, template=""
    ) as executor:
        result = executor.run([WorkItem(function=_function, args=(0,))])[0]
    assert isinstance(result, ExecutorFailure)
    assert "No module named 'ropt'" in result.message
    assert list(tmp_path.glob("*.txt"))
    assert not list(tmp_path.glob("*.in"))


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
@pytest.mark.parametrize("captured", ["", "   \n\n"], ids=["absent", "blank"])
def test_hpc_failure_names_the_output_file_it_could_not_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured: str
) -> None:
    # A shared filesystem need not show the output yet. Saying nothing would
    # leave the one place worth looking unnamed.
    class _SilentJob(MockedHPCAdapter):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.submitted = ""

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            self.submitted = job_name
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            if captured:
                (self._path / f"{job_name}.txt").write_text(captured)
            return self._job_id

        def live_job_ids(self) -> set[int]:
            return set()

    adapter = _SilentJob(tmp_path)
    _mock_scheduler(monkeypatch, adapter)
    with HPCExecutor(
        workdir=tmp_path, workers=1, interval=0, retries=0, template=""
    ) as executor:
        result = executor.run([WorkItem(function=_function, args=(0,))])[0]
    assert isinstance(result, ExecutorFailure)
    assert str(tmp_path / f"{adapter.submitted}.txt") in result.message


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_stopping_hpc_executor_cancels_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = threading.Event()
    cancelled = threading.Event()

    class _StuckAdapter(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
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
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    outcome: list[BaseException] = []

    def _run() -> None:
        try:
            executor.run([WorkItem(function=_function, args=(0,))])
        except ExecutorStopped as exc:
            outcome.append(exc)

    with executor:
        runner = threading.Thread(target=_run)
        runner.start()
        assert submitted.wait(timeout=5.0)
        executor.close()
    runner.join(5.0)
    assert isinstance(outcome[0], ExecutorStopped)
    assert cancelled.wait(timeout=5.0)
    assert adapter.deleted == [1]
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_job_submitted_during_close_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitting = threading.Event()
    stopped = threading.Event()

    class _SlowAdapter(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            submitting.set()
            stopped.wait(timeout=5.0)
            self._job_id += 1
            self._jobs[self._job_id] = job_name
            return self._job_id

    adapter = _SlowAdapter(tmp_path)
    _mock_scheduler(monkeypatch, adapter)
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    outcome: list[BaseException] = []

    def _run() -> None:
        try:
            executor.run([WorkItem(function=_function, args=(0,))])
        except ExecutorStopped as exc:
            outcome.append(exc)

    with executor:
        runner = threading.Thread(target=_run)
        runner.start()
        assert submitting.wait(timeout=5.0)
        executor.close()
    stopped.set()
    runner.join(5.0)
    assert isinstance(outcome[0], ExecutorStopped)
    assert adapter.deleted == [1]
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_queued_hpc_work_resumes_on_free_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
    items = [WorkItem(function=_function, args=(idx,)) for idx in range(4)]
    with HPCExecutor(
        workdir=tmp_path, workers=1, interval=0.01, template=""
    ) as executor:
        assert executor.run(items) == [1, 2, 3, 4]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_executor_refuses_to_overwrite_existing_work_item_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
    executor = HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="")
    item_id = uuid4()
    (tmp_path / f"{item_id}.out").touch()
    with executor:
        with pytest.raises(ExecutionError, match="already exist"):
            executor._launch_job(item_id, [WorkItem(function=_function, args=(0,))])


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_failing_submission_fails_own_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _RejectsFirst(MockedHPCAdapter):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.rejected = False

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            if not self.rejected:
                self.rejected = True
                msg = "sbatch: error: Batch job submission failed"
                raise RuntimeError(msg)
            return super().submit_job(job_name, command, **kwargs)

    _mock_scheduler(monkeypatch, _RejectsFirst(tmp_path))
    with HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="") as executor:
        with pytest.raises(RuntimeError, match="submission failed"):
            executor.run([WorkItem(function=_function, args=(0,))])
        assert executor.run([WorkItem(function=_function, args=(1,))]) == [2]


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_rejected_hpc_submission_leaves_no_input_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The input file is written before the job is handed over, so a scheduler
    # that rejects it would otherwise block a retry under the same name.
    class _RejectingScheduler(MockedHPCAdapter):
        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            msg = "sbatch: error: Batch job submission failed"
            raise RuntimeError(msg)

    _mock_scheduler(monkeypatch, _RejectingScheduler(tmp_path))
    with HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="") as executor:
        with pytest.raises(RuntimeError, match="submission failed"):
            executor.run([WorkItem(function=_function, args=(0,))])
    assert not list(tmp_path.iterdir())


def test_worker_records_the_worker_traceback_as_a_note() -> None:
    payload = dumps((partial(_function, 0, raise_error=True), (), {}))
    ok, blob = _run_payload(payload)
    assert not ok
    exc = loads(blob)
    assert isinstance(exc, ValueError)
    assert any("Traceback" in note for note in exc.__notes__)


def test_worker_wraps_an_unpicklable_exception() -> None:
    payload = dumps((_raise_unpicklable_error, (0,), {}))
    ok, blob = _run_payload(payload)
    assert not ok
    exc = loads(blob)
    assert isinstance(exc, RuntimeError)
    assert any("Traceback" in note for note in exc.__notes__)


def test_a_result_that_cannot_be_sent_is_not_blamed_on_the_function() -> None:
    # The call itself succeeded and only its result could not be serialized.
    payload = dumps((_return_unpicklable, (), {}))
    ok, blob = _run_payload(payload)
    assert not ok
    exc = loads(blob)
    assert any("Could not send the result back" in note for note in exc.__notes__)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_job_exit_writes_output_file(tmp_path: Path) -> None:
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(dumps((_exit_task, (), {})))
    assert run_task(str(input_file), str(output_file)) == 1
    assert isinstance(loads(output_file.read_bytes()), SystemExit)


def test_job_wraps_an_exception_the_standard_library_cannot_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `cloudpickle` is optional on the job path, so whether an exception can
    # travel has to be decided with the serializer that will do the sending.
    monkeypatch.setattr("ropt.components.executors._picklable.dumps", pickle.dumps)
    monkeypatch.setattr("ropt.components.executors.__main__.dump", pickle.dump)
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(pickle.dumps((_raise_locally_defined_error, (), {})))
    assert run_task(str(input_file), str(output_file)) == 1
    result = pickle.loads(output_file.read_bytes())  # ruff: ignore[suspicious-pickle-usage]
    assert isinstance(result, RuntimeError)
    assert "_LocalError" in str(result)
    assert any("Traceback" in note for note in result.__notes__)


def _vanishing_work_item() -> int:
    return 1


def _vanishing_job_task() -> int:
    return 1


def test_a_work_item_the_worker_cannot_rebuild_reports_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # It pickles, because the name resolves here. The worker is where it does
    # not, which is why no send-side check can catch this.
    payload = pickle.dumps((_vanishing_work_item, (), {}))
    monkeypatch.delattr(sys.modules[__name__], "_vanishing_work_item")
    ok, blob = _run_payload(payload)
    assert not ok
    exc = loads(blob)
    assert isinstance(exc, AttributeError)
    assert any("Could not rebuild the work item" in note for note in exc.__notes__)


def test_a_task_the_job_cannot_rebuild_reports_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same failure on the job path, where it matters most: the job has no
    # channel back, so the note in the result file is the whole diagnosis.
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(pickle.dumps((_vanishing_job_task, (), {})))
    monkeypatch.delattr(sys.modules[__name__], "_vanishing_job_task")
    assert run_task(str(input_file), str(output_file)) == 1
    result = loads(output_file.read_bytes())
    assert isinstance(result, AttributeError)
    assert any("Could not rebuild the task" in note for note in result.__notes__)


def test_the_hint_stays_off_the_task_s_own_exception(tmp_path: Path) -> None:
    # The load and the call are separated precisely so this exception, which is
    # the task's own, carries no advice about rebuilding it.
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(dumps((partial(_function, 0, raise_error=True), (), {})))
    assert run_task(str(input_file), str(output_file)) == 1
    result = loads(output_file.read_bytes())
    assert isinstance(result, ValueError)
    assert not any("Could not rebuild" in note for note in result.__notes__)


def test_a_result_that_cannot_be_written_still_reaches_the_executor(
    tmp_path: Path,
) -> None:
    # The task ran to completion; only its result could not be written.
    input_file = tmp_path / "job.in"
    output_file = tmp_path / "job.out"
    input_file.write_bytes(dumps((_return_unpicklable, (), {})))
    assert run_task(str(input_file), str(output_file)) == 1
    result = loads(output_file.read_bytes())
    assert any("Could not send the result back" in note for note in result.__notes__)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["job.in", "job.out"]


@pytest.mark.skipif(not HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
def test_job_sends_an_exception_the_standard_library_cannot_send() -> None:
    # The same exception, with `cloudpickle` present: it survives as itself.
    try:
        _raise_locally_defined_error()
    except Exception as exc:  # ruff: ignore[blind-except]
        result = picklable_exception(exc)
    assert type(result).__name__ == "_LocalError"


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_unserializable_work_item_fails_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Serializing happens before the job exists, so this failure belongs to the
    # work item, and it says what to install rather than what broke inside.
    _mock_scheduler(monkeypatch, MockedHPCAdapter(tmp_path))
    with HPCExecutor(workdir=tmp_path, workers=1, interval=0, template="") as executor:
        with pytest.raises(ExecutionError, match="could not be sent to a job"):
            executor.run([WorkItem(function=_function, args=(threading.Lock(),))])
    assert not list(tmp_path.iterdir())


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_worker_may_construct_workflow_objects() -> None:
    with ProcessExecutor(workers=1) as executor:
        assert executor.run(
            [WorkItem(function=_construct_handler_in_worker, args=(0,))]
        ) == ["ResultsHandler"]


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_max_tasks_per_child_restarts_worker() -> None:
    items = [WorkItem(function=_worker_pid, args=(index,)) for index in range(3)]
    with ProcessExecutor(workers=1, max_tasks_per_child=1) as executor:
        collected = executor.run(items)
    assert len(set(collected)) == 3


@pytest.mark.slow
@pytest.mark.timeout(30)
@pytest.mark.parametrize("public_api", [True, False])
def test_stopping_kills_a_busy_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, public_api: bool
) -> None:
    # An idle worker leaves on its own when the pool shuts down, so only a
    # worker busy with work that never ends shows whether stopping stops it.
    if not public_api:
        monkeypatch.delattr(ProcessPoolExecutor, "terminate_workers", raising=False)
    listener = Listener(str(tmp_path / "worker"))
    executor = ProcessExecutor(workers=1)
    connection = None
    try:
        outcome: list[BaseException] = []

        def _run() -> None:
            try:
                executor.run(
                    [
                        WorkItem(
                            function=_block_until_disconnected, args=(listener.address,)
                        )
                    ]
                )
            except ExecutorStopped as exc:
                outcome.append(exc)

        with executor:
            runner = threading.Thread(target=_run)
            runner.start()
            connection = listener.accept()
            executor.close()
        assert connection.poll(10.0)
        with pytest.raises(EOFError):
            connection.recv()
        runner.join(5.0)
        assert isinstance(outcome[0], ExecutorStopped)
    finally:
        if connection is not None:
            connection.close()
        listener.close()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_dying_worker_reported_as_infrastructure_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A worker lost while the executor is running is not something it asked for,
    # so it is reported, unlike the workers a stop kills on purpose.
    with caplog.at_level(logging.WARNING, logger="ropt"):
        with ProcessExecutor(workers=1) as executor:
            result = executor.run([WorkItem(function=_kill_own_process)])[0]
    assert isinstance(result, ExecutorFailure)
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
def test_executor_error(
    executor_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [
        WorkItem(function=_function, args=(idx,), kwargs={"raise_error": True})
        for idx in range(2)
    ]
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
    with executor:
        with pytest.raises(ValueError, match="Test error in function") as excinfo:
            executor.run(items)
        if executor_name in {"hpc", "multiprocessing"}:
            notes = getattr(excinfo.value, "__notes__", [])
            assert any("Traceback" in note for note in notes)


initial_values = np.array([0.0, 0.0, 0.1])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {"max_functions": 8},
        "backend": {"convergence_tolerance": 1e-2},
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.001,
        },
        "gradient": {"number_of_perturbations": 3},
        "objectives": {"weights": [0.75, 0.25]},
    }


def _opt_function(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    test_functions: Sequence[
        Callable[[NDArray[np.float64], EvaluationFunctionContext], float]
    ],
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

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
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

    def _mock_scheduler(
        monkeypatch: pytest.MonkeyPatch, adapter: MockedHPCAdapter
    ) -> None:
        monkeypatch.setattr(
            "ropt.components.executors._hpc_executor.pysqa.QueueAdapter",
            lambda *args, **kwargs: adapter,
        )
        monkeypatch.setattr(
            HPCExecutor, "_live_job_ids", lambda _self: adapter.live_job_ids()
        )

else:

    class MockedHPCAdapter:  # type: ignore[no-redef]
        def __init__(self, path: Path) -> None:
            raise RuntimeError(path)

    def _mock_scheduler(
        monkeypatch: pytest.MonkeyPatch, adapter: MockedHPCAdapter
    ) -> None:
        raise RuntimeError((monkeypatch, adapter))


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
def test_executor_evaluator_ok(
    config: dict[str, Any],
    eval_func: Any,
    executor_name: str,
    monkeypatch: pytest.MonkeyPatch,
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
    with executor:
        results = _opt_workflow(executor, config, eval_func())
    assert results is not None
    assert np.allclose(results.variables, [0.0, 0.0, 0.5], atol=0.02)


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
def test_executor_evaluator_error(
    config: dict[str, Any],
    test_functions: Sequence[
        Callable[[NDArray[np.float64], EvaluationFunctionContext], float]
    ],
    executor_name: str,
    monkeypatch: pytest.MonkeyPatch,
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
    with executor, pytest.raises(ValueError, match="Test error in function"):
        _opt_workflow(
            executor,
            config,
            partial(_opt_function, test_functions=test_functions, raise_error=True),
        )


@pytest.mark.parametrize(
    "executor_name",
    ["threading", pytest.param("multiprocessing", marks=pytest.mark.slow)],
)
def test_executor_survives_user_code_error_and_is_reusable(
    config: dict[str, Any],
    test_functions: Sequence[
        Callable[[NDArray[np.float64], EvaluationFunctionContext], float]
    ],
    eval_func: Any,
    executor_name: str,
) -> None:
    match executor_name:
        case "threading":
            executor: ExecutorBase = ThreadExecutor(workers=2)
        case "multiprocessing":
            executor = ProcessExecutor(workers=2)
    with executor:
        # A user-code error aborts only its own evaluation; the executor stays usable.
        with pytest.raises(ValueError, match="Test error in function"):
            _opt_workflow(
                executor,
                config,
                partial(_opt_function, test_functions=test_functions, raise_error=True),
            )
        results = _opt_workflow(executor, config, eval_func())
    assert results is not None
    assert np.allclose(results.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_error_escaping_the_body_closes_the_executor(
    config: dict[str, Any],
    test_functions: Sequence[
        Callable[[NDArray[np.float64], EvaluationFunctionContext], float]
    ],
) -> None:
    # No explicit close: an error escaping the block must still close the executor.
    executor = ThreadExecutor(workers=2)
    with pytest.raises(ValueError, match="Test error in function"), executor:
        _opt_workflow(
            executor,
            config,
            partial(_opt_function, test_functions=test_functions, raise_error=True),
        )
    assert executor.closed


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
def test_executor_evaluator_two_optimizations(
    config: dict[str, Any],
    eval_func: Any,
    executor_name: str,
    monkeypatch: pytest.MonkeyPatch,
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
    results_list: list[FunctionResults | None] = []

    def _run_once() -> None:
        results_list.append(_opt_workflow(executor, config, eval_func()))

    with executor:
        threads = [threading.Thread(target=_run_once) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30.0)
            assert not thread.is_alive()

    assert len(results_list) == 2
    for results in results_list:
        assert results is not None
        assert np.allclose(results.variables, [0.0, 0.0, 0.5], atol=0.02)


class _RecordingExecutor(ThreadExecutor):
    def __init__(self) -> None:
        super().__init__(workers=2)
        self.sizes: list[int] = []

    def _run_bundles(
        self, bundles: list[list[WorkItem]], store: Callable[[int, Any], None]
    ) -> None:
        self.sizes.extend(len(bundle) for bundle in bundles)
        super()._run_bundles(bundles, store)


@pytest.mark.parametrize("bundle_size", [1, 2, 4, 0])
def test_executor_bundles_work_items(bundle_size: int) -> None:
    executor = _RecordingExecutor()
    with executor:
        assert executor.run(
            [WorkItem(function=_function, args=(idx,)) for idx in range(5)],
            bundle_size=bundle_size,
        ) == [1, 2, 3, 4, 5]
    expected_max = max(executor.sizes) if bundle_size == 0 else bundle_size
    assert executor.sizes
    assert all(1 <= size <= expected_max for size in executor.sizes)


def test_invalid_bundle_size() -> None:
    with pytest.raises(ValueError, match="bundle_size"):
        ThreadExecutor(bundle_size=-1)
    with ThreadExecutor() as executor:
        with pytest.raises(ValueError, match="bundle_size"):
            executor.run([WorkItem(function=_function, args=(0,))], bundle_size=-1)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_relative_workdir_rejected() -> None:
    # The workdir is shared with the cluster nodes, which do not necessarily
    # share this process's working directory.
    with pytest.raises(ValueError, match="must be an absolute path"):
        HPCExecutor(workdir="relative/path", template="")


def test_broken_worker_pool_reported_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The same failure the unguarded-main test triggers out of process, checked
    # here without paying for real subprocesses.
    class _BrokenPool:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            # Closing the half-built executor reaches into these.
            self._shutdown_lock = threading.Lock()
            self._processes: dict[int, Any] = {}

        @staticmethod
        def submit(*_args: Any, **_kwargs: Any) -> None:
            raise BrokenProcessPool

        def shutdown(self, *_args: Any, **_kwargs: Any) -> None: ...

    monkeypatch.setattr(
        "ropt.components.executors._process_executor.ProcessPoolExecutor",
        _BrokenPool,
    )
    with pytest.raises(ExecutionError, match="guard the program entry point"):
        ProcessExecutor(workers=1)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_missing_workdir_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        HPCExecutor(workdir=tmp_path / "nowhere", template="")


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_unconfigured_hpc_executor_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No template and no usable pysqa configuration: there is nothing to submit
    # jobs with, so this must fail at construction rather than at submit time.
    monkeypatch.setattr(
        "ropt.components.executors._hpc_executor._get_config_path",
        lambda config_path: None,
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
def test_hpc_out_of_range_setting_rejected(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        HPCExecutor(workdir=tmp_path, template="", **kwargs)


@pytest.fixture
def pysqa_config(tmp_path: Path) -> Path:
    root = tmp_path / "pysqa"
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
    return root


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_executor_builds_from_a_configuration_directory(
    tmp_path: Path, pysqa_config: Path
) -> None:
    # The other branch of the constructor: every other test passes a template.
    executor = HPCExecutor(
        workdir=tmp_path, config_path=pysqa_config, cluster="cluster_b"
    )
    try:
        adapter = executor._queue_adapter
        assert adapter.list_clusters() == ["cluster_a", "cluster_b"]
        assert adapter.queue_list == ["bulk", "shared"]
    finally:
        executor.close()


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_executor_selects_the_cluster_holding_the_queue(
    tmp_path: Path, pysqa_config: Path
) -> None:
    executor = HPCExecutor(workdir=tmp_path, config_path=pysqa_config, queue="bulk")
    try:
        assert executor._queue_adapter.queue_list == ["bulk", "shared"]
    finally:
        executor.close()


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
def test_hpc_cluster_selection_rejects_what_it_cannot_resolve(
    tmp_path: Path, pysqa_config: Path, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ExecutionError, match=match):
        HPCExecutor(workdir=tmp_path, config_path=pysqa_config, **kwargs)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
@pytest.mark.parametrize("name", ["config_path", "cluster", "queue"])
def test_hpc_template_rejects_configuration_arguments(
    tmp_path: Path, pysqa_config: Path, name: str
) -> None:
    kwargs: dict[str, Any] = {
        name: pysqa_config if name == "config_path" else "cluster_a"
    }
    with pytest.raises(ValueError, match=f"cannot be combined with: {name}"):
        HPCExecutor(workdir=tmp_path, template="", **kwargs)


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_configuration_rejects_a_scheduler(
    tmp_path: Path, pysqa_config: Path
) -> None:
    with pytest.raises(ValueError, match="applies to a template only"):
        HPCExecutor(workdir=tmp_path, config_path=pysqa_config, scheduler="slurm")


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_template_uses_the_scheduler_it_is_given(tmp_path: Path) -> None:
    executor = HPCExecutor(workdir=tmp_path, template="", scheduler="lsf")
    try:
        adapter = executor._queue_adapter._adapter
        assert type(adapter._commands).__name__.lower().startswith("lsf")
    finally:
        executor.close()


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_pysqa_still_accepts_what_the_executor_passes_by_name() -> None:
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
        "memory_max",
        "run_time_max",
    ):
        assert name in signature.parameters, name
    assert "output" not in signature.parameters
    assert any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _scheduler_wrappers() -> dict[str, Any]:
    from pysqa.wrapper import abstract  # ruff: ignore[import-outside-top-level]

    package = importlib.import_module("pysqa.wrapper")
    wrappers: dict[str, Any] = {}
    for info in pkgutil.iter_modules(package.__path__):
        try:
            module = importlib.import_module(f"pysqa.wrapper.{info.name}")
        except ImportError:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, abstract.SchedulerCommands)
                and cls is not abstract.SchedulerCommands
                and cls.__module__ == module.__name__
            ):
                wrappers[info.name] = cls
    return wrappers


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_pysqa_names_the_job_id_column_jobid() -> None:
    # `_live_job_ids` indexes the result with `"jobid"`, so the name is a contract.
    checked = []
    for name, cls in _scheduler_wrappers().items():
        if "convert_queue_status" not in vars(cls):
            continue
        try:
            frame = cls.convert_queue_status("")
        except Exception:  # ruff: ignore[blind-except, try-except-continue]
            continue
        assert "jobid" in frame.columns, name
        checked.append(name)
    assert "slurm" in checked, checked


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_submit_options_reach_the_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _RecordingKwargs(MockedHPCAdapter):
        seen: dict[str, Any] = {}

        def submit_job(self, job_name: str, command: str, **kwargs: Any) -> int:
            type(self).seen = kwargs
            return super().submit_job(job_name, command, **kwargs)

    _mock_scheduler(monkeypatch, _RecordingKwargs(tmp_path))
    with HPCExecutor(
        workdir=tmp_path,
        workers=1,
        interval=0,
        template="",
        memory_max=8,
        run_time_max=600,
        submit_options={"account": "proj", "reservation": None},
    ) as executor:
        assert executor.run([WorkItem(function=_function, args=(1,))]) == [2]
    assert _RecordingKwargs.seen["memory_max"] == 8
    assert _RecordingKwargs.seen["run_time_max"] == 600
    assert _RecordingKwargs.seen["account"] == "proj"
    assert "reservation" not in _RecordingKwargs.seen


@pytest.mark.skipif(not _TEST_HPC, reason="hpc requirements are not installed")
def test_hpc_submit_options_refuse_what_the_executor_sets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="set by the executor itself: cores, queue"):
        HPCExecutor(
            workdir=tmp_path,
            template="",
            submit_options={"cores": 4, "queue": "fast", "account": "proj"},
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
def test_run_on_closed_executor_raises_workflow_error(
    executor_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    executor.close()
    executor.close()
    assert executor.closed
    with pytest.raises(WorkflowError, match="closed"):
        executor.run([WorkItem(function=_function, args=(0,))])


class _FatalError(BaseException):
    pass


def test_fatal_work_item_error_reaches_caller() -> None:
    # The subject still matters without a task group: a BaseException raised by
    # the work item must be the one observed by the caller.
    def _raise_fatal(input_value: int) -> int:
        msg = f"fatal {input_value}"
        raise _FatalError(msg)

    with ThreadExecutor(workers=1) as executor:
        with pytest.raises(_FatalError, match="fatal 0"):
            executor.run([WorkItem(function=_raise_fatal, args=(0,))])


def test_handle_result_raises_on_executor_failure() -> None:
    with pytest.raises(
        ExecutionError,
        match=r"An evaluation could not be run: the job wrote to item\.txt",
    ):
        _handle_result(
            0,
            ExecutorFailure("the job wrote to item.txt"),
            np.zeros((2, 1), dtype=np.float64),
            {},
            objective_count=1,
        )


def test_wrong_evaluation_result_type_rejected() -> None:
    with pytest.raises(WorkflowError, match="got str"):
        _handle_result(
            0,
            "not one",
            np.zeros((1, 1), dtype=np.float64),
            {},
            objective_count=1,
        )


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_multiprocessing_unguarded_main_reports_startup_error(tmp_path: Path) -> None:
    script = tmp_path / "unguarded.py"
    script.write_text(
        "from ropt.components.executors import ProcessExecutor\n\n"
        "executor = ProcessExecutor(workers=1)\n"
        "executor.close()\n"
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, check=False
    )
    assert proc.returncode != 0
    assert b"Could not start worker processes" in proc.stderr


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CLOUDPICKLE, reason="cloudpickle is not installed")
def test_multiprocessing_cloudpickles_functions_and_results() -> None:
    def make_adder(offset: int) -> Callable[[int], int]:
        def add(value: int) -> int:
            return value + offset

        return add

    def local_double(value: int) -> int:
        return value * 2

    def make_callable(value: int) -> Callable[[], int]:
        return lambda: value

    items = [
        WorkItem(function=lambda value: value + 100, args=(1,)),
        WorkItem(function=make_adder(10), args=(2,)),
        WorkItem(function=local_double, args=(3,)),
        WorkItem(function=make_callable, args=(42,)),
    ]
    with ProcessExecutor(workers=2) as executor:
        results = executor.run(items)
    assert sorted(value for value in results if isinstance(value, int)) == [6, 12, 101]
    returned = [value for value in results if callable(value)]
    assert len(returned) == 1
    assert returned[0]() == 42


@pytest.mark.slow
def test_multiprocessing_unserializable_payload_reports_error() -> None:
    lock = threading.Lock()

    def use_lock() -> Any:
        return lock

    with ProcessExecutor(workers=1) as executor:
        # The serialization failure is delivered as ours rather than as an
        # opaque pool error.
        with pytest.raises(ExecutionError, match="could not be sent"):
            executor.run([WorkItem(function=use_lock)])
        assert not executor.closed


@pytest.mark.slow
def test_multiprocessing_without_cloudpickle_rejects_lambda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ropt.components.executors._process_executor.dumps", pickle.dumps
    )
    with ProcessExecutor(workers=1) as executor:
        with pytest.raises(ExecutionError, match="could not be sent"):
            executor.run([WorkItem(function=lambda: 1)])
        assert not executor.closed


@pytest.mark.slow
def test_multiprocessing_without_cloudpickle_rejects_an_unpicklable_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ParallelEvaluator submits a picklable module-level function and passes the
    # user callback as an argument, so the arguments must be checked too.
    monkeypatch.setattr(
        "ropt.components.executors._process_executor.dumps", pickle.dumps
    )
    with ProcessExecutor(workers=1) as executor:
        with pytest.raises(ExecutionError, match="could not be sent"):
            executor.run([WorkItem(function=_call, args=(lambda: 1,))])


def _return_captured(_handler: Any) -> int:
    return 0


def _opt_function_capturing_handler(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    test_functions: Sequence[
        Callable[[NDArray[np.float64], EvaluationFunctionContext], float]
    ],
    handler: ResultsHandler,
) -> EvaluationFunctionResult:
    del handler
    return EvaluationFunctionResult(
        objectives=np.fromiter(
            (func(variables, context) for func in test_functions), dtype=np.float64
        )
    )


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_work_item_capturing_a_workflow_object_is_refused() -> None:
    # The handler holds a lock, so the work item cannot be serialized at all.
    with ProcessExecutor(workers=1) as executor:
        with pytest.raises(ExecutionError, match="could not be sent to a worker"):
            executor.run(
                [WorkItem(function=_return_captured, args=(ResultsHandler(),))]
            )
        assert not executor.closed


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_transfer_error_from_parallel_evaluation_bubbles_up(
    config: dict[str, Any],
    test_functions: Sequence[
        Callable[[NDArray[np.float64], EvaluationFunctionContext], float]
    ],
) -> None:
    with ProcessExecutor(workers=1) as executor, pytest.raises(ExecutionError):
        _opt_workflow(
            executor,
            config,
            partial(
                _opt_function_capturing_handler,
                test_functions=test_functions,
                handler=ResultsHandler(),
            ),
        )


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_local_jobs_evaluate_work() -> None:
    executor = LocalJobExecutor(workers=2)
    workdir = executor.workdir
    with executor:
        assert executor.run(
            [WorkItem(function=_function, args=(i,)) for i in range(4)]
        ) == [
            1,
            2,
            3,
            4,
        ]
    _wait_for_local_cleanup(executor)
    # Nothing failed and cleanup is on, so there is nothing in there to read.
    assert not workdir.exists()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_local_job_error_carries_its_traceback(tmp_path: Path) -> None:
    # The job is the only place the traceback existed, and it left no channel
    # back: it travels as a note on the exception or not at all.
    with LocalJobExecutor(workdir=tmp_path, workers=1) as executor:
        with pytest.raises(ValueError, match="Test error") as info:
            executor.run(
                [WorkItem(function=_function, args=(0,), kwargs={"raise_error": True})]
            )
    assert any("Traceback" in note for note in info.value.__notes__)


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_local_jobs_run_in_separate_processes() -> None:
    # The point of a job over a thread: its own interpreter, which is also what
    # makes it killable.
    executor = LocalJobExecutor(workers=2)
    with executor:
        collected = executor.run([WorkItem(function=os.getpid) for _ in range(2)])
    assert len(set(collected)) == 2
    assert os.getpid() not in collected
    _wait_for_local_cleanup(executor)


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_stopping_kills_a_local_job_and_its_children(tmp_path: Path) -> None:
    # A job that started a process of its own: stopping has to reach that too,
    # or it is orphaned and outlives the run that asked for it.
    listener = Listener(str(tmp_path / "job"))
    executor = LocalJobExecutor(workdir=tmp_path, workers=1)
    connection = None
    try:
        outcome: list[BaseException] = []

        def _run() -> None:
            try:
                executor.run(
                    [
                        WorkItem(
                            function=_spawn_child_and_block, args=(listener.address,)
                        )
                    ]
                )
            except ExecutorStopped as exc:
                outcome.append(exc)

        with executor:
            runner = threading.Thread(target=_run)
            runner.start()
            connection = listener.accept()
            executor.close()
        assert connection.poll(10.0)
        with pytest.raises(EOFError):
            connection.recv()
        runner.join(5.0)
        assert isinstance(outcome[0], ExecutorStopped)
    finally:
        if connection is not None:
            connection.close()
        listener.close()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_local_job_that_dies_without_a_result_fails_work(tmp_path: Path) -> None:
    # Killed outright, so nothing was written: the only account of the job is
    # what it printed, and that has to reach the caller.
    with LocalJobExecutor(workdir=tmp_path, workers=1) as executor:
        result = executor.run([WorkItem(function=_print_and_die, args=(0,))])[0]
    assert isinstance(result, ExecutorFailure)
    assert "never appeared" in result.message
    assert "about to be killed" in result.message


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_local_job_ids_are_not_pids(caplog: pytest.LogCaptureFixture) -> None:
    # Pids come round again, and job ids that were pids would come round with
    # them, letting a finished job be mistaken for one that is still running.
    executor = LocalJobExecutor(workers=1)
    with caplog.at_level(logging.DEBUG, logger="ropt"), executor:
        for value in range(3):
            assert executor.run([WorkItem(function=_function, args=(value,))]) == [
                value + 1
            ]
    started = [line for line in caplog.messages if line.startswith("Started local job")]
    assert [line.rsplit(" ", 1)[-1] for line in started] == ["1)", "2)", "3)"]
    _wait_for_local_cleanup(executor)


def test_local_executor_removes_its_directory_when_it_closes() -> None:
    # The directory belongs to the executor, so it goes when the executor does.
    # It is the teardown thread that removes it, once the jobs it waits for are
    # gone.
    with LocalJobExecutor() as executor:
        workdir = executor.workdir
        assert workdir.exists()
    _wait_for_local_cleanup(executor)
    assert not workdir.exists()


def test_local_executor_keeps_a_directory_it_was_given(tmp_path: Path) -> None:
    executor = LocalJobExecutor(workdir=tmp_path, workers=1, cleanup=False)
    executor.close()
    assert tmp_path.exists()


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_local_executor_keeps_its_directory_when_a_job_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Removing the directory would take the failed job's captured output with
    # it, which is the only account of why it failed.
    executor = LocalJobExecutor(workers=1)
    workdir = executor.workdir
    with caplog.at_level(logging.WARNING, logger="ropt"):
        with executor:
            result = executor.run([WorkItem(function=_print_and_die, args=(0,))])[0]
            assert isinstance(result, ExecutorFailure)
        _wait_for_local_cleanup(executor)
    assert workdir.exists()
    output = workdir / "".join(str(path.name) for path in workdir.glob("*.txt"))
    assert "about to be killed" in output.read_text()
    # A random name kept and never mentioned is a directory nobody can find.
    assert any(
        str(workdir) in message and "a work item failed" in message
        for message in caplog.messages
    )
    shutil.rmtree(workdir)


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_local_executor_keeps_its_directory_when_polling_gives_up() -> None:
    # Giving up on polling fails whatever was out, and those items never reach
    # the pass that removes their files, so this route has to keep the directory too.
    executor = LocalJobExecutor(workers=1)
    workdir = executor.workdir

    def _unreachable() -> set[int]:
        msg = "cannot tell whether the job is alive"
        raise RuntimeError(msg)

    executor._live_job_ids = _unreachable  # type: ignore[method-assign]
    with executor:
        result = executor.run([WorkItem(function=_function, args=(0,))])[0]
    _wait_for_local_cleanup(executor)
    assert isinstance(result, ExecutorFailure)
    assert "could not be queried" in result.message
    assert workdir.exists()
    assert executor._processes == {}
    shutil.rmtree(workdir)


@pytest.mark.slow
@pytest.mark.timeout(30)
def test_local_executor_keeps_its_directory_when_cleanup_is_off() -> None:
    # `cleanup=False` means nothing here is removed; a directory that removed
    # itself anyway would make the flag mean the opposite of what it says.
    executor = LocalJobExecutor(workers=1, cleanup=False)
    workdir = executor.workdir
    with executor:
        assert executor.run([WorkItem(function=_function, args=(0,))]) == [1]
    _wait_for_local_cleanup(executor)
    assert workdir.exists()
    assert list(workdir.glob("*.out")) != []
    shutil.rmtree(workdir)


def test_local_executor_refuses_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        LocalJobExecutor(workdir=tmp_path / "nowhere")


def test_local_executor_refuses_a_non_posix_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "name", "nt")
    with pytest.raises(ExecutionError, match="POSIX"):
        LocalJobExecutor()


@pytest.mark.skipif(os.name != "posix", reason="local jobs are POSIX only")
def test_local_executor_refuses_to_start_a_job_after_releasing(tmp_path: Path) -> None:
    # Starting and registering happen under one acquisition of the process lock,
    # which is also what publishes the release, so a job cannot become this
    # executor's after the teardown thread was told there would be no more: it
    # would never be waited for, and its directory could go while it writes.
    executor = LocalJobExecutor(workdir=tmp_path, workers=1)
    executor.close()
    with pytest.raises(ExecutorStopped):
        executor._start_job(uuid4(), [sys.executable, "-c", ""])
    assert not list(tmp_path.iterdir())


class _ControlledJobExecutor(JobExecutorBase):
    _query_retries = 2

    def __init__(
        self, workdir: Path, *, interval: float = 0.0, workers: int = 1
    ) -> None:
        super().__init__(
            workdir=workdir,
            workers=workers,
            interval=interval,
            retries=0,
            cleanup=True,
        )
        self.started: list[tuple[UUID, int]] = []
        self.cancelled: list[int] = []
        self.cancel_daemon: list[bool] = []
        self.passes = 0
        self.active_passes = 0
        self.max_active_passes = 0
        self.results: dict[UUID, Any] = {}
        self.raise_in_pass: BaseException | None = None
        self.pass_entered = threading.Event()
        self.release_pass = threading.Event()

    def _start_job(self, item_id: UUID, command: list[str]) -> int:
        raise AssertionError((item_id, command))

    def _launch_job(self, item_id: UUID, bundle: list[WorkItem]) -> int:
        job_id = len(self.started) + 1
        self.started.append((item_id, job_id))
        self.results[item_id] = [
            item.function(*item.args, **item.kwargs) for item in bundle
        ]
        return job_id

    def _live_job_ids(self) -> set[int]:
        return set()

    def _cancel_job(self, job_id: int) -> None:
        self.cancelled.append(job_id)
        self.cancel_daemon.append(threading.current_thread().daemon)

    def _launch_jobs(self, update: _StateUpdate) -> None:
        with self._state._lock:
            self.passes += 1
            self.active_passes += 1
            self.max_active_passes = max(self.max_active_passes, self.active_passes)
        self.pass_entered.set()
        self.release_pass.wait(timeout=5.0)
        try:
            if self.raise_in_pass is not None:
                error = self.raise_in_pass
                self.raise_in_pass = None
                raise error
            for item_id, _caller, _index, bundle in update.jobs_to_launch:
                update.launched_jobs[item_id] = self._launch_job(item_id, bundle)
                update.results[item_id] = self.results[item_id]
        finally:
            with self._state._lock:
                self.active_passes -= 1


def test_job_executor_cancels_on_a_thread_the_interpreter_waits_for(
    tmp_path: Path,
) -> None:
    # Cancelling runs on a thread of its own, so that a second interrupt breaks
    # the join rather than the cancelling. A new thread inherits the daemon flag
    # of the thread that creates it, and `run_concurrent` drives every run from
    # a daemon thread, which the interpreter would not wait for.
    executor = _ControlledJobExecutor(tmp_path)
    executor.release_pass.set()
    closed = threading.Event()

    def _close() -> None:
        with executor._state._lock:
            executor._state._jobs[uuid4()] = 1
        executor.close()
        closed.set()

    thread = threading.Thread(target=_close, daemon=True)
    thread.start()
    assert closed.wait(5.0)
    thread.join(5.0)
    assert not thread.is_alive()
    assert executor.cancelled == [1]
    assert executor.cancel_daemon == [False]


def test_job_executor_two_callers_share_one_backend(tmp_path: Path) -> None:
    # `_backend_in_use` lets only one caller reach the backend at a time; each
    # caller must still receive the result for its own batch.
    executor = _ControlledJobExecutor(tmp_path, workers=2)
    executor.release_pass.set()
    results: list[list[Any]] = []

    def _run(value: int) -> None:
        results.append(executor.run([WorkItem(function=_function, args=(value,))]))

    with executor:
        threads = [threading.Thread(target=_run, args=(idx,)) for idx in (0, 10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)
            assert not thread.is_alive()
    assert sorted(results) == [[1], [11]]
    assert executor.max_active_passes == 1


def test_job_executor_store_exception_releases_the_backend(tmp_path: Path) -> None:
    # Results are stored with the backend not reserved; if user code rejects a
    # result, later runs must not find `_backend_in_use` stranded.
    executor = _ControlledJobExecutor(tmp_path)
    executor.release_pass.set()

    def _reject(_index: int, _result: Any) -> None:
        msg = "caller rejected result"
        raise ValueError(msg)

    with executor:
        with pytest.raises(ValueError, match="caller rejected"):
            executor._run_bundles([[WorkItem(function=_function, args=(0,))]], _reject)
        assert executor.run([WorkItem(function=_function, args=(1,))]) == [2]


def test_job_executor_result_stored_during_a_wait_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A caller woken while waiting must take what it already has before trying
    # to claim the backend again, or it can sleep past a result it owns.
    executor = _ControlledJobExecutor(tmp_path, interval=100.0)
    state = executor._state
    caller: list[tuple[int, Any]] = []

    def _wait(timeout: float | None = None) -> bool:
        caller.append((0, [7]))
        return True

    monkeypatch.setattr(state._condition, "wait", _wait)
    with state._lock:
        state._backend_busy = True
    assert state.pop_results(caller) == []
    executor._launch_and_collect(caller)
    assert state.pop_results(caller) == [(0, [7])]
    assert executor.passes == 0


def test_job_executor_base_exception_releases_waiting_callers(tmp_path: Path) -> None:
    # A fatal backend pass must notify other callers after releasing the
    # backend, so they do not remain blocked behind a dead one.
    executor = _ControlledJobExecutor(tmp_path, workers=2)
    executor.raise_in_pass = _FatalError("boom")
    fatal: list[_FatalError] = []
    survivor: list[list[Any]] = []

    def _failing_run() -> None:
        try:
            executor.run([WorkItem(function=_function, args=(0,))])
        except _FatalError as exc:
            fatal.append(exc)

    def _surviving_run() -> None:
        survivor.append(executor.run([WorkItem(function=_function, args=(1,))]))

    with executor:
        first = threading.Thread(target=_failing_run)
        second = threading.Thread(target=_surviving_run)
        first.start()
        assert executor.pass_entered.wait(timeout=5.0)
        second.start()
        executor.release_pass.set()
        first.join(5.0)
        second.join(5.0)
        assert not first.is_alive()
        assert not second.is_alive()
    assert len(fatal) == 1
    assert survivor == [[2]]


def test_job_executor_unlaunched_item_fails_its_caller(tmp_path: Path) -> None:
    # Work given a slot before the backend is claimed must still release its
    # caller if the claim fails in that gap.
    executor = _ControlledJobExecutor(tmp_path)
    state = executor._state
    caller: list[tuple[int, Any]] = []
    update = _StateUpdate()
    with state._lock:
        state._queue.append((caller, 0, [WorkItem(function=_function, args=(0,))]))
        state._pick_jobs_to_launch(update)
    update.error = RuntimeError("claim failed")
    state.apply_update(update, release_backend=False)
    index, result = caller[0]
    assert index == 0
    assert isinstance(result, ExecutorFailure)
    assert "claim failed" in result.message


def test_job_executor_drops_late_results_after_a_caller_left(tmp_path: Path) -> None:
    # A caller that has left must not keep a list that a later claim can fill
    # with results nobody will read.
    executor = _ControlledJobExecutor(tmp_path)
    state = executor._state
    caller: list[tuple[int, Any]] = [(0, [1])]
    item_id = uuid4()
    with state._lock:
        state._active[item_id] = (caller, 0)
        state._jobs[item_id] = 7
    assert state.drop(caller) == [(item_id, 7)]
    state.apply_update(_StateUpdate(results={item_id: [2]}), release_backend=False)
    assert caller == []
    assert item_id not in state._active


def test_job_executor_failed_query_spends_one_retry_per_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After a failed query, the retry budget is paced by the polling interval;
    # a tight caller loop must not spend the whole budget at once.
    executor = _ControlledJobExecutor(tmp_path, interval=10.0)
    state = executor._state
    caller: list[tuple[int, Any]] = []
    waits: list[float | None] = []

    def _wait(timeout: float | None = None) -> bool:
        waits.append(timeout)
        return True

    monkeypatch.setattr(state._condition, "wait", _wait)
    with state._lock:
        state._query_failures = 1
        state._last_query = 100.0
    monkeypatch.setattr(
        "ropt.components.executors._job_executor.time.monotonic", lambda: 100.0
    )
    executor._launch_and_collect(caller)
    assert waits == [10.0]
    assert executor.passes == 0
    assert state._query_failures == 1


def test_job_executor_launch_only_claim_spends_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When a retry grace period is active, a claim that only launches new jobs
    # must not count as another scheduler query.
    executor = _ControlledJobExecutor(tmp_path, interval=10.0)
    state = executor._state
    executor.release_pass.set()
    clock = 100.0
    monkeypatch.setattr(
        "ropt.components.executors._job_executor.time.monotonic", lambda: clock
    )
    with state._lock:
        state._query_failures = 1
        state._last_query = clock
    with executor:
        results: list[Any] = [None]

        def _store(index: int, result: Any) -> None:
            results[index] = result

        executor._run_bundles([[WorkItem(function=_function, args=(0,))]], _store)
    assert results == [[1]]
    assert state._last_query == clock
    assert state._query_failures == 1
