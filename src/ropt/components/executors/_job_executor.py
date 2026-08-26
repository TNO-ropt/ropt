"""The machinery shared by executors that run work as separate processes.

Unlike the thread and process executors, there is no channel back from a job:
work travels over the filesystem, as a serialized `<id>.in` file the job reads
and an `<id>.out` file it writes, with `<id>.txt` for whatever it printed. The
job runs `ropt.components.executors` as a module, with the interpreter that
started it, which is the one `ropt` is installed in.

So there is nothing to await, only a backend to ask. One thread does all the
blocking work — starting jobs and asking after them — while the async worker
loop alternates between handing it work and waiting for more. A job that
disappeared without leaving a readable result is retried a bounded number of
times, because a shared filesystem may take a while to show it.

Two threads reach the backend, and which one matters. `_start_job` and
`_live_job_ids` run on the single poll thread, one after the other. `_cancel_job`
runs on the loop thread, because `_cleanup` does. Anything kept between those
calls is therefore shared by two threads and needs its own lock: `_jobs_closed`
is this module's instance of that problem, and a subclass that keeps state of
its own has the same obligation.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import threading
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pickle import UnpicklingError  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from ropt._logging import get_logger
from ropt.exceptions import ExecutionError, ExecutorFailure, WorkflowError

from ._serialize import CANNOT_SERIALIZE, dump, load
from .base import ExecutorBase, Submission, WorkItem

if TYPE_CHECKING:
    from uuid import UUID

_logger = get_logger(__name__)

# How much of a failed job's captured output travels with its failure.
_OUTPUT_TAIL_LINES: Final = 20

# Returned for a work item that has not settled: its result is either not there
# yet, or not whole yet with budget left to wait for the rest of it.
_PENDING: Final = object()


class JobExecutorBase(ExecutorBase):
    """Runs work items as separate processes that communicate through files.

    Subclasses decide what a job is: how one is started, how to tell which are
    still running, and how to cancel one. Everything between those three
    answers — the file layout, the poll loop, the worker cap, cancellation on
    shutdown — lives here.
    """

    # Names this kind of job in log messages, and the thing that runs them.
    _kind = "job"
    _backend = "backend"

    # Extra attempts to query the backend after one fails. A backend that can
    # have a bad moment raises this; one that cannot has nothing to wait for.
    _query_retries = 0

    def __init__(
        self,
        *,
        workdir: Path,
        workers: int,
        interval: float,
        retries: int,
        cleanup: bool,
    ) -> None:
        """Initialize the shared state.

        Subclasses validate and resolve `workdir` themselves, because where a
        job may read and write is the one thing they do not agree on.

        Args:
            workers:  Maximum number of jobs running at once.
            workdir:  Directory holding each work item's files.
            interval: Polling interval in seconds.
            retries:  Extra polls to wait for a result after the first attempt.
            cleanup:  Whether to remove work item files once they are done with.

        Raises:
            ValueError: If `workers`, `interval` or `retries` is out of range.
        """
        super().__init__()
        if workers < 1:
            msg = f"The number of workers must be at least one: {workers}"
            raise ValueError(msg)
        if interval < 0:
            msg = f"The polling interval must not be negative: {interval}"
            raise ValueError(msg)
        if retries < 0:
            msg = f"The number of retries must not be negative: {retries}"
            raise ValueError(msg)
        self._workdir = workdir
        self._workers = workers
        self._interval = interval
        self._retries_limit = retries
        self._remove_files = cleanup
        self._worker_task: asyncio.Task[None] | None = None
        self._pool: ThreadPoolExecutor | None = None

        self._items: dict[str | UUID, tuple[Submission, WorkItem]] = {}
        self._jobs: dict[str | UUID, int] = {}
        self._retries: dict[str | UUID, int] = {}
        # The started jobs are reached from the poll thread and from cleanup on
        # the loop thread; `_jobs_closed` closes the door between them, so a job
        # cannot be started after cleanup has passed it by.
        self._jobs_lock = threading.Lock()
        self._jobs_closed = False
        self._work_arrived = asyncio.Event()
        self._query_failures = 0
        # Set once a work item's captured output has been kept for the user to
        # read. A subclass that owns its working directory needs to know, since
        # removing the directory would take that output with it.
        self._output_kept = False

    @abstractmethod
    def _start_job(self, item_id: str | UUID, command: list[str]) -> int:
        """Start a job running `command` and return the id that identifies it.

        Runs on the poll thread. The job's output belongs in `<item_id>.txt` in
        the working directory, which is the only record of a job that died
        before writing a result.

        Args:
            item_id: The work item the job runs, and the stem of its files.
            command: The command line the job must run.

        Returns:
            An id that `_live_job_ids` and `_cancel_job` accept.
        """

    @abstractmethod
    def _live_job_ids(self) -> set[int]:
        """Return the ids of the jobs that have not finished yet.

        Runs on the poll thread. An id that is absent is taken to mean the job
        ended, whether it succeeded, failed or was killed; what became of it is
        read from its result file.

        Returns:
            The ids of the jobs still running.
        """

    @abstractmethod
    def _cancel_job(self, job_id: int) -> None:
        """Cancel a job that is still running.

        Runs on the loop thread, so it must not wait for the job to die: a
        Ctrl-C that has to wait for cancellation to finish is the thing this
        design avoids.

        Args:
            job_id: The job to cancel.
        """

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.
        """
        self._begin_start()
        self._work_arrived = asyncio.Event()
        with self._jobs_lock:
            self._jobs_closed = False
        # A new run answers this question again from scratch; the previous run's
        # verdict says nothing about the files this one will write.
        self._output_kept = False
        # A single thread, because it is the only one that talks to the backend:
        # starting and polling stay in one order, and off the loop thread.
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ropt-job-poll"
        )
        self._worker_task = task_group.create_task(self._worker(self._pool))
        _logger.info(
            "Starting %s executor (%d max workers, %.2fs poll interval)",
            self._kind,
            self._workers,
            self._interval,
        )
        await self._finish_start(task_group)

    async def _worker(self, pool: ThreadPoolExecutor) -> None:
        # Every backend call blocks, so the whole start-and-poll round trip is
        # handed to the single poll thread; this loop only decides when.
        loop = asyncio.get_running_loop()
        while self._running.is_set():
            pending = self._take_work_items()
            if self._items:
                results = await loop.run_in_executor(
                    pool, self._run_work_items, pending
                )
                self._deliver_results(results)
            await self._wait_for_work()

    async def _wait_for_work(self) -> None:
        self._work_arrived.clear()
        # There is room and work waiting: go round again without pausing.
        if len(self._items) < self._workers and not self._work_queue.empty():
            return
        # Poll while jobs are out, but wait indefinitely when there is nothing
        # to poll for, so an idle executor costs nothing.
        idle = not self._items and self._work_queue.empty()
        timeout = None if idle else self._interval
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._work_arrived.wait(), timeout)

    def _accept(self, submission: Submission) -> None:
        super()._accept(submission)
        self._work_arrived.set()

    def _take_work_items(self) -> list[tuple[str | UUID, WorkItem]]:
        # Takes no more than there is room for: `_items` holds the jobs that are
        # out, so `workers` caps how many run at once.
        pending: list[tuple[str | UUID, WorkItem]] = []
        while len(self._items) < self._workers:
            try:
                submission, work_item = self._work_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if submission.is_finished:
                # Its caller has already left, so this would be a job whose
                # result nobody reads.
                continue
            try:
                item_id = self._register(submission, work_item)
            except WorkflowError as exc:
                self._fail(submission, exc)
                continue
            pending.append((item_id, work_item))
        return pending

    def _register(self, submission: Submission, work_item: WorkItem) -> str | UUID:
        # The ID is the file name the job reads and writes, so a caller-given
        # name is refused if it is already in flight, rather than overwritten.
        item_id = work_item.name or uuid4()
        if item_id in self._items:
            msg = f"Work item ID '{item_id}' is already in use; names must be unique."
            raise WorkflowError(msg)
        self._items[item_id] = (submission, work_item)
        return item_id

    def _deliver_results(self, results: dict[str | UUID, Any]) -> None:
        for item_id, result in results.items():
            entry = self._items.pop(item_id, None)
            if entry is None:
                continue
            submission, work_item = entry
            if isinstance(result, Exception) and not isinstance(
                result, ExecutorFailure
            ):
                # Deliver to the evaluator; keep the executor alive (no raise).
                self._fail(submission, result)
            else:
                self._deliver(submission, work_item, result)

    def _cleanup(self) -> None:
        """Clean up the executor resources."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
        self._worker_task = None
        self._cancel_jobs()
        self._items.clear()
        self._cleanup_submissions()

    def _cancel_jobs(self) -> None:
        with self._jobs_lock:
            self._jobs_closed = True
            jobs = dict(self._jobs)
            self._jobs.clear()
        for item_id, job_id in jobs.items():
            self._delete_job(item_id, job_id)

    def _delete_job(self, item_id: str | UUID, job_id: int) -> None:
        try:
            self._cancel_job(job_id)
        except Exception as exc:  # ruff: ignore[blind-except]
            _logger.warning(
                "Could not cancel %s job %s (job id: %s): %s",
                self._kind,
                item_id,
                job_id,
                exc,
            )
        else:
            _logger.debug(
                "Cancelled %s job %s (job id: %s)", self._kind, item_id, job_id
            )
        self._retries.pop(item_id, None)
        if self._remove_files:
            self._cleanup_files(item_id)

    def _run_work_items(
        self, pending: list[tuple[str | UUID, WorkItem]]
    ) -> dict[str | UUID, Any]:
        # Runs on the poll thread: start what was taken, then ask the backend
        # about everything that is out.
        results: dict[str | UUID, Any] = {}
        for item_id, work_item in pending:
            try:
                if not self._submit(item_id, work_item):
                    # Shutting down: leave the rest, cleanup releases them.
                    return results
            except Exception as exc:  # ruff: ignore[blind-except]
                results[item_id] = exc
        return results | self._poll()

    def _submit(self, item_id: str | UUID, work_item: WorkItem) -> bool:
        with self._jobs_lock:
            if self._jobs_closed:
                return False
        existing = any(
            (self._workdir / f"{item_id}{suffix}").exists()
            for suffix in (".in", ".out", ".txt")
        )
        if existing:
            msg = (
                f"Work item files for '{item_id}' already exist in {self._workdir}; "
                "give each executor its own working directory."
            )
            raise ExecutionError(msg)
        input_file = self._workdir / f"{item_id}.in"
        output_file = self._workdir / f"{item_id}.out"
        self._write_input(item_id, input_file, work_item)
        try:
            job_id = self._start_job(
                item_id,
                # The interpreter that started this, not whatever `python` the
                # job's PATH resolves to: only this one is known to import ropt.
                [
                    sys.executable,
                    "-m",
                    "ropt.components.executors",
                    str(input_file),
                    str(output_file),
                ],
            )
        except BaseException:
            if self._remove_files:
                self._cleanup_files(item_id)
            raise
        with self._jobs_lock:
            stopped = self._jobs_closed
            if not stopped:
                self._jobs[item_id] = job_id
        if stopped:
            # Cleanup ran while this job was being started, so it never made the
            # list it would have been cancelled from: cancel it here.
            self._delete_job(item_id, job_id)
            return False
        _logger.debug("Started %s job %s (job id: %s)", self._kind, item_id, job_id)
        return True

    def _write_input(
        self, item_id: str | UUID, input_file: Path, work_item: WorkItem
    ) -> None:
        # Written to a temporary file and renamed, so the job can never observe
        # a half-written input: on a shared filesystem the rename is what makes
        # it visible, and fsync is what makes the content precede it.
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=self._workdir)
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "wb") as fp:
                dump((work_item.function, work_item.args, work_item.kwargs), fp)
                fp.flush()
                os.fsync(fp.fileno())
            tmp_path.rename(input_file)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            msg = (
                f"Work item '{item_id}' could not be sent to a job: {CANNOT_SERIALIZE}."
            )
            raise ExecutionError(msg) from exc
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _poll(self) -> dict[str | UUID, Any]:
        results: dict[str | UUID, Any] = {}
        try:
            jobs = self._live_job_ids()
        except Exception as exc:  # ruff: ignore[blind-except]
            # A backend that cannot be reached looks exactly like "nothing has
            # finished", so failed queries are acted on rather than ignored.
            return self._handle_query_failure(exc)
        self._query_failures = 0
        with self._jobs_lock:
            submitted = dict(self._jobs)
        for item_id, job_id in submitted.items():
            # Gone from the backend is the only sign that a job has ended; what
            # became of it has to be read from its output file.
            if job_id in jobs:
                continue
            result = self._read_result(item_id)
            if result is not _PENDING:
                results[item_id] = result
        if self._remove_files:
            for item_id, result in results.items():
                self._cleanup_files(
                    item_id, keep_output=isinstance(result, ExecutorFailure)
                )
        return results

    def _read_result(self, item_id: str | UUID) -> Any:  # ruff: ignore[any-type]
        output_file = self._workdir / f"{item_id}.out"
        try:
            with output_file.open("rb") as fp:
                result = load(fp)
        except FileNotFoundError:
            # The file may simply not be visible yet, so give the filesystem a
            # bounded number of further polls to show it.
            return self._retry_or_fail(
                item_id,
                f"Output file for work item {item_id} never appeared",
                "output file never appeared",
            )
        except (OSError, EOFError, UnpicklingError):
            # Present but unreadable, which a partially visible file also looks
            # like: retried on the same budget before giving up.
            return self._retry_or_fail(
                item_id,
                f"No valid result for work item {item_id} after {self._retries_limit} retries",
                f"no valid result after {self._retries_limit} retries",
            )
        except (ImportError, AttributeError) as exc:
            # The unpickler got as far as looking a name up, so the bytes were
            # already complete: reading them again cannot change the answer, and
            # spending the retry budget here would only delay the failure and
            # then blame the filesystem, which is the one thing not at fault.
            msg = (
                f"The result of work item {item_id} could not be reconstructed: "
                f"{exc}. This process must be able to import whatever the job "
                "returned."
            )
            return self._fail_item(item_id, msg, exc)
        except Exception as exc:  # ruff: ignore[blind-except]
            # Unpickling runs the code that rebuilds the object, and that can
            # raise anything at all. Whatever it was belongs to this work item
            # rather than to the executor, which anything escaping here would
            # take down: `_poll` runs outside `_run_work_items`' own guard.
            msg = f"The result of work item {item_id} could not be read: {exc}"
            return self._fail_item(item_id, msg, exc)
        self._retries.pop(item_id, None)
        self._drop_job(item_id)
        return result

    def _retry_or_fail(self, item_id: str | UUID, msg: str, reason: str) -> Any:  # ruff: ignore[any-type]
        # A shared filesystem may take a while to show a finished job's result,
        # so the same bounded budget covers "not there yet" and "not whole yet".
        retry_count = self._retries.get(item_id, 0) + 1
        self._retries[item_id] = retry_count
        if retry_count <= self._retries_limit:
            return _PENDING
        return self._fail_item(item_id, msg, reason)

    def _fail_item(
        self, item_id: str | UUID, msg: str, reason: object
    ) -> ExecutorFailure:
        # Give up on this work item: its retry budget is either spent or beside
        # the point, and its job is no longer something to wait for.
        self._retries.pop(item_id, None)
        self._drop_job(item_id)
        self._output_kept = True
        _logger.warning("%s work item %s failed: %s", self._kind, item_id, reason)
        return ExecutorFailure(msg + self._job_output(item_id))

    def _job_output(self, item_id: str | UUID) -> str:
        # A job that died before writing a result left its only trace here, so
        # the tail travels with the failure and the file itself is kept.
        output_file = self._workdir / f"{item_id}.txt"
        try:
            lines = output_file.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        tail = [line for line in lines if line.strip()][-_OUTPUT_TAIL_LINES:]
        if not tail:
            return ""
        body = "\n".join(tail)
        return f"; the job wrote to {output_file}:\n{body}"

    def _handle_query_failure(self, exc: BaseException) -> dict[str | UUID, Any]:
        # Only a run of failures is fatal where a backend can have a bad moment:
        # giving up at the first would end runs it is still working on. Once the
        # run is long enough, every job that is out fails, because nothing can
        # be said about a job that cannot be asked after.
        results: dict[str | UUID, Any] = {}
        self._query_failures += 1
        _logger.warning(
            "Querying the %s failed (%d/%d): %s",
            self._backend,
            self._query_failures,
            self._query_retries + 1,
            exc,
        )
        if self._query_failures > self._query_retries:
            msg = (
                f"The {self._backend} could not be queried after "
                f"{self._query_retries + 1} attempts: {exc}"
            )
            with self._jobs_lock:
                submitted = list(self._jobs)
            for item_id in submitted:
                self._drop_job(item_id)
                self._retries.pop(item_id, None)
                results[item_id] = ExecutorFailure(msg)
            if submitted:
                # These items never reach the cleanup pass in `_poll`, so their
                # output survives here too, and a directory holding it must not
                # be removed.
                self._output_kept = True
            self._query_failures = 0
        return results

    def _drop_job(self, item_id: str | UUID) -> None:
        with self._jobs_lock:
            self._jobs.pop(item_id, None)

    def _cleanup_files(self, item_id: str | UUID, *, keep_output: bool = False) -> None:
        suffixes = (".in", ".out") if keep_output else (".in", ".out", ".txt")
        for suffix in suffixes:
            path = self._workdir / f"{item_id}{suffix}"
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
