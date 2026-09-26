"""The machinery shared by executors that run work as separate processes.

Unlike the thread and process executors, there is no channel back from a job:
work travels over the filesystem, as a serialized `<id>.in` file the job reads
and an `<id>.out` file it writes, with `<id>.txt` for whatever it printed. The
job runs `ropt.components.executors` as a module, with the interpreter that
started it, which is the one `ropt` is installed in.

So there is nothing to await, only a backend to ask, and asking blocks. The
executor owns no thread for it. Every caller blocked in `run` loops over two
steps: take the results that are ready for it, or else claim the backend and
launch the queued jobs and collect the finished ones on everyone's behalf. Only
one caller holds the backend at a time, which is what keeps launching jobs and
asking after them in one order.

Three things carry that. `_Results` is a plain list, one per `run` call, where
whichever caller collected a result leaves it for the call waiting on it.
`_State` is the bookkeeping they all share, and the only thing its lock guards.
`_StateUpdate` is what one caller changed while the backend was its own: filled
with no lock held, then merged by `apply_update`. That merge is not an
overwrite, because the others change the state meanwhile, so each entry is
written only if it is still wanted.

The lock is never held across a backend call or across user code. Cancelling a
job is the one backend call that may run while another caller holds the backend:
making it wait would put a cancellation behind a status query, which is what a
caller pressing Ctrl-C is waiting for.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
import time
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from pickle import UnpicklingError  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from ropt._logging import get_logger
from ropt._serialize import CANNOT_SERIALIZE, dump, load
from ropt.exceptions import ExecutionError

from .base import (
    ExecutorBase,
    ExecutorFailure,
    WorkItem,
    _calls,
    _run_bundle,
    _stopped,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

_logger = get_logger(__name__)

# How much of a failed job's captured output travels with its failure.
_OUTPUT_TAIL_LINES: Final = 20

# Returned for a work item whose result is not there yet, or not whole yet.
_NOT_READY: Final = object()

# One `run` call's finished results, each with the position it was asked for.
# The list object also identifies the call: whichever caller collects a result
# appends to it, and the call itself empties it.
_Results = list[tuple[int, Any]]


@dataclass
class _StateUpdate:
    # What one caller changed while the backend was its own. Every field names
    # the shared state it feeds. It is filled with no lock held, so that the
    # slow work does not block anyone, and merged by `apply_update`.
    jobs_to_launch: list[tuple[UUID, _Results, int, list[WorkItem]]] = field(
        default_factory=list
    )
    launched_jobs: dict[UUID, int] = field(default_factory=dict)
    results: dict[UUID, Any] = field(default_factory=dict)
    retries: set[UUID] = field(default_factory=set)
    queried: bool = False
    query_error: BaseException | None = None
    # Why the caller stopped early, used as the reason for whatever it was
    # given to launch but never did.
    error: BaseException | None = None


class _State:
    # The bookkeeping every caller shares, and the only thing the lock guards.
    # Nothing outside this class touches it: a caller claims the backend, works
    # with no lock held, and hands back an `_StateUpdate` to be merged.
    #
    # The merge cannot be a wholesale overwrite, because other callers change
    # this state meanwhile: one may leave, or queue new work, or close the
    # executor. So every entry is written only if it is still wanted.

    def __init__(
        self, *, workers: int, interval: float, query_retries: int, backend_name: str
    ) -> None:
        self._workers = workers
        self._interval = interval
        self._query_retries = query_retries
        self._backend_name = backend_name
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self._queue: deque[tuple[_Results, int, list[WorkItem]]] = deque()
        self._active: dict[UUID, tuple[_Results, int]] = {}
        self._jobs: dict[UUID, int] = {}
        self._retries: dict[UUID, int] = {}
        self._last_query = time.monotonic() - interval
        self._query_failures = 0
        self._backend_busy = False
        self._output_kept = False

    @property
    def output_kept(self) -> bool:
        # A subclass that owns its working directory needs to know: removing it
        # would take the kept output with it.
        with self._lock:
            return self._output_kept

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def clear(self) -> list[tuple[UUID, int]]:
        with self._lock:
            jobs = list(self._jobs.items())
            self._jobs.clear()
            self._active.clear()
            self._retries.clear()
            self._queue.clear()
            return jobs

    def queue(self, results: _Results, bundles: list[list[WorkItem]]) -> None:
        with self._condition:
            self._check_open()
            for index, bundle in enumerate(bundles):
                self._queue.append((results, index, bundle))
            self._condition.notify_all()

    def pop_results(self, results: _Results) -> _Results:
        # Emptied in place, because whichever caller has the backend holds this
        # same list. Handed over before the caller stores them, so user code
        # that raises does not do so under the lock.
        with self._lock:
            self._check_open()
            ready = list(results)
            results.clear()
            return ready

    def claim_backend(
        self, results: _Results, update: _StateUpdate
    ) -> tuple[bool, bool]:
        # Returns whether the backend is now this caller's, and whether it
        # should ask which jobs have finished.
        with self._condition:
            self._check_open()
            # Re-checked under the acquisition that does the waiting: a result
            # arriving between the two would otherwise be slept through.
            if results:
                return False, False
            if self._backend_busy:
                # Everything that could release this caller notifies, so unlike
                # the wait below this one needs no timeout.
                self._condition.wait()
                return False, False
            self._pick_jobs_to_launch(update)
            waited = time.monotonic() - self._last_query
            if not update.jobs_to_launch and waited < self._interval:
                self._condition.wait(self._interval - waited)
                return False, False
            # Eager while the backend is answering, which is the rate without
            # this check; paced by the interval once a query has failed, so the
            # query budget spans the grace period it names.
            query_due = waited >= self._interval or self._query_failures == 0
            self._backend_busy = True
            return True, query_due

    def jobs_to_check(self) -> list[tuple[UUID, int, int]]:
        # Jobs launched by earlier claims, with the polls each has already been
        # given. What this caller just launched is still in its update, so it
        # never asks after a job that cannot have finished.
        with self._lock:
            return [
                (item_id, job_id, self._retries.get(item_id, 0))
                for item_id, job_id in self._jobs.items()
            ]

    def apply_update(
        self, update: _StateUpdate, *, release_backend: bool
    ) -> list[tuple[UUID, int]]:
        with self._condition:
            if release_backend:
                self._backend_busy = False
            cancel: list[tuple[UUID, int]] = []
            for item_id, job_id in update.launched_jobs.items():
                if item_id in self._active and not self._closed:
                    self._jobs[item_id] = job_id
                else:
                    # Its run left, or the executor closed, while it was being
                    # launched: nobody is waiting for it now.
                    cancel.append((item_id, job_id))
            for item_id, result in update.results.items():
                if item_id in self._active:
                    self._complete(item_id, result)
                    self._output_kept |= isinstance(result, ExecutorFailure)
            for item_id in update.retries:
                if item_id in self._active:
                    self._retries[item_id] = self._retries.get(item_id, 0) + 1
            if update.queried:
                self._last_query = time.monotonic()
                self._note_query(update.query_error)
            self._fail_unlaunched(update)
            self._condition.notify_all()
            return cancel

    def drop(self, results: _Results) -> list[tuple[UUID, int]]:
        # A run leaving is the only thing that takes its work out of here, so
        # nothing can write into its list afterwards.
        with self._condition:
            results.clear()
            self._queue = deque(
                entry for entry in self._queue if entry[0] is not results
            )
            jobs: list[tuple[UUID, int]] = []
            for item_id in [
                item_id
                for item_id, (item_results, _) in self._active.items()
                if item_results is results
            ]:
                del self._active[item_id]
                self._retries.pop(item_id, None)
                job_id = self._jobs.pop(item_id, None)
                if job_id is not None:
                    jobs.append((item_id, job_id))
            self._condition.notify_all()
            return jobs

    def _check_open(self) -> None:
        if self._closed:
            raise _stopped()

    def _pick_jobs_to_launch(self, update: _StateUpdate) -> None:
        # `_active` holds the work that is out, so `workers` caps how many run
        # at once, and the slot must be claimed before the job exists. The
        # update records them, so work that never starts can still be failed
        # back to the run waiting for it.
        while len(self._active) < self._workers and self._queue:
            results, index, bundle = self._queue.popleft()
            # The id is the stem of the files this job reads and writes.
            item_id = uuid4()
            update.jobs_to_launch.append((item_id, results, index, bundle))
            self._active[item_id] = (results, index)

    def _complete(self, item_id: UUID, result: Any) -> None:  # ruff: ignore[any-type]
        results, index = self._active.pop(item_id)
        self._jobs.pop(item_id, None)
        self._retries.pop(item_id, None)
        results.append((index, result))

    def _fail_unlaunched(self, update: _StateUpdate) -> None:
        # Given a slot but never launched: the caller stopped on close, or
        # raised before reaching them.
        reason = (
            "the executor was closed" if update.error is None else f"{update.error}"
        )
        for item_id, _results, _index, _bundle in update.jobs_to_launch:
            if item_id in update.launched_jobs or item_id in update.results:
                continue
            if item_id not in self._active:
                continue
            self._complete(
                item_id, ExecutorFailure(f"The work item was not run: {reason}")
            )

    def _note_query(self, error: BaseException | None) -> None:
        if error is None:
            self._query_failures = 0
            return
        # Only a run of failures is fatal where a backend can have a bad moment:
        # giving up at the first would end runs it is still working on. Once the
        # run is long enough, every job that is out fails, because nothing can
        # be said about a job that cannot be asked after.
        self._query_failures += 1
        _logger.warning(
            "Querying the %s failed (%d/%d): %s",
            self._backend_name,
            self._query_failures,
            self._query_retries + 1,
            error,
        )
        if self._query_failures <= self._query_retries:
            return
        msg = (
            f"The {self._backend_name} could not be queried after "
            f"{self._query_retries + 1} attempts: {error}"
        )
        outstanding = list(self._active)
        for item_id in outstanding:
            self._complete(item_id, ExecutorFailure(msg))
        if outstanding:
            # These never reach the cleanup in `_collect_finished`, so their
            # output survives here too.
            self._output_kept = True
        self._query_failures = 0


class JobExecutorBase(ExecutorBase):
    """Runs work items as separate processes that communicate through files.

    Subclasses decide what a job is: how one is started, how to tell which are
    still running, and how to cancel one. Everything between those three
    answers — the file layout, the worker cap, starting and collecting,
    cancellation — lives here.
    """

    # Names this kind of job in log messages, and the thing that runs them.
    _kind = "job"
    _backend_name = "backend"

    # Extra attempts to query the backend after one fails. A backend that can
    # have a bad moment raises this; one that cannot has nothing to wait for.
    _query_retries = 0

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        *,
        workdir: Path,
        workers: int,
        interval: float,
        retries: int,
        cleanup: bool,
        bundle_size: int = 1,
    ) -> None:
        """Initialize the shared state.

        Subclasses validate and resolve `workdir` themselves, because where a
        job may read and write is the one thing they do not agree on.

        Args:
            workdir:     Directory holding each work item's files.
            workers:     Maximum number of jobs running at once.
            interval:    Polling interval in seconds.
            retries:     Extra polls to wait for a result after the first attempt.
            cleanup:     Whether to remove work item files once they are done with.
            bundle_size: Calls per job, `0` for a whole batch.

        Raises:
            ValueError: If `workers`, `interval` or `retries` is out of range.
        """
        super().__init__(bundle_size=bundle_size)
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
        self._retries_limit = retries
        self._remove_files = cleanup
        self._state = _State(
            workers=workers,
            interval=interval,
            query_retries=self._query_retries,
            backend_name=self._backend_name,
        )
        _logger.info(
            "Started %s executor (%d max workers, %.2fs poll interval)",
            self._kind,
            workers,
            interval,
        )

    @abstractmethod
    def _start_job(self, item_id: UUID, command: list[str]) -> int:
        # On the caller that has claimed the backend. The job's output belongs
        # in `<item_id>.txt` in the working directory: the only record of a job
        # that died before writing a result. Returns an id that `_live_job_ids`
        # and `_cancel_job` accept.
        ...

    @abstractmethod
    def _live_job_ids(self) -> set[int]:
        # On the caller that has claimed the backend. An absent id means the
        # job ended, however it ended; what became of it is read from its result
        # file.
        ...

    @abstractmethod
    def _cancel_job(self, job_id: int) -> None:
        # On a departing caller's own thread, so it must not wait for the job to
        # die: a Ctrl-C that waits for cancellation to finish is what this
        # design avoids. It may run while another caller has claimed the
        # backend, so a subclass keeping state of its own needs a lock for it.
        ...

    def _on_close(self) -> None:
        self._state.close()

    def _release(self) -> None:
        self._cancel_jobs(self._state.clear())

    def _run_bundles(
        self,
        bundles: list[list[WorkItem]],
        store: Callable[[int, Any], None],
    ) -> None:
        results: _Results = []
        try:
            self._state.queue(results, bundles)
            remaining = len(bundles)
            while remaining > 0:
                ready = self._state.pop_results(results)
                if ready:
                    remaining -= len(ready)
                    for index, result in ready:
                        if isinstance(result, BaseException):
                            raise result
                        store(index, result)
                else:
                    self._launch_and_collect(results)
        finally:
            self._cancel_jobs(self._state.drop(results))

    def _launch_and_collect(self, results: _Results) -> None:
        # Launch the queued jobs and collect the finished ones, for every caller
        # at once. Only one caller does this at a time; the rest wait here.
        update = _StateUpdate()
        claimed = False
        try:
            claimed, query_due = self._state.claim_backend(results, update)
            if claimed:
                self._launch_jobs(update)
                if query_due:
                    self._collect_finished(update)
        except BaseException as exc:
            update.error = exc
            raise
        finally:
            # Work may have been given a slot without the backend being
            # claimed, and the runs waiting for it must still be released.
            if claimed or update.jobs_to_launch:
                self._cancel_jobs(
                    self._state.apply_update(update, release_backend=claimed)
                )

    def _launch_jobs(self, update: _StateUpdate) -> None:
        for item_id, _results, _index, bundle in update.jobs_to_launch:
            if self.closed:
                # `apply_update` fails what is left back to its callers.
                break
            try:
                update.launched_jobs[item_id] = self._launch_job(item_id, bundle)
            except Exception as exc:  # ruff: ignore[blind-except]
                update.results[item_id] = exc

    def _collect_finished(self, update: _StateUpdate) -> None:
        update.queried = True
        try:
            live = self._live_job_ids()
        except Exception as exc:  # ruff: ignore[blind-except]
            # A backend that cannot be reached looks exactly like "nothing has
            # finished", so failed queries are acted on rather than ignored.
            update.query_error = exc
            return
        for item_id, job_id, retries_used in self._state.jobs_to_check():
            # Gone from the backend is the only sign that a job has ended; what
            # became of it has to be read from its output file.
            if job_id in live:
                continue
            result = self._read_result(item_id, retries_used)
            if result is _NOT_READY:
                update.retries.add(item_id)
                continue
            update.results[item_id] = result
            if self._remove_files:
                self._cleanup_files(
                    item_id, keep_output=isinstance(result, ExecutorFailure)
                )

    def _cancel_jobs(self, jobs: list[tuple[UUID, int]]) -> None:
        if not jobs:
            return
        # On a thread of its own, joined: cancelling is one backend call per
        # job, and this runs on a thread that may have been interrupted. A
        # second interrupt then breaks the join rather than the cancelling, and
        # the thread is not a daemon, so the interpreter still waits for it.
        thread = threading.Thread(
            target=self._cancel_jobs_now,
            args=(jobs,),
            name=f"ropt-{self._kind}-cancel",
            daemon=False,
        )
        try:
            thread.start()
        except RuntimeError:
            # No new threads at interpreter shutdown.
            self._cancel_jobs_now(jobs)
            return
        thread.join()

    def _cancel_jobs_now(self, jobs: list[tuple[UUID, int]]) -> None:
        for item_id, job_id in jobs:
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
            if self._remove_files:
                self._cleanup_files(item_id)

    def _launch_job(self, item_id: UUID, bundle: list[WorkItem]) -> int:
        existing = any(
            (self._workdir / f"{item_id}{suffix}").exists()
            for suffix in (".in", ".out", ".txt")
        )
        if existing:
            msg = f"Work item files for '{item_id}' already exist in {self._workdir}."
            raise ExecutionError(msg)
        input_file = self._workdir / f"{item_id}.in"
        output_file = self._workdir / f"{item_id}.out"
        self._write_input(item_id, input_file, bundle)
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
        _logger.debug("Started %s job %s (job id: %s)", self._kind, item_id, job_id)
        return job_id

    def _write_input(
        self, item_id: UUID, input_file: Path, bundle: list[WorkItem]
    ) -> None:
        # Written to a temporary file and renamed, so the job can never observe
        # a half-written input: on a shared filesystem the rename is what makes
        # it visible, and fsync is what makes the content precede it.
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=self._workdir)
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "wb") as fp:
                dump((_run_bundle, (_calls(bundle),), {}), fp)
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

    def _read_result(self, item_id: UUID, retries_used: int) -> Any:  # ruff: ignore[any-type]
        output_file = self._workdir / f"{item_id}.out"
        try:
            with output_file.open("rb") as fp:
                return load(fp)
        except FileNotFoundError:
            # The file may simply not be visible yet, so give the filesystem a
            # bounded number of further polls to show it.
            return self._retry_or_fail(
                item_id,
                retries_used,
                f"Output file for work item {item_id} never appeared",
                "output file never appeared",
            )
        except (OSError, EOFError, UnpicklingError):
            # Present but unreadable, which a partially visible file also looks
            # like: retried on the same budget before giving up.
            return self._retry_or_fail(
                item_id,
                retries_used,
                f"No valid result for work item {item_id} after "
                f"{self._retries_limit} retries",
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
            return self._failure_for(item_id, msg, exc)
        except Exception as exc:  # ruff: ignore[blind-except]
            # Unpickling runs the code that rebuilds the object, and that can
            # raise anything at all. Whatever it was belongs to this work item
            # rather than to the executor, which anything escaping here would
            # take down.
            msg = f"The result of work item {item_id} could not be read: {exc}"
            return self._failure_for(item_id, msg, exc)

    def _retry_or_fail(
        self, item_id: UUID, retries_used: int, msg: str, reason: str
    ) -> Any:  # ruff: ignore[any-type]
        # A shared filesystem may take a while to show a finished job's result,
        # so the same bounded budget covers "not there yet" and "not whole yet".
        if retries_used < self._retries_limit:
            return _NOT_READY
        return self._failure_for(item_id, msg, reason)

    def _failure_for(self, item_id: UUID, msg: str, reason: object) -> ExecutorFailure:
        _logger.warning("%s work item %s failed: %s", self._kind, item_id, reason)
        return ExecutorFailure(msg + self._job_output_tail(item_id))

    def _job_output_tail(self, item_id: UUID) -> str:
        # A job that died before writing a result left its only trace here, so
        # the tail travels with the failure and the file itself is kept. A
        # shared filesystem may not show the content yet, and a submission
        # script may not have redirected the job's output at all, so the path is
        # named either way: it is the one place left to look.
        output_file = self._workdir / f"{item_id}.txt"
        try:
            lines = output_file.read_text(errors="replace").splitlines()
        except OSError:
            return f"; no job output could be read from {output_file}"
        tail = [line for line in lines if line.strip()][-_OUTPUT_TAIL_LINES:]
        if not tail:
            return f"; the job wrote nothing to {output_file}"
        body = "\n".join(tail)
        return f"; the job wrote to {output_file}:\n{body}"

    def _cleanup_files(self, item_id: UUID, *, keep_output: bool = False) -> None:
        suffixes = (".in", ".out") if keep_output else (".in", ".out", ".txt")
        for suffix in suffixes:
            path = self._workdir / f"{item_id}{suffix}"
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
