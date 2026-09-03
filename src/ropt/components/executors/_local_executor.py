"""Defines a class for running evaluations as local processes.

A local job is what [`JobExecutorBase`][] calls a job: this module supplies the
three answers it needs — how to start one, how to tell which are still running,
and how to cancel one — for the case where the machine running `ropt` is also
the machine running the work.

It sits between the process pool and the cluster: like a cluster job, each
evaluation gets an interpreter of its own that can be killed outright and whose
output is captured to a file; unlike one, there is no queueing system, no shared
filesystem and nothing to install.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import signal
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import tempfile
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

from ropt._logging import get_logger
from ropt.exceptions import ExecutionError

from ._job_executor import JobExecutorBase

if TYPE_CHECKING:
    import asyncio
    from uuid import UUID

_logger = get_logger(__name__)


def _run_teardown(jobs: queue.Queue[subprocess.Popen[bytes] | Path | None]) -> None:
    # Finishes closing a `LocalJobExecutor`, off the loop thread: cancelling a
    # job may not wait there, but a killed job that is never waited for lingers
    # as a zombie, and its directory may not go until it is truly gone.
    # Nothing is read from the executor: a restart may already have replaced
    # both the queue and the directory, so both arrive as values.
    while True:
        item = jobs.get()
        if not isinstance(item, subprocess.Popen):
            workdir = item
            break
        with contextlib.suppress(OSError, ValueError):
            item.wait()
    if workdir is not None:
        shutil.rmtree(workdir, ignore_errors=True)
        _logger.debug("Removed the local working directory %s", workdir)


def _remove_unused_workdir(workdir: Path) -> None:
    # For an executor that is never started: nothing else would remove its
    # directory. `rmdir` refuses a directory with anything in it, which is what
    # limits this to the unused case -- a started executor either had its
    # directory removed by the teardown thread already, or is keeping it
    # deliberately because there is output in it to read.
    with contextlib.suppress(OSError):
        workdir.rmdir()


class LocalJobExecutor(JobExecutorBase):
    """An executor that runs each work item as a separate local process.

    Needs no extras and no configuration. See
    [Parallel Evaluation](../workflows/parallel.md#localjobexecutor) for details.

    POSIX only: cancelling kills a job's whole process group, so that whatever
    the job started itself goes with it, and Windows has no equivalent.
    """

    _kind = "local"
    _backend = "local job backend"

    def __init__(
        self,
        *,
        workdir: Path | str | None = None,
        workers: int = 1,
        interval: float = 0.1,
        retries: int = 0,
        cleanup: bool = True,
    ) -> None:
        """Initialize the local job executor.

        Args:
            workdir:  Directory for each work item's serialized I/O files and
                      captured output. The default is a private temporary
                      directory that this executor creates and removes when it
                      closes — unless there is something in it to read, that is:
                      if a work item failed, or `cleanup` is off, the directory
                      is kept and its path logged.
            workers:  Maximum number of jobs running at once.
            interval: Polling interval in seconds. Small by default: a local
                      process is finished the moment it exits, so this is dead
                      time rather than politeness towards a scheduler.
            retries:  Number of extra polls to wait for a work item's result.
                      The default of `0` is enough, because a local job writes
                      and renames its result before it exits, so the result is
                      there the moment the process is gone.
            cleanup:  Whether to remove a work item's files once its result is
                      retrieved or its job is cancelled. A work item that failed
                      keeps its captured output, which is the only record of why.

        Raises:
            ValueError:     If `workdir` is not an existing directory, or if
                            `workers`, `interval` or `retries` is out of range.
            ExecutionError: If the system is not POSIX.
        """
        if os.name != "posix":
            msg = (
                "The local job executor requires a POSIX system: cancelling a "
                "job kills its whole process group, which Windows has no "
                "equivalent for."
            )
            raise ExecutionError(msg)
        self._own_workdir = workdir is None
        if workdir is None:
            resolved = Path(tempfile.mkdtemp(prefix="ropt-local-"))
            weakref.finalize(self, _remove_unused_workdir, resolved)
        else:
            resolved = Path(workdir).resolve()
            if not resolved.is_dir():
                msg = f"The local working directory does not exist: {resolved}"
                raise ValueError(msg)
        super().__init__(
            workdir=resolved,
            workers=workers,
            interval=interval,
            retries=retries,
            cleanup=cleanup,
        )
        self._next_job_id = 0
        # The running jobs are reached from the poll thread (`_start_job` and
        # `_live_job_ids`) and from the loop thread (`_cancel_job`), so they
        # need a lock of their own, as the module docstring of the base says.
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self._processes_lock = threading.Lock()
        self._teardown_queue: queue.Queue[subprocess.Popen[bytes] | Path | None] = (
            queue.Queue()
        )
        self._teardown_thread: threading.Thread | None = None
        self._workdir_released = False

    @property
    def workdir(self) -> Path:
        """The directory the jobs read and write.

        Worth asking for when you did not pass one: that directory is temporary,
        and this is the only way to find it while the executor is running.

        Returns:
            The working directory.
        """
        return self._workdir

    async def start(self, task_group: asyncio.TaskGroup) -> None:
        """Start the executor.

        Args:
            task_group: The task group to use.
        """
        if self._own_workdir and self._workdir_released:
            # A directory of its own for every run: the previous one belongs to
            # that run's teardown thread now, which either has taken it away
            # already or is about to, and may not have got round to it yet.
            self._workdir = Path(tempfile.mkdtemp(prefix="ropt-local-"))
        self._workdir_released = False
        self._teardown_queue = queue.Queue()
        self._teardown_thread = threading.Thread(
            target=_run_teardown,
            args=(self._teardown_queue,),
            name="ropt-local-teardown",
            daemon=True,
        )
        self._teardown_thread.start()
        await super().start(task_group)

    def _start_job(self, item_id: str | UUID, command: list[str]) -> int:
        output_file = self._workdir / f"{item_id}.txt"
        with output_file.open("wb") as fp:
            process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
                command,
                # The working directory is inherited rather than set to the one
                # the files live in: a job has to be able to import whatever the
                # caller could, and for a script that means its own directory.
                stdin=subprocess.DEVNULL,
                stdout=fp,
                stderr=subprocess.STDOUT,
                # A session of its own, so the job leads a process group that
                # cancelling can reach as a whole: killing only the job itself
                # would orphan anything it started.
                start_new_session=True,
            )
        with self._processes_lock:
            self._next_job_id += 1
            job_id = self._next_job_id
            self._processes[job_id] = process
        return job_id

    def _live_job_ids(self) -> set[int]:
        with self._processes_lock:
            entries = list(self._processes.items())
        live = {job_id for job_id, process in entries if process.poll() is None}
        if len(live) < len(entries):
            # `poll` collected their exit status on the way past, so this thread
            # is where a job that ended on its own stops being a child.
            with self._processes_lock:
                for job_id, _ in entries:
                    if job_id not in live:
                        self._processes.pop(job_id, None)
        return live

    def _cancel_job(self, job_id: int) -> None:
        with self._processes_lock:
            process = self._processes.pop(job_id, None)
        if process is None or process.poll() is not None:
            return
        with contextlib.suppress(OSError):
            # The group rather than the process: `start_new_session` made the
            # job lead one, so this reaches whatever it started as well.
            os.killpg(process.pid, signal.SIGTERM)
        # This runs on the loop thread, which must never wait for a process to
        # die: a Ctrl-C that waits is the thing this design is here to avoid.
        # The teardown thread waits instead, and it is the only thing that does.
        self._teardown_queue.put(process)

    def _cleanup(self) -> None:
        """Clean up the executor resources."""
        # Cancels the jobs first, which is what fills the teardown queue; the
        # sentinel goes in behind them, so every one of them is waited for.
        super()._cleanup()
        # Anything the base did not know to cancel: giving up on polling drops
        # its jobs, and a dropped local job is a process of ours that would
        # otherwise outlive the executor.
        with self._processes_lock:
            leftover = list(self._processes)
        for job_id in leftover:
            self._cancel_job(job_id)
        # Decided here rather than in the teardown thread because by now the
        # jobs are cancelled and no further work item can fail, so the answer is
        # final.
        keep_workdir = self._own_workdir and (
            not self._remove_files or self._output_kept
        )
        if keep_workdir:
            # The name is random, so without this the directory is kept and
            # unfindable, which is the same as not keeping it.
            _logger.warning(
                "Keeping the local working directory %s: %s.",
                self._workdir,
                "a work item failed"
                if self._output_kept
                else "cleanup is off, so nothing here is removed",
            )
        # Doubles as the sentinel: a directory to remove, or nothing to do.
        self._teardown_queue.put(
            self._workdir if self._own_workdir and not keep_workdir else None
        )
        self._workdir_released = True
