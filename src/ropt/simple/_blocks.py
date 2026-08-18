"""The execution blocks: ``threads()``, ``processes()`` and ``hpc()``.

Each opens an executor on the ambient session for the duration of the block and
removes it on exit. The session itself is acquired from `_session`, and is shared
with any block already open, so only the outermost block owns its lifetime.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ropt.components.executors import (
    HPCExecutor,
    MultiprocessingExecutor,
    ThreadingExecutor,
)
from ropt.exceptions import WorkflowError

from ._session import _acquire_session, _release_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextvars import Token

    from ropt.components.executors import Executor

    from ._session import _Session


class _ExecutionScope:
    """Install an executor on the ambient session for the duration of a block."""

    def __init__(self, make_executor: Callable[[], Executor]) -> None:
        self._make_executor = make_executor
        self._session: _Session | None = None
        self._token: Token[_Session | None] | None = None

    def __enter__(self) -> None:
        # Entering twice would overwrite the first block's session and token,
        # so the first would never be released. `open_executor` refuses the
        # second block anyway, but only by accident of ordering.
        if self._session is not None:
            msg = (
                "This execution block is already open and cannot be entered "
                "again; open a separate block instead."
            )
            raise WorkflowError(msg)
        session, token = _acquire_session()
        try:
            session.open_executor(self._make_executor)
        except BaseException:
            _release_session(session, token)
            raise
        self._session = session
        self._token = token

    def __exit__(self, *_exc: object) -> None:
        assert self._session is not None
        session, token = self._session, self._token
        # Drop the block's state before tearing it down, so a closed scope does
        # not keep a stopped session, its loop and its thread alive, and can be
        # opened again for a fresh one.
        self._session = None
        self._token = None
        try:
            session.close_executor()
        finally:
            _release_session(session, token)


def threads(*, workers: int = 1) -> _ExecutionScope:
    """Run evaluations in a thread pool for the duration of the block.

    See [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        workers: The number of worker threads.

    Returns:
        A context manager backing evaluations with a thread pool.
    """
    return _ExecutionScope(lambda: ThreadingExecutor(workers=workers))


def processes(*, workers: int = 1) -> _ExecutionScope:
    """Run evaluations in a process pool for the duration of the block.

    The objective must be picklable. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        workers: The number of worker processes.

    Returns:
        A context manager backing evaluations with a process pool.
    """
    return _ExecutionScope(lambda: MultiprocessingExecutor(workers=workers))


def hpc(  # ruff: ignore[too-many-arguments]
    *,
    workers: int = 1,
    cores: int = 1,
    cluster: str | None = None,
    queue: str | None = None,
    workdir: Path | str | None = None,
    config_path: Path | str | None = None,
    template: str | None = None,
    queue_type: str = "slurm",
) -> _ExecutionScope:
    """Run evaluations on an HPC cluster for the duration of the block.

    Interfaces with a cluster queue (e.g. Slurm) through `pysqa`; requires the
    `ropt[hpc]` extra, and the objective must be picklable. The cluster is
    selected from `cluster`/`queue`: give a queue to search for its cluster, a
    cluster to use its default queue, or both to be explicit. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    Args:
        workers:     The maximum number of concurrent cluster jobs.
        cores:       The number of CPUs per job.
        cluster:     The cluster name, when the `pysqa` config defines several.
        queue:       The queue or partition name.
        workdir:     The shared-filesystem working directory (defaults to the
                     current directory).
        config_path: The path to the `pysqa` configuration directory.
        template:    An inline submission-script template, instead of a config.
        queue_type:  The queueing system type.

    Returns:
        A context manager backing evaluations with an HPC cluster.
    """
    resolved = Path.cwd() if workdir is None else Path(workdir).resolve()
    return _ExecutionScope(
        lambda: HPCExecutor(
            workers=workers,
            cores=cores,
            cluster=cluster,
            queue=queue,
            workdir=resolved,
            config_path=config_path,
            template=template,
            queue_type=queue_type,
        )
    )
