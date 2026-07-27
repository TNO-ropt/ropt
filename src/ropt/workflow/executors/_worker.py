"""Process-local flag marking a ropt worker process."""

from __future__ import annotations

_IS_WORKER = False


def _mark_worker_process() -> None:
    global _IS_WORKER  # ruff: ignore[global-statement]
    _IS_WORKER = True


def is_worker_process() -> bool:
    """Check whether the current process is a ropt worker process.

    Returns `True` when running inside a worker spawned by a
    [`MultiprocessingExecutor`][ropt.workflow.executors.MultiprocessingExecutor]
    or a job launched by an
    [`HPCExecutor`][ropt.workflow.executors.HPCExecutor], and `False` in the
    driver process or in a
    [`ThreadingExecutor`][ropt.workflow.executors.ThreadingExecutor] worker,
    which share the driver's process.

    Returns:
        True if running inside a ropt worker process.
    """
    return _IS_WORKER
