"""The worker pool: the workers that a run's evaluations are given to.

A pool is handed to a run explicitly, so a run's behaviour never depends on
where it is called from.

A pool with workers comes from a session factory (`thread_pool`,
`process_pool`, `local_pool`, `hpc_pool`), which is what ties its lifetime to a
session: closing the session releases the pools it built.

A [`SerialPool`][ropt.simple.SerialPool] is the exception, and needs no session,
because it owns nothing to release. It is what a run given no pool evaluates on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    from ropt.components.executors import Executor

    from ._session import Session

_RELEASED = (
    "This pool was released when its session closed; build a new one inside an "
    "open session, for example `with session() as s: pool = s.thread_pool()`."
)


class WorkerPool(ABC):
    """The workers that a run's evaluations are given to.

    Passed to a run with `pool=`. A pool with workers comes from a session
    factory such as [`thread_pool`][ropt.simple.Session.thread_pool] and lives
    until that session closes; a [`SerialPool`][ropt.simple.SerialPool] is built
    directly and evaluates in-process. See
    [Running Optimizations](../running/running.md) for a walkthrough.

    This is the type to annotate with when your own code takes a pool.
    """

    # The session that built this pool: a run evaluating here registers with
    # it, so the session's `stop()` reaches that run.
    _session: Session | None = None

    @property
    @abstractmethod
    def executor(self) -> Executor | None:
        """The executor this pool's evaluations run on.

        Returns:
            The executor, or `None` to evaluate on the calling thread.
        """


class SerialPool(WorkerPool):
    """A pool that evaluates in-process, on the calling thread.

    It has no workers, needs no session, and needs no releasing, so it is built
    directly:

    ```python
    result = optimize(config, x0, objective, pool=SerialPool())
    ```

    A run given no pool at all evaluates the same way.
    """

    @property
    def executor(self) -> Executor | None:
        """The executor this pool's evaluations run on.

        Returns:
            Always `None`: evaluations run on the calling thread.
        """
        return None


class _ExecutorPool(WorkerPool):
    def __init__(self, session: Session, executor: Executor) -> None:
        self._session: Session | None = session
        self._executor: Executor | None = executor

    @property
    def executor(self) -> Executor | None:
        if self._executor is None:
            raise WorkflowError(_RELEASED)
        return self._executor

    def _release(self) -> None:
        # Dropping the executor is what releases its workers. The pool itself
        # may outlive this, as a name the caller still holds.
        self._executor = None


def _session_of(pool: WorkerPool | None) -> Session | None:
    return None if pool is None else pool._session  # ruff: ignore[private-member-access]
