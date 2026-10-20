"""The worker pool: the workers that evaluations are given to.

A pool comes from a session factory (`thread_pool`, `process_pool`,
`hpc_pool`), which is what ties its lifetime to a session:
closing the session releases the pools it built.

A pool is passed to `optimize`, `evaluate`, `evaluate_batch` or `run` as their
`pool` argument, so where the work happens is stated at the call site, and the
session the call is nested in follows from the pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ropt.components.executors.base import ExecutorBase
from ropt.exceptions import WorkflowError

if TYPE_CHECKING:
    from ropt.components.executors import Executor

    from ._session import Session

_RELEASED = (
    "This pool was released when its session closed; build a new one inside an "
    "open session, for example `with session() as s: pool = s.thread_pool()`."
)


class WorkerPool:
    """The workers that evaluations are given to.

    Built by a session factory such as
    [`thread_pool`][ropt.Session.thread_pool], and released when that
    session closes. A pool has no methods of its own: pass it as the `pool`
    argument of [`optimize`][ropt.optimize], [`evaluate`][ropt.evaluate],
    [`evaluate_batch`][ropt.evaluate_batch] or [`run`][ropt.run], which then
    run their work on its workers, nested in its session. See
    [Evaluating in Parallel](../running/parallel.md) for a walkthrough.

    This is the type to annotate with when your own code takes a pool.
    """

    def __init__(self, session: Session, executor: Executor) -> None:
        """Initialize a pool owned by a session.

        Args:
            session:  The session that built this pool.
            executor: The executor its evaluations run on.
        """
        self._session = session
        self._executor: Executor | None = executor

    @property
    def _live_executor(self) -> Executor:
        if self._executor is None:
            raise WorkflowError(_RELEASED)
        return self._executor

    def _release(self) -> ExecutorBase | None:
        # Dropping the executor is what releases its workers. The pool itself
        # may outlive this, as a name the caller still holds. The executor is
        # returned so that the session can wait for it, at most until a
        # deadline it shares with its other pools.
        executor, self._executor = self._executor, None
        if not isinstance(executor, ExecutorBase):
            return None
        executor._stop()  # ruff: ignore[private-member-access]
        return executor

    def _retire(self) -> None:
        executor = self._executor
        if isinstance(executor, ExecutorBase):
            executor._retire()  # ruff: ignore[private-member-access]


def pool_parts(pool: WorkerPool | None) -> tuple[Session | None, Executor | None]:
    if pool is None:
        return None, None
    executor = pool._live_executor  # ruff: ignore[private-member-access]
    return pool._session, executor  # ruff: ignore[private-member-access]
