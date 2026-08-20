"""The worker pool: where evaluations run, and which batch IDs they draw from.

A pool bundles the two things that always travel together: the
[`Executor`][ropt.components.executors.Executor] that runs the evaluations, and
the [`BatchIdCounter`][ropt.components.evaluators.BatchIdCounter] their batch IDs
come from. Runs that share a pool therefore share a batch-ID sequence without
the caller having to arrange it.

Pools are created by the session factories (`thread_pool`, `process_pool`,
`hpc_pool`, `serial_pool`) and handed to a run explicitly, so a run's behaviour
never depends on where it is called from.

A serial pool is the degenerate case: no executor, only a counter. Evaluations
then run on the calling thread, which is what a run does when it is given no
pool at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from ropt.components._transferred import _make_placeholder
from ropt.components.evaluators import BatchIdCounter

if TYPE_CHECKING:
    from ropt.components.executors import Executor

    from ._session import _Session


class WorkerPool:
    """The workers that run evaluations, and the batch IDs they share.

    Created by a session factory such as
    [`thread_pool`][ropt.simple.Session.thread_pool]. Every run using the same
    pool draws its batch IDs from the same counter, so concurrent runs never
    produce the same batch ID.

    A pool lives until its session closes, which releases it. Release it earlier
    with [`close`][ropt.simple.WorkerPool.close], or by using it as a context
    manager. See [Running Optimizations](../running/running.md) for a
    walkthrough.

    A pool built by [`serial_pool`][ropt.simple.serial_pool] has no executor and
    no workers to release; it exists so that runs sharing one batch-ID sequence
    can say so, whether or not they run in parallel.
    """

    def __init__(
        self,
        executor: Executor | None = None,
        session: _Session | None = None,
        bundle_size: int = 1,
    ) -> None:
        """Initialize the pool.

        Args:
            executor:    The started executor, or `None` to evaluate in-process.
            session:     The session that owns the pool, if it has one.
            bundle_size: Evaluations per worker task, `0` for the whole batch.

        Raises:
            ValueError: If `bundle_size` is negative.
        """
        if bundle_size < 0:
            # A serial pool builds no evaluator, so nothing downstream checks it.
            msg = f"bundle_size must be >= 0, got {bundle_size}"
            raise ValueError(msg)
        self._executor = executor
        self._session = session
        self._bundle_size = bundle_size
        self._batch_ids = BatchIdCounter()
        self._closed = False

    def __reduce__(self) -> tuple[object, tuple[str]]:
        # A pool belongs to the session that built it, so it cannot follow work
        # into a worker process; it arrives there as an inert placeholder that
        # the entry points reject by name.
        return (_make_placeholder, ("A worker pool",))

    @property
    def closed(self) -> bool:
        """Whether the pool has been released.

        Returns:
            `True` once the pool, or the session that built it, was closed.
        """
        return self._closed

    @property
    def executor(self) -> Executor | None:
        """The executor that runs this pool's evaluations.

        Returns:
            The executor, or `None` for a serial pool.
        """
        return self._executor

    @property
    def batch_ids(self) -> BatchIdCounter:
        """The counter every run on this pool draws its batch IDs from.

        Returns:
            The batch ID counter.
        """
        return self._batch_ids

    @property
    def bundle_size(self) -> int:
        """How many evaluations are sent to a worker as one task.

        Returns:
            The number of evaluations per task, `0` meaning the whole batch.
        """
        return self._bundle_size

    def close(self) -> None:
        """Release the pool's workers without waiting for the session to close.

        Closing is final: a pool cannot be reopened, and a run still using it
        ends with [`ExecutorStopped`][ropt.exceptions.ExecutorStopped]. Closing
        twice is a no-op, and closing a serial pool releases nothing, since it
        holds nothing.

        Most code does not need this — a session releases every pool it created.
        It matters when pools are created in a loop within one long-lived
        session, above all process pools, which hold worker interpreters.
        """
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            self._session.discard_extra(self)
        if self._executor is not None:
            self._executor.cancel()

    def __enter__(self) -> Self:
        """Enter a block that closes the pool on exit.

        Returns:
            The pool itself.
        """
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the pool."""
        self.close()


def serial_pool() -> WorkerPool:
    """Create a pool that evaluates in-process, on the calling thread.

    A serial pool has no workers: it carries only the batch-ID counter that the
    runs sharing it draw from. Use it to give concurrent runs one batch-ID
    sequence without running their evaluations in parallel, or as an explicit
    way to say that a run should evaluate in-process.

    It needs no session, and needs no releasing.

    Returns:
        A pool without an executor.
    """
    return WorkerPool()
