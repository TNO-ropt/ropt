"""The abort scope of a task, and of the tasks nested in it."""

from __future__ import annotations

import contextlib
import threading
from contextvars import ContextVar
from functools import partial
from typing import TYPE_CHECKING

from ropt._logging import get_logger
from ropt.enums import ExitCode

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_logger = get_logger(__name__)


def _run_callback(callback: Callable[[], None]) -> None:
    # One observer's callback must not withhold the abort from the observers
    # after it, nor leave the caller that registered it holding an exception.
    try:
        callback()
    except Exception:
        _logger.exception("An abort callback raised")


class AbortScope:
    """An abort request for a task and the tasks nested in it.

    A step is given one at construction and polls it at the same points it
    polls its own [`stop`][ropt.components.compute_steps.ComputeStep.stop]
    request, so one scope reaches any number of steps at once. The two differ
    in kind: `stop` ends a run on a criterion it was given, while a scope aborts
    it without consulting it. They also differ in lifetime, since `stop` is
    cleared by the next `run` and a scope is not: a step that starts while the
    scope is aborting is aborted from the outset.

    Code that cannot poll registers a callback instead, which is how a blocked
    [`Executor.run`][ropt.components.executors.Executor.run] is woken rather
    than left waiting for work that will not run.

    A scope can be nested in a parent scope, so that it aborts with it, and can
    be the current scope of the code that a task runs: see `nested_in` and
    `as_current`.

    Aborting a scope is thread-safe, and an aborted scope cannot be reset.
    """

    def __init__(self) -> None:
        """Initialize a scope that is not aborting."""
        self._lock = threading.Lock()
        self._aborting = False
        self._callbacks: list[Callable[[], None]] = []
        self._exit_code = ExitCode.ABORTED

    def abort(self, exit_code: ExitCode = ExitCode.ABORTED) -> None:
        """Abort everything observing this scope.

        Calling this more than once has no further effect: the first call
        decides the exit code, so a later abort for another reason cannot
        overwrite the code a run is already ending with.

        Args:
            exit_code: The exit code the steps on this scope end with.
        """
        with self._lock:
            if self._aborting:
                return
            self._exit_code = exit_code
            self._aborting = True
            callbacks = list(self._callbacks)
        for callback in callbacks:
            _run_callback(callback)

    @property
    def exit_code(self) -> ExitCode:
        """The exit code a step on this scope ends with.

        Returns:
            The exit code passed to the first `abort` call.
        """
        with self._lock:
            return self._exit_code

    @property
    def aborting(self) -> bool:
        """Whether an abort has been requested.

        Returns:
            `True` once `abort` has been called.
        """
        with self._lock:
            return self._aborting

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to run when this scope aborts.

        A scope that is already aborting runs the callback immediately, so a
        caller that registers late is not left waiting.

        The callback runs on the thread that aborts the scope, or on this one
        when the scope is already aborting, so it must return promptly. One
        that raises is logged instead of raised, and the callbacks after it
        still run.

        Args:
            callback: The zero-argument callable to run.
        """
        with self._lock:
            if not self._aborting:
                self._callbacks.append(callback)
                return
        _run_callback(callback)

    def remove_callback(self, callback: Callable[[], None]) -> None:
        """Deregister a callback.

        Removing one that is not registered does nothing.

        Args:
            callback: The callable to remove.
        """
        with self._lock, contextlib.suppress(ValueError):
            self._callbacks.remove(callback)

    @contextlib.contextmanager
    def nested_in(self, parent: AbortScope | None) -> Generator[None]:
        """Abort this scope when `parent` aborts, for as long as the block runs.

        This scope takes the exit code of `parent`. A `parent` that is already
        aborting aborts it at once.

        Args:
            parent: The scope to abort with, or `None` for none.

        Yields:
            Nothing; `parent` no longer reaches this scope once the block ends.
        """
        if parent is None:
            yield
            return
        abort = partial(self._abort_with, parent)
        parent.add_callback(abort)
        try:
            yield
        finally:
            parent.remove_callback(abort)

    def _abort_with(self, parent: AbortScope) -> None:
        # Registered before `parent` aborts, so its exit code is read when it fires.
        self.abort(parent.exit_code)

    @contextlib.contextmanager
    def as_current(self) -> Generator[None]:
        """Make this the current abort scope inside the block.

        Inside the block,
        [`current_abort_scope`][ropt.components.concurrency.current_abort_scope]
        returns this scope.

        Yields:
            Nothing; the previous current scope returns when the block ends.
        """
        token = _current.set(self)
        try:
            yield
        finally:
            _current.reset(token)


# A context variable rather than a thread-local, so that a thread started with
# `contextvars.copy_context().run` inherits it.
_current: ContextVar[AbortScope | None] = ContextVar(
    "current_abort_scope", default=None
)


def current_abort_scope() -> AbortScope | None:
    """Return the abort scope of the calling code.

    That is the scope of the task whose code is running: its evaluation
    function, an event handler, or a function given to `run`. In the `with`
    block of a session, outside any task, it is the scope of that session. A
    task started from the calling code is nested in this scope.

    Returns:
        The current scope, or `None` outside any task and session block.
    """
    return _current.get()


def aborting() -> bool:
    """Return whether the task that runs the calling code is being aborted.

    Code that a task runs — an evaluation function, an event handler, a report
    callback, or a function given to [`run`][ropt.Session.run] — can call this
    to stop early once its task is aborted. In the `with` block of a session it
    returns whether the session is aborted. It returns `False` outside any task
    and session block, which includes the jobs of an HPC pool. In the worker
    processes of a process pool it turns `True` within a fraction of a second
    of the abort. See
    [Failures and Aborts](../running/failures.md#stopping-early-when-aborted).

    Returns:
        Whether the task that runs the calling code is being aborted.
    """
    scope = _current.get()
    return scope is not None and scope.aborting
