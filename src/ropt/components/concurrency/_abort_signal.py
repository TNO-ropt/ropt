"""An abort request shared by whatever observes it."""

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


class AbortSignal:
    """An abort request that compute steps can be told to observe.

    A step is given one at construction and polls it at the same points it
    polls its own [`stop`][ropt.components.compute_steps.ComputeStep.stop]
    request, so one signal reaches any number of steps at once. The two differ
    in kind: `stop` ends a run on a criterion it was given, while a signal aborts
    it without consulting it. They also differ in lifetime, since `stop` is
    cleared by the next `run` and a signal is not: a step that starts while the
    signal is aborting is aborted from the outset.

    Code that cannot poll registers a callback instead, which is how a blocked
    [`Executor.run`][ropt.components.executors.Executor.run] is woken rather
    than left waiting for work that will not run.

    A signal can abort with a parent signal, and can be the parent signal of the
    runs and offloads started while it is marked as one: see `aborts_with` and
    `as_parent`.

    Setting the signal is thread-safe, and it cannot be reset.
    """

    def __init__(self) -> None:
        """Initialize a signal that is not aborting."""
        self._lock = threading.Lock()
        self._aborting = False
        self._callbacks: list[Callable[[], None]] = []
        self._exit_code = ExitCode.ABORTED

    def abort(self, exit_code: ExitCode = ExitCode.ABORTED) -> None:
        """Abort everything observing this signal.

        Calling this more than once has no further effect: the first call
        decides the exit code, so a later abort for another reason cannot
        overwrite the code a run is already ending with.

        Args:
            exit_code: The exit code the steps on this signal end with.
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
        """The exit code a step on this signal ends with.

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
        """Register a callback to run when this signal aborts.

        A signal that is already aborting runs the callback immediately, so a
        caller that registers late is not left waiting.

        The callback runs on the thread that aborts the signal, or on this one
        when the signal is already aborting, so it must return promptly. One
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
    def aborts_with(self, parent: AbortSignal | None) -> Generator[None]:
        """Abort this signal when `parent` aborts, for as long as the block runs.

        This signal takes the exit code of `parent`. A `parent` that is already
        aborting aborts it at once.

        Args:
            parent: The signal to abort with, or `None` for none.

        Yields:
            Nothing; `parent` no longer reaches this signal once the block ends.
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

    def _abort_with(self, parent: AbortSignal) -> None:
        # Registered before `parent` aborts, so its exit code is read when it fires.
        self.abort(parent.exit_code)

    @contextlib.contextmanager
    def as_parent(self) -> Generator[None]:
        """Make this the parent signal of runs and offloads started in the block.

        Inside the block, [`parent_signal`][ropt.components.concurrency.parent_signal]
        returns this signal.

        Yields:
            Nothing; the previous parent signal returns when the block ends.
        """
        token = _parent.set(self)
        try:
            yield
        finally:
            _parent.reset(token)


# A context variable rather than a thread-local, so that a thread started with
# `contextvars.copy_context().run` inherits it.
_parent: ContextVar[AbortSignal | None] = ContextVar("parent_signal", default=None)


def parent_signal() -> AbortSignal | None:
    """Return the parent signal of a run or offload started from the calling code.

    That is the signal of the run or offload whose code is running: its
    evaluation function, an event handler, or an offloaded function.

    Returns:
        The parent signal, or `None` for code that no run or offload is running.
    """
    return _parent.get()
