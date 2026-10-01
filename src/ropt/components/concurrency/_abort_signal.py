"""An abort request shared by whatever observes it."""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

from ropt._logging import get_logger
from ropt.enums import ExitReason

if TYPE_CHECKING:
    from collections.abc import Callable

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
    in kind: `stop` ends a run on a criterion it was given, while a signal cuts
    it off without consulting it. They also differ in lifetime, since `stop` is
    cleared by the next `run` and a signal is not: a step that starts while the
    signal is aborting is cut off from the outset.

    Code that cannot poll registers a callback instead, which is how a blocked
    [`Executor.run`][ropt.components.executors.Executor.run] is woken rather
    than left waiting for work it is about to abandon.

    Setting the signal is thread-safe, and it cannot be reset.
    """

    def __init__(self) -> None:
        """Initialize a signal that is not aborting."""
        self._lock = threading.Lock()
        self._aborting = False
        self._callbacks: list[Callable[[], None]] = []
        self._exit_reason = ExitReason.ABORTED

    def abort(self, exit_reason: ExitReason = ExitReason.ABORTED) -> None:
        """Cut off everything observing this signal.

        Calling this more than once has no further effect: the first call
        decides the exit reason, so a later abort for another reason cannot
        overwrite the reason a run is already ending for.

        Args:
            exit_reason: The reason the steps on this signal end with.
        """
        with self._lock:
            if self._aborting:
                return
            self._exit_reason = exit_reason
            self._aborting = True
            callbacks = list(self._callbacks)
        for callback in callbacks:
            _run_callback(callback)

    @property
    def exit_reason(self) -> ExitReason:
        """The reason a step on this signal ends with.

        Returns:
            The reason passed to the first `abort` call.
        """
        with self._lock:
            return self._exit_reason

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
