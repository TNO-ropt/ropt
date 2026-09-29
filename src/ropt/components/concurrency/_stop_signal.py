"""A stop request shared by whatever observes it."""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

from ropt.enums import ExitCode

if TYPE_CHECKING:
    from collections.abc import Callable


class StopSignal:
    """A stop request that compute steps can be told to observe.

    A step is given one at construction and polls it at the same points it
    polls its own [`stop`][ropt.components.compute_steps.ComputeStep.stop]
    request, so one signal stops any number of steps at once. Unlike `stop`,
    which is cleared by the next `run`, a signal keeps its state: a step that
    starts while the signal is stopping is stopped from the outset.

    Code that cannot poll registers a callback instead, which is how a blocked
    [`Executor.run`][ropt.components.executors.Executor.run] is woken rather
    than left waiting for work it is about to abandon.

    Setting the signal is thread-safe, and it cannot be reset.
    """

    def __init__(self) -> None:
        """Initialize a signal that is not stopping."""
        self._lock = threading.Lock()
        self._flag = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._exit_code = ExitCode.CANCELLED

    def stop(self, exit_code: ExitCode = ExitCode.CANCELLED) -> None:
        """Request that everything observing this signal stops.

        Calling this more than once has no further effect: the first call
        decides the exit code, so a later stop for another reason cannot
        overwrite the reason a run is already stopping for.

        Args:
            exit_code: The code the steps stopping on this signal end with.
        """
        with self._lock:
            if self._flag.is_set():
                return
            self._exit_code = exit_code
            self._flag.set()
            callbacks = list(self._callbacks)
        for callback in callbacks:
            callback()

    @property
    def exit_code(self) -> ExitCode:
        """The exit code a step stopping on this signal ends with.

        Returns:
            The code passed to the first `stop` call.
        """
        with self._lock:
            return self._exit_code

    @property
    def stopping(self) -> bool:
        """Whether a stop has been requested.

        Returns:
            `True` once `stop` has been called.
        """
        return self._flag.is_set()

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to run when this signal stops.

        A signal that is already stopping runs the callback immediately, so a
        caller that registers late is not left waiting.

        The callback runs on the thread that calls `stop`, so it must return
        promptly and must not raise.

        Args:
            callback: The zero-argument callable to run.
        """
        with self._lock:
            if not self._flag.is_set():
                self._callbacks.append(callback)
                return
        callback()

    def remove_callback(self, callback: Callable[[], None]) -> None:
        """Deregister a callback.

        Removing one that is not registered does nothing.

        Args:
            callback: The callable to remove.
        """
        with self._lock, contextlib.suppress(ValueError):
            self._callbacks.remove(callback)
