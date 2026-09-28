"""A stop request shared by whatever observes it."""

from __future__ import annotations

import threading


class StopSignal:
    """A stop request that compute steps can be told to observe.

    A step is given one at construction and polls it at the same points it
    polls its own [`stop`][ropt.components.compute_steps.ComputeStep.stop]
    request, so one signal stops any number of steps at once. Unlike `stop`,
    which is cleared by the next `run`, a signal keeps its state: a step that
    starts while the signal is stopping is stopped from the outset.

    Setting the signal is thread-safe, and it cannot be reset.
    """

    def __init__(self) -> None:
        """Initialize a signal that is not stopping."""
        self._flag = threading.Event()

    def stop(self) -> None:
        """Request that everything observing this signal stops."""
        self._flag.set()

    @property
    def stopping(self) -> bool:
        """Whether a stop has been requested.

        Returns:
            `True` once `stop` has been called.
        """
        return self._flag.is_set()
