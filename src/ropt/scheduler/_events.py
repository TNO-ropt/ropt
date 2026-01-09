import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base class for all events.

    This class serves as the base for all event objects that are passed between
    different components of the task runner system.

    Attributes:
        source: The identifier of the event's origin.
    """

    source: str | uuid.UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorEvent(Event):
    """Events that can carry an exception.

    This class extends the `Event` class to include an optional `error`
    attribute, which can be used to report exceptions.

    Attributes:
        error: An optional exception object.
    """

    error: BaseException | None = None
