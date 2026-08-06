"""Detection of workflow objects transferred into a worker process.

Workflow objects (compute steps, evaluators, event handlers, the event
dispatcher, and the batch-id counter) are process-local: they may be pickled as
part of a task payload, but they must not be *used* in a worker. When such an
object is unpickled it becomes an inert
[`_Placeholder`][ropt.components._transferred._Placeholder] and records its type
here. The executor worker runners call
[`check_transferred`][ropt.components._transferred.check_transferred] right after
deserializing a task and raise a
[`TransferError`][ropt.exceptions.TransferError] if anything was captured.
"""

from __future__ import annotations

from ropt.exceptions import TransferError

_transferred_subjects: list[str] = []


class _Placeholder:
    """Inert stand-in for a workflow object unpickled in another process."""

    __slots__ = ("_subject",)

    def __init__(self, subject: str) -> None:
        self._subject = subject

    def __getattr__(self, name: str) -> _Placeholder:
        # Bound-method unpickling does getattr(obj, name); returning a
        # sub-placeholder (rather than raising) lets that reconstruction succeed.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Placeholder(self._subject)

    def __reduce__(self) -> tuple[type[_Placeholder], tuple[str]]:
        return (_Placeholder, (self._subject,))


def _make_placeholder(subject: str) -> _Placeholder:
    _transferred_subjects.append(subject)
    return _Placeholder(subject)


def reset_transferred() -> None:
    """Clear the record of transferred objects before deserializing a task."""
    _transferred_subjects.clear()


def check_transferred() -> None:
    """Raise if a workflow object was transferred into this worker.

    Raises:
        TransferError: If a deserialized task captured a workflow object.
    """
    if not _transferred_subjects:
        return
    subjects = ", ".join(sorted(set(_transferred_subjects)))
    _transferred_subjects.clear()
    msg = f"Workflow objects cannot be used in a worker process: {subjects}."
    raise TransferError(msg)
