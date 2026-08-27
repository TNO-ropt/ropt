"""Making worker exceptions safe to send back to the parent process."""

import traceback

from ropt._serialize import dumps, loads


def picklable_exception(exc: BaseException) -> BaseException:
    """Attach the worker traceback and return a serializable exception.

    The active traceback is recorded as a note so it survives serialization back
    to the parent process. If `exc` itself cannot be serialized, it is replaced
    by a `RuntimeError` carrying its `repr` and notes.

    Which serializer is used depends on what is installed, so the test has to be
    made with the one that will do the sending.

    Args:
        exc: The exception raised while running the task.

    Returns:
        A serializable exception with the worker traceback attached as a note.
    """
    exc.add_note(traceback.format_exc())
    try:
        loads(dumps(exc))
    except Exception:  # ruff: ignore[blind-except]
        wrapped = RuntimeError(repr(exc))
        for note in getattr(exc, "__notes__", []):
            wrapped.add_note(note)
        return wrapped
    return exc
