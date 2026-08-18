"""Making worker exceptions safe to send back to the parent process."""

import traceback

import cloudpickle


def picklable_exception(exc: BaseException) -> BaseException:
    """Attach the worker traceback and return a cloudpickle-safe exception.

    The active traceback is recorded as a note so it survives serialization back
    to the parent process. If `exc` itself cannot be pickled, it is replaced by
    a `RuntimeError` carrying its `repr` and notes.

    Args:
        exc: The exception raised while running the task.

    Returns:
        A picklable exception with the worker traceback attached as a note.
    """
    exc.add_note(traceback.format_exc())
    try:
        cloudpickle.loads(cloudpickle.dumps(exc))
    except Exception:  # ruff: ignore[blind-except]
        wrapped = RuntimeError(repr(exc))
        for note in getattr(exc, "__notes__", []):
            wrapped.add_note(note)
        return wrapped
    return exc
