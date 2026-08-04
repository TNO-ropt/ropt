"""Script for running functions with pickled arguments and return values."""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import cloudpickle

from ropt.workflow._transferred import check_transferred, reset_transferred


def main() -> int:
    """Run the script."""
    sys.exit(run_task(sys.argv[1], sys.argv[2]))


def run_task(input_path: str, output_path: str) -> int:
    """Run a cloudpickled task and write the cloudpickled result.

    Args:
        input_path:  File holding the cloudpickled `(function, args, kwargs)`.
        output_path: File the cloudpickled result is written to.

    Returns:
        `0` if the task succeeded, `1` if the task function raised.
    """
    reset_transferred()
    try:
        with Path(input_path).open("rb") as fp:
            function, args, kwargs = cloudpickle.load(fp)
        check_transferred()
        result = function(*args, **kwargs)
        exit_code = 0
    except Exception as exc:  # ruff: ignore[blind-except]
        exc.add_note(traceback.format_exc())
        result = _picklable_exception(exc)
        exit_code = 1
    finally:
        out_path = Path(output_path)
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=out_path.parent)
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "wb") as fp:
                cloudpickle.dump(result, fp)
                fp.flush()
                os.fsync(fp.fileno())
            tmp_path.rename(out_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return exit_code


def _picklable_exception(exc: BaseException) -> BaseException:
    try:
        cloudpickle.loads(cloudpickle.dumps(exc))
    except Exception:  # ruff: ignore[blind-except]
        wrapped = RuntimeError(repr(exc))
        for note in getattr(exc, "__notes__", []):
            wrapped.add_note(note)
        return wrapped
    return exc


if __name__ == "__main__":
    main()
