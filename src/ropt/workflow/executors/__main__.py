"""Script for running functions with pickled arguments and return values."""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import cloudpickle


def main() -> int:
    """Run the script."""
    try:
        with Path(sys.argv[1]).open("rb") as fp:
            function, args, kwargs = cloudpickle.load(fp)
        result = function(*args, **kwargs)
        exit_code = 0
    except Exception as exc:  # ruff: ignore[blind-except]
        exc.add_note(traceback.format_exc())
        result = _picklable_exception(exc)
        exit_code = 1
    finally:
        output_path = Path(sys.argv[2])
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=output_path.parent)
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "wb") as fp:
                cloudpickle.dump(result, fp)
                fp.flush()
                os.fsync(fp.fileno())
            tmp_path.rename(output_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    sys.exit(exit_code)


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
