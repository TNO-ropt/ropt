"""Script for running functions with pickled arguments and return values.

The other end of the job-based executors: this is what a cluster job or a local
job runs. It has no channel back to the executor, so the outcome travels as the
result file alone, and a failure is written to it as an exception rather than
being raised into a void.
"""

import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ropt._serialize import CANNOT_DESERIALIZE, CANNOT_SERIALIZE, dump, load
from ropt.components.executors._picklable import picklable_exception


def _load_task(
    input_path: str,
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]:
    try:
        with Path(input_path).open("rb") as fp:
            function, args, kwargs = load(fp)
    except Exception as exc:
        exc.add_note(f"Could not rebuild the task: {CANNOT_DESERIALIZE}.")
        raise
    return function, args, kwargs


def _write_result(result: object, out_path: Path) -> None:
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=out_path.parent)
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "wb") as fp:
            dump(result, fp)
            fp.flush()
            os.fsync(fp.fileno())
        tmp_path.rename(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    """Run the script."""
    sys.exit(run_task(sys.argv[1], sys.argv[2]))


def run_task(input_path: str, output_path: str) -> int:
    """Run a serialized task and write the serialized result.

    Args:
        input_path:  File holding the serialized `(function, args, kwargs)`.
        output_path: File the serialized result is written to.

    Returns:
        `0` if the task succeeded, `1` if it raised or its result could not be
            written.
    """
    try:
        function, args, kwargs = _load_task(input_path)
        result = function(*args, **kwargs)
        exit_code = 0
    except BaseException as exc:  # ruff: ignore[blind-except]
        result = picklable_exception(exc)
        exit_code = 1
    finally:
        out_path = Path(output_path)
        try:
            _write_result(result, out_path)
        except Exception as exc:  # ruff: ignore[blind-except]
            exc.add_note(f"Could not send the result back: {CANNOT_SERIALIZE}.")
            _write_result(picklable_exception(exc), out_path)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    main()
