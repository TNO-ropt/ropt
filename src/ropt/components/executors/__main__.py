"""Script for running functions with pickled arguments and return values.

The other end of the job-based executors: this is what a cluster job or a local
job runs. It has no channel back to the executor, so the outcome travels as the
result file alone, and a failure is written to it as an exception rather than
being raised into a void.
"""

import os
import sys
import tempfile
from pathlib import Path

from ropt.components._transferred import check_transferred, reset_transferred
from ropt.components.executors._picklable import picklable_exception
from ropt.components.executors._serialize import dump, load


def main() -> int:
    """Run the script."""
    sys.exit(run_task(sys.argv[1], sys.argv[2]))


def run_task(input_path: str, output_path: str) -> int:
    """Run a serialized task and write the serialized result.

    Args:
        input_path:  File holding the serialized `(function, args, kwargs)`.
        output_path: File the serialized result is written to.

    Returns:
        `0` if the task succeeded, `1` if the task function raised.
    """
    reset_transferred()
    try:
        with Path(input_path).open("rb") as fp:
            function, args, kwargs = load(fp)
        check_transferred()
        result = function(*args, **kwargs)
        exit_code = 0
    except BaseException as exc:  # ruff: ignore[blind-except]
        # Anything at all, including SystemExit: this process exists only to run
        # the task, and the executor needs to hear how it went.
        result = picklable_exception(exc)
        exit_code = 1
    finally:
        # Written and renamed into place, so the executor polling for the file
        # never reads a partial result.
        out_path = Path(output_path)
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
    return exit_code


if __name__ == "__main__":
    main()
