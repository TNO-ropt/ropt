"""Script for running functions with pickled arguments and return values."""

import os
import sys
import tempfile
from pathlib import Path

import cloudpickle


def main() -> int:
    """Run the script."""
    try:
        with Path(sys.argv[1]).open("rb") as fp:
            function, args, kwargs = cloudpickle.load(fp)
        result = function(*args, **kwargs)
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        result = exc
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


if __name__ == "__main__":
    main()
