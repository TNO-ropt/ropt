"""Script for running functions with pickled arguments and return values."""

import os
import sys
from pathlib import Path

import cloudpickle


def main() -> int:
    """Run the script."""
    try:
        cwd = Path.cwd()
        with Path(sys.argv[2]).open("rb") as fp:
            function, args, kwargs = cloudpickle.load(fp)
        os.chdir(Path(sys.argv[1]))
        exit_code = 0
        result = function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        result = exc
        exit_code = 1
    finally:
        os.chdir(cwd)
        with Path(sys.argv[3]).open("wb") as fp:
            cloudpickle.dump(result, fp)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
