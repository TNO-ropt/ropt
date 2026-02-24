"""Script for running functions with pickled arguments and return values."""

import sys
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
        with Path(sys.argv[2]).open("wb") as fp:
            cloudpickle.dump(result, fp)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
