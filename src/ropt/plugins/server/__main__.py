"""Script for running functions with pickled arguments and return values."""

import sys
from pathlib import Path

import cloudpickle


def main() -> None:
    """Run the script."""
    try:
        with Path(sys.argv[1]).open("rb") as fp:
            function, args = cloudpickle.load(fp)
        with Path(sys.argv[2]).open("wb") as fp:
            cloudpickle.dump(function(*args), fp)
    except Exception as exc:  # noqa: BLE001
        with Path(sys.argv[2]).open("wb") as fp:
            cloudpickle.dump(exc, fp)
        sys.exit(1)


if __name__ == "__main__":
    main()
