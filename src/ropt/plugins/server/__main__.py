"""Script for running external optimizers."""

import sys

import cloudpickle


def main() -> None:
    """Run the script."""
    try:
        input_data = sys.stdin.buffer.read()
        if not input_data:
            output_data = cloudpickle.dumps(ValueError("No input data"))
            sys.exit(1)
        function, args = cloudpickle.loads(input_data)
        output_data = cloudpickle.dumps(function(*args))
    except Exception as exc:  # noqa: BLE001
        output_data = cloudpickle.dumps(exc)
        sys.exit(1)
    finally:
        sys.stdout.buffer.write(output_data)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
