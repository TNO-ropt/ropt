"""Script for running external optimizers."""

import sys

import cloudpickle


def main() -> None:
    """Run the script."""
    input_data = sys.stdin.buffer.read()

    if not input_data:
        sys.stderr.buffer.write(cloudpickle.dumps(ValueError("No input data")))
        sys.exit(1)

    try:
        function, args = cloudpickle.loads(input_data)
        result = function(*args)
        output_data = cloudpickle.dumps(result)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.buffer.write(cloudpickle.dumps(exc))
        sys.exit(1)

    sys.stdout.buffer.write(output_data)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
