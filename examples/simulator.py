"""A stand-in for an external simulator, run by `programs.py`.

Reads the variables from an input file, writes the objective to an output file,
and prints its progress to standard output, as many simulators do. When the
input sets `hang`, it stops responding instead, as a simulator can on an input
it does not handle.

Usage: python simulator.py INPUT OUTPUT
"""

import json
import sys
import time
from pathlib import Path


def main() -> int:
    """Run the simulation.

    Returns:
        The exit status: 0 on success, 2 on a usage error.
    """
    try:
        _, input_file, output_file = sys.argv
    except ValueError:
        print("usage: simulator.py INPUT OUTPUT", file=sys.stderr)
        return 2
    data = json.loads(Path(input_file).read_text(encoding="utf-8"))
    print(f"simulating variables {data['variables']}", flush=True)
    if data.get("hang", False):
        print("not responding", flush=True)
        time.sleep(3600)
    objective = sum((x - data["target"]) ** 2 for x in data["variables"])
    Path(output_file).write_text(json.dumps({"objective": objective}), encoding="utf-8")
    print(f"objective {objective}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
