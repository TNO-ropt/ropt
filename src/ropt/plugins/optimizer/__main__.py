"""Script for running external optimizers."""

import sys
from pathlib import Path

from ropt.plugins.optimizer.external import _PluginOptimizer


def main() -> int:
    """Run the script.

    Returns:
        The exit code.
    """
    fifo1 = Path(sys.argv[1])
    fifo2 = Path(sys.argv[2])
    parent_pid = int(sys.argv[3])

    assert fifo1.exists()
    assert fifo2.exists()

    return _PluginOptimizer(parent_pid).run(fifo1, fifo2)


if __name__ == "__main__":
    sys.exit(main())
