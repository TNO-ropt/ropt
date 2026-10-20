"""Running an external program from the objective with `run_program`.

The objective writes the variables to an input file, runs a simulator as a
separate program, and reads the objective from the file the simulator writes.
The simulator here is `simulator.py`, a stand-in for a real one.

The program is started with `run_program` rather than `subprocess.run`. Both
wait for the program to exit, but only `run_program` stops it when the
optimization is aborted: by Ctrl-C, by `Session.abort`, or by a failure
elsewhere. With `subprocess.run`, an evaluation on a thread pool waits for the
program until it exits, however long that takes, and on other pools the
program keeps running after the optimization has ended.

The script runs the optimization twice. The first run completes. In the second,
the simulator of one realization stops responding, and a time limit calls
`Session.abort`, which terminates it.
"""

import json
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt import EvaluationFunctionContext, ExitCode, run_program, session
from ropt.exceptions import AbortedError
from ropt.utils import restore_keyboard_interrupt

restore_keyboard_interrupt()

DIM = 2
REALIZATIONS = 4
TARGETS = [0.8, 0.9, 1.1, 1.2]
SIMULATOR = Path(__file__).with_name("simulator.py")
TIME_LIMIT = 2.0  # seconds, for the second run
CONFIG: dict[str, Any] = {
    "variables": {"variable_count": DIM, "perturbation_magnitudes": 1e-3},
    "realizations": {"weights": [1.0] * REALIZATIONS},
    "gradient": {"number_of_perturbations": 1},
    "optimizer": {"max_functions": 10},
}
INITIAL_VALUES = np.zeros(DIM)


# --8<-- [start:objective]
def objective(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    *,
    hang_realization: int | None = None,
) -> float:
    """Run the simulator for one realization and read its objective.

    Args:
        variables:        The variable vector to evaluate.
        context:          Identifies the realization being evaluated.
        hang_realization: A realization whose simulator stops responding.

    Returns:
        The objective, or NaN if the simulator failed.
    """
    with tempfile.TemporaryDirectory() as workdir:
        cwd = Path(workdir)
        simulator_input = {
            "variables": variables.tolist(),
            "target": TARGETS[context.realization],
            "hang": context.realization == hang_realization,
        }
        (cwd / "input.json").write_text(json.dumps(simulator_input), encoding="utf-8")
        status = run_program(
            [sys.executable, SIMULATOR, "input.json", "output.json"],
            cwd=cwd,
            stdout="simulator.log",
            stderr=subprocess.STDOUT,
        )
        if status != 0:
            return np.nan
        output = json.loads((cwd / "output.json").read_text(encoding="utf-8"))
        return float(output["objective"])


# --8<-- [end:objective]


def main() -> None:
    """Run the optimization once to completion, and once with a hanging simulator."""
    # --8<-- [start:run]
    with session() as s:
        result = s.thread_pool(workers=4).optimize(CONFIG, INITIAL_VALUES, objective)
    # --8<-- [end:run]
    assert result.exit_code == ExitCode.MAX_FUNCTIONS_REACHED
    assert result.results is not None
    print(f"optimal variables: {result.results.variables}")
    assert np.allclose(result.results.variables, np.mean(TARGETS), atol=0.05)

    # --8<-- [start:abort]
    hanging = partial(objective, hang_realization=2)
    exit_code = None
    start = time.monotonic()
    with session() as s:
        timer = threading.Timer(TIME_LIMIT, s.abort)
        timer.start()
        try:
            s.thread_pool(workers=4).optimize(CONFIG, INITIAL_VALUES, hanging)
        except AbortedError as error:
            print(f"aborted after {time.monotonic() - start:.1f} s: {error}")
            exit_code = error.exit_code
        finally:
            timer.cancel()
    # --8<-- [end:abort]
    assert exit_code == ExitCode.USER_ABORT
    # The hanging simulator would run for an hour; the abort ended it.
    assert time.monotonic() - start < TIME_LIMIT + 5.0


if __name__ == "__main__":
    main()
