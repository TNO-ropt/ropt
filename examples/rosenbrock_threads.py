"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters.

For demonstration purposes multiple optimizations are run in parallel using
threading. To simulate that the function to optimize does some significant work,
an adjustable delay is set.

Run this script with the `--help` option to see how to switch between different
scenarios:

```
    usage: rosenbrock_threads.py [-h] [-n NOPT] [-d DELAY]

    options:
      -h, --help            show this help message and exit
      -n NOPT, --nopt NOPT  number of optimizations (default: 2)
      -d DELAY, --delay DELAY
                            synchronous delay in seconds (default, no delay: 0.0)
```

Compare for instance, a case where the objective function needs significant time:
```
    time python examples/rosenbrock_threads.py -d 0.005 -n 5
```

with no delay, simulating a case where the objective function offloaded to a background task:
```
    time python examples/rosenbrock_threads.py -n 5
```

The intended application for this approach is for cases where the work to be
done by the objective function is substantial, but can be offloaded, e.g. to a
HPC cluster. In this case, the relative work done by the HPC is dominant, and
the optimization code can run reasonable well in threads while waiting for the
results.
"""

import argparse
import asyncio
import time
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.config import EnOptConfig
from ropt.plugins.server.base import EvaluatorServer
from ropt.results import FunctionResults
from ropt.workflow import (
    create_compute_step,
    create_evaluator,
    create_event_handler,
    create_server,
)

DIM = 5
UNCERTAINTY = 0.1
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * 10,
    },
    "gradient": {
        "number_of_perturbations": 5,
    },
}
initial_values = 2 * np.arange(DIM) / DIM + 0.5


async def rosenbrock(
    variables: NDArray[np.float64],
    realization: int,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    delay: float,
) -> NDArray[np.float64]:
    """Function evaluator for the multi-dimensional rosenbrock function.

    Args:
        variables:    The variables to evaluate.
        realization:  Realization number.
        a:            The 'a' parameters.
        b:            The 'b' parameters.
        delay:        The delay in the objective function.

    Returns:
        The calculated objective.
    """
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (a[realization] - x) ** 2 + b[realization] * (y - x * x) ** 2
    time.sleep(delay)  # noqa: ASYNC251
    await asyncio.sleep(0.0)
    return np.asarray([objective])


def run_optimization(
    server: EvaluatorServer,
    function: Callable[[NDArray[np.float64], int], Awaitable[NDArray[np.float64]]],
    config: EnOptConfig,
) -> FunctionResults:
    """Run the optimization.

    Returns:
        The optimal results.
    """
    evaluator = create_evaluator("async_evaluator", function=function, server=server)
    step = create_compute_step("optimizer", evaluator=evaluator)
    tracker = create_event_handler("tracker")
    step.add_event_handler(tracker)
    step.run(variables=initial_values, config=config)
    results: FunctionResults = tracker["results"]
    return results


async def async_run(
    config: EnOptConfig,
    a_list: list[NDArray[np.float64]],
    b_list: list[NDArray[np.float64]],
    delay: float,
) -> list[FunctionResults]:
    """Run the asynchronous code.

    Returns:
        The optimal results.
    """
    function_server = create_server("function_server", workers=2)
    assert isinstance(function_server, EvaluatorServer)
    async with asyncio.TaskGroup() as tg:
        function_server.start(tg)
        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    run_optimization,
                    function_server,
                    partial(rosenbrock, a=a, b=b, delay=delay),
                    config,
                )
                for a, b in zip(a_list, b_list, strict=True)
            ),
        )
        function_server.stop()
    return results


def run(n_opt: int, delay: float) -> None:
    """Run the optimization.

    Args:
        n_opt: The number of optimizations to run in threads.
        delay:  Synchronous delay in the objective function.
    """
    rng = default_rng(seed=123)

    realizations = len(CONFIG["realizations"]["weights"])
    a = [
        rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations) for _ in range(n_opt)
    ]
    b = [
        rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)
        for _ in range(n_opt)
    ]

    results = asyncio.run(async_run(EnOptConfig.model_validate(CONFIG), a, b, delay))
    for optimal_result in results:
        assert optimal_result is not None
        assert optimal_result.functions is not None
        print(f"Optimal variables: {optimal_result.evaluations.variables}")
        print(f"Optimal objective: {optimal_result.functions.weighted_objective}\n")
        assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=1e-1)
        assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-1)


def main(n_opt: int = 2, delay: float = 0.0) -> None:
    """Run the example and check the result.

    Arguments:
        n_opt:  Number of optimizations
        delay:  Synchronous delay in the objective function.

    """
    run(n_opt, delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nopt",
        type=int,
        default=2,
        help="number of optimizations (default: 2)",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.0,
        help="synchronous delay in seconds (default, no delay: 0.0)",
    )
    args = parser.parse_args()
    main(args.nopt, args.delay)
