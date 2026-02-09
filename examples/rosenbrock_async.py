"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters.

For demonstration purposes multiple optimizations are run in parallel using
threading or multiprocessing.

This script has the following options:

```
usage: rosenbrock_async.py [-h] [-m] [-d DELAY] [-w WORKERS] [-o OPTIMIZATIONS]

options:
  -h, --help            show this help message and exit
  -m, --multiprocessing
                        Use multiprocessing instead of asyncio for function evaluation
  -d DELAY, --delay DELAY
                        Delay in seconds before evaluating the rosenbrock function
                        (default: 0.0)
  -w WORKERS, --workers WORKERS
                        The number of workers for function evaluation (default: 2)
  -o OPTIMIZATIONS, --optimizations OPTIMIZATIONS
                        The number of parallel optimizations (default: 2)
```
"""

import argparse
import asyncio
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.config import EnOptConfig
from ropt.plugins.server.base import Server
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


def rosenbrock(  # noqa: PLR0913
    variables: NDArray[np.float64],
    realization: int,
    perturbation: int,  # noqa: ARG001
    batch_id: int,  # noqa: ARG001
    *,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    delay: float,
) -> NDArray[np.float64]:
    """Function evaluator for the multi-dimensional rosenbrock function.

    Args:
        variables:    The variables to evaluate.
        realization:  Realization number.
        perturbation: Perturbation number.
        batch_id:     Batch ID.
        a:            The 'a' parameters.
        b:            The 'b' parameters.
        delay:        The delay before starting the evaluation.

    Returns:
        The calculated objective.
    """
    time.sleep(delay)
    objective = 0.0
    for d_idx in range(DIM - 1):
        x, y = variables[d_idx : d_idx + 2]
        objective += (a[realization] - x) ** 2 + b[realization] * (y - x * x) ** 2
    return np.asarray([objective])


def run_optimization(
    server: Server,
    function: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
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


async def async_run(  # noqa: PLR0913
    config: EnOptConfig,
    a_list: list[NDArray[np.float64]],
    b_list: list[NDArray[np.float64]],
    *,
    multiprocessing: bool = False,
    delay: float = 0.0,
    workers: int = 4,
) -> list[FunctionResults]:
    """Run the asynchronous code.

    Args:
        config:          The configuration of the optimizer.
        a_list:          The list of 'a' parameters.
        b_list:          The list of 'b' parameters.
        multiprocessing: If True, use multiprocessing for function evaluations.
        delay:           Delay in seconds while running the rosenbrock function.
        workers:         The number of workers to use.

    Returns:
        The optimal results.
    """
    async_server = create_server(
        "multiprocessing_server" if multiprocessing else "async_server",
        workers=workers,
    )
    assert isinstance(async_server, Server)
    async with asyncio.TaskGroup() as tg:
        await async_server.start(tg)
        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    run_optimization,
                    async_server,
                    partial(rosenbrock, a=a, b=b, delay=delay),
                    config,
                )
                for a, b in zip(a_list, b_list, strict=True)
            ),
        )
        async_server.cancel()
    return results


async def main(
    *,
    multiprocessing: bool = False,
    delay: float = 0.0,
    workers: int = 4,
    optimizations: int = 2,
) -> None:
    """Run the optimization."""
    rng = default_rng(seed=1234)

    realizations = len(CONFIG["realizations"]["weights"])
    a = [
        rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations)
        for _ in range(optimizations)
    ]
    b = [
        rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)
        for _ in range(optimizations)
    ]

    start_time = time.perf_counter()
    results = await async_run(
        EnOptConfig.model_validate(CONFIG),
        a,
        b,
        multiprocessing=multiprocessing,
        delay=delay,
        workers=workers,
    )
    end_time = time.perf_counter()

    for optimal_result in results:
        assert optimal_result is not None
        assert optimal_result.functions is not None
        print(f"Optimal variables: {optimal_result.evaluations.variables}")
        print(f"Optimal objective: {optimal_result.functions.weighted_objective}\n")
        assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=2e-1)
        assert np.allclose(optimal_result.evaluations.variables, 1, atol=2e-1)

    print(f"Elapsed time: {end_time - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--multiprocessing",
        action="store_true",
        help="Use multiprocessing instead of asyncio for function evaluation",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds before evaluating the rosenbrock function (default: 0.0)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=2,
        help="The number of workers for function evaluation (default: 2)",
    )
    parser.add_argument(
        "-o",
        "--optimizations",
        type=int,
        default=2,
        help="The number of parallel optimizations (default: 2)",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            multiprocessing=args.multiprocessing,
            delay=args.delay,
            workers=args.workers,
            optimizations=args.optimizations,
        )
    )
