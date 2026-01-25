"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters.

For demonstration purposes multiple optimizations are run in parallel using
threading.
"""

import asyncio
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

N_OPT = 2

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


def rosenbrock(
    variables: NDArray[np.float64],
    realization: int,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Function evaluator for the multi-dimensional rosenbrock function.

    Args:
        variables:    The variables to evaluate.
        realization:  Realization number.
        a:            The 'a' parameters.
        b:            The 'b' parameters.

    Returns:
        The calculated objective.
    """
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


async def async_run(
    config: EnOptConfig,
    a_list: list[NDArray[np.float64]],
    b_list: list[NDArray[np.float64]],
) -> list[FunctionResults]:
    """Run the asynchronous code.

    Returns:
        The optimal results.
    """
    async_server = create_server("async_server", workers=2)
    assert isinstance(async_server, Server)
    async with asyncio.TaskGroup() as tg:
        async_server.start(tg)
        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    run_optimization,
                    async_server,
                    partial(rosenbrock, a=a, b=b),
                    config,
                )
                for a, b in zip(a_list, b_list, strict=True)
            ),
        )
        async_server.cancel()
    return results


def main() -> None:
    """Run the optimization."""
    rng = default_rng(seed=1234)

    realizations = len(CONFIG["realizations"]["weights"])
    a = [
        rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations) for _ in range(N_OPT)
    ]
    b = [
        rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)
        for _ in range(N_OPT)
    ]

    results = asyncio.run(async_run(EnOptConfig.model_validate(CONFIG), a, b))
    for optimal_result in results:
        assert optimal_result is not None
        assert optimal_result.functions is not None
        print(f"Optimal variables: {optimal_result.evaluations.variables}")
        print(f"Optimal objective: {optimal_result.functions.weighted_objective}\n")
        assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=1e-1)
        assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    main()
