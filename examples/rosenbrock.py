"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
write a minimal configuration and how to run and monitor the optimization.

This script demonstrate different ways to run the optimization:
   1. Use multiple perturbations or only one with the `merge_realizations` option.
   2. Use a function callback or an batch evaluation callback.
   3. Use a basic optimizer a workflow.

You can select each of these options independently using the command line arguments:

    usage: python rosenbrock.py [-h] [--merge] [--function] [--workflow]

    options:
    -h, --help  show this help message and exit
    --merge     Merge the realizations in gradient calculation
    --function  Use a function callback instead of an batch evaluation callback
    --workflow  Use a workflow instead of basic optimizer
"""

import argparse
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.context import EnOptContext
from ropt.enums import EnOptEventType
from ropt.evaluation import (
    EvaluationBatchCallback,
    EvaluationBatchContext,
    EvaluationBatchResult,
)
from ropt.events import EnOptEvent
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer
from ropt.workflow.compute_steps import OptimizationStep
from ropt.workflow.evaluators import BatchEvaluator, Evaluator, FunctionEvaluator
from ropt.workflow.event_handlers import Observer, Tracker

DIM = 5
CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}
INITIAL_VALUES = 2 * np.arange(DIM) / DIM + 0.5
UNCERTAINTY = 0.1


def rosenbrock_evaluator_callback(
    variables: NDArray[np.float64],
    context: EvaluationBatchContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationBatchResult:
    """Batch evaluation callback for the multi-dimensional rosenbrock function.

    This batch evaluation callback should handle multiple variable vectors,
    since it is internally used by the `BasicOptimizer` in combination with a
    `BatchEvaluator` to handle all variable vectors in a batch.

    Args:
        variables: The variables to evaluate.
        context:   Evaluator context.
        a:         The 'a' parameters.
        b:         The 'b' parameters.

    Returns:
        An `EvaluationBatchResult` object containing the calculated objectives.
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluationBatchResult(objectives=objectives)


def rosenbrock_function_callback(  # noqa: PLR0913, PLR0917
    variables: NDArray[np.float64],
    realization: int,
    perturbation: int,  # noqa: ARG001
    batch_id: int,  # noqa: ARG001
    eval_idx: int,  # noqa: ARG001
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64] | dict[str, Any]:
    """Function callback for the multi-dimensional rosenbrock function.

    This function callback should handle a single variable vector, since it is
    used in combination with a `FunctionEvaluator` that calls this function for
    each variable vector in a batch.

    Args:
        variables:    1-D variable vector for this evaluation.
        realization:  The realization index.
        perturbation: The perturbation index (`-1` when unperturbed).
        batch_id:     Integer identifying the current evaluation batch.
        eval_idx:     Row index within the batch.
        a:         The 'a' parameters.
        b:         The 'b' parameters.

    Returns:
        The evaluation result as an array or a dictionary.
    """
    objective = 0.0
    for idx in range(DIM - 1):
        x, y = variables[idx : idx + 2]
        objective += (a[realization] - x) ** 2 + b[realization] * (y - x * x) ** 2
    return np.array([[objective]], dtype=np.float64)


def report(event_or_results: EnOptEvent | tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    This function can be used as a callback for both events and results.
    Normally, you would only handle one type depending on if you use a basic
    optimizer or a workflow.

    Args:
        event_or_results: The event or results to process.
    """
    results = (
        event_or_results.results
        if isinstance(event_or_results, EnOptEvent)
        else event_or_results
    )
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")


def run_optimization(
    config: dict[str, Any],
    evaluator: Evaluator | EvaluationBatchCallback,
    *,
    workflow: bool = False,
) -> FunctionResults:
    """Run the optimization.

    Args:
        config:    The optimization configuration.
        evaluator: The evaluator to use.
        workflow:  If True, use a workflow instead of the basic optimizer.

    Returns:
        The optimal results.
    """
    optimal_result: FunctionResults | None
    if workflow:
        if not isinstance(evaluator, Evaluator):
            evaluator = BatchEvaluator(callback=evaluator)
        step = OptimizationStep(evaluator=evaluator)
        tracker = Tracker()
        step.add_event_handler(tracker)
        reporter = Observer(
            callback=report, event_types={EnOptEventType.FINISHED_EVALUATION}
        )
        step.add_event_handler(reporter)
        step.run(variables=INITIAL_VALUES, context=EnOptContext.model_validate(CONFIG))
        optimal_result = tracker["results"]
    else:
        optimizer = BasicOptimizer(config=config, evaluator=evaluator)
        optimizer.set_results_callback(report)
        optimizer.run(INITIAL_VALUES)
        optimal_result = optimizer.results
    assert optimal_result is not None
    return optimal_result


def main(**kwargs: dict[str, Any]) -> None:
    """Run the example and check the result."""
    merge = kwargs.get("merge", False)
    realizations = 50 if merge else 10
    CONFIG.update(
        {
            "realizations": {
                "weights": [1.0] * realizations,
            },
            "gradient": {
                "number_of_perturbations": 1 if merge else 5,
                "merge_realizations": merge,
            },
        }
    )
    rng = default_rng(seed=123)
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)

    evaluator = (
        FunctionEvaluator(function=partial(rosenbrock_function_callback, a=a, b=b))
        if kwargs.get("function")
        else partial(rosenbrock_evaluator_callback, a=a, b=b)
    )

    optimal_result = run_optimization(
        CONFIG, evaluator, workflow=bool(kwargs.get("workflow"))
    )
    assert optimal_result is not None
    assert optimal_result.functions is not None

    print(f"Optimal variables: {optimal_result.evaluations.variables}")
    print(f"Optimal objective: {optimal_result.functions.target_objective}\n")

    assert np.allclose(optimal_result.functions.target_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python rosenbrock.py")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="merge the realizations in gradient calculation",
    )
    parser.add_argument(
        "--function",
        action="store_true",
        help="use a function callback instead of a batch evaluation callback",
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="use a workflow instead of basic optimizer",
    )
    main(**vars(parser.parse_args()))
