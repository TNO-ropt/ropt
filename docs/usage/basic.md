# Running a basic optimization task

The `ropt` library provides a [`BasicOptimizer`][ropt.workflow.BasicOptimizer]
class that simplifies running an optimization task.

This section walks you through an example of how to use `BasicOptimizer` to
solve a simple optimization problem. We minimize the multi-dimensional
Rosenbrock function, where we introduce some uncertainty in its parameters
across an ensemble of realizations.

## The complete example

Below is the full Python script for this example. We will go through each part
of the script in the following sections.

```python
"""Example of optimization of a multi-dimensional Rosenbrock test function.

This example demonstrates optimization of the a modified multi-dimensional
Rosenbrock function that exhibits uncertainty in its parameters. It shows how to
write a minimal configuration and how to run and monitor the optimization.
"""

from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

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
    context: EvaluatorContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluatorResult:
    """Function evaluator for the multi-dimensional rosenbrock function.

    This function returns a tuple containing the calculated objectives and
    `None`, the latter because no constraints are calculated.

    Args:
        variables: The variables to evaluate.
        context:   Evaluator context.
        a:         The 'a' parameters.
        b:         The 'b' parameters.

    Returns:
        The calculated objective, and `None`
    """
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluatorResult(objectives=objectives)


def report(results: tuple[Results, ...]) -> None:
    """Report results of an evaluation.

    Args:
        results: The results.
    """
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


def run_optimization(config: dict[str, Any]) -> FunctionResults:
    """Run the optimization.

    Args:
        config: The configuration of the optimizer.

    Returns:
        The optimal results.
    """
    rng = default_rng(seed=123)

    realizations = len(config["realizations"]["weights"])
    a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=realizations)
    b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=realizations)

    optimizer = BasicOptimizer(CONFIG, partial(rosenbrock, a=a, b=b))
    optimizer.set_results_callback(report)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert optimizer.results.functions is not None

    print(f"Optimal variables: {optimizer.results.evaluations.variables}")
    print(f"Optimal objective: {optimizer.results.functions.weighted_objective}\n")

    return optimizer.results


def main() -> None:
    """Run the example and check the result."""
    optimal_result = run_optimization(CONFIG)
    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=1e-1)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=1e-1)


if __name__ == "__main__":
    main()
```

## Configuration

The `BasicOptimizer` requires a configuration dictionary that is parsed into an
[`EnOptConfing`][ropt.config.EnOptConfig] object. Most of the optimization
parameters are set to their defaults when parsing the dictionary, here we
override only the most essential ones:

- **`variables`**: Specifies details about the optimization variables.
  `variable_count` is the number of variables, and `perturbation_magnitudes` is
  used for generating the perturbations that are used for the gradient
  calculations.
- **`realizations`**: Defines the ensemble. `weights` is a list where each entry
  corresponds to a realization. Here, we have 10 realizations with equal
  weights.
- **`gradient`**: Configures the stochastic gradient approximation.
  `number_of_perturbations` controls how many perturbations are used to estimate
  the gradient at each iteration.

## The objective function

You must provide a Python function (`rosenbrock()` in our example) that `ropt`
can call to evaluate your objective function for a given set of variables. The
evaluator function receives the `variables` to be evaluated and an
[`EvaluatorContext`][ropt.evaluator.EvaluatorContext] object. The context
provides information such as which realizations to compute. The function must
return an [`EvaluatorResult`][ropt.evaluator.EvaluatorResult] containing the
calculated objective values.

## Running the optimization

The `run_optimization` function shows the main steps for setting up and running
the optimizer:

1.  **Initialize uncertain parameters**:
    Before instantiating the optimizer, the `run_optimization` function
    initializes a random number generator (`rng = default_rng(seed=123)`). This
    generator is then used to create the uncertain parameters `a` and `b` for
    the Rosenbrock function, simulating variability across realizations.

2.  **Instantiate `BasicOptimizer`**:
    We create an instance of `BasicOptimizer`, passing the configuration
    dictionary and the objective function. We use `functools.partial` to pass
    the uncertain parameters `a` and `b` to our `rosenbrock` function, ensuring
    they are available during objective function evaluations.

3.  **Set a results callback**:
    The `set_results_callback` method allows you to register a function that
    will be called after each optimization iteration with the current results.
    In this example, the `report` function is used to print the current
    variables and the weighted objective value, providing real-time feedback on
    the optimization progress.

4.  **Run the optimizer**:
    The `run` method starts the optimization process, beginning from the
    `initial_values`. `ropt` will now iteratively call your objective function
    to find the optimal variable values that minimize the weighted average of
    the objective across all realizations.

5.  **Retrieve results**:
    After the optimization completes, the final results are accessible via the
    `optimizer.results` attribute. The optimal variables and the corresponding
    weighted objective are printed to the console, and finally function then
    returns the optimal results,  an instance of
    [`FunctionResults`][ropt.results.FunctionResults].
