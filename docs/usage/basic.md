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
from functools import partial
from typing import Any

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.results import FunctionResults, Results
from ropt.workflow import BasicOptimizer

DIM = 5
UNCERTAINTY = 0.1
REALIZATIONS = 10

def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluationBatchContext,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> EvaluationBatchResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for r_idx, r in enumerate(context.realizations):
        for d_idx in range(DIM - 1):
            x, y = variables[r_idx, d_idx : d_idx + 2]
            objectives[r_idx, 0] += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return EvaluationBatchResult(objectives=objectives)

def report(results: tuple[Results, ...]) -> None:
    for item in results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.target_objective}\n")

CONFIG: dict[str, Any] = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * REALIZATIONS,
    },
}
initial_values = 2 * np.arange(DIM) / DIM + 0.5

rng = default_rng(seed=123)
a = rng.normal(loc=1.0, scale=UNCERTAINTY, size=REALIZATIONS)
b = rng.normal(loc=100.0, scale=100 * UNCERTAINTY, size=REALIZATIONS)

optimizer = BasicOptimizer(CONFIG, partial(rosenbrock, a=a, b=b))
optimizer.set_results_callback(report)
optimizer.run(initial_values)

print(f"Optimal variables: {optimizer.results.evaluations.variables}")
print(f"Optimal objective: {optimizer.results.functions.target_objective}\n")
```

## The objective function

You must provide a Python function (`rosenbrock()` in our example) that `ropt`
can call to evaluate your objective function for a given set of variables. The
evaluator function receives the `variables` to be evaluated and an
[`EvaluationBatchContext`][ropt.evaluation.EvaluationBatchContext] object. The
context provides information such as which realizations to compute. The function
must return an [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult]
containing the calculated objective values.

The `variables` input is a matrix, where each row is a variable vector. We
iterate over the rows and calculate the Rosenbrock function for each variable
vector. The context object has a `realizations` field that maps the row index to
a realization number, which we use to find the correct values of the `a` and `b`
parameters. The resulting values are stored in the rows of the `objectives`
matrix. The number of columns of the `objectives` matrix is equal to one, since
this problem has only a single objective and no non-linear constraints.

## Configuration

The `BasicOptimizer` requires a configuration dictionary that is parsed into an
[`EnOptContext`][ropt.context.EnOptContext] object. Most of the optimization
parameters are set to their defaults when parsing the dictionary, here we
override only the most essential ones:

- **`variables`**: Specifies details about the optimization variables.
  `variable_count` is the number of variables, and `perturbation_magnitudes` is
  used for generating the perturbations that are used for the gradient
  calculations. We set only a single perturbation value that is used for all
  variables.
- **`realizations`**: Defines the ensemble. `weights` is a list where each entry
  corresponds to a realization. Here, we have 10 realizations with equal
  weights. The weights are equal to one, internally they will be normalized to a
  sum of 1, when necessary.
- **`gradient`**: Configures the stochastic gradient approximation.
  `number_of_perturbations` controls how many perturbations are used to estimate
  the gradient at each iteration.

## Reporting

The `report` callback receives intermediate results after each function
evaluation batch, allowing you to monitor optimization progress. The callback
receives a tuple of [Results][ropt.results.Results] objects. Each item in the
tuple may be a [FunctionResults][ropt.results.FunctionResults] (containing
objective and constraint evaluations) or a
[GradientResults][ropt.results.GradientResults] (containing gradient
information). In this example, we filter for `FunctionResults` and print the
current variables and objective value.

## Running the optimization

After defining the necessary functions and configuration, the main steps for
setting up and running the optimizer are:

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
    weighted objective are printed to the console, and the function returns the
    optimal result, an instance of
    [`FunctionResults`][ropt.results.FunctionResults].

## BasicOptimizer reference

[`BasicOptimizer`][ropt.workflow.BasicOptimizer] wraps the workflow framework
into a simple, single-run interface. The table below summarizes its API:

| Member                    | Description
| ------------------------- | ---------------------------------------------------------------------------------------------------
| `__init__(config, evaluator, *, constraint_tolerance=1e-10)` | Create the optimizer from a config dict and a [`EvaluationBatchCallback`][ropt.evaluation.EvaluationBatchCallback] callable or an [`Evaluator`][ropt.workflow.evaluators.Evaluator] object.
| `run(initial_values)`     | Execute the optimization starting from `initial_values`. Returns an [`ExitCode`][ropt.enums.ExitCode].
| `results`                 | Property returning the best [`FunctionResults`][ropt.results.FunctionResults] found, or `None`.
| `set_results_callback(cb)`| Register a callback `cb(results: tuple[FunctionResults, ...]) -> None` invoked after each evaluation batch.
| `set_abort_callback(cb)`  | Register a callback `cb() -> bool`; returning `True` aborts the run with [`ExitCode.USER_ABORT`][ropt.enums.ExitCode].

### Evaluator signature

When you pass a plain Python callable as the evaluator, it must have the
[`EvaluationBatchCallback`][ropt.evaluation.EvaluationBatchCallback] signature:
```python
def evaluator(
    variables: NDArray[np.float64],
    context: EvaluationBatchContext,
) -> EvaluationBatchResult:
    ...
```

See [Writing an Evaluator](evaluation_callbacks.md) for details.

### Under the hood

Internally, `BasicOptimizer` creates an optimization workflow that does the following:

1. Validates `config` into an [`EnOptContext`][ropt.context.EnOptContext].
2. Instantiates a
   [`BatchEvaluator`][ropt.workflow.evaluators.BatchEvaluator] object that
   calls your callable when needed.
3. Creates an [`EnsembleOptimizer`][ropt.workflow.compute_steps.EnsembleOptimizer]
   compute step.
4. Attaches a [`Tracker`][ropt.workflow.event_handlers.Tracker] to remember the
   best result.
5. Runs the step and exposes the best result via the `results` property.

If you need more control (multiple runs, custom handlers, async evaluation),
use the workflow framework directly — see
[Optimization Workflows](workflows.md).

### Customization

Sometimes you want to inject behavior — extra logging, telemetry, a custom
results store — into every `BasicOptimizer` run without modifying the call
sites. `BasicOptimizer` loads additional event handlers from a JSON file at:

```
<prefix>/share/ropt/options.json
```

where `<prefix>` is the Python installation prefix (or system data prefix).
Find it with:

```python
from sysconfig import get_paths
print(get_paths()["data"])
```

The JSON file lists handlers under `basic_optimizer.event_handlers` as
`module.ClassName` strings:

```json
{
    "basic_optimizer": {
        "event_handlers": ["mylogger.Logger"]
    }
}
```

`mylogger` must be importable from the active Python environment. The
referenced class must subclass
[`EventHandler`][ropt.workflow.event_handlers.EventHandler] and accept no
required constructor arguments.

## Where to next

- Look up any configuration key: [Configuration](configuration.md).
- Add per-realization logic or handle simulator failures:
  [Writing Evaluation Callbacks](evaluation_callbacks.md).
- Process or export the result objects:
  [Working with Results](results.md).
- Move beyond a single optimization run:
  [Optimization Workflows](workflows.md).
- Full runnable variants of this example: [examples/rosenbrock.py](https://github.com/TNO-ropt/ropt/blob/main/examples/rosenbrock.py)
