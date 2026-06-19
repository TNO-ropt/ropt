# Quickstart

This page shows the smallest useful `ropt` program: minimizing the deterministic
multi-dimensional Rosenbrock function with the
[`BasicOptimizer`][ropt.workflow.BasicOptimizer] class.

## Prerequisites

Install `ropt` (see [Installation](installation.md)):

```bash
pip install ropt
```

## A minimal optimization

```python
import numpy as np
from numpy.typing import NDArray

from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult
from ropt.workflow import BasicOptimizer

DIM = 5
CONFIG = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
}

def rosenbrock(variables: NDArray[np.float64], _: EvaluationBatchContext) -> EvaluationBatchResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v_idx in range(variables.shape[0]):
        for d_idx in range(DIM - 1):
            x, y = variables[v_idx, d_idx : d_idx + 2]
            objectives[v_idx, 0] += (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return EvaluationBatchResult(objectives=objectives)

initial_values = 2 * np.arange(DIM) / DIM + 0.5
optimizer = BasicOptimizer(CONFIG, rosenbrock)
optimizer.run(initial_values)

print(f"Optimal variables: {optimizer.results.evaluations.variables}")
print(f"Optimal objective: {optimizer.results.functions.target_objective}")
```

Running this script optimizes the 5-dimensional Rosenbrock function from the
starting point `initial_values` and prints the best variables and weighted
objective value found.

## What just happened

Three pieces of information drive every `ropt` optimization:

1. **A configuration dict** — describes the optimization problem (variables,
   objectives, constraints, gradient settings, optimizer choice). Here we set
   only the minimum: how many variables there are, and a small
   `perturbation_magnitudes` value used to estimate gradients numerically.
2. **An evaluator** — a Python callable that takes a matrix of variable vectors
   plus an [`EvaluationBatchContext`][ropt.evaluation.EvaluationBatchContext]
   and returns an
   [`EvaluationBatchResult`][ropt.evaluation.EvaluationBatchResult] with the
  objective values for each row. The variable matrix has one row per requested
  evaluation. `ropt` may request several rows in a single call so that an
  evaluator can compute them in parallel.
3. **A driver** — here, [`BasicOptimizer`][ropt.workflow.BasicOptimizer]. It
   wires the configuration and evaluator together, executes the optimization,
   and exposes the best result through its `results` property.


## Where to next

- A guided walkthrough of `BasicOptimizer` features:
  [Running a basic optimization task](basic.md).
- The full configuration vocabulary, section by section:
  [Configuration](configuration.md).
- Writing more advanced evaluation callbacks (per-realization data, partial failures):
  [Writing Evaluation Callbacks](evaluation_callbacks.md).
- Full runnable example:
  [examples/rosenbrock_deterministic.py](https://github.com/TNO-ropt/ropt/blob/main/examples/rosenbrock_deterministic.py).
