# Running a basic optimization script

Consider the Rosenbrock function, a standard test problem:

$$ f(x,y) = (a - x)^2 + b (y - x^2)^2, $$

which has a global minimum of $f(x, y) = 0$ at $(x, y) = (a, a^2)$ .

Here's an example optimizing the Rosenbrock function for $a = 1$ and $b = 100$:

```python
import numpy as np
from numpy.typing import NDArray

from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer


def rosenbrock(
    variables: NDArray[np.float64],                                   # (1)!
    context: EvaluatorContext                                         # (2)!
) -> EvaluatorResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for idx in range(variables.shape[0]):
        x, y = variables[idx, :]
        objectives[idx, 0] = (1.0 - x) ** 2 + 100 * (y - x * x) ** 2
    return EvaluatorResult(objectives=objectives)                     # (3)!

CONFIG = {                                                            # (4)!
    "variables": {"initial_values": [0.5, 2.0]},
    "gradient": {"perturbation_magnitudes": 1e-5}                     # (5)!
}

optimum = BasicOptimizer(CONFIG, rosenbrock).run().results            # (6)!

print(optimum.evaluations.variables, optimum.functions.weighted_objective)
```

1. The variables to optimize ($x, y$) are passes as a single `numpy` array. The
   function may receive multiple variable vectors to evaluate, hence the input
   is a matrix where the variable vectors are the rows of the matrix.
2. Additional information is passes via an
   [`EvaluatorContext`][ropt.evaluator.EvaluatorContext] object. It is not
   needed in this case.
3. Results are returned via an
   [`EvaluatorResult`][ropt.evaluator.EvaluatorResult] object. The objectives
   result is a matrix since multiple input vectors and multiple objectives may
   be evaluated.
4. Create an optimizer configuration with default values except for initial
   values and perturbation magnitudes.
5. Set perturbation magnitudes to a small value for accurate gradient
   estimation.
6. Make an [`BasicOptimizer`][ropt.plan.BasicOptimizer] plan, run it, and
   retrieve the results.

Running this will print the estimated optimal variables and the corresponding
function value:

```python
[1.00117794 1.0023715 ] 1.4078103983185034e-06
```

This example uses the [BasicOptimizer][ropt.plan.BasicOptimizer] class which
provides a simplified interface for running optimizations. More complex
optimization workflows can be implemented using the [plan][ropt.plan]
functionality.