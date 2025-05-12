# The `BasicOptimizer` class

For straightforward optimization tasks involving a single algorithm run, the
[`BasicOptimizer`][ropt.plan.BasicOptimizer] class typically provides all the
necessary functionality. While more intricate optimization workflows can be
constructed using the general [`plan`][ropt.plan] system, the `BasicOptimizer`
serves as an excellent introduction to several fundamental concepts within
`ropt`:

- [**Function Evaluation**](evaluation.md): You, the user, must supply the
  objective function to be optimized. This is done by providing a Python
  function that adheres to the following signature:
  ```python
  evaluator: Callable[[NDArray[np.float64], EvaluatorContext], EvaluatorResult]
  ```
  This function receives the variables to evaluate, together with a
  [`EvaluatorContext`][ropt.evaluator.EvaluatorContext] object, that provides
  some additional information, and returns a
  [`EvaluatorResult`][ropt.evaluator.EvaluatorResult] object containing the
  results.
- [**Configuration**](configuration.md): The behavior of the optimization
  process is controlled through a configuration object, specifically an instance
  of [`EnOptConfig`][ropt.config.enopt.EnOptConfig].
- [**Domain Transforms**](domain_transforms.md): The numerical range (domain)
  used internally by the optimizer might differ from the domain you define for
  your variables, objectives and/or constraints (e.g., for scaling purposes).
  Domain transforms manage the mapping between these internal optimizer values
  and your user-defined variable and function values.
- [**Results**](results.md): Retrieve and inspect intermediate and final results.

To illustrate these concepts, let's begin with a simple example. We'll optimize
the Rosenbrock function, a standard benchmark problem in optimization:

$$ f(x,y) = (a - x)^2 + b (y - x^2)^2, $$

which has a global minimum of $f(x, y) = 0$ at $(x, y) = (a, a^2)$ .


### Deterministic optimization

Here's an example optimizing the Rosenbrock function for $a = 1$ and $b = 100$,
using a `BasicOptimizer` object (press :material-plus-circle-outline: for
explanations):

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
    for v in range(variables.shape[0]):
        x, y = variables[v, :]
        objectives[v, 0] = (1.0 - x)**2 + 100 * (y - x * x)**2
    return EvaluatorResult(objectives=objectives)                     # (3)!

CONFIG = {                                                            # (4)!
    "variables": {"initial_values": [0.5, 2.0]},
    "gradient": {"perturbation_magnitudes": 1e-5}                     # (5)!
}

optimizer = BasicOptimizer(CONFIG, rosenbrock)                        # (6)!
optimum = optimizer.run().results                                     # (7)!
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
6. Make an [`BasicOptimizer`][ropt.plan.BasicOptimizer] object
7. Run the optimizer and retrieve the optimal [`Results`][ropt.results.Results]
   object.

Running this will print the estimated optimal variables and the corresponding
function value:

```python
[1.00117794 1.0023715 ] 1.4078103983185034e-06
```

This example illustrates how to implement a deterministic objective function
using a standard Python function. For a deeper dive into the function's expected
inputs and outputs, refer to the [Function Evaluation](evaluation.md) section.
The configuration provided here is minimal; it only specifies the mandatory
initial variable values and the perturbation magnitude used for numerical
gradient estimation. You can find comprehensive details on optimizer settings in
the [Configuration](configuration.md) section.


### Robust optimization

Let's modify the example to optimize an ensemble of Rosenbrock functions, where
we draw the values of $a$ and $b$ from a normal distribution (press
:material-plus-circle-outline: for explanations):

```py
from functools import partial
import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer

REALIZATIONS = 10                                                     # (1)!

rng = default_rng(seed=123)                                           # (2)!
a = rng.normal(loc=1.0, scale=0.1, size=REALIZATIONS)
b = rng.normal(loc=100.0, scale=10, size=REALIZATIONS)

def rosenbrock(
    variables: NDArray[np.float64],
    context: EvaluatorContext,
    a: NDArray[np.float64],                                           # (3)!
    b: NDArray[np.float64],
) -> EvaluatorResult:
    objectives = np.zeros((variables.shape[0], 1), dtype=np.float64)
    for v, r in enumerate(context.realizations):                      # (4)!
        x, y = variables[v, :]
        objectives[v, 0] = (a[r] - x)**2 + b[r] * (y - x * x)**2      # (5)!
    return EvaluatorResult(objectives=objectives)

CONFIG = {
    "variables": {"initial_values": [0.5, 2.0]},
    "gradient": {"perturbation_magnitudes": 1e-5},
    "realizations": {"weights": [1.0] * REALIZATIONS},                # (6)!
}

func = partial(rosenbrock, a=a, b=b)                                  # (7)!
optimum = BasicOptimizer(CONFIG, func).run().results
print(optimum.evaluations.variables, optimum.functions.weighted_objective)
```

1.  Define the ensemble size as 10, meaning we'll work with 10 different
    versions (realizations) of the parameters $a$ and $b$.
2.  Generate 10 random values for $a$ and $b$ from normal distributions using
    `numpy.random.default_rng`. These arrays represent the parameters for each
    realization.
3.  Modify the `rosenbrock` function signature to accept the `a` and `b`
    parameter arrays, making it aware of the different realizations.
4.  Inside the function, iterate through the input `variables`. Use
    `context.realizations` to get the index `r` identifying which realization
    corresponds to the current variable vector being evaluated.
5.  Use the realization index `r` to access the specific `a[r]` and `b[r]`
    values needed for the objective function calculation for that realization.
6.  Update the `CONFIG` dictionary. Specify the ensemble by providing a list of
    `weights` under the `realizations` key. The length of this list determines
    the number of realizations (10 in this case), and the values assign a weight
    to each (here, all are weighted equally).
7.  Use `functools.partial` to create a new function `func`. This pre-fills the
    `a` and `b` arguments of the `rosenbrock` function, resulting in a callable
    that matches the signature expected by `BasicOptimizer` (`(variables,
    context) -> EvaluatorResult`).

Running this will print the estimated optimal variables and the corresponding
function value:

```python
[1.00878109 1.01754239] 0.004814803947156314
```
