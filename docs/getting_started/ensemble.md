# Ensemble-Based Optimization

The [Deterministic Optimization](deterministic.md) page minimized a single,
fixed objective. Here we work through an *uncertain* problem: the objective
depends on parameters we do not know exactly. We now have a set of functions,
each with different parameters drawn from some (possibly unknown) probability
distribution. Each member of the set is a **realization**. The full runnable
script is
[examples/simple/ensemble.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/ensemble.py).

`ropt` optimizes the realizations together by combining them into a single
**robust objective** — by default a weighted average over the realizations.
Minimizing the average yields a solution that performs well across the whole set
rather than for one particular case. We minimize the Rosenbrock function again,
but now its two coefficients vary between realizations.

## 1. Describe the problem

The config adds a `realizations` section next to the variables. The `weights`
list has one entry per realization and sets how much each contributes to the
combined objective:

```python
DIM = 5
config = {
    "variables": {
        "variable_count": DIM,
        "perturbation_magnitudes": 1e-6,
    },
    "realizations": {
        "weights": [1.0] * 10,   # ten equally weighted realizations
    },
}
```

The weights need not sum to one; `ropt` normalizes them. Equal weights, as here,
give a plain average. See [Configuration](../optimizer_setup/configuration.md) for the
other realization settings.

## 2. Draw the uncertain parameters

Each realization is one draw of the uncertain parameters. Here the two Rosenbrock
coefficients are sampled once per realization, so that `a[r]` and `b[r]` are the
coefficients for realization `r`:

```python
import numpy as np

rng = np.random.default_rng(seed=123)
a = rng.normal(loc=1.0, scale=0.1, size=10)
b = rng.normal(loc=100.0, scale=10.0, size=10)
```

## 3. Write the evaluation function

`ropt` calls the evaluation function once for every realization at each point it
evaluates, so it must return the value for *its own* realization. The second
argument tells it which one: `context.realization` is the realization number,
which we use to index the parameter arrays:

```python
from ropt.simple import EvaluationFunctionContext


def rosenbrock(variables: np.ndarray, context: EvaluationFunctionContext) -> float:
    r = context.realization
    objective = 0.0
    for i in range(DIM - 1):
        x, y = variables[i : i + 2]
        objective += (a[r] - x) ** 2 + b[r] * (y - x * x) ** 2
    return float(objective)
```

The deterministic example ignored this second argument; an ensemble objective
uses `context.realization` to select the parameters for the realization it is
computing. `ropt` combines the per-realization values into the robust objective
for you.

## 4. Run it

The call is the same as for a deterministic problem:

```python
from ropt.simple import optimize

initial_values = 2 * np.arange(DIM) / DIM + 0.5
result = optimize(config, initial_values, rosenbrock)
```

`ropt` evaluates all ten realizations at each point, averages them into the
robust objective, and optimizes that. As in the deterministic example, you can
pass a `report` callback to watch the run.

## 5. Read the result

`optimize` returns an [`OptimizeResult`][ropt.simple.OptimizeResult], exactly as
in the deterministic case:

```python
print(f"optimal variables: {result.variables}")
print(f"optimal objective: {result.target_objective}")
```

Because the coefficients are centered on the deterministic values, the robust
optimum still lies near all ones — but it minimizes the *average* over the
uncertain coefficients rather than any single realization.

## Where to next

- The complete simple API: [Running Optimizations](../running/running.md).
- All realization settings: [Configuration](../optimizer_setup/configuration.md).
- The ideas and terms behind ensembles:
  [Key Concepts](../optimizer_setup/key_concepts.md).
