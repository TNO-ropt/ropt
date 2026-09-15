# Ensemble-Based Optimization

The [Quickstart](quickstart.md) minimized a single, fixed objective. Here we
work through an *uncertain* problem: the objective depends on parameters we do
not know exactly. We now have a set of functions, each with different parameters
drawn from some (possibly unknown) probability distribution. Each member of the
set is a **realization**. The full runnable script is
[examples/simple/ensemble.py](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/ensemble.py).

`ropt` optimizes the realizations together by combining them into a single
**robust objective** — by default a weighted average over the realizations.
Minimizing the average yields a solution that performs well across the whole set
rather than for one particular case. We minimize the Rosenbrock function again,
but now its two coefficients vary between realizations.

For realization $i$, with coefficients $a_i$ and $b_i$, the per-realization
objective is:

$$ f_i(\mathbf{x}) = \sum_{k=1}^{n-1} \left[ (a_i - x_k)^2 + b_i \left(
x_{k+1} - x_k^2 \right)^2 \right] $$

which is the Rosenbrock function of the [Quickstart](quickstart.md), generalized
to $n$ variables: it reduces to that function when $a_i = 1$ and $b_i = 100$ for
every realization. `ropt` combines the realizations into the robust objective:

$$ f(\mathbf{x}) = \sum_i w_i f_i(\mathbf{x}), $$

with weights $w_i$ that we set in the configuration below.

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

The [Quickstart](quickstart.md) ignored this second argument; an ensemble
objective uses `context.realization` to select the parameters for the
realization it is computing. `ropt` combines the per-realization values into the
robust objective for you.

## 4. Run it

The call is the same as for a deterministic problem:

```python
from ropt.simple import optimize

initial_values = 2 * np.arange(DIM) / DIM + 0.5
result = optimize(config, initial_values, rosenbrock)
```

`ropt` evaluates all ten realizations at each point, averages them into the
robust objective, and optimizes that. To follow the run while it proceeds, pass
a `report` callback; see
[Reporting progress](../running/running.md#reporting-progress).

## 5. Read the result

`optimize` returns an [`OptimizeResult`][ropt.simple.OptimizeResult]:

```python
print(f"exit code:         {result.exit_code}")
print(f"optimal variables: {result.variables}")
print(f"optimal objective: {result.target_objective}")
```

- `result.variables` is the best set of variables found, and
  `result.target_objective` the robust objective value there. Both are `None` if
  the run produced no valid result.
- `result.exit_code` says why the run stopped.
- `result.results` holds the full low-level result, if you need every detail.

See [The result](../running/running.md#the-result) for the remaining fields.

Because the coefficients are centered on the values used in the
[Quickstart](quickstart.md), the robust optimum still lies close to where all
variables equal 1 — but it minimizes the *average* over the uncertain
coefficients rather than any single realization.

## Where to next

- Collect every result, not just the best one:
  [Collecting Results with Handlers](handlers.md).
- The complete simple API: [Running Optimizations](../running/running.md).
- All realization settings: [Configuration](../optimizer_setup/configuration.md).
- The ideas and terms behind ensembles:
  [Key Concepts](../optimizer_setup/key_concepts.md).
