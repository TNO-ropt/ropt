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
rather than for one particular case. We minimize the Rosenbrock function,
generalized to a sum of $n$ terms over $n + 1$ variables, with coefficients that
vary between realizations.

For realization $i$, with coefficients $a_i$ and $b_i$, the per-realization
objective is:

$$ f_i(\mathbf{x}) = \sum_{k=1}^{n} \left[ (a_i - x_k)^2 + b_i \left(
x_{k+1} - x_k^2 \right)^2 \right] $$

which reduces to the standard Rosenbrock function of the
[Quickstart](quickstart.md) when $n = 1$, with $a_i = 1$ and $b_i = 100$ for
every realization.

`ropt` combines the realizations into the robust objective:

$$ f(\mathbf{x}) = \sum_i w_i f_i(\mathbf{x}), $$

with weights $w_i$ that we set in the configuration below.

## 1. Describe the problem

The config adds a `realizations` section: the `weights` list has one entry per
realization that sets how much each contributes to the combined objective:

```python
--8<-- "examples/simple/ensemble.py:config"
```

The weights need not sum to one; `ropt` normalizes them. Equal weights, as here,
give a plain average. See [Configuration Sections](../optimizer_setup/configuration_sections.md#realizations)
for the other realization settings. `INITIAL_VALUES` is the point the
optimization starts from.

## 2. Draw the uncertain parameters

Each realization is one draw of the uncertain parameters. Here the two Rosenbrock
coefficients are sampled once per realization, so that `a[r]` and `b[r]` are the
coefficients for realization `r`:

```python
--8<-- "examples/simple/ensemble.py:draws"
```

## 3. Write the evaluation function

`ropt` calls the evaluation function once for every realization at each point it
evaluates, so it must return the value for *its own* realization. The second
argument identifies which one: `context.realization` is the realization number,
which we use to index the parameter arrays:

```python
--8<-- "examples/simple/ensemble.py:objective"
```

The [Quickstart](quickstart.md) ignored this second argument; an ensemble
objective uses `context.realization` to select the parameters for the
realization it is computing. `ropt` combines the per-realization values into the
robust objective for you.

Returning a single number, as here, is the simplest case. A function with
multiple objectives, or with constraints, returns a sequence instead — the
objectives first, then the constraints; see
[Constraints](../optimizer_setup/constraints.md).

## 4. Follow the progress (optional)

To track a running optimization, pass a `report` callback: `ropt` calls it after
every evaluation, with the
[`FunctionResults`][ropt.results.FunctionResults] describing what was just
computed.

```python
--8<-- "examples/simple/ensemble.py:report"
```

The callback belongs to one run and sees one evaluation at a time. To keep the
results rather than just look at them — or to collect them across several runs
— use a handler instead; see
[Collecting Results with Handlers](handlers.md).

## 5. Run it

The call is the same as for a deterministic problem, with `INITIAL_VALUES` the
start point defined above:

```python
--8<-- "examples/simple/ensemble.py:run"
```

`ropt` evaluates all ten realizations at each point and averages them into the
robust objective, which is what it optimizes.

## 6. Read the result

[`optimize`][ropt.simple.optimize] returns an
[`OptimizationResult`][ropt.simple.OptimizationResult]:

```python
--8<-- "examples/simple/ensemble.py:result"
```

- `result.exit_code` indicates why the run stopped (a member of the
  [`ExitCode`][ropt.enums.ExitCode] enumeration).
- `result.results` is the best evaluation the run reached, a
  [`FunctionResults`][ropt.results.FunctionResults] carrying every value that
  evaluation produced — the same object a handler receives. It is `None` if the
  run produced no valid result, so one check covers every value read from it.
- `result.results.variables` is the best set of variables found, and
  `result.results.target_objective` the robust objective value there.

That evaluation carries much more than those two values — the per-realization
objectives, the aggregates, the ensemble weights; see
[Working with Results](../running/results.md).

Because the coefficients are centered on the values used in the
[Quickstart](quickstart.md), the robust optimum still lies close to where all
variables equal 1 — but it minimizes the *average* over the uncertain
coefficients rather than any single realization.

## See also

- The ideas and terms behind ensembles:
  [Key Concepts](../optimizer_setup/key_concepts.md).
