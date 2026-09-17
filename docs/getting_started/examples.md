# Examples

Every script in the
[examples/simple](https://github.com/TNO-ropt/ropt/tree/main/examples/simple)
folder is listed here. They are short, and the test suite keeps them working, so
they are a reliable starting point to copy from.

They all use the [simple API](../running/running.md), which covers most
optimization tasks. The last column links the page of the manual that walks
through the script; read that page with the script open beside it.

Scripts that assemble a workflow by hand are listed under
[Workflow Examples](../advanced/examples.md).

| Script | What it shows | Explained in |
| --- | --- | --- |
| [`evaluate.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/evaluate.py) | Evaluating variable vectors without optimizing | [Running Optimizations](../running/running.md) |
| [`ensemble.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/ensemble.py) | Optimizing the mean objective over uncertain realizations | [Ensemble-Based Optimization](ensemble.md) |
| [`constrained.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/constrained.py) | Linear and nonlinear constraints | [Constraints](../optimizer_setup/constraints.md) |
| [`discrete.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/discrete.py) | Integer variables, solved with differential evolution | [Discrete and Mixed-Integer Variables](../optimizer_setup/discrete.md) |
| [`mixed.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/mixed.py) | Continuous and integer variables in one problem | [Discrete and Mixed-Integer Variables](../optimizer_setup/discrete.md) |
| [`realization_filter.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/realization_filter.py) | A custom filter that reweights realizations | [Realization Filters](../optimizer_setup/realization_filters.md) |
| [`function_estimator.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/function_estimator.py) | A custom estimator that aggregates realizations | [Function Estimators](../optimizer_setup/function_estimators.md) |
| [`sampler.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/sampler.py) | A custom sampler that perturbs one variable at a time | [Stochastic Gradients](../optimizer_setup/gradients.md) |
| [`metadata.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/metadata.py) | Tagging a run, and recording per-realization data | [Working with Results](../running/results.md) |
| [`export.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/export.py) | Exporting results to a pandas or polars frame | [Working with Results](../running/results.md) |
| [`handlers.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/handlers.py) | Collecting results from runs that overlap in time | [Result Handlers](../running/handlers.md) |
| [`stopping.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/stopping.py) | Stopping a run from the `report` callback | [Running Optimizations](../running/running.md) |
| [`failures.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/failures.py) | What a failing realization does, and how to allow some | [Common Pitfalls](../troubleshooting/index.md) |
| [`restart.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/restart.py) | Restarting from the best point, collecting every result | [Restarting from the Best Point](../running/restart.md) |
| [`parallel.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/parallel.py) | Evaluating on a thread or process pool | [Parallel Execution and Many Runs](../running/parallel.md) |
| [`optimize_many.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/optimize_many.py) | Running several optimizations concurrently | [Parallel Execution and Many Runs](../running/parallel.md) |
| [`nested_optimization.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/nested_optimization.py) | An inner optimization per outer evaluation, on its own pool | [Nested Optimization](../running/nested.md) |
| [`hpc.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/hpc.py) | Submitting evaluations to a cluster queue | [Parallel Execution and Many Runs](../running/parallel.md) |
