# Examples

Every script in the
[examples](https://github.com/TNO-ropt/ropt/tree/main/examples)
folder is listed here. They are short, and the test suite keeps them working, so
they can be copied and adapted as they stand.

The last column links the page of the manual that walks through the script;
read that page with the script open beside it.

| Script | What it shows | Explained in |
| --- | --- | --- |
| [`evaluate.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/evaluate.py) | Evaluating variable vectors without optimizing | [Running Optimizations](../running/running.md) |
| [`ensemble.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/ensemble.py) | Optimizing the mean objective over uncertain realizations | [Ensemble-Based Optimization](ensemble.md) |
| [`constrained.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/constrained.py) | Linear and nonlinear constraints | [Constraints](../optimizer_setup/constraints.md) |
| [`discrete.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/discrete.py) | Integer variables, solved with differential evolution | [Discrete and Mixed-Integer Variables](../optimizer_setup/discrete.md) |
| [`mixed.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/mixed.py) | Continuous and integer variables in one problem | [Discrete and Mixed-Integer Variables](../optimizer_setup/discrete.md) |
| [`realization_filter.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/realization_filter.py) | A custom filter that reweights realizations | [Realization Filters](../optimizer_setup/realization_filters.md) |
| [`function_estimator.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/function_estimator.py) | A custom estimator that aggregates realizations | [Function Estimators](../optimizer_setup/function_estimators.md) |
| [`sampler.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/sampler.py) | A custom sampler that perturbs one variable at a time | [Samplers](../optimizer_setup/samplers.md) |
| [`metadata.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/metadata.py) | Tagging an optimization, and recording per-realization data | [Working with Results](../results/results.md) |
| [`export.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/export.py) | Exporting results to a pandas or polars frame | [Working with Results](../results/results.md) |
| [`scaling.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/scaling.py) | Reading results in the configured and the optimizer's units | [Working with Results](../results/results.md) |
| [`handlers.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/handlers.py) | Collecting results from optimizations that overlap in time | [Result Handlers](../results/handlers.md) |
| [`tables.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/tables.py) | Collecting results into tables of both types with `DataFrameHandler` | [Result Handlers](../results/handlers.md#example-tables-of-both-types) |
| [`stopping.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/stopping.py) | Stopping an optimization from the result callback | [Running Optimizations](../running/running.md) |
| [`failures.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/failures.py) | What a failing realization does, and how to allow some | [Troubleshooting](../troubleshooting/index.md) |
| [`restart.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/restart.py) | Restarting from the best point, collecting every result | [Restarting from the Best Point](../running/restart.md) |
| [`initial_values.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/initial_values.py) | Restarting with a larger ensemble, reusing what is known there | [Reusing Results at the Starting Point](../running/initial_values.md) |
| [`parallel.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/parallel.py) | Evaluating on a thread or process pool | [Evaluating in Parallel](../running/parallel.md) |
| [`multiple_optimizations.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/multiple_optimizations.py) | Running several optimizations concurrently | [Running Multiple Optimizations](../running/many_runs.md) |
| [`nested_optimization.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/nested_optimization.py) | An inner optimization per outer evaluation, with its own pool | [Nested Optimization](../running/nested.md) |
| [`programs.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/programs.py) | Running a simulator from the objective, and stopping it on an abort | [Running External Programs](../running/programs.md) |
| [`simulator.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simulator.py) | A stand-in simulator, run by `programs.py` | [Running External Programs](../running/programs.md) |
| [`hpc.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/hpc.py) | Submitting evaluations to a cluster queue | [Evaluating in Parallel](../running/parallel.md#running-on-an-hpc-cluster) |
