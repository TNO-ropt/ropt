# Examples

Every runnable script in the
[examples](https://github.com/TNO-ropt/ropt/tree/main/examples) folder is listed
here. The scripts are short and are kept working by the test suite, so they are
a reliable starting point to copy from.

The manual explains; these scripts run. Where a page of the manual walks a
script, it is linked in the last column — read that page and keep the script
open beside it. The remaining scripts are documented by their own docstrings,
and pages for them are still being written.

## Simple API

These use the [simple API](../running/running.md), which covers most
optimization tasks.

| Script | What it shows | Explained in |
| --- | --- | --- |
| [`evaluate.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/evaluate.py) | Evaluating variable vectors without optimizing | — |
| [`ensemble.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/ensemble.py) | Optimizing the mean objective over uncertain realizations | [Ensemble-Based Optimization](../getting_started/ensemble.md) |
| [`constrained.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/constrained.py) | Linear and nonlinear constraints | [Constraints](../optimizer_setup/constraints.md) |
| [`discrete.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/discrete.py) | Integer variables, solved with differential evolution | [Mixed-Integer Optimization](discrete.md) |
| [`mixed.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/mixed.py) | Continuous and integer variables in one problem | — |
| [`realization_filter.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/realization_filter.py) | A custom filter that reweights realizations | — |
| [`metadata.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/metadata.py) | Tagging a run, and recording per-realization data | [Attaching metadata](../running/running.md#attaching-metadata) |
| [`restart.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/restart.py) | Restarting from the best point, collecting every result | [Restarting from the Best Point](restart.md) |
| [`parallel.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/parallel.py) | Evaluating on a thread or process pool | — |
| [`optimize_many.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/optimize_many.py) | Running several optimizations concurrently | — |
| [`handlers.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/handlers.py) | Collecting results from runs that overlap in time | — |
| [`nested_optimization.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/nested_optimization.py) | An inner optimization per outer evaluation, on its own pool | — |
| [`hpc.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/simple/hpc.py) | Submitting evaluations to a cluster queue | — |

## Low-level API

These assemble the [workflow components](../workflows/workflows.md) by hand, for
cases the simple API does not cover.

| Script | What it shows | Explained in |
| --- | --- | --- |
| [`workflow.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/workflow.py) | Compute steps and event handlers assembled by hand | [Building a Workflow](workflow.md) |
| [`nested.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested.py) | Nested optimization, sequential and in one process | [Parallel Evaluation](../workflows/parallel.md) |
| [`nested_parallel.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested_parallel.py) | The same flow with process and thread executors | [Parallel Evaluation](../workflows/parallel.md) |
| [`parallel_evaluator.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/parallel_evaluator.py) | Several optimizations in parallel, driven by asyncio | — |
| [`hpc_executor.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/hpc_executor.py) | Managing a cluster executor explicitly | — |
