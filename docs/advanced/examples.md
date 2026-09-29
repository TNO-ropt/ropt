# Workflow Examples

Every script in the
[examples/advanced](https://github.com/TNO-ropt/ropt/tree/main/examples/advanced)
folder is listed here. Each assembles a workflow from
[components](workflows.md) and runs it directly.

Scripts that use `ropt.simple` are listed under
[Examples](../getting_started/examples.md).

| Script | What it shows | Explained in |
| --- | --- | --- |
| [`workflow.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/workflow.py) | Compute steps and event handlers assembled by hand | [Optimization Workflows](workflows.md) |
| [`nested.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested.py) | Nested optimization, sequential and in one process | [Parallel Evaluation](parallel.md) |
| [`nested_parallel.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested_parallel.py) | The same flow with process and thread executors | [Parallel Evaluation](parallel.md) |
| [`parallel_evaluator.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/parallel_evaluator.py) | Several optimizations in parallel on one executor | — |
| [`hpc_executor.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/hpc_executor.py) | Managing a cluster executor explicitly | — |
