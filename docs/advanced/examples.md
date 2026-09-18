# Workflow Examples

Every script in the
[examples/advanced](https://github.com/TNO-ropt/ropt/tree/main/examples/advanced)
folder is listed here. They assemble the [workflow components](workflows.md) by
hand, for the cases the simple API does not cover.

Scripts that use the simple API are listed under
[Examples](../getting_started/examples.md).

| Script | What it shows | Explained in |
| --- | --- | --- |
| [`workflow.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/workflow.py) | Compute steps and event handlers assembled by hand | [Optimization Workflows](workflows.md) |
| [`nested.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested.py) | Nested optimization, sequential and in one process | [Parallel Evaluation](parallel.md) |
| [`nested_parallel.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/nested_parallel.py) | The same flow with process and thread executors | [Parallel Evaluation](parallel.md) |
| [`parallel_evaluator.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/parallel_evaluator.py) | Several optimizations in parallel, driven by asyncio | — |
| [`hpc_executor.py`](https://github.com/TNO-ropt/ropt/blob/main/examples/advanced/hpc_executor.py) | Managing a cluster executor explicitly | — |
