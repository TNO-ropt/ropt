# Executors

Executors run the [`WorkItem`][ropt.components.executors.WorkItem] objects of a
[`Submission`][ropt.components.executors.Submission], produced by a
[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator], on a concrete
execution mechanism (threads, processes, or an HPC cluster).

See [Parallel Evaluation](../workflows/parallel.md) for usage.

::: ropt.components.executors.Executor
::: ropt.components.executors.WorkItem
::: ropt.components.executors.Submission
::: ropt.components.executors.ThreadingExecutor
::: ropt.components.executors.MultiprocessingExecutor
::: ropt.components.executors.HPCExecutor

