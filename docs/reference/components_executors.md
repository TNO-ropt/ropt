# Executors

Executors dispatch [`Task`][ropt.components.executors.Task] objects produced by a
[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator] to a concrete
execution mechanism (threads, processes, or an HPC cluster).

See [Parallel Evaluation](../low_level/parallel.md) for usage.

::: ropt.components.executors.Executor
::: ropt.components.executors.Task
::: ropt.components.executors.ResultsQueue
::: ropt.components.executors.ThreadingExecutor
::: ropt.components.executors.MultiprocessingExecutor
::: ropt.components.executors.HPCExecutor

