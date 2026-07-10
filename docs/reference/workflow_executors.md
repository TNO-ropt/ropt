# Executors

Executors dispatch [`Task`][ropt.workflow.executors.Task] objects produced by a
[`ParallelEvaluator`][ropt.workflow.evaluators.ParallelEvaluator] to a concrete
execution mechanism (threads, processes, or an HPC cluster).

See [Parallel Evaluation](../usage/parallel.md) for usage.

::: ropt.workflow.executors.Executor
::: ropt.workflow.executors.Task
::: ropt.workflow.executors.ThreadingExecutor
::: ropt.workflow.executors.MultiprocessingExecutor
::: ropt.workflow.executors.HPCExecutor

