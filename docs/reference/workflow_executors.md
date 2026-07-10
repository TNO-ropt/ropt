# Executors

Executors dispatch [`Task`][ropt.workflow.executors.Task] objects produced by a
[`ParallelEvaluator`][ropt.workflow.evaluators.ParallelEvaluator] to a concrete
execution mechanism (threads, processes, or an HPC cluster).
[`EventServer`][ropt.workflow.executors.EventServer] dispatches events from
worker threads to handlers running in the asyncio event loop.

See [Parallel Evaluation](../usage/parallel.md) for usage.

::: ropt.workflow.executors.Executor
::: ropt.workflow.executors.Task
::: ropt.workflow.executors.ThreadingExecutor
::: ropt.workflow.executors.MultiprocessingExecutor
::: ropt.workflow.executors.HPCExecutor
::: ropt.workflow.executors.EventServer

