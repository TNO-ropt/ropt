# Executors

Executors run the [`WorkItem`][ropt.components.executors.WorkItem] objects of a
[`Submission`][ropt.components.executors.Submission], produced by a
[`ParallelEvaluator`][ropt.components.evaluators.ParallelEvaluator], on a concrete
execution mechanism (threads, processes, local jobs, or an HPC cluster).

See [Parallel Evaluation](../workflows/parallel.md) for usage.

[`Executor`][ropt.components.executors.Executor] is the interface a compute step
sees; [`ExecutorBase`][ropt.components.executors.ExecutorBase] implements the
submission bookkeeping shared by the built-in executors and is the class to
subclass when adding a new execution mechanism.

::: ropt.components.executors.Executor
::: ropt.components.executors.ExecutorBase
::: ropt.components.executors.WorkItem
::: ropt.components.executors.Submission
::: ropt.components.executors.ThreadExecutor
::: ropt.components.executors.ProcessExecutor
::: ropt.components.executors.LocalJobExecutor
::: ropt.components.executors.HPCExecutor

