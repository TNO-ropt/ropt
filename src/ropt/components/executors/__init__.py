"""Executors: where an evaluator sends work to be run.

An executor runs the [`WorkItem`][ropt.components.executors.WorkItem] objects of
a [`Submission`][ropt.components.executors.Submission] on a concrete mechanism:
threads, processes, local jobs, or an HPC cluster.
[`Executor`][ropt.components.executors.Executor] is the interface a compute step
sees; [`ExecutorBase`][ropt.components.executors.ExecutorBase] adds the
submission bookkeeping the built-in executors share. See
[Parallel Evaluation](../advanced/parallel.md) for usage, and
[Implementing a Component](../advanced/components.md) for writing one.
"""

from __future__ import annotations

from ._hpc_executor import HPCExecutor
from ._local_executor import LocalJobExecutor
from ._process_executor import ProcessExecutor
from ._thread_executor import ThreadExecutor
from .base import Executor, ExecutorBase, ExecutorFailure, Submission, WorkItem

__all__ = [
    "Executor",
    "ExecutorBase",
    "ExecutorFailure",
    "HPCExecutor",
    "LocalJobExecutor",
    "ProcessExecutor",
    "Submission",
    "ThreadExecutor",
    "WorkItem",
]
