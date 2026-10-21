"""Executors: where an evaluator sends work to be run.

An executor runs [`WorkItem`][ropt._executors.WorkItem] objects on a
concrete mechanism: threads, processes, local jobs, or an HPC cluster.
[`Executor`][ropt._executors.Executor] is the interface an evaluator
uses; [`ExecutorBase`][ropt._executors.ExecutorBase] adds the
lifecycle the built-in executors share. See
[Evaluating in Parallel](../running/parallel.md) for usage.
"""

from __future__ import annotations

from ._hpc_executor import HPCExecutor
from ._local_executor import LocalJobExecutor
from ._process_executor import ProcessExecutor
from ._thread_executor import ThreadExecutor
from .base import Executor, ExecutorBase, ExecutorFailure, WorkItem, WorkNotRun

__all__ = [
    "Executor",
    "ExecutorBase",
    "ExecutorFailure",
    "HPCExecutor",
    "LocalJobExecutor",
    "ProcessExecutor",
    "ThreadExecutor",
    "WorkItem",
    "WorkNotRun",
]
