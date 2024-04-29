"""A concurrent evaluator based on parsl.

the `ParslEvaluator` class implements an evaluator using the
[`parsl`](https://parsl-project.org/) library for parallel programming. This
evaluator makes it possible to efficiently run `ropt` optimizations on a wide
range of compute resources.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import parsl
from numpy.typing import NDArray
from parsl.config import Config
from parsl.dataflow.futures import AppFuture
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider
from parsl.providers.base import ExecutionProvider

from ._concurrent import ConcurrentEvaluator, ConcurrentTask
from ._evaluator import EvaluatorContext


class State(Enum):
    """Parsl task state enumeration."""

    UNKNOWN = "unknown"
    PENDING = "pending"
    RUNNING = "running"
    LAUNCHED = "launched"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Task(ConcurrentTask):
    """Dataclass storing task future.

    The `ParslEvaluator` class accepts a callback that, for each job, returns a
    list of tasks to be evaluated. The task contains the future that is being
    evaluated and the state of the task, which sampled at a regular interval by
    the `monitor` method of the `ParslEvaluator`class.

    Attributes:
        future:    The future of the task
        state:     The last sampled state
        exception: Any exception that may have occurred,  None otherwise
    """

    future: AppFuture
    state: State = State.UNKNOWN
    exception: Optional[BaseException] = None


class ParslEvaluator(ConcurrentEvaluator):
    """A function evaluator based on the parsl library."""

    _RUN_INFO_DIR = "runinfo"

    def __init__(  # noqa: PLR0913
        self,
        workflow: Callable[..., Any],
        *,
        monitor: Optional[Callable[..., Any]] = None,
        provider: Optional[ExecutionProvider] = None,
        max_threads: int = 4,
        retries: int = 0,
        enable_cache: bool = True,
    ) -> None:
        """Create a parsl evaluator object.

        Args:
            provider:     Parsl execution provider to use. By default `LocalProvider`
            workflow:     Callback to start a single workflow run
            monitor:      Monitor function
            max_threads:  Maximum number of threads for local execution. Defaults to 4
            retries:      Number of retries upon failure of a task. Defaults to 0
            enable_cache: If `True` enable function value caching.
        """
        super().__init__(enable_cache=enable_cache)

        self._batch_id: int
        self._variables: NDArray[np.float64]
        self._jobs: Dict[int, List[Task]] = {}
        self._workflow = workflow
        self._monitor = monitor

        executor: Union[ThreadPoolExecutor, HighThroughputExecutor]
        if provider is None or isinstance(provider, LocalProvider):
            executor = ThreadPoolExecutor(
                label="local_threads", max_threads=max_threads
            )
        else:
            executor = HighThroughputExecutor(
                label="high_throughput_executor", provider=provider, max_workers=1
            )

        parsl.clear()
        parsl.load(
            Config(
                executors=[executor],
                strategy="htex_auto_scale",
                retries=retries,
                run_dir=self._RUN_INFO_DIR,
            ),
        )

    def launch(
        self,
        batch_id: int,
        variables: NDArray[np.float64],
        context: EvaluatorContext,
        active: Optional[NDArray[np.bool_]],
    ) -> Dict[int, ConcurrentTask]:
        """Launch the parsl task.

        See the [ropt.evaluator.ConcurrentEvaluator][] abstract base class.

        # noqa
        """
        self._batch_id = batch_id
        self._variables = variables
        self._jobs = {
            job_idx: self._workflow(batch_id, job_idx, variables, context)
            for job_idx in range(variables.shape[0])
            if active is None or active[job_idx]
        }
        return {idx: futures[-1] for idx, futures in self._jobs.items()}

    def monitor(self) -> None:
        """Monitor the tasks of all jobs.

        See the [ropt.evaluator.ConcurrentEvaluator][] abstract base class.

        # noqa
        """
        changed = False
        for job in self._jobs.values():
            for task in job:
                # Make sure task.exception is update from the future:
                if task.future.done() and task.exception is None:
                    task.exception = task.future.exception()

                # Default is not running:
                state: State = State.PENDING

                # Set the state:
                if task.exception is not None:
                    state = State.FAILED
                elif task.future.done():
                    state = State.SUCCESS
                elif task.future.running():
                    state = State.RUNNING
                elif task.future.task_status() == "launched":
                    state = State.LAUNCHED

                # If the state changed, remember:
                if state != task.state:
                    task.state = state
                    changed = True
        if self._monitor is not None and changed:
            self._monitor(self._batch_id, self._jobs)
