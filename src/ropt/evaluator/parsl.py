"""A concurrent evaluator based on parsl.

the `ParslEvaluator` class implements an evaluator using the
[`parsl`](https://parsl-project.org/) library for parallel programming. This
evaluator makes it possible to efficiently run `ropt` optimizations on a wide
range of compute resources.
"""

from dataclasses import dataclass
from enum import Enum
from time import sleep
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

    future: Optional[AppFuture]
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
        htex_kwargs: Optional[Dict[str, Any]] = None,
        max_threads: int = 4,
        retries: int = 0,
        worker_restart: int = 0,
        enable_cache: bool = True,
        polling: float = 0.1,
    ) -> None:
        """Create a parsl evaluator object.

        Args:
            provider:       Parsl execution provider to use. By default `LocalProvider`
            workflow:       Callback to start a single workflow run
            monitor:        Monitor function
            htex_kwargs:    Keyword arguments forwarded to the htex executor
            max_threads:    Maximum number of threads for local execution. Defaults to 4
            retries:        Number of retries upon failure of a task. Defaults to 0
            worker_restart: Restart the workers every `worker_restart` batch.
            enable_cache:   If `True` enable function value caching.
            polling:        Time in seconds between checking job status
        """
        super().__init__(enable_cache=enable_cache, polling=polling)

        self._batch_id: int
        self._variables: NDArray[np.float64]
        self._jobs: Dict[int, List[Task]] = {}
        self._workflow = workflow
        self._monitor = monitor
        self._executor: Union[ThreadPoolExecutor, HighThroughputExecutor]

        parsl.clear()

        if provider is None or isinstance(provider, LocalProvider):
            self._executor = ThreadPoolExecutor(
                label="local_threads", max_threads=max_threads
            )
            self._worker_restart = 0
            self._htex_kwargs = {}
        else:
            if htex_kwargs is None:
                htex_kwargs = {}
            htex_kwargs.setdefault("max_workers_per_node", 1)
            htex_kwargs.setdefault("heartbeat_period", 30)
            self._executor = HighThroughputExecutor(
                label="high_throughput_executor", provider=provider, **htex_kwargs
            )
            self._worker_restart = worker_restart
            self._htex_kwargs = htex_kwargs

        parsl.load(
            Config(
                executors=[self._executor],
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
        if (
            self._worker_restart > 0
            and batch_id > 0
            and batch_id % self._worker_restart == 0
        ):
            assert isinstance(self._executor, HighThroughputExecutor)
            while self._executor.connected_workers > 0:
                self._executor.scale_in(self._executor.connected_workers)
                sleep(self._htex_kwargs["heartbeat_period"])
        self._batch_id = batch_id
        self._variables = variables
        self._jobs = {
            job_idx: self._workflow(batch_id, job_idx, variables, context)
            for job_idx in range(variables.shape[0])
            if active is None or active[job_idx]
        }
        return {idx: futures[-1] for idx, futures in self._jobs.items() if futures}

    def monitor(self) -> None:  # noqa: C901
        """Monitor the tasks of all jobs.

        See the [ropt.evaluator.ConcurrentEvaluator][] abstract base class.

        # noqa
        """
        changed = False
        for job in self._jobs.values():
            for task in job:
                if task.future is not None:
                    # Update task.exception from the future:
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
