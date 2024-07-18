import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ._evaluator import EvaluatorContext, EvaluatorResult


@dataclass
class ConcurrentTask(ABC):
    """Abstract data class for tasks in a concurrent evaluator.

    These task objects should contain the future that represents the task and
    return the results of the evaluation.

    It should implement two methods to retrieve the objective function values
    and optionally the constraint values.
    """

    future: Optional[Any]

    @abstractmethod
    def get_objectives(self) -> Optional[NDArray[np.float64]]:
        """Return the objectives calculated by the task.

        This method will only be called after the future is done, and if no
        exception was raised during the execution of the task.

        Info:
            The return value might be None, as derived evaluator classes might
            also employ tasks that do not return results.

        Returns:
            The calculated objectives.
        """

    def get_constraints(self) -> Optional[NDArray[np.float64]]:
        """Return the constraints calculated by the task, if available.

        This method will only be called after the future is done, and if no
        exception was raised during the execution of the task.

        This has a default implementation that returns `None`, which is only
        usable when it is certain that there are no non-linear constraints. If
        there are non-linear constraints, this method should be overridden.

        Returns:
            The calculated constraints or `None`.
        """
        return None


class ConcurrentEvaluator(ABC):
    """Abstract base class for implementing a concurrent evaluator.

    This abstract base class provides the framework for implementing an
    evaluator that uses a concurrent executor.

    The `launch` method must be implemented to start an evaluation for one
    vector of variables, and return a future-like object, compatible with the
    `concurrent.futures` module of Python, implementing at least the `done()`,
    `exception()`, and `result()` methods. The `monitor` method can be overloaded
    to implement specific monitoring functionality, the default implementation does
    nothing.

    The class implements an optional caching mechanism to prevent repeated
    evaluation of functions. When enabled, all evaluation results are cached in
    memory and reused when requested. This is in particular useful in
    optimization plans with multiple or nested optimizations where often
    restarts occur from points that already have been evaluated before.
    """

    def __init__(
        self, *, enable_cache: bool = True, polling: float = 0.1, max_submit: int = 500
    ) -> None:
        """Initialize a concurrent evaluator object.

        Some general properties of the evaluator are fixed at initialization time:

        - The cache may be enabled or disabled using the `enable_cache` parameter.
        - While evaluations are running concurrently the evaluator will
          regularly poll them to check their status. The `polling` parameter
          determine the delay between polling events.
        - When a very large number of jobs is submitted this may overwhelm the
          evaluator, depending on the implementation. To prevent this, the jobs
          may be submitted in smaller portions, the size of which are defined by
          the `max_submit` argument.

        Args:
            enable_cache: Enable the caching mechanism
            polling:      Time in seconds between checking job status
            max_submit:   Maximum number of variables to submit simultaneously
        """
        self._batch_id = 0
        self._polling = polling
        self._max_submit = max_submit
        self._cache: Optional[_Cache] = _Cache() if enable_cache else None

    @abstractmethod
    def launch(
        self,
        batch_id: Any,  # noqa: ANN401
        job_id: int,
        variables: NDArray[np.float64],
        context: EvaluatorContext,
    ) -> Optional[ConcurrentTask]:
        """Launch an evaluation return a future for each job.

        This method must implement the process of launching a a single function
        evaluation for a variable vector passed via the `variables` parameter. A
        unique batch ID is passed via the `batch_id`, which can be optionally
        used by the evaluator to identify the current batch of functions.

        This method should return a dictionary mapping the indices of the jobs
        to the tasks that will contain the result. The tasks are objects
        deriving from the [`ConcurrentTask`][ropt.evaluator.ConcurrentTask]
        class, containing the future object representing the launched task and
        its result. Under the hood, other tasks may be launched, but only those
        that contain results should be returned.

        This method is called by the `__call__` method, which implements the
        [`Evaluator`][ropt.evaluator.Evaluator] callback signature.

        The `context` argument with optional information is passed from the
        `__call__` method unchanged.

        Args:
            batch_id:  The ID of the batch of evaluations to run.
            job_id:    The ID of the job launched for the variables
            variables: The matrix of variables to evaluate.
            context:   Evaluator context.

        Returns:
            The last future or `None` if no tasks were launched.
        """

    def monitor(self) -> None:  # noqa: B027
        """Monitor the states of the running evaluations.

        This method is called regularly in the polling loop that checks the
        futures until all results are collected. If the status of the
        evaluations should be monitored, this method should be overridden.

        The time in seconds between calls in the polling loop can be modified by
        setting the `polling` attribute upon object initialization.
        """

    def __call__(
        self, variables: NDArray[np.float64], context: EvaluatorContext
    ) -> EvaluatorResult:
        """Launch all evaluations and collect the results.

        This method implements the evaluation function, passed to the ensemble
        optimizer constructor.

        Args:
            variables: The matrix of variables to evaluate.
            context:   Evaluator context

        Returns:
            The batch ID and objective and constraint function values.
        """
        objective_results, constraint_results = _init_results(variables, context)

        active = self._try_cache(
            variables, context, objective_results, constraint_results
        )

        var_idx = 0
        tasks: Dict[int, ConcurrentTask] = {}

        # Keep submitting and monitoring until all variables and tasks are done:
        while var_idx < variables.shape[0] or len(tasks) > 0:
            # Add more tasks up to a maximum of max_submit:
            for _ in range(self._max_submit - len(tasks)):
                if var_idx < variables.shape[0]:
                    if active is None or active[var_idx]:
                        task = self.launch(
                            self._batch_id, var_idx, variables[var_idx], context
                        )
                        if task is not None:
                            tasks[var_idx] = task
                    var_idx += 1

            # Monitor the current tasks:
            for idx in list(tasks.keys()):  # tasks changes size, hence list()
                future = tasks[idx].future
                if future is None or future.done():
                    if future is None or future.exception() is None:
                        objective_results[idx, :] = tasks[idx].get_objectives()
                        if constraint_results is not None:
                            constraint_results[idx, :] = tasks[idx].get_constraints()
                    del tasks[idx]
            self.monitor()
            time.sleep(self._polling)

        result = EvaluatorResult(
            objectives=objective_results,
            constraints=constraint_results,
            batch_id=self._batch_id,
        )

        self._update_cache(
            variables, context, active, objective_results, constraint_results
        )

        self._batch_id += 1
        return result

    def _try_cache(
        self,
        variables: NDArray[np.float64],
        context: EvaluatorContext,
        objectives: NDArray[np.float64],
        constraints: Optional[NDArray[np.float64]],
    ) -> Optional[NDArray[np.bool_]]:
        if self._cache is None:
            return context.active
        active = (
            np.ones(variables.shape[0], dtype=np.bool_)
            if context.active is None
            else np.fromiter(
                (context.active[realization] for realization in context.realizations),
                dtype=np.bool_,
            )
        )

        for job_idx, real_id in enumerate(self._get_realization_ids(context)):
            cache_id = self._cache.find_key(real_id, variables[job_idx, ...])
            if cache_id is not None:
                active[job_idx] = False
                objectives[job_idx, ...] = self._cache.get_objectives(cache_id)
                if constraints is not None:
                    constraints[job_idx, ...] = self._cache.get_constraints(cache_id)

        return active

    def _update_cache(
        self,
        variables: NDArray[np.float64],
        context: EvaluatorContext,
        active: Optional[NDArray[np.bool_]],
        objectives: NDArray[np.float64],
        constraints: Optional[NDArray[np.float64]],
    ) -> None:
        if self._cache is not None:
            assert active is not None
            for job_idx, real_id in enumerate(self._get_realization_ids(context)):
                if active[job_idx]:
                    self._cache.add_simulation_results(
                        job_idx, real_id, variables, objectives, constraints
                    )

    def _get_realization_ids(self, context: EvaluatorContext) -> Tuple[Any, ...]:
        names = context.config.realizations.names
        if names is None:
            names = tuple(range(context.config.realizations.weights.size))
        return tuple(names[realization] for realization in context.realizations)


def _init_results(
    variables: NDArray[np.float64], context: EvaluatorContext
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
    objective_count = context.config.objective_functions.weights.size
    constraint_count = (
        0
        if context.config.nonlinear_constraints is None
        else context.config.nonlinear_constraints.rhs_values.size
    )
    objective_results = np.full(
        (variables.shape[0], objective_count), fill_value=np.nan, dtype=np.float64
    )
    if constraint_count > 0:
        constraint_results = np.full(
            (variables.shape[0], constraint_count), fill_value=np.nan, dtype=np.float64
        )
    else:
        constraint_results = None
    if context.active is not None:
        inactive = ~context.active
        objective_results[inactive, :] = 0.0
        if constraint_results is not None:
            constraint_results[inactive, :] = 0.0
    return objective_results, constraint_results


class _Cache:
    def __init__(self) -> None:
        # Stores the realization/controls key, together with an ID.
        self._keys: DefaultDict[int, List[Tuple[NDArray[np.float64], int]]] = (
            defaultdict(list)
        )
        # Store objectives and constraints by ID:
        self._objectives: Dict[int, NDArray[np.float64]] = {}
        self._constraints: Dict[int, NDArray[np.float64]] = {}

        # Generate unique ID's:
        self._counter = count()

    def add_simulation_results(
        self,
        job_idx: int,
        real_id: int,
        control_values: NDArray[np.float64],
        objectives: NDArray[np.float64],
        constraints: Optional[NDArray[np.float64]],
    ) -> None:
        cache_id = next(self._counter)
        self._keys[real_id].append((control_values[job_idx, :].copy(), cache_id))
        self._objectives[cache_id] = objectives[job_idx, ...].copy()
        if constraints is not None:
            self._constraints[cache_id] = constraints[job_idx, ...].copy()

    def find_key(
        self, real_id: int, control_vector: NDArray[np.float64]
    ) -> Optional[int]:
        # Brute-force search, premature optimization is the root of all evil:
        for cached_vector, cache_id in self._keys.get(real_id, []):
            if np.allclose(
                control_vector,
                cached_vector,
                rtol=0.0,
                atol=float(np.finfo(np.float32).eps),
            ):
                return cache_id
        return None

    def get_objectives(self, cache_id: int) -> NDArray[np.float64]:
        return self._objectives[cache_id]

    def get_constraints(self, cache_id: int) -> NDArray[np.float64]:
        return self._constraints[cache_id]
