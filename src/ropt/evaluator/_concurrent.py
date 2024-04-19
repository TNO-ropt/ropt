import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ._evaluator import EvaluatorContext, EvaluatorResult


@dataclass
class ConcurrentTask(ABC):
    """Abstract data class for tasks in a concurrent evaluator.

    These task objects should contain the future that represents the task and
    return the results of the evaluation.

    It should implement two functions to retrieve the objective function values
    and optional constraint values.
    """

    future: Any

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

        This has a default implementation that returns `None` and is only usable
        when it is certain that there are no non-linear constraints. If there
        are non-linear constraints, this method should be overridden.

        Returns:
            The calculated constraints or `None`.
        """
        return None


class ConcurrentEvaluator(ABC):
    """Abstract base class for implementing a concurrent evaluator.

    This abstract base class provides the framework for implementing an
    evaluator that uses a concurrent executor.

    The `launch` method must be implemented to start an evaluation for one
    vector of variables, and return a future-like object, compatible with
    futures from the `concurrent.futures` module of Python, implementing at
    least the `done()`, `exception()`, and `result()` methods.

    To use a class derived from `ConcurrentEvaluator`, pass the object via the
    `evaluator` argument of the
    [`EnsembleOptimizer`][ropt.optimization.EnsembleOptimizer] constructor.
    """

    def __init__(self) -> None:
        """Initialize a concurrent evaluator object."""
        self.polling: float = 0.1
        "The time in seconds between polling for evaluation status."

        self._batch_id = 0

    @abstractmethod
    def launch(
        self,
        batch_id: Any,  # noqa: ANN401
        variables: NDArray[np.float64],
        context: EvaluatorContext,
    ) -> Dict[int, ConcurrentTask]:
        """Launch the evaluations and return futures.

        This method must implement the process of launching a batch of function
        evaluations for a set of variable vectors passed via the `variables`
        parameter. A unique batch ID is passed via the `batch_id`, which can be
        optionally used.

        This method should return a dictionary mapping the indices of the jobs
        to the tasks that will contain the result. The tasks are objects
        deriving from the [`ConcurrentTask`][ropt.evaluator.ConcurrentTask]
        class, containing the future object representing the launched task and
        its result. Under the hood, other tasks may be launched, but only those
        that contain results should be returned.

        This method is called by the `__call__` method, which implements the
        [`Evaluator`][ropt.evaluator.Evaluator] callback signature and can be
        passed by the [`EnsembleOptimizer`][ropt.optimization.EnsembleOptimizer]
        object.

        The `context` argument with optional information is passed from the
        `__call__` method unchanged.

        Args:
            batch_id:  The ID of the batch of evaluations to run.
            variables: The matrix of variables to evaluate.
            context:   Evaluator context.

        Returns:
            A dictionary mapping the indices of launched evaluations to tasks.
        """

    def monitor(self) -> None:  # noqa: B027
        """Monitor the states of the running evaluations.

        This method is called regularly in the polling loop that checks the
        futures until all results are collected. If the status of the
        evaluations should be monitored, this method should be overridden.

        The time in seconds between calls in the polling loop can be modified by
        setting the `polling` attribute.
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
        tasks = self.launch(self._batch_id, variables, context)
        tasks = tasks.copy()  # Use a shallow copy so we can safely modify the dict.
        while tasks:
            # We are modifying the dict while iterating, use a copy of the keys:
            for idx in list(tasks.keys()):
                if tasks[idx].future.done():
                    if tasks[idx].future.exception() is None:
                        objective_results[idx, :] = tasks[idx].get_objectives()
                        if constraint_results is not None:
                            constraint_results[idx, :] = tasks[idx].get_constraints()
                    del tasks[idx]
            self.monitor()
            time.sleep(self.polling)
        result = EvaluatorResult(
            objectives=objective_results,
            constraints=constraint_results,
            batch_id=self._batch_id,
        )
        self._batch_id += 1
        return result


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
