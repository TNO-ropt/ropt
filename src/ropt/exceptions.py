"""Exceptions raised within the `ropt` library."""

from ropt.enums import ExitCode


class OptimizerStop(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised internally to stop an optimization with a specific exit code.

    Used only within the optimizer core to unwind the backend optimization loop
    (for example when a function or batch budget is reached). It carries the
    [`ExitCode`][ropt.enums.ExitCode] the optimization terminates with.
    """

    def __init__(self, exit_code: ExitCode) -> None:
        """Initialize the OptimizerStop exception.

        Args:
            exit_code: The exit code the optimization terminates with.
        """
        self.exit_code = exit_code
        super().__init__()


class TooFewRealizations(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised when too few realizations are available to compute a result.

    A generic signal, carrying no exit code.
    """


class ExecutorStopped(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised when the evaluation executor is no longer running.

    A generic signal, carrying no exit code, raised by the parallel evaluator
    when its executor has stopped and the current evaluation cannot proceed.
    """


class ExecutorFailure(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised when an executor fails to execute a task."""


class TransferError(Exception):
    """Raised when a workflow object is used in a worker process.

    Workflow objects such as compute steps, evaluators, and event handlers are
    process-local. They may be pickled as part of a task payload, but they must
    not be used in a worker: if a task dispatched to a
    [`MultiprocessingExecutor`][ropt.components.executors.MultiprocessingExecutor]
    or [`HPCExecutor`][ropt.components.executors.HPCExecutor] worker captures one,
    the worker detects it and raises this error before running the task.
    """
