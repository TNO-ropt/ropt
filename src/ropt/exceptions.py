"""Exceptions raised within the `ropt` library."""

from ropt.enums import ExitCode


class RoptError(Exception):
    """Base class for all runtime errors raised by `ropt`.

    Catch this to handle any error raised by `ropt` itself. Configuration and
    validation errors are **not** part of this hierarchy; they surface as
    `pydantic.ValidationError`. The internal stop signals
    ([`OptimizerStop`][ropt.exceptions.OptimizerStop],
    [`TooFewRealizations`][ropt.exceptions.TooFewRealizations],
    [`ExecutorStopped`][ropt.exceptions.ExecutorStopped]) are control flow, not
    errors, and are excluded too.
    """


class WorkflowError(RoptError):
    """A workflow or runtime object was used incorrectly.

    For example a compute step, evaluator, or event handler used concurrently or
    out of order, an ownership or registration conflict, submitting to a
    dispatcher that is not running, or reusing a locked context.
    """


class ExecutionError(RoptError):
    """The execution infrastructure failed at runtime.

    For example a worker pool that cannot start, a broken process pool, a task
    that cannot be serialized, or an HPC setup or submission problem.
    """


class UnsupportedError(RoptError):
    """An optional dependency is missing or a requested feature is unsupported.

    For example a required extra (such as pandas or polars) is not installed,
    or the selected plugin does not support the requested method.
    """


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
