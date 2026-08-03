"""Exceptions raised within the `ropt` library."""

from ropt.enums import ExitCode


class Abort(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised when a compute step is aborted prematurely.

    This exception signals that an optimization or another compute step could
    not complete its intended task due to a specific condition (e.g.,
    insufficient valid realizations, user request).

    It must be initialized with an [`ExitCode`][ropt.enums.ExitCode] describing
    the reason for the abortion.
    """

    def __init__(self, exit_code: ExitCode) -> None:
        """Initialize the Abort exception.

        Stores the reason for the abortion, accessible via the `exit_code`
        attribute.

        Args:
            exit_code: The exit code describing why the compute step was aborted.
        """
        self.exit_code = exit_code
        super().__init__()


class ExecutorFailure(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised when an executor fails to execute a task."""


class TransferError(Exception):
    """Raised when a workflow object is used in a worker process.

    Workflow objects such as compute steps, evaluators, and event handlers are
    process-local. They may be pickled as part of a task payload, but they must
    not be used in a worker: if a task dispatched to a
    [`MultiprocessingExecutor`][ropt.workflow.executors.MultiprocessingExecutor]
    or [`HPCExecutor`][ropt.workflow.executors.HPCExecutor] worker captures one,
    the worker detects it and raises this error before running the task.
    """
