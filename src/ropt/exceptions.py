"""Exceptions raised within the `ropt` library."""

from ropt.enums import ExitCode


class Abort(Exception):  # noqa: N818
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


class ServerFailure(Exception):  # noqa: N818
    """Raised when an evaluator server fails to execute a task."""
