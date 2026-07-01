"""Exceptions raised within the `ropt` library."""

from .exit_info import ExitInfo


class Abort(Exception):  # noqa: N818
    """Raised when a compute step is aborted prematurely.

    This exception signals that an optimization or another compute step could
    not complete its intended task due to a specific condition (e.g.,
    insufficient valid realizations, user request).

    It must be initialized with an [`ExitInfo`][ropt.exit_info.ExitInfo]
    describing the reason for the abortion.
    """

    def __init__(self, info: ExitInfo) -> None:
        """Initialize the Abort exception.

        Stores the reason for the abortion, accessible via the `info`
        attribute.

        Args:
            info: The exit info describing why the compute step was aborted.
        """
        self.info = info
        super().__init__()


class ServerFailure(Exception):  # noqa: N818
    """Raised when an evaluator server fails to execute a task."""
