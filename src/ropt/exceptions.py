"""Exceptions raised within the `ropt` library."""

from .enums import ExitCode


class ComputeStepAborted(Exception):  # noqa: N818
    """Raised when a compute step is aborted prematurely.

    This exception signals that an optimization or another compute step could
    not complete its intended task due to a specific condition (e.g.,
    insufficient valid realizations, user request).

    It must be initialized with an [`ExitCode`][ropt.enums.ExitCode] indicating
    the reason for the abortion.
    """

    def __init__(self, exit_code: ExitCode) -> None:
        """Initialize the ComputeStepAborted exception.

        Stores the reason for the abortion, which can be accessed via the
        `exit_code` attribute.

        Args:
            exit_code: The code indicating why the compute step was aborted.
        """
        self.exit_code = exit_code
        super().__init__()
