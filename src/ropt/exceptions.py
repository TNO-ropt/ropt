"""Exceptions raised within the `ropt` library."""

from .enums import OptimizerExitCode


class ConfigError(Exception):
    """Raised when an configuration error occurs."""


class OptimizationAborted(Exception):  # noqa: N818
    """Raised when an optimization is aborted.

    When constructing the exception object an exit code must be passed that
    indicates the reason for aborting (see
    [`OptimizerExitCode`][ropt.enums.OptimizerExitCode]).
    """

    def __init__(self, exit_code: OptimizerExitCode) -> None:
        """Initialize an exception that aborts the optimization.

        Args:
            exit_code: The exit code indicating the reason for the abort.
        """
        self.exit_code = exit_code
        super().__init__()
