"""Exceptions raised within the `ropt` library."""

from typing import Optional

from .enums import OptimizerExitCode


class ConfigError(Exception):
    """Raised when an configuration error occurs."""


class PlanError(Exception):
    """Raised when an error occurs in an optimization plan."""

    def __init__(
        self,
        message: str,
        *,
        name: Optional[str] = None,
    ) -> None:
        """Initialize with a custom error message.

        Args:
            message: Error message
            name:    Optional step or handler object name to add to the message
        """
        msg = f"Step `{name}`: " if name is not None else ""
        super().__init__(msg + message)


class OptimizationAborted(Exception):  # noqa: N818
    """Raised when an optimization is aborted.

    When constructing the exception object an exit code must be passed that
    indicates the reason for aborting (see
    [`OptimizerExitCode`][ropt.enums.OptimizerExitCode]).
    """

    def __init__(self, exit_code: OptimizerExitCode) -> None:
        """Initialize an exception that aborts the optimization.

        Args:
            exit_code: The exit code indicating the reason for the abort
        """
        self.exit_code = exit_code
        super().__init__()
