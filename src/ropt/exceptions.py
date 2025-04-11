"""Exceptions raised within the `ropt` library."""

from .enums import OptimizerExitCode


class ConfigError(Exception):
    """Raised when an configuration error occurs."""


class OptimizationAborted(Exception):  # noqa: N818
    """Raised when an optimization process within a step is aborted prematurely.

    This exception signals that an optimizer or evaluator step could not
    complete its intended task due to a specific condition (e.g., insufficient
    valid realizations, user request).

    It must be initialized with an
    [`OptimizerExitCode`][ropt.enums.OptimizerExitCode] indicating the reason
    for the abortion.
    """

    def __init__(self, exit_code: OptimizerExitCode) -> None:
        """Initialize the OptimizationAborted exception.

        Stores the reason for the abortion, which can be accessed via the
        `exit_code` attribute.

        Args:
            exit_code: The code indicating why the optimization step was aborted.
        """
        self.exit_code = exit_code
        super().__init__()


class PlanAborted(Exception):  # noqa: N818
    """Raised when an optimization plan is aborted prematurely.

    This exception signals that the execution of a [`Plan`][ropt.plan.Plan]
    was stopped before completion. This typically occurs when a step or handler
    within the plan explicitly sets the plan's `aborted` flag (e.g., due to a
    user request via a signal or an unrecoverable error condition detected by
    a handler).

    Any subsequent attempts to execute further steps in an already aborted plan
    will raise this exception.
    """
