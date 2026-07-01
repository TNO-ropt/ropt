"""Structured information about the termination of a compute step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ropt.enums import ExitCode

_DEFAULT_MESSAGES: Final[dict[ExitCode, str]] = {
    ExitCode.UNKNOWN: "Unknown cause of termination",
    ExitCode.TOO_FEW_REALIZATIONS: "Too few realizations were evaluated successfully",
    ExitCode.MAX_FUNCTIONS_REACHED: "Maximum number of function evaluations reached",
    ExitCode.MAX_BATCHES_REACHED: "Maximum number of evaluation batches reached",
    ExitCode.USER_ABORT: "Optimization aborted by the user",
    ExitCode.OPTIMIZER_FINISHED: "Optimization finished successfully",
    ExitCode.ENSEMBLE_EVALUATOR_FINISHED: "Ensemble evaluation finished successfully",
    ExitCode.ABORT_FROM_ERROR: "Aborted due to an error",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ExitInfo:
    """Outcome of a compute step.

    If no message is provided, a default message will be used or generated based
    on the exit code.

    Attributes:
        exit_code: The [`ExitCode`][ropt.enums.ExitCode] indicating why the
                   step terminated.
        message:   Optional message suitable for reporting to end users.
    """

    exit_code: ExitCode
    message: str = ""

    def __post_init__(self) -> None:  # noqa: D105
        if not self.message:
            object.__setattr__(self, "message", _DEFAULT_MESSAGES[self.exit_code])


@dataclass(frozen=True, slots=True, kw_only=True)
class MaxFunctionsReachedInfo(ExitInfo):
    """Exit info for [`ExitCode.MAX_FUNCTIONS_REACHED`][ropt.enums.ExitCode].

    Attributes:
        exit_code: Always
                   [`ExitCode.MAX_FUNCTIONS_REACHED`][ropt.enums.ExitCode].
        limit:     The configured maximum number of function evaluations that
                   was reached.
    """

    exit_code: ExitCode = ExitCode.MAX_FUNCTIONS_REACHED
    limit: int

    def __post_init__(self) -> None:  # noqa: D105
        if not self.message:
            object.__setattr__(
                self,
                "message",
                (f"Maximum number of function evaluations reached ({self.limit})"),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class MaxBatchesReachedInfo(ExitInfo):
    """Exit info for [`ExitCode.MAX_BATCHES_REACHED`][ropt.enums.ExitCode].

    Attributes:
        exit_code: Always
                   [`ExitCode.MAX_BATCHES_REACHED`][ropt.enums.ExitCode].
        limit:     The configured maximum number of evaluation batches that
                   was reached.
    """

    exit_code: ExitCode = ExitCode.MAX_BATCHES_REACHED
    limit: int

    def __post_init__(self) -> None:  # noqa: D105
        if not self.message:
            object.__setattr__(
                self,
                "message",
                f"Maximum number of evaluation batches reached ({self.limit})",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AbortFromErrorInfo(ExitInfo):
    """Exit info for [`ExitCode.ABORT_FROM_ERROR`][ropt.enums.ExitCode].

    Attributes:
        exit_code: Always
                   [`ExitCode.ABORT_FROM_ERROR`][ropt.enums.ExitCode].
        error:     Description of the underlying error that caused the
                   abort. Typically the string form of the exception.
    """

    exit_code: ExitCode = ExitCode.ABORT_FROM_ERROR
    error: str

    def __post_init__(self) -> None:  # noqa: D105
        if not self.message:
            object.__setattr__(
                self, "message", f"Aborted due to an error: {self.error}"
            )
