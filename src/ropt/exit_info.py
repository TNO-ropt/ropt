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
            object.__setattr__(
                self,
                "message",
                _DEFAULT_MESSAGES.get(self.exit_code, self.exit_code.name),
            )


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


@dataclass(frozen=True, slots=True, kw_only=True)
class TooFewRealizationsInfo(ExitInfo):
    """Exit info for [`ExitCode.TOO_FEW_REALIZATIONS`][ropt.enums.ExitCode].

    Carries a batch-level summary of how many results in the failing batch
    could not meet the configured
    [`realization_min_success`][ropt.config.RealizationsConfig] threshold, and
    (for gradient results) how many of those failures were caused by too few
    successful perturbations relative to the
    [`perturbation_min_success`][ropt.config.GradientConfig] threshold.

    Attributes:
        exit_code:                Always
                                  [`ExitCode.TOO_FEW_REALIZATIONS`][ropt.enums.ExitCode].
        failed_functions:         Number of function results in the batch that
                                  failed the realization threshold.
        failed_gradients:         Number of gradient results in the batch that
                                  failed the realization threshold.
        failed_perturbations:     Number of failed gradient results that had at
                                  least one realization failing the perturbation
                                  threshold.
        realization_min_success:  The minimum number of successful realizations
                                  that was required for a result to be accepted.
        perturbation_min_success: The minimum number of successful perturbations
                                  per realization that was required for a
                                  gradient result to be accepted. `None` when
                                  the batch contained no gradient results.
    """

    exit_code: ExitCode = ExitCode.TOO_FEW_REALIZATIONS
    failed_functions: int
    failed_gradients: int
    failed_perturbations: int
    realization_min_success: int
    perturbation_min_success: int | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if not self.message:
            object.__setattr__(self, "message", self._build_message())

    def _build_message(self) -> str:
        parts: list[str] = []
        if self.failed_functions:
            parts.append(f"{self.failed_functions} function result(s)")
        if self.failed_gradients:
            parts.append(f"{self.failed_gradients} gradient result(s)")
        subjects = " and ".join(parts) if parts else "no results"
        message = (
            f"Too few realizations succeeded: {subjects} failed to meet the"
            f" minimum of {self.realization_min_success} successful"
            " realization(s)."
        )
        if self.failed_perturbations:
            assert self.perturbation_min_success is not None
            message += (
                f" {self.failed_perturbations} of the failed gradient"
                " result(s) had realization(s) that fell below the minimum of"
                f" {self.perturbation_min_success} successful perturbation(s)."
            )
        return message
