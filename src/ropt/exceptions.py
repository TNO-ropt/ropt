"""Exceptions raised within the `ropt` library."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt.enums import ExitReason
    from ropt.simple import OptimizationResult


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

    For example a compute step or evaluator used concurrently, an event handler
    re-entered from inside itself, or an executor asked to run work from one of
    its own workers.
    """


class ExecutionError(RoptError):
    """The execution infrastructure failed at runtime.

    For example an executor that cannot start, a broken process pool, a task
    that cannot be serialized, an HPC setup or submission problem, or an
    evaluation whose worker died before producing a result.
    """


class UnsupportedError(RoptError):
    """An optional dependency is missing or a requested feature is unsupported.

    For example a required extra (such as pandas or polars) is not installed,
    or the selected plugin does not support the requested method.
    """


class RunsFailedError(RoptError):
    """One of several concurrent runs raised.

    Several runs produce several outcomes, so there is neither a single
    exception to re-raise nor a single set of results to return. This carries
    both, so the work the other runs did is not thrown away with the one that
    failed. The first exception is chained, so a traceback still shows what
    went wrong.

    The runs that did not fail were cut off when this one did and ended with
    `ExitReason.ABORTED_ON_ERROR`, unless they were started with
    `keep_going=True`. Each kept whatever its completed batches had produced;
    one cut off during its first batch has no result.

    Attributes:
        outcomes: Per run, in the order the runs were given, its
                  [`OptimizationResult`][ropt.simple.OptimizationResult] or the
                  exception it raised.
    """

    def __init__(self, outcomes: Sequence[OptimizationResult | Exception]) -> None:
        """Initialize the error.

        Args:
            outcomes: Per run, its result or the exception it raised.
        """
        self.outcomes = tuple(outcomes)
        failed = sum(1 for item in self.outcomes if isinstance(item, Exception))
        msg = (
            f"{failed} of {len(self.outcomes)} runs raised; `outcomes` holds"
            " what each run raised or reached."
        )
        super().__init__(msg)


class OptimizerStop(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised internally to stop an optimization with a specific exit reason.

    Used only within the optimizer core to unwind the backend optimization loop
    (for example when a function or batch budget is reached). It carries the
    [`ExitReason`][ropt.enums.ExitReason] the optimization terminates with.
    """

    def __init__(self, exit_reason: ExitReason) -> None:
        """Initialize the OptimizerStop exception.

        Args:
            exit_reason: The reason the optimization terminates with.
        """
        self.exit_reason = exit_reason
        super().__init__()


class TooFewRealizations(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised when too few realizations are available to compute a result.

    A generic signal, carrying no exit code.
    """


class ExecutorStopped(Exception):  # ruff: ignore[error-suffix-on-exception-name]
    """Raised when the evaluation executor can no longer run the work.

    A generic signal, carrying no exit code. In practice this happens at
    interpreter shutdown, when the worker pool is gone and a run that outlived
    its program is released rather than left waiting.
    """
