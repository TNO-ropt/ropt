"""Exceptions raised within the `ropt` library.

Two families are defined here. The errors derive from
[`RoptError`][ropt.exceptions.RoptError] and report a failure a caller may
catch: [`WorkflowError`][ropt.exceptions.WorkflowError],
[`ExecutionError`][ropt.exceptions.ExecutionError],
[`UnsupportedError`][ropt.exceptions.UnsupportedError],
[`AbortedError`][ropt.exceptions.AbortedError] and
[`RunsFailedError`][ropt.exceptions.RunsFailedError].

The other three derive from `Exception` directly and are control-flow signals
rather than errors. [`OptimizerStop`][ropt.exceptions.OptimizerStop] and
[`ExecutorStopped`][ropt.exceptions.ExecutorStopped] are raised and caught
inside `ropt`. [`TooFewRealizations`][ropt.exceptions.TooFewRealizations] is
raised by a
[`RealizationFilter`][ropt.realization_filter.RealizationFilter] that can give
no realization a positive weight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ropt import OptimizationResult
    from ropt.enums import ExitCode


class RoptError(Exception):
    """Base class for all runtime errors raised by `ropt`.

    Catch this to handle any error raised by `ropt` itself. Configuration and
    validation errors are **not** part of this hierarchy; they surface as
    `pydantic.ValidationError`. The control-flow signals `OptimizerStop`,
    `TooFewRealizations` and `ExecutorStopped` are excluded too.
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


class AbortedError(RoptError):
    """Work was cut off before it could finish.

    Raised by [`WorkerPool.offload`][ropt.WorkerPool.offload], which
    returns whatever its callables return and so has nowhere to report a reason.
    An optimization or an evaluation carries its reason on the result object it
    returns instead, and does not raise this.

    Attributes:
        exit_code: Why the work was cut off.
    """

    def __init__(self, exit_code: ExitCode) -> None:
        """Initialize the error.

        Args:
            exit_code: Why the work was cut off.
        """
        self.exit_code = exit_code
        msg = f"The work was cut off before it could finish: {exit_code.name}."
        super().__init__(msg)


class RunsFailedError(RoptError):
    """One of several concurrent runs raised.

    Several runs produce several outcomes, so there is neither a single
    exception to re-raise nor a single set of results to return. This carries
    both, so the work the other runs did is not thrown away with the one that
    failed. The first exception is chained, so a traceback still shows what
    went wrong.

    The runs that did not fail were cut off when this one did and ended with
    `ExitCode.ABORTED_ON_ERROR`, unless they were started with
    `keep_going=True`. Each kept whatever its completed batches had produced;
    one cut off during its first batch has no result.

    Attributes:
        outcomes: Per run, in the order the runs were given, its
                  [`OptimizationResult`][ropt.OptimizationResult] or the
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
    """Raised internally to stop an optimization with a specific exit code.

    Used only within the optimizer core to unwind the backend optimization loop
    (for example when a function or batch budget is reached). It carries the
    [`ExitCode`][ropt.enums.ExitCode] the optimization terminates with.
    """

    def __init__(self, exit_code: ExitCode) -> None:
        """Initialize the OptimizerStop exception.

        Args:
            exit_code: The exit code the optimization terminates with.
        """
        self.exit_code = exit_code
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
