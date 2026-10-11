"""Exceptions raised within the `ropt` library.

Two families are defined here. The errors derive from
[`RoptError`][ropt.exceptions.RoptError] and report a failure a caller may
catch: [`WorkflowError`][ropt.exceptions.WorkflowError],
[`ExecutionError`][ropt.exceptions.ExecutionError],
[`UnsupportedError`][ropt.exceptions.UnsupportedError] and
[`AbortedError`][ropt.exceptions.AbortedError].

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

    For example an executor that cannot start, a broken process pool, a function
    that cannot be serialized for a worker, an HPC setup or submission problem,
    or an evaluation whose worker died before producing a result.
    """


class UnsupportedError(RoptError):
    """An optional dependency is missing or a requested feature is unsupported.

    For example a required extra (such as pandas or polars) is not installed,
    or the selected plugin does not support the requested method.
    """


class AbortedError(RoptError):
    """Work was aborted before it could finish.

    Raised by an optimization, an evaluation or a `run` call that was aborted:
    by [`Session.abort`][ropt.Session.abort], by its session closing, by a
    failure elsewhere, or because its pool could no longer run the work. See
    [Failures and Aborts](../running/failures.md).

    Attributes:
        exit_code: Why the work was aborted.
    """

    def __init__(self, exit_code: ExitCode) -> None:
        """Initialize the error.

        Args:
            exit_code: Why the work was aborted.
        """
        self.exit_code = exit_code
        msg = f"The work was aborted before it could finish: {exit_code.name}."
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
    interpreter shutdown, when the worker pool is gone and a task that outlived
    its program is released rather than left waiting.
    """
