"""Enumerations used in the configuration, event and result APIs.

A member of a string-valued enumeration is a `str`, so wherever one is accepted
its value may be written instead:
[`VariableType`][ropt.enums.VariableType],
[`BoundaryType`][ropt.enums.BoundaryType],
[`PerturbationType`][ropt.enums.PerturbationType] and
[`AxisName`][ropt.enums.AxisName].
"""

from enum import IntEnum, StrEnum
from typing import Final


class VariableType(StrEnum):
    """Enumerates the types of optimization variables.

    Specified in [`VariablesConfig`][ropt.config.VariablesConfig], this
    information allows optimization backends to adapt their behavior.
    """

    REAL = "real"
    "Continuous variables represented by real values."

    INTEGER = "integer"
    "Discrete variables represented by integer values."


class BoundaryType(StrEnum):
    """Enumerates strategies for handling variable boundary violations.

    When variables are perturbed during optimization, their values might fall
    outside the defined lower and upper bounds. This enumeration defines
    different methods to adjust these perturbed values back within the valid
    range. The chosen strategy is configured in the
    [`GradientConfig`][ropt.config.GradientConfig].
    """

    NONE = "none"
    """Do not modify the value."""

    TRUNCATE = "truncate"
    r"""Truncate the value $v_i$ at the lower or upper boundary ($l_i$, $u_i$):

    $$
    \hat{v_i} = \begin{cases}
        l_i & \text{if $v_i < l_i$}, \\
        u_i & \text{if $v_i > u_i$}, \\
        v_i & \text{otherwise}
    \end{cases}
    $$
    """

    MIRROR = "mirror"
    r"""Mirror the value $v_i$ at the lower or upper boundary ($l_i$, $u_i$):

    $$
    \hat{v_i} = \begin{cases}
        2l_i - v_i & \text{if $v_i < l_i$}, \\
        2u_i - v_i & \text{if $v_i > u_i$}, \\
        v_i        & \text{otherwise}
    \end{cases}
    $$
    """


class PerturbationType(StrEnum):
    """Enumerates methods for scaling perturbation samples.

    Before a generated perturbation sample is added to a variable's current
    value (during gradient estimation, for example), it can be scaled. This
    enumeration defines the available scaling methods, configured in the
    [`GradientConfig`][ropt.config.GradientConfig].
    """

    ABSOLUTE = "absolute"
    "Use the perturbation value as is."

    RELATIVE = "relative"
    r"""Multiply the perturbation value $p_i$ by the range defined by the bounds
    of the variables $c_i$: $\hat{p}_i = (c_{i,\text{max}} - c_{i,\text{min}})
    \times p_i$. The bounds will generally be defined in the configuration for
    the variables (see [`VariablesConfig`][ropt.config.VariablesConfig]).
    """


class EnOptEventType(IntEnum):
    """Enumerates the types of events emitted during a run.

    See [Handling Results](../results/handlers.md#writing-your-own-handler) for
    when each event type fires and what data it carries.
    """

    START_EVALUATION = 1
    """Emitted before a batch of function or gradient evaluations."""

    FINISHED_EVALUATION = 2
    """Emitted after that batch completes; carries the results produced."""

    START_OPTIMIZER = 3
    """Emitted just before starting an optimizer."""

    FINISHED_OPTIMIZER = 4
    """Emitted immediately after an optimizer finishes."""

    START_ENSEMBLE_EVALUATOR = 5
    """Emitted before an evaluation without an optimizer begins."""

    FINISHED_ENSEMBLE_EVALUATOR = 6
    """Emitted after an evaluation without an optimizer finishes."""


class ExitCode(IntEnum):
    """Enumerates the reasons a run ends.

    A run is an optimization or a single evaluation. It either **stops** or is
    **aborted**. It stops when a condition on the run is met — the optimizer
    converged, a budget ran out, a handler decided the results were good enough
    — so it ends at a point that satisfied a stated criterion. It is aborted
    when something ends it regardless of the state of the run, and the
    result is then whatever it had reached, not a considered endpoint.

    Each member carries a short description in
    [`message`][ropt.enums.ExitCode.message], for reporting.
    """

    UNKNOWN = 0
    """Unknown cause of termination."""

    TOO_FEW_REALIZATIONS = 1
    """Returned when too few realizations are evaluated successfully."""

    MAX_FUNCTIONS_REACHED = 2
    """Returned when the maximum number of function evaluations is reached."""

    MAX_BATCHES_REACHED = 3
    """Returned when the maximum number of evaluation batches is reached."""

    STOPPED = 4
    """Returned when an event handler asked the run to stop.

    A graceful end at an evaluation boundary, on a criterion the caller
    supplied. A `report` callback is such a handler.
    """

    FINISHED = 5
    """Returned when the run terminated normally."""

    EXECUTOR_SHUT_DOWN = 6
    """Returned when the executor could no longer run the evaluation.

    Not a failure: in practice the interpreter was shutting down and the worker
    pool was gone, so the run is released rather than left waiting. An executor
    that breaks raises [`ExecutionError`][ropt.exceptions.ExecutionError]
    instead.
    """

    ABORTED = 7
    """Returned when the run was aborted from outside, without regard to where
    it had got to.

    This is what closing a session reports for a run still under way.
    """

    ABORTED_ON_ERROR = 8
    """Returned when another run this one shares a session with raised."""

    USER_ABORT = 9
    """Returned when [`Session.abort`][ropt.Session.abort] aborted the run.

    Set apart from [`ABORTED`][ropt.enums.ExitCode.ABORTED] so that an abort
    that was asked for can be told from one the library performed itself.
    """

    @property
    def message(self) -> str:
        """A short description of this exit code, for reporting.

        Note that `str()` and f-strings give the integer value, as they do for
        any [`IntEnum`][enum.IntEnum]; use this or `name` to report a run.

        Returns:
            A capitalized phrase with no trailing period.
        """
        return _MESSAGES[self]


# Kept beside the members rather than in the docs, so that one call site reports
# a run the same way everywhere.
_MESSAGES: Final[dict[ExitCode, str]] = {
    ExitCode.UNKNOWN: "Exit code not set",
    ExitCode.TOO_FEW_REALIZATIONS: "Too few realizations evaluated successfully",
    ExitCode.MAX_FUNCTIONS_REACHED: "Maximum number of function evaluations reached",
    ExitCode.MAX_BATCHES_REACHED: "Maximum number of evaluation batches reached",
    ExitCode.STOPPED: "Stopped by an event handler",
    ExitCode.FINISHED: "Run completed normally",
    ExitCode.EXECUTOR_SHUT_DOWN: "Executor could no longer run the evaluations",
    ExitCode.ABORTED: "Run aborted",
    ExitCode.ABORTED_ON_ERROR: "Aborted because another run on the session failed",
    ExitCode.USER_ABORT: "Run aborted on request",
}


class AxisName(StrEnum):
    """Enumerates the semantic meaning of axes in data arrays.

    Labels what each dimension of a [`Results`][ropt.results.Results] field's
    multidimensional array represents, and is used to look up axis labels via
    [`get_axes`][ropt.results.AxisMetadata.get_axes]. See
    [Working with Results](../results/results.md#axes-and-dimensionality)
    for a full table of fields and their axes.
    """

    VARIABLE = "variable"
    """The axis index corresponds to the index of the variable."""

    OBJECTIVE = "objective"
    """The axis index corresponds to the index of the objective function."""

    LINEAR_CONSTRAINT = "linear_constraint"
    """The axis index corresponds to the index of the linear constraint."""

    NONLINEAR_CONSTRAINT = "nonlinear_constraint"
    """The axis index corresponds to the index of the constraint function."""

    REALIZATION = "realization"
    """The axis index corresponds to the index of the realization."""

    PERTURBATION = "perturbation"
    """The axis index corresponds to the index of the perturbation."""
