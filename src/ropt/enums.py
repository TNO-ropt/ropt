"""Enumerations used by `ropt`."""

from enum import Enum, IntEnum


class VariableType(IntEnum):
    """Enumerates the variable types."""

    REAL = 1
    "Continuous variables represented by real values."

    INTEGER = 2
    "Discrete variables represented by integer values."


class ConstraintType(IntEnum):
    """Enumerates the types of linear or non-linear constraints."""

    LE = 1
    r"Less or equal constraint ($\le$)."

    GE = 2
    r"Greater or equal constraint ($\ge$)."

    EQ = 3
    r"Equality constraint ($=$)."


class BoundaryType(IntEnum):
    """Enumerates the ways boundaries should be treated.

    When variables are perturbed their values may violate boundary constraints.
    This enumeration lists the ways these values can be modified to fix this.
    """

    NONE = 1
    """Do not modify the value."""

    TRUNCATE_BOTH = 2
    r"""Truncate the value $v_i$ at the lower or upper boundary ($l_i$, $u_i$):

    $$
    \hat{v_i} = \begin{cases}
        l_i & \text{if $v_i < l_i$}, \\
        b_i & \text{if $v_i > b_i$}, \\
        v_i & \text{otherwise}
    \end{cases}
    $$
    """

    MIRROR_BOTH = 3
    r"""Mirror the value $v_i$ at the lower or upper boundary ($l_i$, $u_i$):

    $$
    \hat{v_i} = \begin{cases}
        2l_i - v_i & \text{if $v_i < l_i$}, \\
        2b_i - v_i & \text{if $v_i > b_i$}, \\
        v_i        & \text{otherwise}
    \end{cases}
    $$
    """


class PerturbationType(IntEnum):
    """Enumerates the types of perturbations that can be applied.

    When applying a perturbation to a variable, generally, some value is
    generated, which is then applied to the unperturbed values (usually by
    addition). This enumeration lists the ways how this perturbation value can
    be modified before being added to the unperturbed variable.
    """

    ABSOLUTE = 1
    "Use the perturbation value as is."

    RELATIVE = 2
    r"""Multiply the perturbation value $p_i$ by the range defined by the bounds
    of the variables $c_i$: $\hat{p}_i = (c_{i,\text{max}} - c_{i,\text{min}})
    \cdot p_i$. The bounds will generally be defined in the configuration for
    the variables (see [`VariablesConfig`][ropt.config.enopt.VariablesConfig]).
    """

    SCALED = 3
    r"""Scale each perturbation $p_i$ according to the scales ($s_i$) for each
    variable: $\hat{p}_i = p_i/s_i$. The scales $s_i$ will generally be defined
    in the configuration of the variables (see
    [`VariablesConfig`][ropt.config.enopt.VariablesConfig]).
    """


class EventType(IntEnum):
    """Enumerates the events handled by the event broker.

    During the execution of the optimization plan, events may be emitted and
    callbacks can be connected to these events . When triggered by an event, the
    callbacks receive an `Event` object. This object contains at least the type
    of the event (a value of this enumeration) and the current configuration of
    the step that is executing. Additionally, depending on the event type, a
    tuple of result objects or an exit code may be present. Refer to the
    documentation of the individual event types for details.
    """

    START_EVALUATION = 1
    """Emitted before evaluating new functions."""

    FINISHED_EVALUATION = 2
    """Emitted after finishing the evaluation.

    Results may be passed to callback reacting to this event.
    """

    START_OPTIMIZER_STEP = 3
    """Emitted just before starting an optimizer step."""

    FINISHED_OPTIMIZER_STEP = 4
    """Emitted immediately after an optimizer step finishes.

    Results and an exit code may be passed to callback reacting to this event.
    """

    START_EVALUATOR_STEP = 5
    """Emitted just before starting an evaluation step."""

    FINISHED_EVALUATOR_STEP = 6
    """Emitted immediately after an evaluation step finishes.

    Results and an exit code may be passed to callback reacting to this event.
    """


class OptimizerExitCode(IntEnum):
    """Enumerates the reasons for terminating an optimization."""

    UNKNOWN = 0
    """Unknown cause of termination."""

    TOO_FEW_REALIZATIONS = 1
    """Returned when too few realizations are evaluated successfully."""

    MAX_FUNCTIONS_REACHED = 2
    """Returned when the maximum number of function evaluations is reached."""

    USER_ABORT = 3
    """Returned when the optimization is aborted by the user."""

    OPTIMIZER_STEP_FINISHED = 4
    """Returned when an optimization step terminates normally."""

    EVALUATION_STEP_FINISHED = 5
    """Returned when an evaluation step terminates normally."""


class ResultAxisName(Enum):
    """Enumerates the possible axis names in a Results data object."""

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
