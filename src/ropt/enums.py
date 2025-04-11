"""Enumerations used within the `ropt` library."""

from enum import IntEnum, StrEnum


class VariableType(IntEnum):
    """Enumerates the types of optimization variables.

    Specified in [`VariablesConfig`][ropt.config.enopt.VariablesConfig], this
    information allows optimization backends to adapt their behavior.
    """

    REAL = 1
    "Continuous variables represented by real values."

    INTEGER = 2
    "Discrete variables represented by integer values."


class BoundaryType(IntEnum):
    """Enumerates strategies for handling variable boundary violations.

    When variables are perturbed during optimization, their values might fall
    outside the defined lower and upper bounds. This enumeration defines
    different methods to adjust these perturbed values back within the valid
    range. The chosen strategy is configured in the
    [`GradientConfig`][ropt.config.enopt.GradientConfig].
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
    """Enumerates methods for scaling perturbation samples.

    Before a generated perturbation sample is added to a variable's current
    value (during gradient estimation, for example), it can be scaled. This
    enumeration defines the available scaling methods, configured in the
    [`GradientConfig`][ropt.config.enopt.GradientConfig].
    """

    ABSOLUTE = 1
    "Use the perturbation value as is."

    RELATIVE = 2
    r"""Multiply the perturbation value $p_i$ by the range defined by the bounds
    of the variables $c_i$: $\hat{p}_i = (c_{i,\text{max}} - c_{i,\text{min}})
    \times p_i$. The bounds will generally be defined in the configuration for
    the variables (see [`VariablesConfig`][ropt.config.enopt.VariablesConfig]).
    """


class EventType(IntEnum):
    """Enumerates the types of events emitted during optimization plan execution.

    Events signal significant occurrences within the optimization process, such
    as the start or end of a plan step or an evaluation. Callbacks can be
    registered to listen for specific event types.

    When an event occurs, registered callbacks receive an
    [`Event`][ropt.plan.Event] object containing:

    - `event_type`: The type of the event (a value from this enumeration).
    - `config`: The configuration object associated with the source.
    - `source`: The unique ID (UUID) of the plan component that emitted the event.
    - `data`: A dictionary containing event-specific data, such as
      [`Results`][ropt.results.Results] objects.

    Refer to the documentation of individual event types and plan components for
    details on the specific data they provide.
    """

    START_EVALUATION = 1
    """Emitted before evaluating new functions."""

    FINISHED_EVALUATION = 2
    """Emitted after finishing the evaluation."""

    START_OPTIMIZER_STEP = 3
    """Emitted just before starting an optimizer step."""

    FINISHED_OPTIMIZER_STEP = 4
    """Emitted immediately after an optimizer step finishes."""

    START_EVALUATOR_STEP = 5
    """Emitted just before starting an evaluation step."""

    FINISHED_EVALUATOR_STEP = 6
    """Emitted immediately after an evaluation step finishes."""


class OptimizerExitCode(IntEnum):
    """Enumerates the reasons for terminating an optimization."""

    UNKNOWN = 0
    """Unknown cause of termination."""

    TOO_FEW_REALIZATIONS = 1
    """Returned when too few realizations are evaluated successfully."""

    MAX_FUNCTIONS_REACHED = 2
    """Returned when the maximum number of function evaluations is reached."""

    MAX_BATCHES_REACHED = 3
    """Returned when the maximum number of evaluation batches is reached."""

    NESTED_OPTIMIZER_FAILED = 4
    """Returned when a nested optimization fails to find an optimal value."""

    USER_ABORT = 5
    """Returned when the optimization is aborted by the user."""

    OPTIMIZER_STEP_FINISHED = 6
    """Returned when an optimization step terminates normally."""

    EVALUATION_STEP_FINISHED = 7
    """Returned when an evaluation step terminates normally."""


class ResultAxis(StrEnum):
    """Enumerates the semantic meaning of axes in `Results` data arrays.

    [`Results`][ropt.results.Results] objects store optimization data (like
    variable values, objective function values, constraint values, etc.) in
    multidimensional NumPy arrays. This enumeration provides standardized labels
    to identify what each dimension (axis) of these arrays represents.

    For example, an array might have dimensions corresponding to different
    realizations, different objective functions, or different variables.

    This information is stored as metadata within the `Results` object and can
    be accessed using methods like
    [`get_axes`][ropt.results.ResultField.get_axes] on result fields. It is
    used internally, for instance, during data export to correctly label axes
    or retrieve associated names (like variable names) from the configuration.
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
