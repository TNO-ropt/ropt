"""The optimization step configuration classes."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field

from ropt.config.utils import Array2D  # noqa: TCH001

from ._plan_config import PlanConfig  # noqa: TCH001


class TrackerStepConfig(BaseModel):
    """The configuration class for tracker steps.

    This configuration class is used within an optimization plan to specify a
    tracker object. The `source` option identifies the steps that produce the
    results to track. The `type` option determines what type of result is
    tracked:

    - `optimal` : The optimal result produced so far
    - `last`    : The last result produced

    If `constraint_tolerance` is given, the results are tested against this
    tolerance to filter out constraint violations.

    Attributes:
        id:                   The ID of the step
        source:               Labels(s) of the step(s) that produce results
        type:                 What type of result to track
        constraint_tolerance: Optional constraint tolerance
    """

    id: str
    source: Union[str, Set[str]]
    type_: Literal["optimal_result", "last_result"] = Field(
        default="optimal_result", alias="type"
    )
    constraint_tolerance: Optional[float] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class UpdateConfigStepConfig(BaseModel):
    """Configuration class for update steps.

    This class is employed within an optimization plan to modify the
    configuration of optimizer or evaluation steps that follow it.



    It can also be used to set initial variables from a tracker. The
    `initial_variables` field should contain the ID of a tracker step within the
    plan. The new values of the variables will be retrieved from the result
    tracked by the specified step.

    Attributes:
        updates:         : Updates for the configuration
        initial_variables: Step to set the initial variables from
    """

    updates: Dict[str, Any] = {}
    initial_variables: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class OptimizerStepConfig(BaseModel):
    """The primary configuration class for an optimizer step.

    This configuration class is utilized within an optimization plan to specify
    that an optimizer should be executed.

    The optional `nested_plan` field can be employed to specify nested
    optimization plans that should be executed by the optimizer during each
    function evaluation. The `nested_result` is mandatory in this case and
    indicates which result tracker should be used to retrieve the final result
    of the nested optimization.

    Attributes:
        id:            Optional ID of the step.
        nested_plan:   Optional nested optimization plan.
        nested_result: The optimal nested result.
    """

    id: Optional[str] = None
    nested_plan: Optional[PlanConfig] = None
    nested_result: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class RestartStepConfig(BaseModel):
    """The configuration class for restart steps.

    This configuration class is employed within an optimization plan to specify
    that the plan should restart from the beginning at that point. The
    `max_restarts` field is utilized to determine the maximum number of times
    the plan is allowed to restart. For instance, with `max_restarts=1`, the
    plan will run twice up to the restart step. If the plan has already
    restarted `max_restarts` times, it will proceed with the next step after the
    restart step or finish if it was the last step.

    The `label` fields is optional and gives the ID of a label step. If given,
    the plan will restart at the position of the given label step.

    The `metadata_key` field can be used to provide a key into the metadata
    dictionary of every result produced during the restart. If present, the
    index of the current restart round will be added to the metadata with that
    key.

    Attributes:
        max_restarts: The maximum number of restarts
        label:        Label of the step to restart at
        metadata_key: Optional key to use for storing a restart index
    """

    max_restarts: int
    label: Optional[str] = None
    metadata_key: Optional[str] = "restart"

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class EvaluatorStepConfig(BaseModel):
    """Configuration class for evaluation steps.

    An evaluator step performs a single function evaluation using the
    optimization configuration currently set by the step.

    Attributes:
        id: Optional ID of the step

    Info:
        This configuration class has its `extra` property set to `"allow"`, as
        it is expected that external code may parse additional fields for
        further configuration.
    """

    id: Optional[str] = None
    variables: Optional[Array2D] = None

    model_config = ConfigDict(
        extra="allow",
        validate_default=True,
        arbitrary_types_allowed=True,
    )
