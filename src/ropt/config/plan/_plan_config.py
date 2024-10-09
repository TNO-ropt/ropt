"""The optimization plan configuration class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class EventHandlerConfig(BaseModel):
    """Configuration of a single event handler object.

    Event handler objects process events emitted by the steps of the
    optimization plan. They usually store information in plan variables that are
    accessible to the steps and to the user via the plan object.

    The `init` string identifies the code that is run to initialize the handler
    object. It is used by the plugin manager to load the code.

    Additional parameters needed by the handler objects are configured using the
    `with_` attribute. The contents of the `with_` attribute depend on the type
    of the handler object.

    Note: `with` is an alias for `with_`
        When parsing dictionaries into a `EventHandlerConfig` object, the name of the
        `with_` attribute should be replaced by by `with`, i.e. without the `_`
        suffix.

    Attributes:
        name:  An identifier used to refer to the handler object
        init:  Identifies the code that initializes the object
        with_: Additional parameters passed to the object
    """

    name: Optional[str] = None
    init: str
    with_: Any = Field(default_factory=dict, alias="with")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class StepConfig(BaseModel):
    """Configuration of a single step.

    A step is a single action within an optimization plan. The `run` string
    identifies the code that executes te step. It is used by the plugin manager
    to load the code.

    A step may be named using the optional `name` field, which will only be used
    for informational purposes, such as in error messages or in generated reports.

    Additional parameters needed by the step may be configured using the `with_`
    attribute. The content of the `with_` attribute depends on the type of the
    step.

    Execution of the step can be made conditional by providing an expression via
    the `if_` attribute. The expression will be parsed and evaluated, and the
    step will only be executed if the result is `True`.

    Note: `with` and `if` aliases
        When parsing dictionaries into a `StepConfig` object, the name of the
        `with_` attribute should be replaced by by `with`, and the name of the
        `if_` attribute by `if`, i.e. without the `_` suffix

    Info: Conditional evaluation
        Conditions defined via the `if_` attribute are evaluated by passing them
        to the [`eval`][ropt.plan.Plan.eval] method of the plan object that is
        executing the steps. Consult the documentation of the method for more
        details on the expressions that can be evaluated.

    Attributes:
        name:  An optional name used to refer to the step
        run:   Identifies the code that runs the step
        with_: Additional parameters passed to the step
        if_:   Optional expression for conditional evaluation
    """

    name: Optional[str] = None
    run: str
    with_: Union[Dict[str, Any], str] = Field(default_factory=dict, alias="with")
    if_: Optional[str] = Field(default=None, alias="if")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class PlanConfig(BaseModel):
    """Configuration class for optimization plans.

    An optimization plan configuration consists of two sections: a event handlers
    section defined using the `handlers` attribute, and a section that defines
    the tasks to perform by the `steps` attribute.

    The `handlers` attribute contains the configuration of the objects that
    process events emitted by the steps. Event handler objects are initialized
    before creating and running the steps.

    When running a plan, arguments can be passed. The `inputs` attributes
    denotes a list of input variables, that will be initialized with the passed
    values.

    After the plan has finished, a tuple of outputs can be returned. The
    `outputs` attribute contains the names of the variables that will used to
    generate the output tuple.

    Variables can be created on the fly by the steps, or by the event handler
    objects, but can also be predefined by the `variables` attribute, giving
    their name and value.

    After initializing the event handler objects, the steps are configured by
    the entries given by the `steps` attribute and are initialized and executed
    in order.

    Attributes:
        handlers:  The event handler objects to initialize
        steps:     The steps that are executed by the plan
        inputs:    The names of input variables
        outputs:   The names of output variables
        variables: Names and values of preset variables.
    """

    handlers: List[EventHandlerConfig] = []
    steps: List[StepConfig]
    inputs: List[str] = []
    outputs: List[str] = []
    variables: Dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )
