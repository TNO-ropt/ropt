"""The optimization plan configuration class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class SetStepConfig(BaseModel):
    """Configuration for a single set step within an optimization plan.

    A set step is used to change the value of one or more plan variables. The
    attributes of the step should either be a dictionary of variable-name/value
    pairs, or a list of such dictionaries. When the set step is run by the plan,
    the variables are set to the given values. Variables that are referenced may
    be dictionaries or objects with attributes. These can be modified by using
    the `[]` or `.` operators (possibly nested) with the specified variable
    name.

    Info: Using Expressions in the Value
        Optionally, the supplied value may be an expression, following the rules
        of the [`eval`][ropt.plan.Plan.eval] method of the plan object executing
        the run steps. Refer to the method's documentation for more information
        on supported expressions.

    Note: Dictionaries vs. Lists
        The arguments of the set step may be dictionaries or lists of
        dictionaries. Multiple variables can be set in this manner in both
        cases. However, care should be taken if a dictionary is used where the
        order of the keys is not well defined, for instance, if it has been read
        from a JSON file. If the order in which variables are set is important,
        it is recommended to use a list.
    """

    set: Union[Dict[str, Any], List[Dict[str, Any]]]

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class RunStepConfig(BaseModel):
    """Configuration for a single run step within an optimization plan.

    A run step represents a single action in the optimization process. It can
    access and modify plan variables, execute tasks such as optimization runs,
    and emit [`events`][ropt.plan.Event], for example, when intermediate
    optimization results are generated.

    The `run` string specifies the code that executes the step, and is used by
    the plugin manager to load the appropriate code.

    Additional parameters required by the run step can be configured using the
    `with_` attribute, with its content varying based on the step type.

    Execution of the run step can be made conditional by providing an expression
    through the `if_` attribute. This expression is evaluated, and the step is
    executed only if the result is `True`.

    Note: `with` and `if` aliases
        When parsing dictionaries into a `StepConfig` object, replace the
        `with_` attribute with `with`, and `if_` with `if` (without the
        underscore).

    Info: Conditional evaluation
        Conditions specified via the `if_` attribute are evaluated using the
        [`eval`][ropt.plan.Plan.eval] method of the plan object executing the
        run steps. Refer to the method's documentation for more information on
        supported expressions.

        While mathematical expressions often need to be enclosed within
        `${{ ... }}` (double braces) in a plan configuration string, this is
        optional for expressions passed via the `if_` attribute.

    Attributes:
        run:   Specifies the code that runs the step.
        with_: Additional parameters passed to the step.
        if_:   An optional expression for conditional execution.
    """

    run: str
    with_: Union[List[Any], Dict[str, Any], str] = Field(
        default_factory=dict, alias="with"
    )
    if_: Optional[str] = Field(default=None, alias="if")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class ResultHandlerConfig(BaseModel):
    """Configuration for a single result handler object.

    Result handler objects process events emitted by the run steps of an
    optimization plan. These objects can receive [`events`][ropt.plan.Event]
    directly from the plan's run steps, or from another result handler in a
    chain of handlers, as defined in the `results` section of a
    [`PlanConfig`][ropt.config.plan.PlanConfig] object. Upon receiving events,
    handlers may perform actions such as modifying plan variables, generating
    output, or updating the result objects included in the event.

    The `run` string specifies the code that initializes the result handler.
    This string is used by the plugin manager to load the handler's code.

    Additional parameters for the handler are configured using the `with_`
    attribute, which varies depending on the type of handler.

    Note: `with` is an alias for `with_`
        When parsing dictionaries into a `ResultHandlerConfig` object, the
        `with_` attribute should be replaced by `with` (without the underscore).

    Attributes:
        run:   Specifies the code used to initialize the result handler.
        with_: Additional parameters passed to the result handler.
    """

    run: str
    with_: Any = Field(default_factory=dict, alias="with")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class PlanConfig(BaseModel):
    """Configuration class for optimization plans.

    This class configures the workflows managed by a [`Plan`][ropt.plan.Plan]
    object, specifying key sections that define the plan's behavior and data
    flow. Each section aligns with a distinct aspect of the optimization plan:

    `inputs`
    : Specifies the names of variables to hold input values when the
      optimization workflow is started using the [`run`][ropt.plan.Plan.run]
      method.

    `outputs`
    : Lists the names of variables whose final values will be returned as a
      tuple when the optimization completes and the `run` method finishes.

    `variables`
    : Defines a dictionary of variable names and initial values, used during
      plan execution.

    `steps`
    : Outlines each step executed once the plan begins. Steps can be set steps,
      which update plan variables, or run steps loaded from plan plugins. Run
      steps support a variety of actions, such as initiating an optimization,
      accessing or modifying variables, and emitting events. Emitted events are
      processed by the handlers specified in the `results` section.

    `results`
    : Specifies the event handlers that process events emitted by run steps.
      Handlers receive events sequentially, with each handler passing events to
      the next in the chain.

    Attributes:
        inputs:    List of input variable names.
        outputs:   List of output variable names.
        variables: Dictionary of variable names with initial values.
        steps:     List of steps defining plan actions.
        results:   List of result handler instances.
    """

    inputs: List[str] = []
    outputs: List[str] = []
    variables: Dict[str, Any] = {}
    steps: List[Union[SetStepConfig, RunStepConfig]] = []
    results: List[ResultHandlerConfig] = []

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )
