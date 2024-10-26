"""The optimization plan configuration class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class StepConfig(BaseModel):
    """Configuration for a single step within an optimization plan.

    A step represents a single action in the optimization process. It can access
    and modify plan variables, execute tasks such as optimization runs, and emit
    [`events`][ropt.plan.Event], for example, when intermediate optimization
    results are generated.

    The `run` string specifies the code that executes the step, and is used by
    the plugin manager to load the appropriate code.

    Additional parameters required by the step can be configured using the
    `with_` attribute, with its content varying based on the step type.

    Execution of the step can be made conditional by providing an expression
    through the `if_` attribute. This expression is evaluated, and the step is
    executed only if the result is `True`.

    Note: `with` and `if` aliases
        When parsing dictionaries into a `StepConfig` object, replace the
        `with_` attribute with `with`, and `if_` with `if` (without the
        underscore).

    Info: Conditional evaluation
        Conditions specified via the `if_` attribute are evaluated using the
        [`eval`][ropt.plan.Plan.eval] method of the plan object executing the
        steps. Refer to the method's documentation for more information on
        supported expressions.

        While mathematical expressions often need to be enclosed within
        `{{ ... }}` (double braces) in a plan configuration string, this is
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

    Result handler objects process events emitted by the steps of an
    optimization plan. These objects can receive [`events`][ropt.plan.Event]
    directly from the plan's steps, or from another result handler in a chain of
    handlers, as defined in the `results` section of a
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

    This class is used to configure the optimization workflows executed by a
    [`Plan`][ropt.plan.Plan] object. A `PlanConfig` object is passed when the
    plan is created, and it defines several key sections, each corresponding to
    a different aspect of the plan's configuration:

    `inputs`
    : Defines the names of variables that will store the input values passed
      when the optimization workflow is started via the
      [`run`][ropt.plan.Plan.run] method.

    `outputs`
    : Specifies the names of variables whose values will be returned as a tuple
      after the optimization workflow completes and the
      [`run`][ropt.plan.Plan.run] method finishes execution.

    `variables`
    : A dictionary of variable names and their associated values, which are set
      and accessed during the execution of the plan.

    `steps`
    : Describes the individual steps that are executed once the plan starts.
      These steps can access and modify the variables defined in the `inputs`,
      `outputs`, and `variables` sections. Additionally, steps may emit
      [`events`][ropt.plan.Event], which are processed by the result handlers
      defined in the `results` section.

    `results`
    : Defines the result handlers that process events emitted by the steps.
      Events are passed sequentially through each handler, with the first handler
      receiving events directly from the steps and passing them along the chain.

    Attributes:
        inputs:    A list of input variable names.
        outputs:   A list of output variable names.
        variables: A dictionary of preset variable names and values.
        steps:     The steps to be executed in the plan.
        results:   The result handler objects to initialize.
    """

    inputs: List[str] = []
    outputs: List[str] = []
    variables: Dict[str, Any] = {}
    steps: List[StepConfig] = []
    results: List[ResultHandlerConfig] = []

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )
