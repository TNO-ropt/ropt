"""The optimization plan configuration class."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlanStepConfig(BaseModel):
    """Configuration for a single step within an optimization plan.

    A step represents a single action in the optimization process. It can
    access and modify plan variables, execute tasks such as optimization runs,
    and emit [`events`][ropt.plan.Event], for example, when intermediate
    optimization results are generated.

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

        While mathematical expressions are generally evaluated using the
        `$eval()` function, this is optional for expressions passed via the
        `if_` attribute. For instance, `$eval(x > 0)` and `x > 0` are equivalent
        values for the `if_` attribute.

    Info: Alternative specification of the step config.
        The standard format for defining a step configuration follows the
        attribute-based structure shown here. Typically, a dictionary
        initializing a step would resemble:

        ```python
        {
            "run": "optimizer",
            "with": {
                "config": "$opt_config",
                "tags": "opt",
            }
            "if": "x > 0"
        }
        ```

        However, a pre-processing step allows a short-hand alternative notation,
        converting it into the above format before constructing the
        configuration object:

        ```python
        {
            "optimizer": {
                "config": "$opt_config",
                "tags": "opt",
            }
            "if": "x > 0"
        }

        For very simple steps that have only a single configuration field, this may
        even be simpler by just providing the value, like this:

        ```python
        {
            "print": "hello world!",
        }
        ```

        However, whereas first short-hand approach is generic and works for all
        steps, this last form may or may not be implemented the step. Consult
        the step documentation to make sure it works.

    Attributes:
        run:   Specifies the code that runs the step.
        with_: Additional parameters passed to the step.
        if_:   An optional expression for conditional execution.
    """

    run: str
    with_: list[Any] | dict[str, Any] | str = Field(default_factory=dict, alias="with")  # type: ignore[arg-type]
    if_: str | None = Field(default=None, alias="if")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _parse_dict(cls, data: Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(data, Mapping) and "run" not in data:
            if len(data) == 1:
                key, value = next(iter(data.items()))
                return {"run": key, "with": value}
            if len(data) == 2 and "if" in data:  # noqa: PLR2004
                key = next(key for key in data if key != "if")
                return {"run": key, "with": data[key], "if": data["if"]}
        return data


class ResultHandlerConfig(BaseModel):
    """Configuration for a single result handler object.

    Result handler objects process events emitted by the steps of an
    optimization plan. These objects can receive [`events`][ropt.plan.Event]
    directly from the plan's steps, or from another result handler in a
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

    Info: Alternative specification of the result handler config.
        The standard format for defining a result handler configuration follows
        the attribute-based structure shown here. Typically, a dictionary
        initializing a step would resemble:

        ```python
        {
            "run": "tracker,
            "with": {
                "var": "result",
                "tags": "opt",
            }
        }
        ```

        However, a pre-processing step allows a short-hand alternative notation,
        converting it into the above format before constructing the
        configuration object:

        ```python
        {
            "tracker": {
                "var": "result",
                "tags": "opt",
            }
        }
        ```

    Attributes:
        run:   Specifies the code used to initialize the result handler.
        with_: Additional parameters passed to the result handler.
    """

    run: str
    with_: Any = Field(default_factory=dict, alias="with")  # type: ignore[arg-type]

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _parse_dict(cls, data: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(data, Mapping):
            msg = "Input should be a mapping"
            raise ValueError(msg)  # noqa: TRY004
        if len(data) == 1:
            key, value = next(iter(data.items()))
            if key != "run":
                return {"run": key, "with": value}
        return data


class PlanConfig(BaseModel):
    """Configuration class for optimization plans.

    This class configures the workflows managed by a [`Plan`][ropt.plan.Plan]
    object, specifying key sections that define the plan's behavior and data
    flow. Each section aligns with a distinct aspect of the optimization plan:

    `inputs`
    : Specifies the names of plan variables to hold input values when the
      optimization workflow is started using the [`run`][ropt.plan.Plan.run]
      method.

    `outputs`
    : Lists the names of plan variables whose final values will be returned as a
      tuple when the optimization completes and the `run` method finishes.

    `bubble_up`
    : Lists the tags of results that should be bubbled up to the parent plan (if
      present).

    `variables`
    : Defines a dictionary of plan variable names and initial values, used
      during plan execution.

    `steps`
    : Outlines each step executed once the plan begins Steps support a variety
      of actions, such as initiating an optimization, accessing or modifying
      variables, and emitting events. Emitted events are processed by the
      handlers specified in the `results` section.

    `handlers`
    : Specifies the event handlers that process events emitted by steps.
      Handlers receive events sequentially, with each handler passing events to
      the next in the chain.

    Attributes:
        inputs:    List of input variable names.
        outputs:   List of output variable names.
        bubble_up: List of tags denoting results that should be bubbled up.
        variables: Dictionary of variable names with initial values.
        steps:     List of steps defining plan actions.
        handlers:  List of result handler instances.
    """

    inputs: list[str] = []
    outputs: list[str] = []
    bubble_up: set[str] = set()
    variables: dict[str, Any] = {}
    steps: list[PlanStepConfig] = []
    handlers: list[ResultHandlerConfig] = []

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        frozen=True,
    )
