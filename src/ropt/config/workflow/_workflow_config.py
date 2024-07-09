"""The optimization workflow configuration class."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextConfig(BaseModel):
    """Configuration of a single context object.

    Context objects process information that is provided by the steps of the
    workflow. They usually store information in workflow variables that are
    accessible to the steps and after the workflow has finished. In most cases,
    the context objects store a single result in a variable with a name equal to
    the `id` of the context object, but additional variables can be defined also.

    The `init` string identifies the code that is run to initialize the context
    object. It is used by the plugin manager to load the code.

    Additional parameters needed by the context objects are configured using the
    `with_` attribute. The contents of the `with_` attribute depend on the type
    of the context object.

    Context objects are referred to by their `id`, which is mandatory.

    Note: `with` is an alias for `with_`
        When parsing dictionaries into a `ContextConfig` object the name of the
        `with_` attribute should be replaced by by `with`, i.e. without the `_`
        suffix.

    Attributes:
        id:    An identifier used to refer to the context object
        init:  Identifies the code that initializes the object
        with_: Additional parameters passed to the object
    """

    id: str
    init: str
    with_: Optional[Any] = Field(default=None, alias="with")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _after_validator(self) -> ContextConfig:
        if not self.id.isidentifier():
            msg = f"Invalid ID: {self.id}"
            raise ValueError(msg)
        return self


class StepConfig(BaseModel):
    """Configuration of a single step in the workflow.

    A step is a single action within a workflow. The `run` string identifies the
    code that executes te step. It is used by the plugin manager to load the
    code.

    A step may be named using the optional `name` field, which will only be used
    for informational purposes, such as in error messages or in generated reports.

    Additional parameters needed by the step may be configured using the `with_`
    attribute. The content of the `with_` attribute depends on the type of the
    step.

    Execution of the step can be made conditional by providing an expression via
    the `if_` attribute. The expression will be parsed and evaluated, and the
    step will only be executed if the result is `True`.

    Note: `with` and `if` aliases
        When parsing dictionaries into a `StepConfig` object the name of the
        `with_` attribute should be replaced by by `with`, and the name of the
        `if_` attribute by `if`, i.e. without the `_` suffix

    Attributes:
        name:  An optional name used to refer to the step
        run:   Identifies the code that runs the step
        with_: Additional parameters passed to the step
        if_:   Optional expression for conditional evaluation
    """

    name: Optional[str] = None
    run: str
    with_: Dict[str, Any] = Field(default_factory=dict, alias="with")
    if_: Optional[str] = Field(default=None, alias="if")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class WorkflowConfig(BaseModel):
    """Configuration for a workflow.

    A workflow consists of two sections: a context section defined using the
    `context` attribute, and a section that defines the tasks to perform by the
    `steps` attribute.

    The `context` attribute contains the configuration of the objects that
    create and maintain the environment in which the workflow runs. Context
    objects are initialized before creating and running the workflow steps.

    After initializing the context objects, workflow steps are configured by the
    entries given by the `steps` attribute and are initialized and executed in
    order.

    Attributes:
        context: The context objects to initialize
        steps:   The steps that are executed by the workflow
    """

    context: List[ContextConfig] = []
    steps: List[StepConfig]

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _after_validator(self) -> WorkflowConfig:
        duplicates = [
            id_
            for id_, count in Counter([item.id for item in self.context]).items()
            if count > 1
        ]
        if duplicates:
            raise ValueError("Duplicate Context ID(s): " + ", ".join(duplicates))
        return self
