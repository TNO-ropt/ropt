"""The optimization workflow configuration class."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextConfig(BaseModel):
    """Configuration of a single context object.

    Context objects store information that is accessible to and updated by the
    steps of the workflow. They are referred to by their `id`, which is
    mandatory.

    The `init` string identifies the code that is run to initialize the context
    object. It is used by the plugin manager to load the code.

    Additional parameters needed by the context objects are configured using the
    `with` attribute. The contents of the `with` attribute depend on the type
    of the context object.

    Attributes:
        id:   An identifier used to refer to the context object
        init: Identifies the code that initializes the object
        with: Additional parameters passed to the object
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

    Additional parameter needed by the step may be configured using the `with`
    attribute. The content of the `with` attribute depends on the type of the step.

    Attributes:
        name: An optional name used to refer to the step
        run:  Identifies the code that runs the step
        with: Additional parameters passed to the step
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
    `context` attribute, and a `steps` attribute that defines the tasks to
    perform.

    The `context` attribute contains the configuration of the objects that
    create and maintain the context in which the workflow runs. Context objects
    are initialized before creating and running the workflow steps.

    After initializing the context objects, workflow steps are configured by the
    entries given by the `steps` attribute and are initialized and executed in
    order. During workflow execution, the context objects may be inspected and
    updated by the steps. After finishing the workflow, each context object can
    be inspected to retrieve any stored results.

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
