"""The optimization workflow configuration class."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextConfig(BaseModel):
    """Configuration of a single context object.

    Context objects store information that is accessible to and updated by the
    steps of the workflow. They are referred to by their `id`, which is
    therefore mandatory.

    The `init` string identifies the code that is run to initialize the context
    object. It is used by the plugin manager to load the code.

    Additional information needed by the context objects is configured using the
    `with` field.

    Attributes:
        id:   An identifier used to refer to the context object
        init: Identifies the code that initializes the object
        with: Additional information passed to the object
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

    The `run` string identifies the code that executes te step. It is used by
    the plugin manager to load the code.

    Additional information needed by the steps is configured using the `with`
    field.

    Attributes:
        name: An optional name used to refer to the step
        run:  Identifies the code that runs the step
        with: Additional information passed to the step
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
    `context` field, and a `steps` field that defines the tasks to perform.

    The `context` field contains the configuration of the objects that create
    and maintain the context in which the workflow runs. Context objects are
    initialized before creating and running the workflow steps.

    After initializing the context objects, workflow steps are configured by the
    entries in the `steps` field and are initialized and executed in order.
    During workflow execution, the any context object may be inspected and
    updated by the steps. After finishing the workflow, each context object can
    be inspected to retrieve any stored results.

    Attributes:
        context: The Context objects to initialize
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
