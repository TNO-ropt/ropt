"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ropt.exceptions import WorkflowError
from ropt.plugins.workflow.base import ContextObj
from ropt.results import Results

from ._utils import _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    from ropt.config.workflow import ContextConfig
    from ropt.results import FunctionResults
    from ropt.workflow import Workflow


class DefaultResultsTrackerWith(BaseModel):
    """Parameters for the results_tracker context object.

    The `type` parameter determines what result is tracked:
    - "optimal": Track the best result added
    - "last": Track the last result added

    Attributes:
        type:                 The type of result to store
        constraint_tolerance: Optional constraint tolerance
    """

    type_: Literal["optimal", "last"] = Field(default="optimal", alias="type")
    constraint_tolerance: Optional[float] = 1e-10

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultResultsTrackerContext(ContextObj):
    """The default results_tracker context object."""

    def __init__(self, config: ContextConfig, workflow: Workflow) -> None:
        """Initialize a default results_tracker context object.

        Args:
            config:   The configuration of the step
            workflow: The workflow
        """
        super().__init__(config, workflow)
        self._with = (
            DefaultResultsTrackerWith()
            if config.with_ is None
            else DefaultResultsTrackerWith.model_validate(config.with_)
        )
        self._value: Optional[FunctionResults] = None

    def update(self, value: Tuple[Results, ...]) -> None:
        """Update the result object.

        Updates the stored results. If the `constraint_tolerance` value set in
        the object configuration is `None`, the object value will be updated
        according to the `type`. If `constraint_tolerance` is not `None`, it
        will only be updated if the results do not violate any constraints.

        Args:
            value: The value to set.
        """
        msg = "attempt to update with invalid data."
        if not isinstance(value, tuple):
            raise WorkflowError(msg, context_id=self.context_config.id)
        for item in value:
            if not isinstance(item, Results):
                raise WorkflowError(msg, context_id=self.context_config.id)

        results: Optional[FunctionResults] = None
        if self._with.type_ == "optimal":
            results = _update_optimal_result(
                self._value, value, self._with.constraint_tolerance
            )
        if self._with.type_ == "last":
            results = _get_last_result(value, self._with.constraint_tolerance)
        if results is not None:
            results = deepcopy(results)
            self._value = results

    def value(self) -> Optional[FunctionResults]:
        """Return the optimal or last results object."""
        return self._value

    def reset(self) -> None:
        """Clear the stored values."""
        self._value = None
