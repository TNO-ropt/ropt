"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ropt.plan import ContextUpdate, ContextUpdateResults
from ropt.plugins.plan.base import ContextObj

from ._utils import _get_last_result, _update_optimal_result

if TYPE_CHECKING:
    from ropt.config.plan import ContextConfig
    from ropt.plan import Plan
    from ropt.results import FunctionResults


class DefaultTrackerWith(BaseModel):
    """Parameters for the track_results context object.

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


class DefaultTrackerContext(ContextObj):
    """The default results context object."""

    def __init__(self, config: ContextConfig, plan: Plan) -> None:
        """Initialize a default tracker context object.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)
        self._with = (
            DefaultTrackerWith()
            if config.with_ is None
            else DefaultTrackerWith.model_validate(config.with_)
        )
        self.set_variable(None)

    def update(self, value: ContextUpdate) -> None:
        """Update the tracker object.

        Updates the stored results. If the `constraint_tolerance` value set in
        the object configuration is `None`, the object value will be updated
        according to the `type`. If `constraint_tolerance` is not `None`, it
        will only be updated if the results do not violate any constraints.

        Args:
            value: The value to set.
        """
        if isinstance(value, ContextUpdateResults):
            results: Optional[FunctionResults] = None
            if self._with.type_ == "optimal":
                results = _update_optimal_result(
                    self.get_variable(), value.results, self._with.constraint_tolerance
                )
            if self._with.type_ == "last":
                results = _get_last_result(
                    value.results, self._with.constraint_tolerance
                )
            if results is not None:
                results = deepcopy(results)
                self.set_variable(results)

    def reset(self) -> None:
        """Clear the stored values."""
        self.set_variable(None)
