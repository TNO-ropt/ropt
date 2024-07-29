"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from ropt.enums import EventType
from ropt.plugins.plan.base import ContextObj
from ropt.report import ResultsTable
from ropt.results import convert_to_maximize

if TYPE_CHECKING:
    from ropt.config.plan import ContextConfig
    from ropt.events import Event
    from ropt.plan import ContextUpdate, Plan


class DefaultTableWith(BaseModel):
    """Parameters for the table context object.

    Attributes:
        columns:        The columns to produce
        path:           The path of the output file
        table_type:     The type of table to produce
        min_header_len: The minimal number of header lines
        steps:          List of steps to respond to
        maximize:       Convert results to maximization
    """

    columns: Dict[str, str]
    path: Path
    table_type: Literal["functions", "gradients"] = "functions"
    min_header_len: Optional[int] = None
    steps: List[str] = []
    maximize: bool = False

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultTableContext(ContextObj):
    """The default table context object."""

    def __init__(self, config: ContextConfig, plan: Plan) -> None:
        """Initialize a default tracker context object.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)
        self._with = DefaultTableWith.model_validate(config.with_)
        self._table = ResultsTable(
            self._with.columns,
            self._with.path,
            table_type=self._with.table_type,
            min_header_len=self._with.min_header_len,
        )
        for event_type in (
            EventType.FINISHED_EVALUATION,
            EventType.FINISHED_EVALUATOR_STEP,
        ):
            self.plan.optimizer_context.events.add_observer(
                event_type, self._handle_results
            )

    def _handle_results(self, event: Event) -> None:
        if not self._with.steps or event.step_name in self._with.steps:
            assert event.results is not None
            results = (
                tuple(convert_to_maximize(result) for result in event.results)
                if self._with.maximize
                else event.results
            )
            self._table.add_results(event.config, results)

    def update(self, _: ContextUpdate) -> None:
        """Update the tracker object, does nothing."""
