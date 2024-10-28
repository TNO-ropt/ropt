"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

import sys
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict

from ropt.config.validated_types import ItemOrSet  # noqa: TCH001
from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler
from ropt.report import ResultsTable
from ropt.results import convert_to_maximize

if sys.version_info >= (3, 11):
    pass
else:
    pass


if TYPE_CHECKING:
    from ropt.config.plan import ResultHandlerConfig
    from ropt.plan import Event, Plan


class DefaultTableHandler(ResultHandler):
    """The default tracker results handler object."""

    class DefaultTableHandlerWith(BaseModel):
        """Parameters for the table results handler.

        This results handler generates a table summarizing the results of an
        optimization. This is implemented via a
        [`ResultsTable`][ropt.report.ResultsTable] object. Refer to its
        documentation for more details.

        Attributes:
            tags:           Tags of the sources to track
            columns:        A mapping of column names for the results table
            path:           The location where the results file will be saved
            table_type:     The type of table to generate
            min_header_len: The minimum number of header lines to generate
            maximize:       If `True`, interpret the results as a maximization
                            problem rather than the default minimization
        """

        tags: ItemOrSet[str]
        columns: Dict[str, str]
        path: Path
        table_type: Literal["functions", "gradients"] = "functions"
        min_header_len: Optional[int] = None
        maximize: bool = False

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
        )

    def __init__(self, config: ResultHandlerConfig, plan: Plan) -> None:
        """Initialize a default tracker results handler object.

        Args:
            config: The configuration of the step
            plan:   The plan that runs this step
        """
        super().__init__(config, plan)
        self._with = self.DefaultTableHandlerWith.model_validate(config.with_)
        self._table = ResultsTable(
            self._with.columns,
            self._with.path,
            table_type=self._with.table_type,
            min_header_len=self._with.min_header_len,
        )

    def handle_event(self, event: Event) -> Event:
        """Handle an event.

        Args:
            event: The event to handle

        Returns:
            The (possibly modified) event.
        """
        if (
            event.event_type
            in {
                EventType.FINISHED_EVALUATION,
                EventType.FINISHED_EVALUATOR_STEP,
            }
            and event.results is not None
            and (event.tags & self._with.tags)
        ):
            self._table.add_results(
                event.config,
                (
                    (convert_to_maximize(item) for item in event.results)
                    if self._with.maximize
                    else event.results
                ),
            )
        return event
