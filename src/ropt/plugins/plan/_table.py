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
    """The default table results handler object.

    The table results handler generates a table that summarizes the results it
    receives. This is implemented through a
    [`ResultsTable`][ropt.report.ResultsTable] object; refer to its
    documentation for detailed information on table generation and formatting.

    This table handler uses the [`DefaultTableHandlerWith`]
    [ropt.plugins.plan._table.DefaultTableHandler.DefaultTableHandlerWith]
    configuration class to parse the `with` field in the
    [`ResultHandler`][ropt.config.plan.ResultHandlerConfig] used to specify this
    handler in a plan configuration.
    """

    class DefaultTableHandlerWith(BaseModel):
        """Parameters for the table results handler.

        The `columns`, `path`, `table_type`, and `min_header_len` fields
        correspond to the parameters of the
        [`ResultsTable`][ropt.report.ResultsTable] constructor. Refer to its
        documentation for more detailed information on these arguments.

        The `maximize` field allows adaptation for maximization problems.
        Typically, results are generated for minimization problems, but setting
        `maximize` to `True` will invert this interpretation for scenarios where
        the negative of an objective function is minimized.

        The `tags` field enables optional labels to be attached to results,
        which assists in filtering relevant results for the table.

        Attributes:
            tags:           Tags of the sources to track.
            columns:        A mapping of column names for the results table.
            path:           The file path where the results table will be saved.
            table_type:     The format or type of table to generate.
            min_header_len: The minimum number of header lines to include.
            maximize:       If `True`, interprets results as a maximization
                            problem rather than the default minimization.
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
            config: The configuration of the step.
            plan:   The plan that runs this step.
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
            event: The event to handle.

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
