"""This module implements the default meta data handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ropt.config.validated_types import ItemOrSet  # noqa: TCH001
from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler

if TYPE_CHECKING:
    from ropt.config.plan import ResultHandlerConfig
    from ropt.plan import Event, Plan


class DefaultMetadataHandler(ResultHandler):
    """The default metadata results handler.

    This handler adds arbitrary metadata to results produced by steps by
    merging them into the `metadata` field of the
    [`Results`][ropt.results.Results] objects that it receives. It is configured
    using a dictionary that maps keys to the data to store. These data entries
    can be of any type; however, if they are strings, they will be evaluated
    using the [`eval`][ropt.plan.Plan.eval] method of the executing
    [`Plan`][ropt.plan.Plan] object. This evaluation occurs when the metadata
    handler processes an event, enabling dynamic data insertion based on the
    current content of plan variables.

    The metadata step uses the [`DefaultMetadataHandlerWith`]
    [ropt.plugins.plan._metadata.DefaultMetadataHandler.DefaultMetadataHandlerWith]
    configuration class to parse the `with` field of the
    [`ResultHandler`][ropt.config.plan.ResultHandlerConfig] used to specify this
    handler in a plan configuration.
    """

    class DefaultMetadataHandlerWith(BaseModel):
        """Parameters used by the default metadata results handler.

        The data to merge into the metadata of a result are required.

        The `tags` field allows optional labels to be attached to each result,
        which can assist result handlers in filtering relevant results.

        Attributes:
            data: Data to merge into the metadata of the results.
            tags: Optional tags specifying which result sources to modify.
        """

        data: dict[str, Any]
        tags: ItemOrSet[str]

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    def __init__(self, config: ResultHandlerConfig, plan: Plan) -> None:
        """Initialize a default metadata results handler.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)

        self._with = self.DefaultMetadataHandlerWith.model_validate(config.with_)

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
            for results in event.results:
                for key, expr in self._with.data.items():
                    results.metadata[key] = self._plan.eval(expr)
        return event
