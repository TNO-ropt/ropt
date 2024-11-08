"""This module implements the default save handler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel, ConfigDict

from ropt.config.validated_types import ItemOrSet  # noqa: TCH001
from ropt.enums import EventType
from ropt.plugins.plan.base import ResultHandler
from ropt.utils.misc import format_tuple

if TYPE_CHECKING:
    from ropt.config.plan import ResultHandlerConfig
    from ropt.plan import Event, Plan


class _Formatter:
    def __init__(self, value: Sequence[Any], delimiters: str) -> None:
        self._value = value
        self._delimiters = delimiters

    def __format__(self, spec: str) -> str:
        return format_tuple(
            tuple(f"{item:{spec}}" for item in self._value), delimiters=self._delimiters
        )


class DefaultSaveHandler(ResultHandler):
    """The default save handler object.

    This handler tracks the [`Results`][ropt.results.Results] objects that it
    receives saves them. It uses the [`DefaultSaveHandlerWith`]
    [ropt.plugins.plan._save.DefaultSaveHandler.DefaultSaveHandlerWith]
    configuration class to parse the `with` field of the
    [`ResultHandler`][ropt.config.plan.ResultHandlerConfig] used to specify this
    handler in a plan configuration.
    """

    class DefaultSaveHandlerWith(BaseModel):
        """Parameters for the save handler.

        The `path` field specifies the output path, including the file name. It
        may contain one or more replacement strings for the following variables,
        which will be substituted with the corresponding value of the result
        objects:

        - `plan_id`:   Replace with a formatted `plan_id` field of the result object.
        - `result_id`: Replace with the `result_id` field of the result object.
        - `batch_id`:  Replace with the `batch_id` field of the result object.

        For example the following string will construct file names using the
        result ID padded with zeros: "output_directory/result{result_id:03d}.nc".

        Since the `plan_id` field of a results objects is a tuple, it needs to
        formatted before substitution. This is done by joining them using
        delimiters from the `delimiters` attribute. The `delimiters` argument
        can contain multiple characters, each of which will be used in turn to
        separate the items in `name`. If fewer delimiters are provided than
        needed, the final delimiter is reused for any remaining items. If
        `delimiters` is not specified, `-` is used by default.

        The `tags` field allows optional labels to be attached to each result,
        assisting result handlers in filtering relevant results.

        Currently the output is stored in netCDf format. A `.nc` extension is added
        to the file name if not present already.

        Attributes:
            path:       The path to the output file names.
            tags:       Tags to filter the sources to track.
            delimiters: The delimiters to use to format plan_id values, applied sequentially.
        """

        path: str
        tags: ItemOrSet[str]
        delimiters: str = "-"

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            frozen=True,
        )

    def __init__(self, config: ResultHandlerConfig, plan: Plan) -> None:
        """Initialize a default save handler object.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        self._with = self.DefaultSaveHandlerWith.model_validate(config.with_)

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
                path = Path(
                    self.plan.eval(
                        self._with.path.format(
                            plan_id=_Formatter(results.plan_id, self._with.delimiters),
                            result_id=results.result_id,
                            batch_id=results.batch_id,
                        )
                    )
                )
                if path.parent.exists() and not path.parent.is_dir():
                    msg = f"Not a directory to store results: {path.parent}"
                    raise RuntimeError(msg)
                if not path.parent.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix != ".nc":
                    path = path.with_suffix(".nc")
                results.to_netcdf(event.config, path)
        return event
