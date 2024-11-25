"""This module implements the load step."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import PlanStepConfig
    from ropt.plan import Plan


class DefaultLoadStep(PlanStep):
    """The default load step.

    The load step loads data from a file and deserializes it into a variable.

    This step uses the
    [`DefaultLoadStep`][ropt.plugins.plan._load_data.DefaultLoadStep.DefaultLoadStepWith]
    configuration class to parse the `with` field of the
    [`PlanStepConfig`][ropt.config.plan.PlanStepConfig], which defines the
    settings for this step within a plan configuration.
    """

    class DefaultLoadStepWith(BaseModel):
        """Parameters used by the load step.

        This configuration specifies the name of the variable to store the
        result and the file path for the file to load.

        The `format` option specifies the file format. Currently, the following
        formats are supported:

        - `json`:   Load the data from a JSON file.
        - `pickle`: Load the data from a pickle file.

        The `format` option is optional, if `None` or not provided (the
        default), the step will attempt to derive the data format from the
        extension of the input path.

        Attributes:
            var:    Name of the variable to store the result.
            path:   File path to load.
            format: The format of the file.
        """

        var: str
        path: str | Path
        format: Literal["json", "pickle"] | None = None

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a default load step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        self._with = self.DefaultLoadStepWith.model_validate(config.with_)
        if self._with.var not in self.plan:
            msg = f"Plan variable does not exist: {self._with.var}"
            raise ValueError(msg)
        if self._with.format is not None:
            _check_format(self._with.format)

    def run(self) -> None:
        """Run the load step."""
        path = Path(self.plan.eval(self._with.path))
        if not path.exists():
            msg = f"The file does not exist: {path}"
            raise RuntimeError(msg)

        file_format = path.suffix if self._with.format is None else self._with.format
        match _check_format(file_format):
            case "json":
                with path.open("r", encoding="utf-8") as file_obj:
                    self.plan[self._with.var] = json.load(file_obj)
            case "pickle":
                with path.open("rb") as file_obj:
                    self.plan[self._with.var] = pickle.load(file_obj)  # noqa: S301


def _check_format(file_format: str | None) -> str:
    if file_format is None or not file_format:
        msg = "No data format specified"
        raise ValueError(msg)
    if file_format.startswith("."):
        file_format = file_format[1:]
    if file_format not in {"json", "pickle"}:
        msg = f"data format not supported: {file_format}"
        raise ValueError(msg)
    return file_format
