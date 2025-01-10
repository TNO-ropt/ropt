"""This module implements the save step."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import PlanStepConfig
    from ropt.plan import Plan


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DefaultSaveStep(PlanStep):
    """The default save step.

    The save step saves data to a file using the specified format.

    This step uses the
    [`DefaultSaveStepWith`][ropt.plugins.plan._save.DefaultSaveStep.DefaultSaveStepWith]
    configuration class to parse the `with` field of the
    [`PlanStepConfig`][ropt.config.plan.PlanStepConfig]. The configuration
    specifies the settings for this step in a plan setup.
    """

    class DefaultSaveStepWith(BaseModel):
        """Configuration parameters for the save step.

        This configuration defines what data is saved by and specifies the file
        path for storing the output. If the path includes directories that do
        not yet exist, they will be created.

        The `format` option defines the file format for saving the data.
        Supported formats are:

        - `json`:   Saves the data in JSON format.
        - `pickle`: Saves the data in a pickle file.

        The `format` option is optional, if `None` or not provided (the
        default), the step will attempt to derive the data format from the
        extension of the output path.

        Attributes:
            data:   The data to save.
            path:   The file path where the output will be stored.
            format: The format used for file storage, determining the serialization method.
        """

        data: Any
        path: str | Path
        format: Literal["json", "pickle"] | None = None

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a default save step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        self._with = self.DefaultSaveStepWith.model_validate(config.with_)
        if self._with.format is not None:
            _check_format(self._with.format)

    def run(self) -> None:
        """Run the save step."""
        path = Path(self.plan.eval(self._with.path))
        if path.parent.exists() and not path.parent.is_dir():
            msg = f"Not a directory to store results: {path.parent}"
            raise RuntimeError(msg)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        data = self.plan.eval(self._with.data)
        file_format = path.suffix if self._with.format is None else self._with.format
        match _check_format(file_format):
            case "json":
                with path.open("w", encoding="utf-8") as file_obj:
                    json.dump(data, file_obj, cls=NumpyEncoder)
            case "pickle":
                with path.open("wb") as file_obj:
                    pickle.dump(data, file_obj)


def _check_format(file_format: str | None) -> str:
    if file_format is None or not file_format:
        msg = "No data format specified"
        raise ValueError(msg)
    file_format = file_format.removeprefix(".")
    if file_format not in {"json", "pickle"}:
        msg = f"data format not supported: {file_format}"
        raise ValueError(msg)
    return file_format
