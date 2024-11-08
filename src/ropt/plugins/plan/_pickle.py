"""This module implements the pickle step."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ropt.config.validated_types import ItemOrTuple  # noqa: TCH001
from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import PlanStepConfig
    from ropt.plan import Plan


class DefaultPickleStep(PlanStep):
    """The default pickle step.

    The pickle step saves specified plan variables to a pickle file by
    constructing a dictionary that maps each variable name to its value.
    This dictionary is then serialized and saved as a pickle file.

    The pickle step uses the
    [`DefaultPickleStepWith`][ropt.plugins.plan._pickle.DefaultPickleStep.DefaultPickleStepWith]
    configuration class to parse the `with` field of the
    [`PlanStepConfig`][ropt.config.plan.PlanStepConfig], which defines the
    settings for this step in a plan configuration.
    """

    class DefaultPickleStepWith(BaseModel):
        """Parameters used by the pickle step.

        This configuration specifies the variables to be saved by the pickle
        step and the file path for the saved pickle file. Directories in the
        path will be created if they do not already exist.

        Attributes:
            vars: List of variable names to save to the pickle file.
            path: File path where the pickle file will be stored.
        """

        vars: ItemOrTuple[str]
        path: str

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a default pickle step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        self._with = self.DefaultPickleStepWith.model_validate(config.with_)
        for name in self._with.vars:
            if name not in self.plan:
                msg = f"Plan variable does not exist: {name}"
                raise ValueError(msg)

    def run(self) -> None:
        """Run the pickle step."""
        path = Path(self.plan.eval(self._with.path))
        if path.parent.exists() and not path.parent.is_dir():
            msg = f"Not a directory to store results: {path.parent}"
            raise RuntimeError(msg)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file_obj:
            pickle.dump({name: self.plan[name] for name in self._with.vars}, file_obj)
