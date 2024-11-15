"""This module implements the default print step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ropt.plugins.plan.base import PlanStep

if TYPE_CHECKING:
    from ropt.config.plan import PlanStepConfig
    from ropt.plan import Plan


class DefaultPrintStep(PlanStep):
    """The default print step.

    The print step is used to display messages on the console, using Python's
    `print` function. The message to be printed must be a string evaluated by
    the [`eval`][ropt.plan.Plan.eval] method of the executing
    [`Plan`][ropt.plan.Plan] object. The message should be enclosed in `[[...]]`
    delimiters; if these delimiters are missing, they are implicitly added
    around the message. This format allows for optional interpolation of
    embedded expressions, delimited by `${ ... $}`.

    The print step uses the [`DefaultPrintStepWith`]
    [ropt.plugins.plan._print.DefaultPrintStep.DefaultPrintStepWith]
    configuration class to parse the `with` field of the
    [`PlanStepConfig`][ropt.config.plan.PlanStepConfig] used to specify this step
    in a plan configuration.

    Note: Shorthand configuration
        With this step it is possible to configure the message in shorthand notation.
        Instead of

        ```python
        {
            "run": "print",
            "with": {
                "message": "Hello World!",
            }
        }
        ```

        or

        ```python
        {
            "print": {
                "message": "Hello World!",
            }
        }
        ```

        You can use:

        ```python
        {
            "print":  "Hello World!",
        }
        ```
    """

    class DefaultPrintStepWith(BaseModel):
        """Parameters used by the print step.

        Attributes:
            message: The message to print.
        """

        message: str

        model_config = ConfigDict(
            extra="forbid",
            validate_default=True,
            arbitrary_types_allowed=True,
            frozen=True,
        )

    def __init__(self, config: PlanStepConfig, plan: Plan) -> None:
        """Initialize a default print step.

        Args:
            config: The configuration of the step.
            plan:   The plan that runs this step.
        """
        super().__init__(config, plan)
        if isinstance(config.with_, str):
            self._message = config.with_.strip()
        else:
            _with = self.DefaultPrintStepWith.model_validate(config.with_)
            self._message = _with.message.strip()

    def run(self) -> None:
        """Run the print step."""
        print(self.plan.eval(self._message))  # noqa: T201
