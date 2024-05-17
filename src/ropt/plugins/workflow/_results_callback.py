"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from ropt.plugins.workflow.base import ContextObj
from ropt.results import Results  # noqa: TCH001

if TYPE_CHECKING:
    from ropt.config.workflow import ContextConfig
    from ropt.workflow import Workflow


class DefaultResultsCallbackWith(BaseModel):
    """Parameters for the result_callback context object.

    Attributes:
        callback: The function to call
    """

    callback: Callable[[Tuple[Results, ...]], Any]
    kwargs: Dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultResultsCallbackContext(ContextObj):
    """The default result_callback context object."""

    def __init__(self, config: ContextConfig, workflow: Workflow) -> None:
        """Initialize a default result_callback context object.

        Args:
            config:   The configuration of the step
            workflow: The workflow
        """
        super().__init__(config, workflow)
        with_ = DefaultResultsCallbackWith.model_validate(config.with_)
        self._callback = with_.callback
        self._kwargs = with_.kwargs
        self._value: Optional[Any] = None

    def update(self, value: Any) -> None:  # noqa: ANN401
        """Update the result_callback object with new results.

        Args:
            value: The results to store.
            kwargs: Optional keyword arguments.
        """
        kwargs = {
            key: self.workflow.parse_value(value) for key, value in self._kwargs.items()
        }
        self._value = self._callback(value, **kwargs)

    def value(self) -> Any:  # noqa: ANN401
        """Return the value."""
        return self._value
