"""This module defines the default callback context object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

from pydantic import BaseModel, ConfigDict

from ropt.plugins.workflow.base import ContextObj
from ropt.results import Results  # noqa: TCH001

if TYPE_CHECKING:
    from ropt.config.workflow import ContextConfig
    from ropt.workflow import Workflow


class DefaultCallbackWith(BaseModel):
    """Parameters for the `callback` context object.

    Attributes:
        function: The function to call
        kwargs:   A dictionary of keyword arguments for the call
    """

    function: Callable[[Tuple[Results, ...]], Any]
    kwargs: Dict[str, Any] = {}

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultCallbackContext(ContextObj):
    """The default `callback` context object.

    The `callback` context object is used to call a user-defined function
    whenever it is updated with new results. For instance, an optimizer step may
    add a `callback` context object to its `update` parameter, so that it will
    be updated any time new results are produced.

    When updated, the `callback` object calls the function defined by its
    `function` parameter with the new result as it first argument. Optionally,
    additional keyword arguments can be defined via the `kwargs` parameter,
    which are added to each function call.

    The parameters of the context object are parsed using the
    [`DefaultCallbackWith`][ropt.plugins.workflow.DefaultCallbackWith] object.
    """

    def __init__(self, config: ContextConfig, workflow: Workflow) -> None:
        super().__init__(config, workflow)
        with_ = DefaultCallbackWith.model_validate(config.with_)
        self._callback = with_.function
        self._kwargs = with_.kwargs
        self.set_variable(None)

    def update(self, value: Any) -> None:  # noqa: ANN401
        kwargs = {
            key: self.workflow.parse_value(value) for key, value in self._kwargs.items()
        }
        self.set_variable(self._callback(value, **kwargs))
