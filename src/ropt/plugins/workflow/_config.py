"""This module defines default configuration context object."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Union

from pydantic import BaseModel, ConfigDict

from ropt.config.enopt import EnOptConfig
from ropt.plugins.workflow.base import ContextObj
from ropt.workflow import ContextUpdate, ContextUpdateDict

if TYPE_CHECKING:
    from ropt.config.workflow import ContextConfig
    from ropt.workflow import Workflow


class DefaultConfigWith(BaseModel):
    """Parameters for the `config` context object.

    Attributes:
        config: The configuration
    """

    config: Union[EnOptConfig, Dict[str, Any]]

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class DefaultConfigContext(ContextObj):
    """The default `config` context object.

    The `config` object is used store and modify configuration objects for
    optimizer and evaluation steps. When initializing the object, the `with`
    field is parsed as an [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object.
    Optimization and evaluation steps can refer to a workflow variable of the
    same name as the `config` ID to retrieve the parsed configuration object.

    The `config` object can be updated with a dictionary of values to construct
    a new `EnOptConfig` object with the updated values. The original value is
    kept with the `config` object and can be recovered by resetting the `config`
    object.
    """

    def __init__(self, config: ContextConfig, workflow: Workflow) -> None:
        super().__init__(config, workflow)
        if "config" in config.with_:
            with_ = DefaultConfigWith.model_validate(config.with_)
            enopt_config = EnOptConfig.model_validate(with_.config)
        else:
            enopt_config = EnOptConfig.model_validate(config.with_)

        self._backup = enopt_config
        self.set_variable(enopt_config)

    def update(self, updates: ContextUpdate) -> None:
        if isinstance(updates, ContextUpdateDict):
            enopt_config = self.get_variable()
            assert enopt_config.original_inputs is not None
            enopt_config = EnOptConfig.model_validate(
                _update_dict(enopt_config.original_inputs, updates.data)
            )
            self.set_variable(enopt_config)

    def reset(self) -> None:
        """Clear the stored values."""
        self.set_variable(self._backup)


def _update_dict(mapping: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    mapping = mapping.copy()
    for key, value in values.items():
        if (
            key in mapping
            and isinstance(mapping[key], dict)
            and isinstance(value, dict)
        ):
            mapping[key] = _update_dict(mapping[key], value)
        else:
            mapping[key] = deepcopy(value)
    return mapping
