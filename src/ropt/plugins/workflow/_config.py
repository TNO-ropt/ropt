"""This module defines default configuration context object."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict

from ropt.config.enopt import EnOptConfig
from ropt.exceptions import WorkflowError
from ropt.plugins.workflow.base import ContextObj

if TYPE_CHECKING:
    from ropt.config.workflow import ContextConfig
    from ropt.workflow import Workflow


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
        enopt_config = EnOptConfig.model_validate(config.with_)
        self._backup = enopt_config
        self.set_variable(enopt_config)

    def update(self, updates: Dict[str, Any]) -> None:
        if not isinstance(updates, dict):
            msg = "attempt to update with invalid data."
            raise WorkflowError(msg, context_id=self.context_config.id)
        enopt_config = self.get_variable()
        assert enopt_config.original_inputs is not None
        enopt_config = EnOptConfig.model_validate(
            _update_dict(enopt_config.original_inputs, updates)
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
