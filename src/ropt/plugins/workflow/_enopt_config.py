"""This module defines the abstract base class for optimization steps."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict

from ropt.config.enopt import EnOptConfig
from ropt.exceptions import WorkflowError
from ropt.plugins.workflow.base import ContextObj

if TYPE_CHECKING:
    from ropt.config.workflow import ContextConfig
    from ropt.workflow import Workflow


class DefaultEnOptConfigContext(ContextObj):
    """The default configuration context object."""

    def __init__(self, config: ContextConfig, workflow: Workflow) -> None:
        """Initialize a default enopt_config context object.

        Args:
            config:   The configuration of the step
            workflow: The workflow
        """
        super().__init__(config, workflow)
        self._enopt_config = EnOptConfig.model_validate(config.with_)
        self._backup = self._enopt_config

    def update(self, updates: Dict[str, Any]) -> None:
        """Update the enopt_config object.

        Updates the stored configuration. Fields that are present in `updates`
        are overwritten, others are kept intact.

        Args:
            updates: The updates to apply.
        """
        if not isinstance(updates, dict):
            msg = "attempt to update with invalid data."
            raise WorkflowError(msg, context_id=self.context_config.id)
        assert self._enopt_config.original_inputs is not None
        self._enopt_config = EnOptConfig.model_validate(
            _update_dict(self._enopt_config.original_inputs, updates)
        )

    def value(self) -> EnOptConfig:
        """Return the optimal or last results object."""
        return self._enopt_config

    def reset(self) -> None:
        """Clear the stored values."""
        self._enopt_config = self._backup


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
