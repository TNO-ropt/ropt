"""This module provides the default plugin implementations for operations.

**Supported Components:**
  - `ensemble_evaluator`: Performs ensemble evaluations
      ([`DefaultEnsembleEvaluatorOperation`][ropt.plugins.operation.ensemble_evaluator.DefaultEnsembleEvaluatorOperation]).
  - `optimizer`: Runs an optimization algorithm using a configured optimizer
      plugin
      ([`DefaultOptimizerOperation`][ropt.plugins.operation.optimizer.DefaultOptimizerOperation]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from .base import OperationPlugin
from .ensemble_evaluator import DefaultEnsembleEvaluatorOperation
from .optimizer import DefaultOptimizerOperation

if TYPE_CHECKING:
    from ropt.plugins.operation.base import Operation

_ACTION_OBJECTS: Final[dict[str, type[Operation]]] = {
    "ensemble_evaluator": DefaultEnsembleEvaluatorOperation,
    "optimizer": DefaultOptimizerOperation,
}


class DefaultOperationPlugin(OperationPlugin):
    """The default plugin for creating workflow operations.

    This plugin acts as a factory for the standard `Operation` implementations
    provided by `ropt`.

    **Supported Operations:**

    - `ensemble_evaluator`: Creates a
        [`DefaultEnsembleEvaluatorOperation`][ropt.plugins.operation.ensemble_evaluator.DefaultEnsembleEvaluatorOperation]
        instance, which performs ensemble evaluations.
    - `optimizer`: Creates a
        [`DefaultOptimizerOperation`][ropt.plugins.operation.optimizer.DefaultOptimizerOperation]
        instance, which runs an optimization algorithm using a configured
        optimizer plugin.
    """

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> Operation:
        """Create an operation.

        # noqa
        """
        _, _, name = name.lower().rpartition("/")
        op_obj = _ACTION_OBJECTS.get(name)
        if op_obj is not None:
            return op_obj(**kwargs)

        msg = f"Unknown operation type: {name}"
        raise TypeError(msg)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in _ACTION_OBJECTS
