"""This module provides the default plugin implementations for compute steps.

**Supported Components:**
  - `ensemble_evaluator`: Performs ensemble evaluations
      ([`DefaultEnsembleEvaluatorComputeStep`][ropt.plugins.compute_step.ensemble_evaluator.DefaultEnsembleEvaluatorComputeStep]).
  - `optimizer`: Runs an optimization algorithm using a configured optimizer
      plugin
      ([`DefaultOptimizerComputeStep`][ropt.plugins.compute_step.optimizer.DefaultOptimizerComputeStep]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from .base import ComputeStepPlugin
from .ensemble_evaluator import DefaultEnsembleEvaluatorComputeStep
from .optimizer import DefaultOptimizerComputeStep

if TYPE_CHECKING:
    from ropt.plugins.compute_step.base import ComputeStep

_ACTION_OBJECTS: Final[dict[str, type[ComputeStep]]] = {
    "ensemble_evaluator": DefaultEnsembleEvaluatorComputeStep,
    "optimizer": DefaultOptimizerComputeStep,
}


class DefaultComputeStepPlugin(ComputeStepPlugin):
    """The default plugin for creating compute_steps.

    This plugin acts as a factory for the standard `ComputeStep` implementations
    provided by `ropt`.

    **Supported Compute Steps:**

    - `ensemble_evaluator`: Creates a
        [`DefaultEnsembleEvaluatorComputeStep`][ropt.plugins.compute_step.ensemble_evaluator.DefaultEnsembleEvaluatorComputeStep]
        instance, which performs ensemble evaluations.
    - `optimizer`: Creates a
        [`DefaultOptimizerComputeStep`][ropt.plugins.compute_step.optimizer.DefaultOptimizerComputeStep]
        instance, which runs an optimization algorithm using a configured
        optimizer plugin.
    """

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> ComputeStep:
        """Create a compute step.

        # noqa
        """
        _, _, name = name.lower().rpartition("/")
        op_obj = _ACTION_OBJECTS.get(name)
        if op_obj is not None:
            return op_obj(**kwargs)

        msg = f"Unknown compute step type: {name}"
        raise TypeError(msg)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in _ACTION_OBJECTS
