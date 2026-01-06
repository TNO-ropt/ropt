"""This module provides the default plugin implementations evaluators.

**Supported Components:**

  - `function_evaluator`: Evaluator that forwards calculations to a given evaluation function.
    ([`DefaultFunctionEvaluator`][ropt.plugins.evaluator._function_evaluator.DefaultFunctionEvaluator])
  - `caching_evaluator`: Evaluator that uses caching to find results that were
    already evaluated before forwarding to another evaluator.
    ([`DefaultCachedEvaluator`][ropt.plugins.evaluator.cached_evaluator.DefaultCachedEvaluator])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from ._function_evaluator import DefaultFunctionEvaluator
from .base import EvaluatorPlugin
from .cached_evaluator import DefaultCachedEvaluator

if TYPE_CHECKING:
    from .base import Evaluator


_EVALUATOR_OBJECTS: Final[dict[str, type[Evaluator]]] = {
    "function_evaluator": DefaultFunctionEvaluator,
    "cached_evaluator": DefaultCachedEvaluator,
}


class DefaultEvaluatorPlugin(EvaluatorPlugin):
    """The default plugin for creating evaluators.

    This plugin acts as a factory for the standard evaluator implementations
    provided by `ropt`.

    **Supported Evaluators:**

    - `function_evaluator`: Creates a
        [`DefaultFunctionEvaluator`][ropt.plugins.evaluator._function_evaluator.DefaultFunctionEvaluator]
        instance, which uses function calls to calculated individual objectives
        and constraints.
    """

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> Evaluator:
        """Create an evaluator.

        # noqa
        """  # noqa: DOC201, DOC501
        _, _, name = name.lower().rpartition("/")
        evaluator_obj = _EVALUATOR_OBJECTS.get(name)
        if evaluator_obj is not None:
            return evaluator_obj(**kwargs)

        msg = f"Unknown evaluator type: {name}"
        raise TypeError(msg)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in _EVALUATOR_OBJECTS
