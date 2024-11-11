"""This module implements the functions provided by the default plan plugin."""

from __future__ import annotations

from typing import Any, Dict, Hashable, Sequence


class ExpressionFunctions:
    """This class provides functions for the expression evaluator.

    The methods in this class are returned by the `data` property of the
    [`DefaultPlanPlugin`][ropt.plugins.plan.default.DefaultPlanPlugin] class and
    added to the expression evaluator used during execution of plans.
    """

    @staticmethod
    def mkdict(keys: Sequence[Hashable], values: Sequence[Any]) -> Dict[Hashable, Any]:
        """Create a dictionary from lists of keys and values.

        Args:
            keys:   The keys of the dictionary.
            values: The values of the dictionary.

        Returns:
            A dictionary constructed from keys and values.
        """
        return dict(zip(keys, values))
