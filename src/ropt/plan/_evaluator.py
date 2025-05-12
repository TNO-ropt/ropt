"""This module defines the optimization plan object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ropt.plugins.plan.base import Evaluator

if TYPE_CHECKING:
    from ropt.plugins import PluginManager


def create_evaluator(
    name: str,
    plugin_manager: PluginManager,
    **kwargs: Any,  # noqa: ANN401
) -> Evaluator:
    """Create an evaluator object.

    Creates an evaluator of a type that is determined by the provided `name`,
    which the plugin system uses to locate the corresponding evaluator class.
    Any additional keyword arguments are passed to the evaluators's constructor.

    Args:
        name:           The name of the evaluator to add.
        plugin_manager: Plugin manager.
        kwargs:         Additional arguments for the evaluators's constructor.

    Returns:
        The new evaluator object.
    """
    evaluator = plugin_manager.get_plugin("evaluator", method=name).create(
        name, **kwargs
    )
    assert isinstance(evaluator, Evaluator)
    return evaluator
