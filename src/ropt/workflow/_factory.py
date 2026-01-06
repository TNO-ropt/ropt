"""The plugin manager."""

from __future__ import annotations

from typing import Any

from ropt.plugins import plugin_manager
from ropt.plugins.compute_step.base import ComputeStep
from ropt.plugins.evaluator.base import Evaluator
from ropt.plugins.event_handler.base import EventHandler


def create_evaluator(method: str, **kwargs: Any) -> Evaluator:  # noqa: ANN401
    """Create a new evaluator.

    Args:
        method: The method string to find the evaluator.
        kwargs: Optional keyword arguments passed to the evaluator init.

    Returns:
        The new evaluator.
    """
    evaluator = plugin_manager.get_plugin("evaluator", method=method).create(
        method, **kwargs
    )
    assert isinstance(evaluator, Evaluator)
    return evaluator


def create_event_handler(method: str, **kwargs: Any) -> EventHandler:  # noqa: ANN401
    """Create a new event handler.

    Args:
        method: The method string to find the handler.
        kwargs: Optional keyword arguments passed to the handler init.

    Returns:
        The new event handler.
    """
    handler = plugin_manager.get_plugin("event_handler", method=method).create(
        method, **kwargs
    )
    assert isinstance(handler, EventHandler)
    return handler


def create_compute_step(method: str, **kwargs: Any) -> ComputeStep:  # noqa: ANN401
    """Create a new compute step.

    Args:
        method: The method string to find the compute step.
        kwargs: Optional keyword arguments passed to the compute step init.

    Returns:
        The new compute step.
    """
    compute_step = plugin_manager.get_plugin("compute_step", method=method).create(
        method, **kwargs
    )
    assert isinstance(compute_step, ComputeStep)
    return compute_step
