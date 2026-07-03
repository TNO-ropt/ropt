"""Tests for ropt logging output."""

# mypy: disable-error-code="arg-type"

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ropt.workflow import BasicOptimizer
from ropt.workflow.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    FunctionEvaluator,
)

if TYPE_CHECKING:
    import pytest
    from numpy.typing import NDArray


def _objective(
    variables: NDArray[np.float64],
    _context: EvaluationFunctionContext,
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(objectives=np.array([float(np.sum(variables**2))]))


_CONFIG: dict[str, Any] = {
    "optimizer": {"max_batches": 3},
    "variables": {"variable_count": 2, "perturbation_magnitudes": 0.01},
    "objectives": {"weights": [1.0]},
}

_INITIAL = np.ones(2)


def test_null_handler_installed() -> None:
    assert any(
        isinstance(h, logging.NullHandler) for h in logging.getLogger("ropt").handlers
    )


def test_optimization_start_message(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="ropt"):
        BasicOptimizer(_CONFIG, FunctionEvaluator(function=_objective)).run(_INITIAL)
    assert any("Starting optimization" in r.message for r in caplog.records)


def test_optimization_finish_message(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="ropt"):
        BasicOptimizer(_CONFIG, FunctionEvaluator(function=_objective)).run(_INITIAL)
    assert any("Optimization finished" in r.message for r in caplog.records)


def test_function_batch_statistics(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="ropt"):
        BasicOptimizer(_CONFIG, FunctionEvaluator(function=_objective)).run(_INITIAL)
    assert any(
        "Function evaluation:" in r.message and "realizations succeeded" in r.message
        for r in caplog.records
    )


def test_gradient_batch_statistics(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="ropt"):
        BasicOptimizer(_CONFIG, FunctionEvaluator(function=_objective)).run(_INITIAL)
    assert any(
        "Gradient evaluation:" in r.message and "realizations succeeded" in r.message
        for r in caplog.records
    )


def test_stopping_max_batches_message(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="ropt"):
        BasicOptimizer(_CONFIG, FunctionEvaluator(function=_objective)).run(_INITIAL)
    assert any(
        "Stopping:" in r.message and "evaluation batches" in r.message
        for r in caplog.records
    )


def test_debug_optimizer_callback_messages(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="ropt"):
        BasicOptimizer(_CONFIG, FunctionEvaluator(function=_objective)).run(_INITIAL)
    assert any("Optimizer callback:" in r.message for r in caplog.records)


def test_plugin_registration_messages(caplog: pytest.LogCaptureFixture) -> None:
    from ropt.plugins.manager import PluginManager  # noqa: PLC0415

    with caplog.at_level(logging.DEBUG, logger="ropt.plugins"):
        PluginManager()
    assert any("Registering plugin:" in r.message for r in caplog.records)


def test_logger_names_identify_source(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="ropt"):
        BasicOptimizer(_CONFIG, FunctionEvaluator(function=_objective)).run(_INITIAL)
    names = {r.name for r in caplog.records}
    assert "ropt.workflow.compute_steps" in names
    assert "ropt.core" in names


def test_no_ropt_messages_below_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="ropt"):
        BasicOptimizer(_CONFIG, FunctionEvaluator(function=_objective)).run(_INITIAL)
    assert not any(r.name.startswith("ropt") for r in caplog.records)
