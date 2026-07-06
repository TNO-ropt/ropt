"""Tests for TableHandler: column population, ordering, and renaming."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import (
    FunctionEvaluations,
    FunctionResults,
    Functions,
    Realizations,
)

pytest.importorskip("pandas")

from ropt.context import EnOptContext
from ropt.workflow.event_handlers import TableHandler

_CONFIG: dict[str, Any] = {
    "variables": {"variable_count": 2},
    "objectives": {"weights": [1.0]},
    "realizations": {"weights": [1.0]},
}

_CONFIG_TWO_REALIZATIONS: dict[str, Any] = {
    "variables": {"variable_count": 2},
    "objectives": {"weights": [1.0]},
    "realizations": {"weights": [1.0, 1.0]},
}


def _make_result(batch_id: int, objective: float = 1.0) -> FunctionResults:
    evaluations = FunctionEvaluations.create(
        variables=np.array([0.5, 1.5]),
        objectives=np.array([[objective]]),
    )
    functions = Functions.create(
        target_objective=np.array(objective),
        objectives=np.array([objective]),
    )
    context = EnOptContext.model_validate(_CONFIG)
    return FunctionResults(
        batch_id=batch_id,
        metadata={},
        names=context.names,
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(1, dtype=np.bool_),
            objective_weights=np.ones((1, 1)),
        ),
        functions=functions,
    )


def _make_event(batch_id: int, objective: float = 1.0) -> EnOptEvent:
    context = EnOptContext.model_validate(_CONFIG)
    return EnOptEvent(
        event_type=EnOptEventType.FINISHED_EVALUATION,
        context=context,
        results=(_make_result(batch_id, objective),),
    )


def _make_result_two_realizations(batch_id: int) -> FunctionResults:
    evaluations = FunctionEvaluations.create(
        variables=np.array([0.5, 1.5]),
        objectives=np.array([[1.0], [2.0]]),
    )
    functions = Functions.create(
        target_objective=np.array(1.5),
        objectives=np.array([1.0, 2.0]),
    )
    context = EnOptContext.model_validate(_CONFIG_TWO_REALIZATIONS)
    return FunctionResults(
        batch_id=batch_id,
        metadata={},
        names=context.names,
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(2, dtype=np.bool_),
            objective_weights=np.ones((1, 2)),
        ),
        functions=functions,
    )


def _make_event_two_realizations(batch_id: int) -> EnOptEvent:
    context = EnOptContext.model_validate(_CONFIG_TWO_REALIZATIONS)
    return EnOptEvent(
        event_type=EnOptEventType.FINISHED_EVALUATION,
        context=context,
        results=(_make_result_two_realizations(batch_id),),
    )


def test_table_handler_populates_table_from_events() -> None:
    handler = TableHandler()
    handler.add_table("t", "functions", {"functions.target_objective": "Obj"})
    handler.handle_event(_make_event(1))
    df = handler["t"]
    assert not df.empty
    assert "Obj" in df.columns


def test_table_handler_returns_empty_dataframe_before_any_event() -> None:
    handler = TableHandler()
    handler.add_table("t", "functions", {"functions.target_objective": "Obj"})
    assert handler["t"].empty


def test_table_handler_accumulates_multiple_events() -> None:
    handler = TableHandler()
    handler.add_table("t", "functions", {"functions.target_objective": "Obj"})
    handler.handle_event(_make_event(1, objective=1.0))
    handler.handle_event(_make_event(2, objective=2.0))
    df = handler["t"]
    assert len(df) == 2
    assert list(df["Obj"]) == [1.0, 2.0]


def test_table_handler_uses_display_title_not_field_name() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {"functions.target_objective": "Total Objective"},
    )
    handler.handle_event(_make_event(1))
    df = handler["t"]
    assert "Total Objective" in df.columns
    assert "functions.target_objective" not in df.columns


def test_table_handler_renames_batch_id_column() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {
            "batch_id": "Batch",
            "functions.target_objective": "Obj",
        },
    )
    handler.handle_event(_make_event(5))
    df = handler["t"]
    assert "Batch" in df.columns
    assert "batch_id" not in df.columns


def test_table_handler_batch_id_value() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {
            "batch_id": "Batch",
            "functions.target_objective": "Obj",
        },
    )
    handler.handle_event(_make_event(batch_id=7))
    df = handler["t"]
    assert int(df["Batch"].iloc[0]) == 7


def test_table_handler_column_order_objective_before_batch() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {
            "functions.target_objective": "Obj",
            "batch_id": "Batch",
        },
    )
    handler.handle_event(_make_event(1))
    cols = list(handler["t"].columns)
    assert cols.index("Obj") < cols.index("Batch")


def test_table_handler_column_order_batch_before_objective() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {
            "batch_id": "Batch",
            "functions.target_objective": "Obj",
        },
    )
    handler.handle_event(_make_event(1))
    cols = list(handler["t"].columns)
    assert cols.index("Batch") < cols.index("Obj")


def test_table_handler_omits_batch_id_when_not_in_columns() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {"functions.target_objective": "Obj"},
    )
    handler.handle_event(_make_event(1))
    df = handler["t"]
    assert "batch_id" not in df.columns
    assert "Obj" in df.columns


def test_table_handler_includes_realization_when_requested() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {
            "realization": "Realization",
            "evaluations.objectives": "Obj",
        },
    )
    handler.handle_event(_make_event_two_realizations(1))
    df = handler["t"]
    assert "Realization" in df.columns


def test_table_handler_renames_realization_column() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {
            "realization": "Run",
            "evaluations.objectives": "Obj",
        },
    )
    handler.handle_event(_make_event_two_realizations(1))
    df = handler["t"]
    assert "Run" in df.columns
    assert "realization" not in df.columns


def test_table_handler_omits_realization_when_not_in_columns() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {"evaluations.objectives": "Obj"},
    )
    handler.handle_event(_make_event_two_realizations(1))
    df = handler["t"]
    assert "realization" not in df.columns
    assert "Obj,0" in df.columns


def test_table_handler_realization_column_order() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {
            "evaluations.objectives": "Obj",
            "realization": "Run",
        },
    )
    handler.handle_event(_make_event_two_realizations(1))
    cols = list(handler["t"].columns)
    assert cols.index("Obj,0") < cols.index("Run")


def test_table_handler_realization_column_order_first() -> None:
    handler = TableHandler()
    handler.add_table(
        "t",
        "functions",
        {
            "realization": "Run",
            "evaluations.objectives": "Obj",
        },
    )
    handler.handle_event(_make_event_two_realizations(1))
    cols = list(handler["t"].columns)
    assert cols.index("Run") < cols.index("Obj,0")
