"""Tests for DataFrameHandler: column population, ordering, and renaming."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from ropt.components.event_handlers import DataFrameHandler
from ropt.context import EnOptContext
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import (
    FunctionEvaluations,
    FunctionResults,
    Functions,
    Realizations,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ropt.results._frame_support import DataFrameEngine

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


@pytest.fixture(name="engine", params=["pandas", "polars"])
def engine_fixture(request: pytest.FixtureRequest) -> DataFrameEngine:
    # The parameters are the module names, so each one skips on its own install.
    pytest.importorskip(request.param)
    return cast("DataFrameEngine", request.param)


def _is_empty(frame: Any) -> bool:
    return bool(frame.is_empty() if hasattr(frame, "is_empty") else frame.empty)


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
        evaluation_point=evaluations.variables,
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(1, dtype=np.bool_),
            objective_weights=np.ones((1, 1)),
        ),
        functions=functions,
    )


def _make_event(
    batch_id: int, objective: float = 1.0, output_dir: Path | None = None
) -> EnOptEvent:
    context = EnOptContext.model_validate(
        _CONFIG
        if output_dir is None
        else _CONFIG | {"optimizer": {"output_dir": output_dir}}
    )
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
        evaluation_point=evaluations.variables,
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


def test_table_handler_populates_table_from_events(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine)
    handler.add_table("t", "functions", {"functions.target_objective": "Obj"})
    handler.handle_event(_make_event(1))
    df = handler["t"]
    assert not _is_empty(df)
    assert "Obj" in df.columns


def test_table_handler_returns_empty_dataframe_before_any_event(
    engine: DataFrameEngine,
) -> None:
    handler = DataFrameHandler(engine=engine)
    handler.add_table("t", "functions", {"functions.target_objective": "Obj"})
    assert _is_empty(handler["t"])


def test_table_handler_accumulates_multiple_events(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine)
    handler.add_table("t", "functions", {"functions.target_objective": "Obj"})
    handler.handle_event(_make_event(1, objective=1.0))
    handler.handle_event(_make_event(2, objective=2.0))
    df = handler["t"]
    assert len(df) == 2
    assert list(df["Obj"]) == [1.0, 2.0]


def test_table_handler_uses_display_title_not_field_name(
    engine: DataFrameEngine,
) -> None:
    handler = DataFrameHandler(engine=engine)
    handler.add_table(
        "t",
        "functions",
        {"functions.target_objective": "Total Objective"},
    )
    handler.handle_event(_make_event(1))
    df = handler["t"]
    assert "Total Objective" in df.columns
    assert "functions.target_objective" not in df.columns


def test_table_handler_renames_batch_id_column(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine)
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


def test_table_handler_batch_id_value(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine)
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
    assert int(next(iter(df["Batch"]))) == 7


def test_table_handler_column_order_objective_before_batch(
    engine: DataFrameEngine,
) -> None:
    handler = DataFrameHandler(engine=engine)
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


def test_table_handler_column_order_batch_before_objective(
    engine: DataFrameEngine,
) -> None:
    handler = DataFrameHandler(engine=engine)
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


def test_table_handler_omits_batch_id_when_not_in_columns(
    engine: DataFrameEngine,
) -> None:
    handler = DataFrameHandler(engine=engine)
    handler.add_table(
        "t",
        "functions",
        {"functions.target_objective": "Obj"},
    )
    handler.handle_event(_make_event(1))
    df = handler["t"]
    assert "batch_id" not in df.columns
    assert "Obj" in df.columns


def test_table_handler_includes_realization_when_requested(
    engine: DataFrameEngine,
) -> None:
    handler = DataFrameHandler(engine=engine)
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


def test_table_handler_renames_realization_column(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine)
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


def test_table_handler_omits_realization_when_not_in_columns(
    engine: DataFrameEngine,
) -> None:
    handler = DataFrameHandler(engine=engine)
    handler.add_table(
        "t",
        "functions",
        {"evaluations.objectives": "Obj"},
    )
    handler.handle_event(_make_event_two_realizations(1))
    df = handler["t"]
    assert "realization" not in df.columns
    assert "Obj,0" in df.columns


def test_table_handler_realization_column_order(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine)
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


def test_table_handler_realization_column_order_first(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine)
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


def test_table_handler_invalid_engine() -> None:
    with pytest.raises(ValueError, match="Invalid frame engine: nonsense"):
        DataFrameHandler(engine="nonsense")  # type: ignore[arg-type]


def test_table_handler_callback_runs_on_every_update(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine)
    handler.add_table("t", "functions", {"functions.target_objective": "Obj"})
    calls = 0

    def _record(_: Path | None) -> None:
        nonlocal calls
        calls += 1

    handler.set_callback(_record)
    handler.handle_event(_make_event(1))
    handler.handle_event(_make_event(2))
    assert calls == 2


def test_table_handler_callback_receives_configured_output_dir(
    engine: DataFrameEngine, tmp_path: Path
) -> None:
    # The callback typically persists the tables, so it is passed the output
    # directory of the run rather than having to dig it out of the context.
    handler = DataFrameHandler(engine=engine)
    handler.add_table("t", "functions", {"functions.target_objective": "Obj"})
    received: list[Path | None] = []

    handler.set_callback(received.append)
    handler.handle_event(_make_event(1))
    handler.handle_event(_make_event(2, output_dir=tmp_path))
    assert received == [None, tmp_path]


def test_table_handler_callback_skipped_when_no_table_grew(
    engine: DataFrameEngine,
) -> None:
    # Function results leave a gradients table untouched, so a callback that
    # persists the tables has nothing to persist.
    handler = DataFrameHandler(engine=engine)
    handler.add_table("t", "gradients", {"gradients.target_objective": "Grad"})
    called = False

    def _record(_: Path | None) -> None:
        nonlocal called
        called = True

    handler.set_callback(_record)
    handler.handle_event(_make_event(1))
    assert not called
    assert _is_empty(handler["t"])


def test_table_handler_engine_property(engine: DataFrameEngine) -> None:
    assert DataFrameHandler(engine=engine).engine == engine


def test_table_handler_sep(engine: DataFrameEngine) -> None:
    handler = DataFrameHandler(engine=engine, sep="::")
    handler.add_table("t", "functions", {"evaluations.objectives": "Obj"})
    handler.handle_event(_make_event_two_realizations(1))
    assert "Obj::0" in list(handler["t"].columns)


def test_table_handler_engine_parity() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    handlers = {
        engine: DataFrameHandler(engine=engine) for engine in ("pandas", "polars")
    }
    for handler in handlers.values():
        handler.add_table(
            "t",
            "functions",
            {
                "batch_id": "Batch",
                "realization": "Run",
                "evaluations.objectives": "Obj",
                "evaluations.variables": "Var",
            },
        )
        handler.handle_event(_make_event_two_realizations(1))
        handler.handle_event(_make_event_two_realizations(2))

    pandas_table = handlers["pandas"]["t"]
    polars_table = handlers["polars"]["t"]
    assert list(polars_table.columns) == list(pandas_table.columns)
    assert len(polars_table) == len(pandas_table)
    for column in pandas_table.columns:
        assert list(polars_table[column]) == list(pandas_table[column])


def test_table_handler_polars_keeps_keys_for_mixed_granularity() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    columns = {
        "batch_id": "Batch",
        "realization": "Run",
        "evaluations.objectives": "Obj",
        "functions.target_objective": "Total",
    }
    tables = {}
    for engine in ("pandas", "polars"):
        handler = DataFrameHandler(engine=engine)
        handler.add_table("t", "functions", dict(columns))
        handler.handle_event(_make_event_two_realizations(1))
        tables[engine] = handler["t"]

    # Pandas cannot align the per-batch and per-realization fields, so it loses
    # the key columns entirely; polars joins them and keeps every column.
    assert list(tables["pandas"].columns) == ["Obj,0", "Total"]
    assert list(tables["polars"].columns) == ["Batch", "Run", "Obj,0", "Total"]
    assert list(tables["polars"]["Run"]) == [0, 1]
    assert list(tables["polars"]["Total"]) == [1.5, 1.5]
