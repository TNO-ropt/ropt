from functools import partial
from pathlib import Path
from typing import Any, Sequence

import pytest

from ropt.enums import EventType, ResultAxis
from ropt.plan import BasicOptimizer, Event
from ropt.report import ResultsTable

# Requires pandas:
pd = pytest.importorskip("pandas")


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "speculative": True,
            "max_functions": 3,
        },
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
            "upper_bounds": 1.0,
            "lower_bounds": -1.0,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def _handle_event(
    event: Event,
    *,
    table: ResultsTable,
    names: dict[str, Sequence[str | int] | None] | None = None,
) -> None:
    if event.event_type == EventType.FINISHED_EVALUATION and "results" in event.data:
        added = False
        for item in event.data["results"]:
            if table.add_results(item, names=names):
                added = True
        if added:
            table.save()


def test_tabular_report_no_results(
    enopt_config: Any, evaluator: Any, tmp_path: Path
) -> None:
    path = tmp_path / "results.txt"
    optimizer = BasicOptimizer(enopt_config, evaluator())
    table = ResultsTable({}, path)
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION, partial(_handle_event, table=table)
    )
    optimizer.run()
    assert not path.exists()


def test_tabular_report_results(
    enopt_config: Any, evaluator: Any, tmp_path: Path
) -> None:
    path = tmp_path / "results.txt"
    optimizer = BasicOptimizer(enopt_config, evaluator())
    table = ResultsTable({"evaluations.variables": "Variables"}, path)
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION, partial(_handle_event, table=table)
    )
    optimizer.run()
    assert path.exists()
    results = pd.read_fwf(tmp_path / "results.txt", header=[0, 1], skiprows=[2])
    assert results.columns.get_level_values(level=0).to_list() == ["Variables"] * 3
    assert results.columns.get_level_values(level=1).to_list() == ["0", "1", "2"]
    assert len(results) == 3


def test_tabular_report_data_frames_results_formatted_names(
    enopt_config: Any, evaluator: Any, tmp_path: Path
) -> None:
    path = tmp_path / "results.txt"
    optimizer = BasicOptimizer(enopt_config, evaluator())
    table = ResultsTable({"evaluations.variables": "Variables"}, path)
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        partial(
            _handle_event,
            table=table,
            names={"variable": tuple(f"a:{idx + 1}" for idx in range(3))},
        ),
    )
    optimizer.run()
    assert path.exists()
    results = pd.read_fwf(path, header=[0, 1], skiprows=[2])
    assert results.columns.get_level_values(level=0).to_list() == ["Variables"] * 3
    assert list(results.columns.get_level_values(level=1).to_list()) == [
        f"a:{idx + 1}" for idx in range(3)
    ]


def test_tabular_report_data_frames_gradients(
    enopt_config: Any, evaluator: Any, tmp_path: Path
) -> None:
    path = tmp_path / "gradients.txt"
    optimizer = BasicOptimizer(enopt_config, evaluator())
    table = ResultsTable(
        {"gradients.weighted_objective": "Total Objective"},
        path,
        table_type="gradients",
    )
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        partial(
            _handle_event,
            table=table,
            names={ResultAxis.VARIABLE: tuple(f"a:{idx + 1}" for idx in range(3))},
        ),
    )
    optimizer.run()
    assert path.exists()
    gradients = pd.read_fwf(path, header=[0, 1], skiprows=[2])
    assert (
        gradients.columns.get_level_values(level=0).to_list() == ["Total Objective"] * 3
    )
    assert gradients.columns.get_level_values(level=1).to_list() == [
        "a:1",
        "a:2",
        "a:3",
    ]
    assert len(gradients) == 3


@pytest.mark.parametrize("min_header_len", [None, 4])
def test_tabular_report_data_frames_min_header_len(
    enopt_config: Any, evaluator: Any, tmp_path: Path, min_header_len: int | None
) -> None:
    path = tmp_path / "results.txt"
    optimizer = BasicOptimizer(enopt_config, evaluator())
    table = ResultsTable(
        {"evaluations.variables": "Variables"}, path, min_header_len=min_header_len
    )
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        partial(_handle_event, table=table),
    )
    optimizer.run()
    assert path.exists()
    with path.open() as fp:
        lines = fp.readlines()
    header_idx = 2 if min_header_len is None else min_header_len
    assert lines[header_idx][0] == "-"
    results = pd.read_fwf(
        path,
        header=list(range(header_idx)),
        skip_blank_lines=False,
        skiprows=[header_idx],
    )
    assert len(results.columns[0]) == header_idx
    assert len(results) == 3
