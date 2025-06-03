from functools import partial
from typing import Any, Literal

import pytest

from ropt.enums import AxisName
from ropt.plan import BasicOptimizer
from ropt.results import Results, results_to_dataframe

# Requires pandas:
pytest.importorskip("pandas")

import pandas as pd


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
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
            "evaluation_policy": "speculative",
        },
        "names": {
            AxisName.VARIABLE: tuple(f"a:{idx}" for idx in range(1, 4)),
        },
    }


def _handle_results(
    results: tuple[Results, ...],
    frames: list[pd.DataFrame],
    fields: set[str],
    result_type: Literal["functions", "gradients"],
    metadata: dict[str, Any] | None = None,
) -> None:
    if metadata is not None:
        for item in results:
            item.metadata = metadata
    frame = results_to_dataframe(results, fields, result_type=result_type)
    if not frame.empty:
        frames.append(frame)


def test_dataframe_results_no_results(enopt_config: Any, evaluator: Any) -> None:
    frames: list[pd.DataFrame] = []
    BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        partial(_handle_results, frames=frames, fields=set(), result_type="functions"),
    ).run()
    assert not frames


def test_dataframe_results_function_results(enopt_config: Any, evaluator: Any) -> None:
    del enopt_config["names"]
    frames: list[pd.DataFrame] = []
    BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        partial(
            _handle_results,
            frames=frames,
            fields={
                "evaluations.variables",
            },
            result_type="functions",
        ),
    ).run()
    frame = pd.concat(frames)
    assert len(frame) == 3
    assert list(frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", idx) for idx in range(3)
    ]


def test_dataframe_results_function_results_formatted_names(
    enopt_config: Any, evaluator: Any
) -> None:
    frames: list[pd.DataFrame] = []
    BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        partial(
            _handle_results,
            frames=frames,
            fields={
                "evaluations.variables",
            },
            result_type="functions",
        ),
    ).run()
    frame = pd.concat(frames)
    assert len(frame) == 3
    assert list(frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", f"a:{idx}") for idx in range(1, 4)
    ]


def test_dataframe_results_gradient_results(enopt_config: Any, evaluator: Any) -> None:
    frames: list[pd.DataFrame] = []
    BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        partial(
            _handle_results,
            frames=frames,
            fields={
                "gradients.weighted_objective",
            },
            result_type="gradients",
        ),
    ).run()
    frame = pd.concat(frames)
    assert len(frame) == 3
    assert list(frame.columns.get_level_values(level=0)) == [
        ("gradients.weighted_objective", f"a:{idx}") for idx in range(1, 4)
    ]


def test_dataframe_results_metadata(enopt_config: Any, evaluator: Any) -> None:
    del enopt_config["names"]
    frames: list[pd.DataFrame] = []
    BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        partial(
            _handle_results,
            frames=frames,
            fields={
                "evaluations.variables",
                "metadata.foo.bar",
                "metadata.not.existing",
            },
            result_type="functions",
            metadata={"foo": {"bar": 1}},
        ),
    ).run()
    frame = pd.concat(frames)
    assert len(frame) == 3
    assert list(frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", idx) for idx in range(3)
    ] + ["metadata.foo.bar"]
