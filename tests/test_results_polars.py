import math
from functools import partial
from typing import Any, Literal

import pytest

from ropt.components.event_handlers import CallbackHandler
from ropt.enums import AxisName, EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import results_to_pandas, results_to_polars
from ropt.simple import optimize

pytest.importorskip("polars")

import polars as pl

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "max_functions": 3,
        },
        "backend": {
            "convergence_tolerance": 1e-5,
        },
        "variables": {
            "variable_count": 3,
            "upper_bounds": 1.0,
            "lower_bounds": -1.0,
            "perturbation_magnitudes": 0.01,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "evaluation_policy": "speculative",
        },
        "names": {
            AxisName.VARIABLE: tuple(f"a:{idx}" for idx in range(1, 4)),
        },
    }


def _handle_results(
    event: EnOptEvent,
    *,
    frames: list[pl.DataFrame],
    fields: set[str],
    result_type: Literal["functions", "gradients"],
    metadata: dict[str, Any] | None = None,
    sep: str = ",",
) -> None:
    results = event.results or ()
    if metadata is not None:
        for item in results:
            item.metadata = metadata
    frame = results_to_polars(results, fields, result_type=result_type, sep=sep)
    if frame.height > 0:
        frames.append(frame)


def _run(
    config: Any,
    eval_func: Any,
    fields: set[str],
    result_type: Literal["functions", "gradients"],
    *,
    metadata: dict[str, Any] | None = None,
    sep: str = ",",
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    optimize(
        config,
        initial_values,
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION},
                callback=partial(
                    _handle_results,
                    frames=frames,
                    fields=fields,
                    result_type=result_type,
                    metadata=metadata,
                    sep=sep,
                ),
            )
        ],
    )
    return frames


def test_polars_results_no_results(config: Any, eval_func: Any) -> None:
    assert not _run(config, eval_func, set(), "functions")


def test_polars_results_function_results(config: Any, eval_func: Any) -> None:
    del config["names"]
    frames = _run(config, eval_func, {"evaluations.variables"}, "functions")
    frame = pl.concat(frames, how="diagonal")
    assert frame.height == 3
    assert frame.columns == [
        "batch_id",
        *(f"evaluations.variables,{idx}" for idx in range(3)),
    ]


def test_polars_results_function_results_formatted_names(
    config: Any, eval_func: Any
) -> None:
    frames = _run(config, eval_func, {"evaluations.variables"}, "functions")
    frame = pl.concat(frames, how="diagonal")
    assert frame.height == 3
    assert frame.columns == [
        "batch_id",
        *(f"evaluations.variables,a:{idx}" for idx in range(1, 4)),
    ]


def test_polars_results_sep(config: Any, eval_func: Any) -> None:
    frames = _run(config, eval_func, {"evaluations.variables"}, "functions", sep="::")
    frame = pl.concat(frames, how="diagonal")
    assert frame.columns == [
        "batch_id",
        *(f"evaluations.variables::a:{idx}" for idx in range(1, 4)),
    ]


def test_polars_results_gradient_results(config: Any, eval_func: Any) -> None:
    frames = _run(config, eval_func, {"gradients.target_objective"}, "gradients")
    frame = pl.concat(frames, how="diagonal")
    assert frame.height == 3
    assert frame.columns == [
        "batch_id",
        *(f"gradients.target_objective,a:{idx}" for idx in range(1, 4)),
    ]


def test_polars_results_metadata(config: Any, eval_func: Any) -> None:
    del config["names"]
    frames = _run(
        config,
        eval_func,
        {"evaluations.variables", "metadata.foo.bar", "metadata.not.existing"},
        "functions",
        metadata={"foo": {"bar": 1}},
    )
    frame = pl.concat(frames, how="diagonal")
    assert frame.height == 3
    assert frame.columns == [
        "batch_id",
        *(f"evaluations.variables,{idx}" for idx in range(3)),
        "metadata.foo.bar",
    ]
    assert frame["metadata.foo.bar"].to_list() == [1, 1, 1]


def test_polars_results_invalid_type() -> None:
    with pytest.raises(TypeError, match="Invalid frame output type: invalid"):
        results_to_polars((), set(), result_type="invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Invalid result type:"):
        results_to_polars(("nonsense",), {"a"}, result_type="functions")  # type: ignore[arg-type]


def test_polars_results_mixed_granularity(config: Any, eval_func: Any) -> None:
    config["gradient"]["number_of_perturbations"] = 2
    frames = _run(
        config,
        eval_func,
        {"gradients.target_objective", "evaluations.perturbed_variables"},
        "gradients",
    )
    frame = pl.concat(frames, how="diagonal")
    assert frame.columns == [
        "batch_id",
        "realization",
        "perturbation",
        *(f"gradients.target_objective,a:{idx}" for idx in range(1, 4)),
        *(f"evaluations.perturbed_variables,a:{idx}" for idx in range(1, 4)),
    ]
    assert frame.null_count().sum_horizontal().item() == 0
    for batch_frame in frame.partition_by("batch_id"):
        assert batch_frame.height == 2
        gradients = batch_frame.select(pl.col("^gradients\\..*$"))
        assert gradients.row(0) == gradients.row(1)


_PARITY_FIELDS: list[tuple[set[str], Literal["functions", "gradients"]]] = [
    ({"evaluations.variables"}, "functions"),
    (
        {
            "batch_id",
            "functions.target_objective",
            "functions.objectives",
            "evaluations.variables",
            "constraint_info.bound_lower",
            "constraint_info.bound_upper",
            "constraint_info.bound_violation",
        },
        "functions",
    ),
    ({"evaluations.variables", "metadata.foo.bar"}, "functions"),
    ({"gradients.target_objective", "gradients.objectives"}, "gradients"),
    (
        {
            "gradients.target_objective",
            "gradients.objectives",
        },
        "gradients",
    ),
    (
        {
            "evaluations.perturbed_variables",
            "evaluations.perturbed_objectives",
        },
        "gradients",
    ),
]


@pytest.mark.parametrize(("fields", "result_type"), _PARITY_FIELDS)
@pytest.mark.parametrize("sep", [",", "::"])
def test_polars_results_pandas_parity(
    config: Any,
    eval_func: Any,
    fields: set[str],
    result_type: Literal["functions", "gradients"],
    *,
    sep: str,
) -> None:
    pytest.importorskip("pandas")

    collected: list[Any] = []

    def _collect(event: EnOptEvent) -> None:
        for item in event.results or ():
            item.metadata = {"foo": {"bar": 1}}
            collected.append(item)

    optimize(
        config,
        initial_values,
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION}, callback=_collect
            )
        ],
    )
    assert collected

    pandas_frame = results_to_pandas(
        collected, fields, result_type=result_type
    ).reset_index()
    pandas_frame.columns = [
        sep.join(str(part) for part in column)
        if isinstance(column, tuple)
        else str(column)
        for column in pandas_frame.columns
    ]
    polars_frame = results_to_polars(
        collected, fields, result_type=result_type, sep=sep
    )

    assert polars_frame.columns == list(pandas_frame.columns)
    assert polars_frame.height == len(pandas_frame)
    for column in pandas_frame.columns:
        expected = list(pandas_frame[column])
        actual = polars_frame[column].to_list()
        for lhs, rhs in zip(actual, expected, strict=True):
            if isinstance(rhs, float) and math.isnan(rhs):
                assert math.isnan(lhs)
            else:
                assert lhs == rhs


def test_polars_results_empty_input() -> None:
    frame = results_to_polars((), set(), result_type="functions")
    assert frame.height == 0
    assert frame.width == 0
