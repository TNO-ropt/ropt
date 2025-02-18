from functools import partial
from typing import Any, Sequence

import pytest

from ropt.config.enopt import EnOptConfig
from ropt.enums import EventType, ResultAxis
from ropt.plan import BasicOptimizer, Event
from ropt.report import ResultsDataFrame

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


def _handle_results(
    event: Event,
    reporter: ResultsDataFrame,
    variable_names: Sequence[str | int] | None = None,
) -> None:
    names: dict[str, Sequence[str | int] | None] | None = (
        None if variable_names is None else {ResultAxis.VARIABLE: variable_names}
    )
    for item in event.data["results"]:
        reporter.add_results(item, names=names)


def test_dataframe_results_no_results(enopt_config: Any, evaluator: Any) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    reporter = ResultsDataFrame(set())
    BasicOptimizer(config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION,
        partial(_handle_results, reporter=reporter),
    ).run()
    assert reporter.frame.empty


def test_dataframe_results_function_results(enopt_config: Any, evaluator: Any) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    reporter = ResultsDataFrame(
        {
            "evaluations.variables",
        },
    )
    BasicOptimizer(config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION,
        partial(_handle_results, reporter=reporter),
    ).run()

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", idx) for idx in range(3)
    ]


def test_dataframe_results_function_results_formatted_names(
    enopt_config: Any, evaluator: Any
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    reporter = ResultsDataFrame(
        {
            "evaluations.variables",
        },
    )
    BasicOptimizer(config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION,
        partial(
            _handle_results,
            reporter=reporter,
            variable_names=[f"a:{idx}" for idx in range(1, 4)],
        ),
    ).run()

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", f"a:{idx}") for idx in range(1, 4)
    ]


def test_dataframe_results_gradient_results(enopt_config: Any, evaluator: Any) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    reporter = ResultsDataFrame(
        {
            "gradients.weighted_objective",
        },
        table_type="gradients",
    )
    BasicOptimizer(config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION,
        partial(
            _handle_results,
            reporter=reporter,
            variable_names=[f"a:{idx}" for idx in range(1, 4)],
        ),
    ).run()

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("gradients.weighted_objective", f"a:{idx}") for idx in range(1, 4)
    ]


def test_dataframe_results_metadata(enopt_config: Any, evaluator: Any) -> None:
    reporter = ResultsDataFrame(
        {
            "evaluations.variables",
            "metadata.foo.bar",
            "metadata.not.existing",
        },
    )

    def even_handler(event: Event) -> None:
        for item in event.data["results"]:
            item.metadata["foo"] = {"bar": 1}
        for item in event.data["results"]:
            reporter.add_results(item)

    BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, even_handler
    ).run()

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", idx) for idx in range(3)
    ] + ["metadata.foo.bar"]


def test_data_frame_table(enopt_config: Any, evaluator: Any) -> None:
    optimizer = BasicOptimizer(enopt_config, evaluator())
    reporter = ResultsDataFrame({"evaluations.variables": "Variables"})
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION, partial(_handle_results, reporter=reporter)
    )
    optimizer.run()
    results = reporter.get_table()
    assert results.columns.to_list() == ["Variables\n0", "Variables\n1", "Variables\n2"]
    assert len(results) == 3


def test_data_frame_table_gradient(enopt_config: Any, evaluator: Any) -> None:
    optimizer = BasicOptimizer(enopt_config, evaluator())
    reporter = ResultsDataFrame(
        {"gradients.weighted_objective": "Total Objective"},
        table_type="gradients",
    )
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        partial(
            _handle_results,
            reporter=reporter,
            variable_names=tuple(f"a:{idx + 1}" for idx in range(3)),
        ),
    )
    optimizer.run()
    gradients = reporter.get_table()
    assert gradients.columns.to_list() == [
        "Total Objective\na:1",
        "Total Objective\na:2",
        "Total Objective\na:3",
    ]
    assert len(gradients) == 3
