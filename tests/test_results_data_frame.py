from functools import partial
from typing import Any

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
    variable_names: tuple[str, ...] | None = None,
) -> None:
    names: dict[ResultAxis, tuple[str, ...] | None] | None = (
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
            "result_id",
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
            "result_id",
            "evaluations.variables",
        },
    )
    BasicOptimizer(config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION,
        partial(
            _handle_results,
            reporter=reporter,
            variable_names=tuple(f"a:{idx}" for idx in range(1, 4)),
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
            "result_id",
            "gradients.weighted_objective",
        },
        table_type="gradients",
    )
    BasicOptimizer(config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION,
        partial(
            _handle_results,
            reporter=reporter,
            variable_names=tuple(f"a:{idx}" for idx in range(1, 4)),
        ),
    ).run()

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("gradients.weighted_objective", f"a:{idx}") for idx in range(1, 4)
    ]


def test_dataframe_results_metadata(enopt_config: Any, evaluator: Any) -> None:
    reporter = ResultsDataFrame(
        {
            "result_id",
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
