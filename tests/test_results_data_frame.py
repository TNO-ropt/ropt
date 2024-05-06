from typing import Any, Dict

import pytest

from ropt.config.enopt import EnOptConfig
from ropt.enums import EventType
from ropt.events import OptimizationEvent
from ropt.optimization import EnsembleOptimizer
from ropt.report import ResultsDataFrame

# Requires pandas:
pd = pytest.importorskip("pandas")


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
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
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_dataframe_results_no_results(enopt_config: Any, evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())
    reporter = ResultsDataFrame(set())
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        lambda event: reporter.add_results(event.config, event.results),  # type: ignore
    )
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])
    assert reporter.frame.empty


def test_dataframe_results_function_results(enopt_config: Any, evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())
    reporter = ResultsDataFrame(
        {
            "result_id",
            "evaluations.variables",
        },
    )
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        lambda event: reporter.add_results(event.config, event.results),  # type: ignore
    )
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", idx) for idx in range(3)
    ]


def test_dataframe_results_function_results_formatted_names(
    enopt_config: Any, evaluator: Any
) -> None:
    enopt_config["variables"]["names"] = [("a", 1), ("a", 2), ("a", 3)]
    optimizer = EnsembleOptimizer(evaluator())
    reporter = ResultsDataFrame(
        {
            "result_id",
            "evaluations.variables",
        },
    )
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        lambda event: reporter.add_results(event.config, event.results),  # type: ignore
    )
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", f"a:{idx}") for idx in range(1, 4)
    ]


def test_dataframe_results_gradient_results(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["names"] = [("a", 1), ("a", 2), ("a", 3)]
    optimizer = EnsembleOptimizer(evaluator())
    reporter = ResultsDataFrame(
        {
            "result_id",
            "gradients.weighted_objective",
        },
        table_type="gradients",
    )
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        lambda event: reporter.add_results(event.config, event.results),  # type: ignore
    )
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("gradients.weighted_objective", f"a:{idx}") for idx in range(1, 4)
    ]


def test_dataframe_results_metadata(enopt_config: Any, evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())
    reporter = ResultsDataFrame(
        {
            "result_id",
            "evaluations.variables",
            "metadata.foo.bar",
            "metadata.not.existing",
        },
    )

    def handler(event: OptimizationEvent) -> None:
        assert event.results is not None
        for item in event.results:
            item.metadata["foo"] = {"bar": 1}
        reporter.add_results(EnOptConfig.model_validate(enopt_config), event.results)

    optimizer.add_observer(EventType.FINISHED_EVALUATION, handler)
    optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])

    assert len(reporter.frame) == 3
    assert list(reporter.frame.columns.get_level_values(level=0)) == [
        ("evaluations.variables", idx) for idx in range(3)
    ] + ["metadata.foo.bar"]


def test_dataframe_results_metadata_step_id(enopt_config: Any, evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())
    reporter = ResultsDataFrame(
        {
            "result_id",
            "evaluations.variables",
            "metadata.step_id",
        },
    )

    def handler(event: OptimizationEvent) -> None:
        assert event.results is not None
        reporter.add_results(EnOptConfig.model_validate(enopt_config), event.results)

    optimizer.add_observer(EventType.FINISHED_EVALUATION, handler)
    optimizer.start_optimization(
        plan=[{"config": enopt_config}, {"optimizer": {"id": "opt"}}]
    )

    assert reporter.frame["metadata.step_id"].to_list() == ["opt"] * 3
