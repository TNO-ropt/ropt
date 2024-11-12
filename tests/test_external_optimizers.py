from typing import Any, Dict

import numpy as np
import pytest

from ropt.enums import EventType, OptimizerExitCode
from ropt.plan import BasicOptimizer, Event
from ropt.results import FunctionResults

pytestmark = [pytest.mark.slow]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "method": "external/slsqp",
            "tolerance": 1e-4,
            "max_iterations": 50,
        },
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_external_run(enopt_config: Any, evaluator: Any) -> None:
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0, 0, 0.5], atol=0.02)


def test_external_max_functions_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 100

    def track_results(event: Event) -> None:
        nonlocal last_evaluation
        assert event.results
        assert isinstance(event.results[0].result_id, int)
        last_evaluation = event.results[0].result_id

    max_functions = 2
    enopt_config["optimizer"]["max_functions"] = max_functions
    optimizer = BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, track_results
    )
    optimizer.run()
    assert last_evaluation == max_functions
    assert optimizer.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED


def test_external_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    def _observer(event: Event) -> None:
        assert event.results
        assert isinstance(event.results[0], FunctionResults)
        assert event.results[0].functions is None

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(enopt_config, evaluator(functions)).add_observer(
        EventType.FINISHED_EVALUATION, _observer
    )
    optimizer.run()
    assert optimizer.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS


def test_external_user_abort(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 100

    def _observer(event: Event) -> None:
        nonlocal last_evaluation
        assert event.results
        assert isinstance(event.results[0].result_id, int)
        last_evaluation = event.results[0].result_id
        if event.results[0].result_id == 1:
            optimizer.abort_optimization()

    optimizer = BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _observer
    )
    optimizer.run()
    assert optimizer.results is not None
    assert last_evaluation == 1
    assert optimizer.exit_code == OptimizerExitCode.USER_ABORT


def test_external_error(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["options"] = {"ftol": "foo"}
    with pytest.raises(
        RuntimeError,
        match="External optimizer error: could not convert string to float: 'foo'",
    ):
        BasicOptimizer(enopt_config, evaluator()).run()
