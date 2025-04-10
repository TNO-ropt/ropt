from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.enums import OptimizerExitCode
from ropt.plan import BasicOptimizer
from ropt.plugins._manager import PluginManager
from ropt.results import FunctionResults, Results

pytestmark = [pytest.mark.slow]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "method": "external/slsqp",
            "tolerance": 1e-4,
            "max_iterations": 50,
        },
        "objectives": {
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


def test_external_invalid_options(enopt_config: Any) -> None:
    enopt_config["optimizer"]["options"] = {"ftol": 0.1, "foo": 1}

    method = enopt_config["optimizer"]["method"]
    plugin = PluginManager().get_plugin("optimizer", method)
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`"
    ):
        plugin.validate_options(method, enopt_config["optimizer"]["options"])


def test_external_max_functions_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation

        last_evaluation += 1

    max_functions = 2
    enopt_config["optimizer"]["max_functions"] = max_functions
    optimizer = BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        track_results
    )
    optimizer.run()
    assert last_evaluation == max_functions + 1
    assert optimizer.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED


def test_external_max_batches_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def track_results(_: tuple[Results, ...]) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_batches = 2
    enopt_config["optimizer"]["max_batches"] = max_batches
    optimizer = BasicOptimizer(enopt_config, evaluator()).set_results_callback(
        track_results
    )
    optimizer.run()
    assert last_evaluation == max_batches
    assert optimizer.exit_code == OptimizerExitCode.MAX_BATCHES_REACHED


def test_external_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    def _observer(results: tuple[Results, ...]) -> None:
        assert isinstance(results[0], FunctionResults)
        assert results[0].functions is None

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(enopt_config, evaluator(functions)).set_results_callback(
        _observer
    )
    optimizer.run()
    assert optimizer.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS


def test_external_user_abort(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def _abort() -> bool:
        nonlocal last_evaluation

        if last_evaluation == 1:
            return True
        last_evaluation += 1
        return False

    optimizer = BasicOptimizer(enopt_config, evaluator()).set_abort_callback(_abort)
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
