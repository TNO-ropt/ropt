from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.plan import BasicOptimizer
from ropt.plugins.sampler.scipy import _SUPPORTED_METHODS
from ropt.results import GradientResults, Results

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "tolerance": 1e-4,
            "max_functions": 20,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


@pytest.mark.parametrize("method", sorted(_SUPPORTED_METHODS))
def test_scipy_samplers_unconstrained(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["samplers"] = [{"method": method}]
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_scipy_indexed_sampler(enopt_config: Any, evaluator: Any) -> None:
    # Removing the second variable will fix its value, since it will not be
    # perturbed and its gradient will always be zero.
    enopt_config["gradient"]["samplers"] = [0, -1, 0]
    enopt_config["variables"]["initial_values"][1] = 0.1

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert pytest.approx(variables[0]) != 0.0
    assert pytest.approx(variables[1]) == 0.1
    assert pytest.approx(variables[2]) != 0.5


@pytest.mark.parametrize("method", sorted(_SUPPORTED_METHODS))
def test_scipy_samplers_shared(enopt_config: Any, method: str, evaluator: Any) -> None:
    enopt_config["realizations"] = {"weights": [1.0, 1.0]}
    enopt_config["samplers"] = [{"method": method}]

    perturbations: dict[str, NDArray[np.float64]] = {}

    def _observer(results: tuple[Results, ...], tag: str) -> None:
        for item in results:
            if isinstance(item, GradientResults) and tag not in perturbations:
                perturbations[tag] = item.evaluations.perturbed_variables

    enopt_config["samplers"][0]["shared"] = False
    variables1 = (
        BasicOptimizer(enopt_config, evaluator())
        .set_results_callback(partial(_observer, tag="result1"))
        .run()
        .variables
    )
    assert variables1 is not None

    enopt_config["samplers"][0]["shared"] = True
    variables2 = (
        BasicOptimizer(enopt_config, evaluator())
        .set_results_callback(partial(_observer, tag="result2"))
        .run()
        .variables
    )
    assert variables2 is not None

    # The perturbations of the two realizations must differ, if not shared:
    assert not np.allclose(
        perturbations["result1"][0, ...], perturbations["result1"][1, ...]
    )

    # The perturbations of the two realizations must be the same, if shared:
    assert np.allclose(
        perturbations["result2"][0, ...], perturbations["result2"][1, ...]
    )

    # The results should be correct, but slightly different:
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.allclose(variables1, variables2, atol=1e-3)
