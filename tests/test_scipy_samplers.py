from functools import partial
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pytest

from ropt.enums import EventType
from ropt.events import OptimizationEvent
from ropt.optimization import EnsembleOptimizer
from ropt.plugins.sampler.scipy import _SUPPORTED_METHODS
from ropt.results import GradientResults

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "tolerance": 1e-4,
            "max_functions": 20,
        },
        "objective_functions": {
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
    optimizer = EnsembleOptimizer(evaluator())
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_scipy_indexed_sampler(enopt_config: Any, evaluator: Any) -> None:
    # Removing the second variable will fix its value, since it will not be
    # perturbed and its gradient will always be zero.
    enopt_config["gradient"]["samplers"] = [0, -1, 0]
    enopt_config["variables"]["initial_values"][1] = 0.1

    optimizer = EnsembleOptimizer(evaluator())
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert pytest.approx(result.evaluations.variables[0]) != 0.0
    assert pytest.approx(result.evaluations.variables[1]) == 0.1
    assert pytest.approx(result.evaluations.variables[2]) != 0.5


@pytest.mark.parametrize("method", sorted(_SUPPORTED_METHODS))
def test_scipy_samplers_shared(enopt_config: Any, method: str, evaluator: Any) -> None:
    enopt_config["realizations"] = {"weights": [1.0, 1.0]}
    enopt_config["samplers"] = [{"method": method}]

    perturbations: Dict[str, NDArray[np.float64]] = {}

    def _observer(event: OptimizationEvent, tag: str) -> None:
        assert event.results is not None
        for item in event.results:
            if isinstance(item, GradientResults) and tag not in perturbations:
                perturbations[tag] = item.evaluations.perturbed_variables

    enopt_config["samplers"][0]["shared"] = False
    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        partial(_observer, tag="result1"),
    )
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    result1 = result.evaluations.variables

    enopt_config["samplers"][0]["shared"] = True
    optimizer = EnsembleOptimizer(evaluator())
    optimizer.add_observer(
        EventType.FINISHED_EVALUATION,
        partial(_observer, tag="result2"),
    )
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    result2 = result.evaluations.variables

    # The perturbations of the two realizations must differ, if not shared:
    assert not np.allclose(
        perturbations["result1"][0, ...], perturbations["result1"][1, ...]
    )

    # The perturbations of the two realizations must be the same, if shared:
    assert np.allclose(
        perturbations["result2"][0, ...], perturbations["result2"][1, ...]
    )

    # The results should be correct, but slightly different:
    assert np.allclose(result1, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(result2, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.allclose(result1, result2, atol=1e-3)
