from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt.components.event_handlers import CallbackHandler
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import GradientResults
from ropt.sampler.scipy import SCIPY_SAMPLER_SUPPORTED_METHODS
from ropt.simple import optimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "optimizer": {
            "max_functions": 20,
        },
        "backend": {
            "convergence_tolerance": 1e-4,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


@pytest.mark.parametrize("method", sorted(SCIPY_SAMPLER_SUPPORTED_METHODS))
def test_scipy_samplers_unconstrained(config: Any, method: str, eval_func: Any) -> None:
    config["samplers"] = [{"method": method}]
    result = optimize(config, initial_values, eval_func())
    assert result.variables is not None
    assert np.allclose(result.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_scipy_indexed_sampler(config: Any, eval_func: Any) -> None:
    # Removing the second variable will fix its value, since it will not be
    # perturbed and its gradient will always be zero.
    config["variables"]["samplers"] = [0, None, 0]

    initial = initial_values.copy()
    initial[1] = 0.1

    result = optimize(config, initial, eval_func())
    assert result.variables is not None
    assert pytest.approx(result.variables[0]) != 0.0
    assert pytest.approx(result.variables[1]) == 0.1
    assert pytest.approx(result.variables[2]) != 0.5


@pytest.mark.parametrize("method", sorted(SCIPY_SAMPLER_SUPPORTED_METHODS))
def test_scipy_samplers_shared(config: Any, method: str, eval_func: Any) -> None:
    config["realizations"] = {"weights": [1.0, 1.0]}
    config["samplers"] = [{"method": method}]

    perturbations: dict[str, NDArray[np.float64]] = {}

    def _observer(event: EnOptEvent, tag: str) -> None:
        for item in event.results:
            if isinstance(item, GradientResults) and tag not in perturbations:
                perturbations[tag] = item.evaluations.perturbed_variables

    config["samplers"][0]["shared"] = False
    result1 = optimize(
        config,
        initial_values,
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION},
                callback=partial(_observer, tag="result1"),
            )
        ],
    )
    assert result1.variables is not None

    config["samplers"][0]["shared"] = True
    result2 = optimize(
        config,
        initial_values,
        eval_func(),
        handlers=[
            CallbackHandler(
                event_types={EnOptEventType.FINISHED_EVALUATION},
                callback=partial(_observer, tag="result2"),
            )
        ],
    )
    assert result2.variables is not None

    # The perturbations of the two realizations must differ, if not shared:
    assert not np.allclose(
        perturbations["result1"][0, ...], perturbations["result1"][1, ...]
    )

    # The perturbations of the two realizations must be the same, if shared:
    assert np.allclose(
        perturbations["result2"][0, ...], perturbations["result2"][1, ...]
    )

    # The results should be correct, but slightly different:
    assert np.allclose(result1.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(result2.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.allclose(result1.variables, result2.variables, atol=1e-3)
