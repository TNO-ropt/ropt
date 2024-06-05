from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pytest
from numpy.random import Generator, default_rng

from ropt.config.enopt import EnOptConfig
from ropt.evaluator._gradient import _perturb_variables
from ropt.exceptions import ConfigError
from ropt.optimization import EnsembleOptimizer
from ropt.plugins import PluginManager
from ropt.plugins.sampler.base import Sampler, SamplerPlugin

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 0.005,
            "max_functions": 4,
        },
        "objective_functions": {
            "weights": [0.75, 0.24],
        },
        "samplers": [
            {
                "method": "test",
            },
        ],
        "gradient": {
            "number_of_perturbations": 3,
            "perturbation_magnitudes": 0.01,
        },
        "variables": {
            "initial_values": [0, 0, 0],
            "upper_bounds": 1.0,
            "lower_bounds": -1.0,
        },
    }


class MockedSampler(Sampler):
    def __init__(
        self,
        enopt_config: EnOptConfig,
        sampler_index: int,
        variable_indices: Optional[NDArray[np.intc]],
        _: Generator,
    ) -> None:
        self._config = enopt_config
        self._sampler = enopt_config.samplers[sampler_index]
        self._variable_indices = variable_indices
        # This sampler only works if the number of perturbation equals the
        # number of variables:
        assert enopt_config.gradient.number_of_perturbations == len(
            enopt_config.variables.initial_values
        )

    def generate_samples(self) -> NDArray[np.float64]:
        variable_count = self._config.variables.initial_values.size
        realization_count = self._config.realizations.weights.size
        perturbation_count = self._config.gradient.number_of_perturbations

        samples: NDArray[np.float64]
        if self._variable_indices is None:
            samples = np.ones(
                (realization_count, perturbation_count, variable_count),
                dtype=np.float64,
            )
        else:
            samples = np.zeros(
                (realization_count, perturbation_count, variable_count),
                dtype=np.float64,
            )
            samples[..., self._variable_indices] = 1.0
        if "scale" in self._sampler.options:
            samples *= self._sampler.options["scale"]
        for idx in range(samples.shape[0]):
            diag = np.diag(samples[idx, ...])
            samples[idx, ...] = np.diag(diag)
        return samples


class MockedSamplerPlugin(SamplerPlugin):
    def create(
        self,
        enopt_config: EnOptConfig,
        sampler_index: int,
        variable_indices: Optional[NDArray[np.intc]],
        rng: Generator,
    ) -> MockedSampler:
        return MockedSampler(enopt_config, sampler_index, variable_indices, rng)

    def is_supported(self, method: str) -> bool:
        return method.lower() in {"test"}


def test_sampler_simple(enopt_config: Any) -> None:
    enopt_config["gradient"]["number_of_perturbations"] = 3
    rng = default_rng(123)
    config = EnOptConfig.model_validate(enopt_config)
    sampler = MockedSampler(config, 0, None, rng)

    perturbed_variables = _perturb_variables(
        variables=np.array([0.0, 0.0, 0.0]),
        variables_config=config.variables,
        gradient_config=config.gradient,
        samplers=[sampler],
    )
    assert np.allclose(perturbed_variables, 0.01 * np.eye(3))


def test_sampler_use_options(enopt_config: Any) -> None:
    rng = default_rng(123)
    samplers: List[Dict[str, Any]] = enopt_config["samplers"]
    samplers[0]["options"] = {"scale": 100.0}
    config = EnOptConfig.model_validate(enopt_config)
    sampler = MockedSampler(config, 0, None, rng)

    perturbed_variables = _perturb_variables(
        variables=np.array([0.0, 0.0, 0.0]),
        variables_config=config.variables,
        gradient_config=config.gradient,
        samplers=[sampler],
    )
    assert np.allclose(perturbed_variables, np.eye(3))


def test_sampler_indexed(enopt_config: Any) -> None:
    rng = default_rng(123)
    samplers: List[Dict[str, Any]] = enopt_config["samplers"]
    samplers.append(copy.deepcopy(samplers[0]))
    samplers[1]["options"] = {"scale": -1}
    enopt_config["gradient"]["samplers"] = [0, 1, 1]
    config = EnOptConfig.model_validate(enopt_config)
    sampler1 = MockedSampler(config, 0, np.array([0]), rng)
    sampler2 = MockedSampler(config, 1, np.array([1, 2]), rng)

    perturbed_variables = _perturb_variables(
        variables=np.array([0.0, 0.0, 0.0]),
        variables_config=config.variables,
        gradient_config=config.gradient,
        samplers=[sampler1, sampler2],
    )
    assert np.allclose(perturbed_variables[0, ...], np.diag([0.01, -0.01, -0.01]))


def test_sampler_order(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["gradient"]["samplers"] = [0, 0, 1]
    enopt_config["samplers"] = [
        {"method": "norm"},
        {"method": "uniform"},
    ]
    optimizer = EnsembleOptimizer(evaluator())
    results1 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results1 is not None
    assert np.allclose(results1.evaluations.variables, [0, 0, 0.5], atol=0.025)

    # Switch the samplers:
    enopt_config["samplers"] = [
        {"method": "uniform"},
        {"method": "norm"},
    ]
    enopt_config["gradient"]["samplers"] = [1, 1, 0]
    optimizer = EnsembleOptimizer(evaluator())
    results2 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results2 is not None
    assert np.allclose(results2.evaluations.variables, [0, 0, 0.5], atol=0.025)

    assert np.allclose(
        results1.evaluations.variables,
        results2.evaluations.variables,
    )


def test_sampler_plugin(enopt_config: Any, evaluator: Any) -> None:
    optimizer = EnsembleOptimizer(evaluator())
    with pytest.raises(ConfigError, match="Method not found: test"):
        optimizer.start_optimization(plan=[{"config": enopt_config}, {"optimizer": {}}])

    plugin_manager = PluginManager()
    plugin_manager.add_plugins("sampler", {"mocked": MockedSamplerPlugin()})
    optimizer = EnsembleOptimizer(evaluator(), plugin_manager)
    result = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert result is not None
    assert np.allclose(result.evaluations.variables, [0, 0, 0.5], atol=0.02)
