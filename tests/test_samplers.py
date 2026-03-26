from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.random import Generator, default_rng

from ropt.config import EnOptConfig, SamplerConfig
from ropt.core._gradient import _perturb_variables
from ropt.sampler import Sampler
from ropt.workflow import BasicOptimizer

if TYPE_CHECKING:
    from numpy.typing import NDArray

initial_values = [0.0, 0.0, 0.0]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "backend": {
            "tolerance": 0.005,
            "max_functions": 4,
        },
        "objectives": {
            "weights": [0.75, 0.24],
        },
        "samplers": [
            {
                "method": "test",
            },
        ],
        "gradient": {
            "number_of_perturbations": 3,
        },
        "variables": {
            "variable_count": len(initial_values),
            "upper_bounds": 1.0,
            "lower_bounds": -1.0,
            "perturbation_magnitudes": 0.01,
        },
    }


class MockedSampler(Sampler):
    def __init__(
        self,
        sampler: SamplerConfig,
    ) -> None:
        assert isinstance(sampler, SamplerConfig)
        self._sampler_config = sampler
        # This sampler only works if the number of perturbation equals the
        # number of variables:

    def init(
        self,
        mask: NDArray[np.bool_] | None,
        _: Generator,
    ) -> None:
        self._mask = mask

    def generate_samples(self, enopt_config: EnOptConfig) -> NDArray[np.float64]:
        assert (
            enopt_config.gradient.number_of_perturbations
            == enopt_config.variables.variable_count
        )
        variable_count = enopt_config.variables.variable_count
        realization_count = enopt_config.realizations.weights.size
        perturbation_count = enopt_config.gradient.number_of_perturbations

        samples: NDArray[np.float64]
        if self._mask is None:
            samples = np.ones(
                (realization_count, perturbation_count, variable_count),
                dtype=np.float64,
            )
        else:
            samples = np.zeros(
                (realization_count, perturbation_count, variable_count),
                dtype=np.float64,
            )
            samples[..., self._mask] = 1.0
        if "scale" in self._sampler_config.options:
            samples *= self._sampler_config.options["scale"]
        for idx in range(samples.shape[0]):
            diag = np.diag(samples[idx, ...])
            samples[idx, ...] = np.diag(diag)
        return samples


def test_sampler_simple(enopt_config: Any) -> None:
    enopt_config["gradient"]["number_of_perturbations"] = 3
    rng = default_rng(123)
    config = EnOptConfig.model_validate(enopt_config)
    sampler_config = config.samplers[0]
    assert isinstance(sampler_config, SamplerConfig)
    sampler = MockedSampler(sampler_config)
    sampler.init(None, rng)

    perturbed_variables = _perturb_variables(
        config,
        np.array([0.0, 0.0, 0.0]),
        [sampler],
    )
    assert np.allclose(perturbed_variables, 0.01 * np.eye(3))


def test_sampler_use_options(enopt_config: Any) -> None:
    rng = default_rng(123)
    samplers: list[dict[str, Any]] = enopt_config["samplers"]
    samplers[0]["options"] = {"scale": 100.0}
    config = EnOptConfig.model_validate(enopt_config)
    sampler_config = config.samplers[0]
    assert isinstance(sampler_config, SamplerConfig)
    sampler = MockedSampler(sampler_config)
    sampler.init(None, rng)

    perturbed_variables = _perturb_variables(
        config,
        np.array([0.0, 0.0, 0.0]),
        [sampler],
    )
    assert np.allclose(perturbed_variables, np.eye(3))


def test_sampler_indexed(enopt_config: Any) -> None:
    rng = default_rng(123)
    samplers: list[dict[str, Any]] = enopt_config["samplers"]
    samplers.append(copy.deepcopy(samplers[0]))
    samplers[1]["options"] = {"scale": -1}
    enopt_config["variables"]["samplers"] = [0, 1, 1]
    config = EnOptConfig.model_validate(enopt_config)

    sampler_config = config.samplers[0]
    assert isinstance(sampler_config, SamplerConfig)
    sampler1 = MockedSampler(sampler_config)
    sampler1.init(np.array([0]), rng)

    sampler_config = config.samplers[1]
    assert isinstance(sampler_config, SamplerConfig)
    sampler2 = MockedSampler(sampler_config)
    sampler2.init(np.array([1, 2]), rng)

    perturbed_variables = _perturb_variables(
        config,
        np.array([0.0, 0.0, 0.0]),
        [sampler1, sampler2],
    )
    assert np.allclose(perturbed_variables[0, ...], np.diag([0.01, -0.01, -0.01]))


def test_sampler_order(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["samplers"] = [0, 0, 1]
    enopt_config["samplers"] = [
        {"method": "norm"},
        {"method": "uniform"},
    ]
    optimizer1 = BasicOptimizer(enopt_config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert np.allclose(
        optimizer1.results.evaluations.variables, [0, 0, 0.5], atol=0.025
    )

    # Switch the samplers:
    enopt_config["samplers"] = [
        {"method": "uniform"},
        {"method": "norm"},
    ]
    enopt_config["variables"]["samplers"] = [1, 1, 0]
    optimizer2 = BasicOptimizer(enopt_config, evaluator())
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables, [0, 0, 0.5], atol=0.025
    )

    assert np.allclose(
        optimizer1.results.evaluations.variables,
        optimizer2.results.evaluations.variables,
    )
