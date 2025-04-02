from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.random import Generator, default_rng

from ropt.config.enopt import EnOptConfig
from ropt.ensemble_evaluator._gradient import _perturb_variables
from ropt.plan import BasicOptimizer
from ropt.plugins.sampler.base import Sampler, SamplerPlugin

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
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
        mask: NDArray[np.bool_] | None,
        _: Generator,
    ) -> None:
        self._config = enopt_config
        self._sampler = enopt_config.samplers[sampler_index]
        self._mask = mask
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
        if "scale" in self._sampler.options:
            samples *= self._sampler.options["scale"]
        for idx in range(samples.shape[0]):
            diag = np.diag(samples[idx, ...])
            samples[idx, ...] = np.diag(diag)
        return samples


class MockedSamplerPlugin(SamplerPlugin):
    @classmethod
    def create(
        cls,
        enopt_config: EnOptConfig,
        sampler_index: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> MockedSampler:
        return MockedSampler(enopt_config, sampler_index, mask, rng)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        return method.lower() in {"test"}


def test_sampler_simple(enopt_config: Any) -> None:
    enopt_config["gradient"]["number_of_perturbations"] = 3
    rng = default_rng(123)
    config = EnOptConfig.model_validate(enopt_config)
    sampler = MockedSampler(config, 0, None, rng)

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
    sampler = MockedSampler(config, 0, None, rng)

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
    enopt_config["gradient"]["samplers"] = [0, 1, 1]
    config = EnOptConfig.model_validate(enopt_config)
    sampler1 = MockedSampler(config, 0, np.array([0]), rng)
    sampler2 = MockedSampler(config, 1, np.array([1, 2]), rng)

    perturbed_variables = _perturb_variables(
        config,
        np.array([0.0, 0.0, 0.0]),
        [sampler1, sampler2],
    )
    assert np.allclose(perturbed_variables[0, ...], np.diag([0.01, -0.01, -0.01]))


def test_sampler_order(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["gradient"]["samplers"] = [0, 0, 1]
    enopt_config["samplers"] = [
        {"method": "norm"},
        {"method": "uniform"},
    ]
    variables1 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables1 is not None
    assert np.allclose(variables1, [0, 0, 0.5], atol=0.025)

    # Switch the samplers:
    enopt_config["samplers"] = [
        {"method": "uniform"},
        {"method": "norm"},
    ]
    enopt_config["gradient"]["samplers"] = [1, 1, 0]
    variables2 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables2 is not None
    assert np.allclose(variables2, [0, 0, 0.5], atol=0.025)

    assert np.allclose(variables1, variables2)
