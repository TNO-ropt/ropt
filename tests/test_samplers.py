from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.random import Generator, default_rng

from ropt.config import SamplerConfig
from ropt.context import EnOptContext
from ropt.core._gradient import _perturb_variables
from ropt.sampler import Sampler
from ropt.workflow import BasicOptimizer

if TYPE_CHECKING:
    from numpy.typing import NDArray

initial_values = [0.0, 0.0, 0.0]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "max_functions": 4,
        },
        "backend": {
            "convergence_tolerance": 0.005,
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
        context: EnOptContext,
        mask: NDArray[np.bool_] | None,
        _: Generator,
    ) -> None:
        self._context = context
        self._mask = mask

    def generate_samples(
        self,
    ) -> NDArray[np.float64]:
        assert (
            self._context.gradient.number_of_perturbations
            == self._context.variables.variable_count
        )
        variable_count = self._context.variables.variable_count
        realization_count = self._context.realizations.weights.size
        perturbation_count = self._context.gradient.number_of_perturbations

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


def test_sampler_simple(config: Any) -> None:
    sampler_config = SamplerConfig.model_validate({"method": "test"})
    sampler = MockedSampler(sampler_config)
    config["gradient"]["number_of_perturbations"] = 3
    config["samplers"] = [sampler]
    context = EnOptContext.model_validate(config)
    rng = default_rng(123)
    sampler.init(context, None, rng)
    perturbed_variables = _perturb_variables(
        context,
        np.array([0.0, 0.0, 0.0]),
        (sampler,),
    )
    assert np.allclose(perturbed_variables, 0.01 * np.eye(3))


def test_sampler_use_options(config: Any) -> None:
    sampler_config = SamplerConfig.model_validate(
        {"method": "test", "options": {"scale": 100.0}}
    )
    sampler = MockedSampler(sampler_config)
    config["samplers"] = [sampler]
    context = EnOptContext.model_validate(config)
    rng = default_rng(123)
    sampler.init(context, None, rng)
    perturbed_variables = _perturb_variables(
        context,
        np.array([0.0, 0.0, 0.0]),
        (sampler,),
    )
    assert np.allclose(perturbed_variables, np.eye(3))


def test_sampler_indexed(config: Any) -> None:
    sampler_config = SamplerConfig.model_validate({"method": "test"})
    sampler0 = MockedSampler(sampler_config)
    sampler_config = SamplerConfig.model_validate(
        {"method": "test", "options": {"scale": -1}}
    )
    sampler1 = MockedSampler(sampler_config)

    config["variables"]["samplers"] = [0, 1, 1]
    config["samplers"] = [sampler0, sampler1]
    context = EnOptContext.model_validate(config)

    rng = default_rng(123)
    sampler0.init(context, np.array([0]), rng)
    sampler1.init(context, np.array([1, 2]), rng)

    perturbed_variables = _perturb_variables(
        context,
        np.array([0.0, 0.0, 0.0]),
        (sampler0, sampler1),
    )
    assert np.allclose(perturbed_variables[0, ...], np.diag([0.01, -0.01, -0.01]))


def test_sampler_order(config: Any, evaluator: Any) -> None:
    config["variables"]["samplers"] = [0, 0, 1]
    config["samplers"] = [
        {"method": "norm"},
        {"method": "uniform"},
    ]
    optimizer1 = BasicOptimizer(config, evaluator())
    optimizer1.run(initial_values)
    assert optimizer1.results is not None
    assert np.allclose(
        optimizer1.results.evaluations.variables, [0, 0, 0.5], atol=0.025
    )

    # Switch the samplers:
    config["samplers"] = [
        {"method": "uniform"},
        {"method": "norm"},
    ]
    config["variables"]["samplers"] = [1, 1, 0]
    optimizer2 = BasicOptimizer(config, evaluator())
    optimizer2.run(initial_values)
    assert optimizer2.results is not None
    assert np.allclose(
        optimizer2.results.evaluations.variables, [0, 0, 0.5], atol=0.025
    )

    assert np.allclose(
        optimizer1.results.evaluations.variables,
        optimizer2.results.evaluations.variables,
    )
