from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np
import pytest
from numpy.random import Generator, default_rng

from ropt import optimize
from ropt._core._gradient import _perturb_variables
from ropt.config import SamplerConfig
from ropt.context import EnOptContext
from ropt.sampler import Sampler

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

    @override
    def init(
        self,
        *,
        realization_count: int,
        perturbation_count: int,
        variable_count: int,
        mask: NDArray[np.bool_] | None,
        rng: Generator,
    ) -> None:
        self._realization_count = realization_count
        self._perturbation_count = perturbation_count
        self._variable_count = variable_count
        self._mask = mask
        self._rng = rng

    @override
    def generate_samples(
        self,
    ) -> NDArray[np.float64]:
        assert self._perturbation_count == self._variable_count
        variable_count = self._variable_count
        realization_count = self._realization_count
        perturbation_count = self._perturbation_count

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


def _init(
    sampler: Sampler,
    context: EnOptContext,
    mask: NDArray[np.bool_] | None,
    rng: Generator,
) -> None:
    sampler.init(
        realization_count=context.realizations.weights.size,
        perturbation_count=context.gradient.number_of_perturbations,
        variable_count=context.variables.variable_count,
        mask=mask,
        rng=rng,
    )


def test_sampler_simple(config: Any) -> None:
    sampler_config = SamplerConfig.model_validate({"method": "test"})
    sampler = MockedSampler(sampler_config)
    config["gradient"]["number_of_perturbations"] = 3
    config["samplers"] = [sampler]
    context = EnOptContext.model_validate(config)
    rng = default_rng(123)
    _init(sampler, context, None, rng)
    perturbed_variables = _perturb_variables(
        context,
        np.array([0.0, 0.0, 0.0]),
        context.samplers,
    )
    assert np.allclose(perturbed_variables, 0.01 * np.eye(3))


@pytest.mark.parametrize("scales", [(1.0, 1.0, 1.0), (1.0, 500.0, 50.0)])
def test_sampler_applies_absolute_magnitudes_with_variable_scaling(
    scales: tuple[float, float, float],
) -> None:
    sampler = MockedSampler(SamplerConfig.model_validate({"method": "test"}))
    offsets = np.array([0.0, 100.0, -10.0])
    context = EnOptContext.model_validate(
        {
            "variables": {
                "variable_count": 3,
                "lower_bounds": [0.0, 100.0, -np.inf],
                "upper_bounds": [1.0, 600.0, np.inf],
                "perturbation_magnitudes": [0.1, 5.0, 1.0],
                "scales": scales,
                "offsets": offsets,
            },
            "gradient": {"number_of_perturbations": 3},
            "samplers": [sampler],
        }
    )
    _init(sampler, context, None, default_rng(123))
    variables = np.array([0.5, 350.0, 0.0])
    perturbed_variables = _perturb_variables(
        context, (variables - offsets) / scales, context.samplers
    )
    assert perturbed_variables.shape == (1, 3, 3)
    np.testing.assert_allclose(
        perturbed_variables[0] * scales + offsets,
        [[0.6, 350.0, 0.0], [0.5, 355.0, 0.0], [0.5, 350.0, 1.0]],
    )


def test_sampler_use_options(config: Any) -> None:
    sampler_config = SamplerConfig.model_validate(
        {"method": "test", "options": {"scale": 100.0}}
    )
    sampler = MockedSampler(sampler_config)
    config["samplers"] = [sampler]
    context = EnOptContext.model_validate(config)
    rng = default_rng(123)
    _init(sampler, context, None, rng)
    perturbed_variables = _perturb_variables(
        context,
        np.array([0.0, 0.0, 0.0]),
        context.samplers,
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
    _init(sampler0, context, np.array([0]), rng)
    _init(sampler1, context, np.array([1, 2]), rng)

    perturbed_variables = _perturb_variables(
        context,
        np.array([0.0, 0.0, 0.0]),
        context.samplers,
    )
    assert np.allclose(perturbed_variables[0, ...], np.diag([0.01, -0.01, -0.01]))


def test_sampler_order(config: Any, eval_func: Any) -> None:
    config["variables"]["samplers"] = [0, 0, 1]
    config["samplers"] = [
        {"method": "norm"},
        {"method": "uniform"},
    ]
    result1 = optimize(config, initial_values, eval_func())
    assert result1.results is not None
    assert np.allclose(result1.results.variables, [0, 0, 0.5], atol=0.025)

    # Switch the samplers:
    config["samplers"] = [
        {"method": "uniform"},
        {"method": "norm"},
    ]
    config["variables"]["samplers"] = [1, 1, 0]
    result2 = optimize(config, initial_values, eval_func())
    assert result2.results is not None
    assert np.allclose(result2.results.variables, [0, 0, 0.5], atol=0.025)

    assert np.allclose(result1.results.variables, result2.results.variables)
