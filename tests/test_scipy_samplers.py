from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.stats.qmc import Halton, LatinHypercube, Sobol

from ropt import optimize
from ropt._handlers import CallbackHandler
from ropt.config import SamplerConfig
from ropt.enums import EnOptEventType
from ropt.events import EnOptEvent
from ropt.results import GradientResults
from ropt.sampler.scipy import SCIPY_SAMPLER_SUPPORTED_METHODS, SciPySampler

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
    config["sampler"] = [{"method": method}]
    result = optimize(config, initial_values, eval_func())
    assert result.results is not None
    assert np.allclose(result.results.variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_scipy_sampler_draw_size_can_change(*, shared: bool, masked: bool) -> None:
    mask = np.array([True, not masked, True])
    rng = np.random.default_rng(123)
    sampler = SciPySampler(SamplerConfig(method="norm", shared=shared))
    sampler.init(
        variable_count=3,
        mask=mask if masked else None,
        rng=np.random.default_rng(123),
    )

    for realization_count, perturbation_count in [(2, 4), (3, 2)]:
        expected = rng.standard_normal(
            (1 if shared else realization_count, perturbation_count, int(mask.sum()))
        )
        if shared:
            expected = np.repeat(expected, realization_count, axis=0)
        samples = sampler.generate_samples(
            realization_count=realization_count, perturbation_count=perturbation_count
        )

        assert samples.shape == (realization_count, perturbation_count, 3)
        np.testing.assert_array_equal(samples[..., mask], expected)
        if masked:
            np.testing.assert_array_equal(samples[..., ~mask], 0)


@pytest.mark.parametrize("method", ["sobol", "halton", "lhs"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_scipy_qmc_sampler_preserves_points(
    method: str, *, shared: bool, masked: bool
) -> None:
    engines = {"sobol": Sobol, "halton": Halton, "lhs": LatinHypercube}
    mask = np.array([True, not masked, True])
    sample_dim = int(mask.sum())
    engine = engines[method](sample_dim, seed=np.random.default_rng(123))

    sampler = SciPySampler(SamplerConfig(method=method, shared=shared))
    sampler.init(
        variable_count=3,
        mask=mask if masked else None,
        rng=np.random.default_rng(123),
    )
    for realization_count, perturbation_count in [(2, 4), (3, 2)]:
        count = perturbation_count * (1 if shared else realization_count)
        expected = 2 * engine.random(count) - 1
        if shared:
            expected = np.tile(expected, (realization_count, 1))
        samples = sampler.generate_samples(
            realization_count=realization_count, perturbation_count=perturbation_count
        )

        assert samples.shape == (realization_count, perturbation_count, 3)
        np.testing.assert_array_equal(
            samples[..., mask].reshape(-1, sample_dim), expected
        )
        if masked:
            np.testing.assert_array_equal(samples[..., ~mask], 0)


@pytest.mark.parametrize("method", ["sobol", "halton", "lhs"])
def test_scipy_qmc_sampler_restarts_on_init(method: str) -> None:
    sampler = SciPySampler(SamplerConfig(method=method))

    samples = []
    for _ in range(2):
        sampler.init(variable_count=3, mask=None, rng=np.random.default_rng(123))
        samples.append(
            sampler.generate_samples(realization_count=1, perturbation_count=4)
        )

    np.testing.assert_array_equal(samples[1], samples[0])


@pytest.mark.parametrize("method", ["sobol", "halton", "lhs"])
def test_scipy_qmc_sampler_reinitializes_the_dimension(method: str) -> None:
    config = SamplerConfig(method=method)
    sampler = SciPySampler(config)
    sampler.init(variable_count=3, mask=None, rng=np.random.default_rng(123))
    sampler.generate_samples(realization_count=1, perturbation_count=4)

    mask = np.array([True, False, True, False])
    samples = []
    for instance in (sampler, SciPySampler(config)):
        instance.init(variable_count=4, mask=mask, rng=np.random.default_rng(456))
        samples.append(
            instance.generate_samples(realization_count=1, perturbation_count=4)
        )

    assert samples[0].shape == (1, 4, 4)
    np.testing.assert_array_equal(samples[0][..., ~mask], 0)
    np.testing.assert_array_equal(samples[0], samples[1])


def test_scipy_indexed_sampler(config: Any, eval_func: Any) -> None:
    # Removing the second variable will fix its value, since it will not be
    # perturbed and its gradient will always be zero.
    config["variables"]["samplers"] = [0, None, 0]

    initial = initial_values.copy()
    initial[1] = 0.1

    result = optimize(config, initial, eval_func())
    assert result.results is not None
    assert pytest.approx(result.results.variables[0]) != 0.0
    assert pytest.approx(result.results.variables[1]) == 0.1
    assert pytest.approx(result.results.variables[2]) != 0.5


@pytest.mark.parametrize("method", sorted(SCIPY_SAMPLER_SUPPORTED_METHODS))
def test_scipy_samplers_shared(config: Any, method: str, eval_func: Any) -> None:
    config["realizations"] = {"weights": [1.0, 1.0]}
    config["sampler"] = [{"method": method}]

    perturbations: dict[str, NDArray[np.float64]] = {}

    def _observer(event: EnOptEvent, tag: str) -> None:
        for item in event.results:
            if isinstance(item, GradientResults) and tag not in perturbations:
                perturbations[tag] = item.perturbations.variables

    config["sampler"][0]["shared"] = False
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
    assert result1.results is not None

    config["sampler"][0]["shared"] = True
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
    assert result2.results is not None

    # The perturbations of the two realizations must differ, if not shared:
    assert not np.allclose(
        perturbations["result1"][0, ...], perturbations["result1"][1, ...]
    )

    # The perturbations of the two realizations must be the same, if shared:
    assert np.allclose(
        perturbations["result2"][0, ...], perturbations["result2"][1, ...]
    )

    # The results should be correct, but slightly different:
    assert np.allclose(result1.results.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(result2.results.variables, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.allclose(
        result1.results.variables, result2.results.variables, atol=1e-3
    )
