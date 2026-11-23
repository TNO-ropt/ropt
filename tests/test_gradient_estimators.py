from typing import Any, ClassVar, override

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from ropt._core import EnsembleEvaluator
from ropt._evaluation import EvaluationFunctionContext
from ropt.config import GradientConfig
from ropt.context import EnOptContext
from ropt.gradient import GradientData, GradientEstimator
from ropt.plugins import MethodSpec, register_plugin
from ropt.results import GradientResults

_STEP = 1e-3


class ForwardDifferences(GradientEstimator):
    methods: ClassVar[MethodSpec] = {"forward"}

    def __init__(self, gradient_config: GradientConfig) -> None:
        self.config = gradient_config
        self.weights: NDArray[np.float64] | None = None

    @override
    def init(self, context: EnOptContext, *, mask: NDArray[np.bool_]) -> None:
        self._realization_count = context.realizations.weights.size
        self._mask = mask

    @override
    def perturbed_variables(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        points = variables + _STEP * np.eye(variables.size)[self._mask]
        return np.repeat(points[np.newaxis], self._realization_count, axis=0)

    @override
    def failed_realizations(self, data: GradientData) -> NDArray[np.bool_]:
        return np.zeros(data.functions.shape[0], dtype=np.bool_)

    @override
    def estimate(
        self, data: GradientData, weights: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        self.weights = weights
        differences = data.perturbed_functions - data.functions[:, np.newaxis, :]
        return np.swapaxes(differences, 1, 2) / _STEP, np.ones_like(weights, bool)


def _linear(variables: NDArray[np.float64], _: EvaluationFunctionContext) -> float:
    return float(np.dot([1.0, 2.0, 3.0], variables))


def _gradient(context: EnOptContext, evaluator: Any) -> GradientResults:
    _, gradient = EnsembleEvaluator(context, evaluator).calculate(
        np.zeros(3), compute_functions=True, compute_gradients=True
    )
    assert isinstance(gradient, GradientResults)
    return gradient


@pytest.mark.parametrize("selection", ["method", "instance"])
def test_gradient_estimator_plugin_estimates_the_masked_variables(
    monkeypatch: Any, evaluator: Any, selection: str
) -> None:
    monkeypatch.setattr("ropt.plugins._manager._plugin_manager", None)
    register_plugin("gradient", "test", ForwardDifferences)
    config = GradientConfig(method="test/forward")
    context = EnOptContext.model_validate(
        {
            "variables": {"variable_count": 3, "mask": [True, False, True]},
            "gradient": (
                config.model_dump()
                if selection == "method"
                else ForwardDifferences(config)
            ),
        }
    )
    assert isinstance(context.gradient, ForwardDifferences)
    assert context.gradient.config.method == "test/forward"

    gradient = _gradient(context, evaluator([_linear]))

    assert gradient.perturbations.variables.shape == (1, 2, 3)
    assert gradient.aggregates is not None
    np.testing.assert_allclose(gradient.aggregates.objectives, [[1.0, 0.0, 3.0]])


def test_gradient_estimator_receives_zero_weight_for_failed_function(
    evaluator: Any,
) -> None:
    estimator = ForwardDifferences(GradientConfig())
    context = EnOptContext.model_validate(
        {
            "variables": {"variable_count": 3},
            "realizations": {"weights": [1.0, 1.0], "realization_min_success": 1},
            "gradient": estimator,
        }
    )

    def fail_second(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        if context.realization == 1 and context.perturbation < 0:
            return np.nan
        return _linear(variables, context)

    _gradient(context, evaluator([fail_second]))

    assert estimator.weights is not None
    np.testing.assert_array_equal(estimator.weights, [[1.0], [0.0]])


def test_stochastic_gradient_svd_threshold_truncates_the_solution(
    evaluator: Any,
) -> None:
    def aggregate(gradient: dict[str, Any]) -> NDArray[np.float64]:
        context = EnOptContext.model_validate(
            {"variables": {"variable_count": 3}, "gradient": gradient}
        )
        aggregates = _gradient(context, evaluator([_linear])).aggregates
        assert aggregates is not None
        return aggregates.objectives

    exact = aggregate({"number_of_perturbations": 3})
    truncated = aggregate(
        {
            "method": "stochastic",
            "number_of_perturbations": 3,
            "options": {"svd_threshold": 0.5},
        }
    )

    np.testing.assert_allclose(exact, [[1.0, 2.0, 3.0]])
    assert not np.allclose(truncated, exact)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"svd_threshold": 0.0}, "greater than 0"),
        ({"svd_threshold": 1.5}, "less than or equal to 1"),
        ({"threshold": 0.5}, "Extra inputs are not permitted"),
    ],
)
def test_stochastic_gradient_rejects_invalid_options(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        EnOptContext.model_validate(
            {"variables": {"variable_count": 3}, "gradient": {"options": options}}
        )
