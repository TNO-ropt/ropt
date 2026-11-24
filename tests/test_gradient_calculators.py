from typing import Any, ClassVar, override

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from ropt import optimize
from ropt._core import EnsembleEvaluator
from ropt._evaluation import EvaluationFunctionContext
from ropt.config import GradientConfig
from ropt.context import EnOptContext
from ropt.gradient import GradientCalculator, GradientData
from ropt.plugins import MethodSpec, register_plugin
from ropt.results import GradientResults

_STEP = 1e-3


class ForwardDifferences(GradientCalculator):
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
    def calculate(
        self, data: GradientData, weights: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        self.weights = weights
        differences = data.perturbed_functions - data.functions[:, np.newaxis, :]
        return np.swapaxes(differences, 1, 2) / _STEP, np.ones_like(weights, bool)


def _linear(variables: NDArray[np.float64], _: EvaluationFunctionContext) -> float:
    return float(np.dot([1.0, 2.0, 3.0], variables))


def _quadratic(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    return _linear(variables, context) + float(np.sum(variables**2))


def _gradient(
    context: EnOptContext,
    evaluator: Any,
    variables: NDArray[np.float64] | None = None,
    g0: GradientResults | None = None,
) -> GradientResults:
    _, gradient = EnsembleEvaluator(context, evaluator, g0=g0).calculate(
        np.zeros(3) if variables is None else variables,
        compute_functions=True,
        compute_gradients=True,
    )
    assert isinstance(gradient, GradientResults)
    return gradient


@pytest.mark.parametrize("selection", ["method", "instance"])
def test_gradient_calculator_plugin_estimates_the_masked_variables(
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


def test_gradient_calculator_receives_zero_weight_for_failed_function(
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


@pytest.mark.parametrize(
    ("method", "perturbation_count"),
    [("forward", 2), ("backward", 2), ("central", 4)],
)
def test_finite_difference_gradient_is_exact_for_linear_functions(
    evaluator: Any, method: str, perturbation_count: int
) -> None:
    context = EnOptContext.model_validate(
        {
            "variables": {
                "variable_count": 3,
                "mask": [True, False, True],
                "scales": [2.0, 1.0, 0.5],
            },
            "gradient": {"method": method},
        }
    )

    gradient = _gradient(context, evaluator([_linear]))

    assert gradient.perturbations.variables.shape == (1, perturbation_count, 3)
    assert gradient.aggregates is not None
    np.testing.assert_allclose(gradient.aggregates.objectives, [[1.0, 0.0, 3.0]])


def test_central_difference_gradient_is_exact_for_quadratic_functions(
    evaluator: Any,
) -> None:
    context = EnOptContext.model_validate(
        {
            "variables": {"variable_count": 3, "perturbation_magnitudes": 0.1},
            "nonlinear_constraints": {"lower_bounds": -np.inf, "upper_bounds": 0.0},
            "gradient": {"method": "central"},
        }
    )

    gradient = _gradient(
        context, evaluator([_quadratic], [_quadratic]), variables=np.ones(3)
    )

    assert gradient.aggregates is not None
    assert gradient.aggregates.constraints is not None
    np.testing.assert_allclose(gradient.aggregates.objectives, [[3.0, 4.0, 5.0]])
    np.testing.assert_allclose(gradient.aggregates.constraints, [[3.0, 4.0, 5.0]])


@pytest.mark.parametrize("method", ["forward", "backward", "central"])
def test_finite_difference_gradient_finds_the_optimum(
    eval_func: Any, method: str
) -> None:
    result = optimize(
        {
            "variables": {"variable_count": 3, "perturbation_magnitudes": 1e-4},
            "backend": {"convergence_tolerance": 1e-6, "max_iterations": 25},
            "gradient": {"method": method},
        },
        [0.0, 0.0, 0.1],
        eval_func([_quadratic]),
    )

    assert result.results is not None
    np.testing.assert_allclose(result.results.variables, [-0.5, -1.0, -1.5], atol=1e-3)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (
            "forward",
            [[0.1, 0.01, 0.1], [0.0, 0.05, 0.1], [0.0, 0.01, 0.0]],
        ),
        (
            "backward",
            [[0.1, 0.01, 0.1], [0.0, 0.05, 0.1], [0.0, 0.01, 0.0]],
        ),
        (
            "central",
            [[0.1, 0.01, 0.1], [0.0, 0.05, 0.1], [0.0, 0.01, 0.0]],
        ),
    ],
)
def test_finite_difference_points_stay_within_the_bounds(
    evaluator: Any, method: str, expected: list[list[float]]
) -> None:
    context = EnOptContext.model_validate(
        {
            "variables": {
                "variable_count": 3,
                "lower_bounds": 0.0,
                "upper_bounds": [1.0, 0.05, 0.12],
                "perturbation_magnitudes": 0.1,
                "boundary_types": "mirror",
            },
            "gradient": {"method": method},
        }
    )

    gradient = _gradient(
        context, evaluator([_linear]), variables=np.array([0.0, 0.01, 0.1])
    )

    np.testing.assert_allclose(gradient.perturbations.variables[0], expected)
    assert gradient.aggregates is not None
    np.testing.assert_allclose(gradient.aggregates.objectives, [[1.0, 2.0, 3.0]])


def test_finite_difference_central_uses_one_point_where_the_step_does_not_fit(
    evaluator: Any,
) -> None:
    context = EnOptContext.model_validate(
        {
            "variables": {
                "variable_count": 3,
                "lower_bounds": 0.0,
                "upper_bounds": 1.0,
                "perturbation_magnitudes": 0.1,
            },
            "gradient": {"method": "central"},
        }
    )
    variables = np.array([0.5, 0.0, 1.0])

    gradient = _gradient(context, evaluator([_linear]), variables=variables)

    np.testing.assert_allclose(
        gradient.perturbations.variables[0],
        [[0.4, 0.0, 1.0], [0.6, 0.0, 1.0], [0.5, 0.1, 1.0], [0.5, 0.0, 0.9]],
    )
    assert gradient.aggregates is not None
    np.testing.assert_allclose(gradient.aggregates.objectives, [[1.0, 2.0, 3.0]])


def test_finite_difference_failed_perturbation_fails_the_realization(
    evaluator: Any,
) -> None:
    context = EnOptContext.model_validate(
        {
            "variables": {"variable_count": 3},
            "realizations": {"weights": [1.0, 1.0], "realization_min_success": 1},
            "gradient": {"method": "forward"},
        }
    )

    def fail_one_perturbation(
        variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        if context.realization == 1 and context.perturbation == 0:
            return np.nan
        return _linear(variables, context)

    gradient = _gradient(context, evaluator([fail_one_perturbation]))

    assert gradient.estimates is not None
    np.testing.assert_allclose(gradient.estimates.objectives[0], [[1.0, 2.0, 3.0]])
    assert np.all(np.isnan(gradient.estimates.objectives[1]))
    assert gradient.aggregates is not None
    np.testing.assert_allclose(gradient.aggregates.objectives, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize(
    ("gradient", "message"),
    [
        ({"options": {"step": 0.1}}, "has no options"),
        ({"number_of_perturbations": 5}, "does not use: number_of_perturbations"),
        ({"perturbation_min_success": 1}, "does not use: perturbation_min_success"),
        ({"merge_realizations": False}, "does not use: merge_realizations"),
    ],
)
def test_finite_difference_rejects_stochastic_settings(
    gradient: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        EnOptContext.model_validate(
            {
                "variables": {"variable_count": 3},
                "gradient": {"method": "central", **gradient},
            }
        )


@pytest.mark.parametrize(
    ("g0_method", "error"), [("central", False), ("stochastic", True)]
)
def test_finite_difference_checks_initial_gradient_points(
    evaluator: Any, g0_method: str, *, error: bool
) -> None:
    def context(method: str) -> EnOptContext:
        return EnOptContext.model_validate(
            {
                "variables": {
                    "variable_count": 3,
                    "scales": [3.0, 7.0, 11.0],
                    "offsets": [1e6, -2e5, 3e4],
                },
                "gradient": {"method": method},
            }
        )

    variables = np.array([0.3, 0.6, 0.9])
    g0 = _gradient(context(g0_method), evaluator([_linear]), variables=variables)

    if error:
        with pytest.raises(ValueError, match="finite-difference pattern"):
            _gradient(context("central"), evaluator([_linear]), variables, g0)
        return
    gradient = _gradient(context("central"), evaluator([_linear]), variables, g0)
    assert gradient.aggregates is not None
    np.testing.assert_allclose(gradient.aggregates.objectives, [[1.0, 2.0, 3.0]])
