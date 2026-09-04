"""Tests for the scaling and offsetting of variables.

Variables reach the optimizer as $y = (x - o)/s$ and are reported as
$x = s\\,y + o$. Both directions come from the same two arrays, so they cannot
drift apart. Linear constraints follow the variables through a change of
variables, and may then be scaled per equation.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.config import VariablesConfig
from ropt.context import EnOptContext
from ropt.enums import PerturbationType
from ropt.results import (
    ConstraintInfo,
    FunctionEvaluations,
    FunctionResults,
    GradientEvaluations,
    GradientResults,
    Gradients,
    Realizations,
)
from ropt.simple import (
    EvaluationFunctionContext,
    HistoryHandler,
    ResultsHandler,
    evaluate,
    optimize,
)
from ropt.utils import scales_and_offsets_from_bounds


def _context(**fields: Any) -> EnOptContext:
    return EnOptContext.model_validate({"variables": {"variable_count": 3}} | fields)


def _function_results(
    variables: NDArray[np.float64],
    constraint_info: ConstraintInfo | None = None,
) -> FunctionResults:
    return FunctionResults(
        batch_id=0,
        metadata={},
        names={},
        evaluations=FunctionEvaluations(
            variables=variables,
            objectives=np.zeros((1, 2)),
            constraints=None,
        ),
        realizations=Realizations(evaluated_realizations=np.array([True])),
        functions=None,
        constraint_info=constraint_info,
    )


def _linear_constraints(**fields: Any) -> Any:
    context = _context(
        linear_constraints={
            "coefficients": [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": [1.0, 1.0],
        }
        | fields.pop("linear_constraints", {}),
        **fields,
    )
    assert context.linear_constraints is not None
    return context.linear_constraints


def test_scales_and_offsets_default_to_the_identity() -> None:
    variables = _context().variables
    assert np.allclose(variables.scales, [1.0, 1.0, 1.0])
    assert np.allclose(variables.offsets, [0.0, 0.0, 0.0])


def test_the_bounds_are_scaled() -> None:
    context = _context(
        variables={
            "variable_count": 3,
            "lower_bounds": [0.0, 100.0, -1.0],
            "upper_bounds": [1.0, 600.0, 1.0],
            "scales": [1.0, 500.0, 2.0],
            "offsets": [0.0, 100.0, 1.0],
        }
    )
    assert np.allclose(context.variables.lower_bounds, [0.0, 0.0, -1.0])
    assert np.allclose(context.variables.upper_bounds, [1.0, 1.0, 0.0])


def test_infinite_bounds_survive_the_map() -> None:
    context = _context(
        variables={
            "variable_count": 3,
            "lower_bounds": [-np.inf, 0.0, -np.inf],
            "upper_bounds": [np.inf, np.inf, 0.0],
            "scales": [2.0, 4.0, 8.0],
            "offsets": [1.0, 2.0, 3.0],
        }
    )
    assert np.allclose(context.variables.lower_bounds, [-np.inf, -0.5, -np.inf])
    assert np.allclose(context.variables.upper_bounds, [np.inf, np.inf, -0.375])


def test_absolute_perturbation_magnitudes_are_scaled() -> None:
    context = _context(
        variables={
            "variable_count": 3,
            "perturbation_magnitudes": [0.1, 0.2, 0.4],
            "perturbation_types": [PerturbationType.ABSOLUTE] * 3,
            "scales": [1.0, 2.0, 4.0],
            "offsets": [10.0, 20.0, 40.0],
        }
    )
    # An offset is a shift, and a magnitude is a distance: only the scale acts.
    assert np.allclose(context.variables.perturbation_magnitudes, [0.1, 0.1, 0.1])


def test_reported_variables_are_mapped_back() -> None:
    scales = np.array([2.0, 4.0, 8.0])
    offsets = np.array([1.0, 2.0, 3.0])
    context = _context(
        variables={"variable_count": 3, "scales": scales, "offsets": offsets}
    )
    user_variables = np.array([0.5, 1.5, 2.5])
    optimizer_variables = (user_variables - offsets) / scales

    unscaled = _function_results(optimizer_variables).unscale(context)
    assert np.allclose(unscaled.evaluations.variables, user_variables)


def test_perturbed_variables_are_mapped_back() -> None:
    context = _context(
        variables={"variable_count": 3, "scales": [2.0, 4.0, 8.0], "offsets": 1.0}
    )
    user_variables = np.array([2.0, 7.0, 21.0])
    results = GradientResults(
        batch_id=0,
        metadata={},
        names={},
        evaluations=GradientEvaluations(
            variables=np.array([0.5, 1.5, 2.5]),
            perturbed_variables=np.array([[[0.5, 1.5, 2.5]]]),
            perturbed_objectives=np.zeros((1, 1, 2)),
            metadata={},
        ),
        realizations=Realizations(evaluated_realizations=np.array([True])),
        gradients=None,
    )
    unscaled = results.unscale(context)
    assert np.allclose(unscaled.evaluations.variables, user_variables)
    assert np.allclose(unscaled.evaluations.perturbed_variables, [[user_variables]])


def test_gradients_are_divided_by_the_variable_scales() -> None:
    context = _context(
        variables={"variable_count": 2, "scales": [2.0, 4.0], "offsets": 1.0},
        objectives={"weights": [1.0], "scales": [5.0]},
        nonlinear_constraints={
            "lower_bounds": [0.0],
            "upper_bounds": [1.0],
            "scales": [10.0],
        },
    )
    # A variable sits in the denominator of a derivative, so the optimizer holds
    # d(f / s_f) / d((x - o) / s_x): the true gradient times s_x / s_f.
    true_gradient = np.array([[3.0, 7.0]])
    results = GradientResults(
        batch_id=0,
        metadata={},
        names={},
        evaluations=GradientEvaluations(
            variables=np.zeros(2),
            perturbed_variables=np.zeros((1, 1, 2)),
            perturbed_objectives=np.zeros((1, 1, 1)),
            metadata={},
        ),
        realizations=Realizations(evaluated_realizations=np.array([True])),
        gradients=Gradients(
            target_objective=np.zeros(2),
            objectives=true_gradient * [2.0, 4.0] / 5.0,
            constraints=true_gradient * [2.0, 4.0] / 10.0,
        ),
    )
    unscaled = results.unscale(context)
    assert unscaled.gradients is not None
    assert np.allclose(unscaled.gradients.objectives, true_gradient)
    assert unscaled.gradients.constraints is not None
    assert np.allclose(unscaled.gradients.constraints, true_gradient)


def test_bound_constraint_diffs_are_scaled_back() -> None:
    context = _context(
        variables={"variable_count": 3, "scales": [2.0, 4.0, 8.0], "offsets": 100.0}
    )
    unscaled = _function_results(
        np.zeros(3),
        constraint_info=ConstraintInfo(
            bound_lower=np.array([0.25, 0.5, 0.75]),
            bound_upper=np.array([-0.25, -0.5, -0.75]),
        ),
    ).unscale(context)

    # A difference is a distance between two values, so the offset cancels.
    assert unscaled.constraint_info is not None
    assert unscaled.constraint_info.bound_lower is not None
    assert np.allclose(unscaled.constraint_info.bound_lower, [0.5, 2.0, 6.0])
    assert unscaled.constraint_info.bound_upper is not None
    assert np.allclose(unscaled.constraint_info.bound_upper, [-0.5, -2.0, -6.0])


def test_linear_constraint_diffs_are_scaled_back() -> None:
    context = _context(
        linear_constraints={
            "coefficients": [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": [1.0, 1.0],
            "scales": [2.0, 5.0],
        }
    )
    unscaled = _function_results(
        np.zeros(3),
        constraint_info=ConstraintInfo(
            linear_lower=np.array([0.25, 0.5]),
            linear_upper=np.array([-0.25, -0.5]),
        ),
    ).unscale(context)

    assert unscaled.constraint_info is not None
    assert unscaled.constraint_info.linear_lower is not None
    assert np.allclose(unscaled.constraint_info.linear_lower, [0.5, 2.5])
    assert unscaled.constraint_info.linear_upper is not None
    assert np.allclose(unscaled.constraint_info.linear_upper, [-0.5, -2.5])


def test_the_estimated_equation_scales_are_undone_in_the_diffs() -> None:
    context = _context(
        linear_constraints={
            "coefficients": [[10.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": [10.0, 1.0],
            "auto_scale": True,
        }
    )
    assert context.linear_constraints is not None
    assert np.allclose(context.linear_constraints.scales, [10.0, 4.0])

    unscaled = _function_results(
        np.zeros(3),
        constraint_info=ConstraintInfo(
            linear_lower=np.array([1.0, 1.0]),
            linear_upper=np.array([-1.0, -1.0]),
        ),
    ).unscale(context)

    # An estimated scale is undone just like a configured one; leaving it in
    # would report distances in the optimizer's units.
    assert unscaled.constraint_info is not None
    assert unscaled.constraint_info.linear_lower is not None
    assert np.allclose(unscaled.constraint_info.linear_lower, [10.0, 4.0])
    assert unscaled.constraint_info.linear_upper is not None
    assert np.allclose(unscaled.constraint_info.linear_upper, [-10.0, -4.0])


def test_the_evaluator_is_called_with_unscaled_variables() -> None:
    seen: list[NDArray[np.float64]] = []

    def objective(
        variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        seen.append(np.asarray(variables, dtype=np.float64).copy())
        return 0.0

    user_variables = np.array([0.5, 1.5, 2.5])
    evaluate(
        {
            "variables": {
                "variable_count": 3,
                "scales": [2.0, 4.0, 8.0],
                "offsets": [1.0, 2.0, 3.0],
            }
        },
        user_variables,
        objective,
    )
    assert seen
    assert np.allclose(seen[0], user_variables)


def test_fixed_variables_are_scaled_too() -> None:
    variables = _context(
        variables={
            "variable_count": 3,
            "lower_bounds": 0.0,
            "upper_bounds": 10.0,
            "mask": [True, False, True],
            "scales": [2.0, 4.0, 8.0],
            "offsets": [1.0, 2.0, 3.0],
        }
    ).variables
    # The map is not restricted to the free variables, so the fixed one is
    # scaled along with the others.
    assert np.allclose(variables.lower_bounds, [-0.5, -0.5, -0.375])
    assert np.allclose(variables.upper_bounds, [4.5, 2.0, 0.875])


def test_a_fixed_variable_reaches_the_evaluator_unchanged() -> None:
    seen: list[NDArray[np.float64]] = []

    def objective(
        variables: NDArray[np.float64], _context: EvaluationFunctionContext
    ) -> float:
        seen.append(np.asarray(variables, dtype=np.float64).copy())
        return 0.0

    user_variables = np.array([0.5, 1.5, 2.5])
    evaluate(
        {
            "variables": {
                "variable_count": 3,
                "mask": [True, False, True],
                "scales": [2.0, 4.0, 8.0],
                "offsets": [1.0, 2.0, 3.0],
            }
        },
        user_variables,
        objective,
    )
    # Scaling a fixed variable is invisible once unscaled: the map and
    # its inverse cancel, whether or not the variable is free.
    assert seen
    assert np.allclose(seen[0], user_variables)


def test_the_linear_constraints_follow_a_change_of_variables() -> None:
    constraints = _linear_constraints(
        variables={
            "variable_count": 3,
            "scales": [2.0, 4.0, 8.0],
            "offsets": [1.0, 2.0, 3.0],
        }
    )
    # A' = A diag(s), b' = b - A o.
    assert np.allclose(constraints.coefficients, [[2.0, 0.0, 0.0], [0.0, 4.0, 8.0]])
    assert np.allclose(constraints.lower_bounds, [-1.0, -5.0])
    assert np.allclose(constraints.upper_bounds, [0.0, -4.0])


def test_the_change_of_variables_preserves_the_distance_to_a_bound() -> None:
    coefficients = np.array([[1.0, 2.0, 3.0]])
    user_point = np.array([0.5, 1.5, 2.5])
    scales = np.array([2.0, 4.0, 8.0])
    offsets = np.array([1.0, 2.0, 3.0])

    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": coefficients,
            "lower_bounds": [0.0],
            "upper_bounds": [10.0],
        },
        variables={"variable_count": 3, "scales": scales, "offsets": offsets},
    )
    optimizer_point = (user_point - offsets) / scales

    # A change of variables is not a rescaling: the equation value shifts by
    # `A o` and so does every bound, leaving the distance between them alone.
    assert np.allclose(
        coefficients @ user_point - np.array([0.0]),
        constraints.coefficients @ optimizer_point - constraints.lower_bounds,
    )
    assert np.allclose(
        coefficients @ user_point - np.array([10.0]),
        constraints.coefficients @ optimizer_point - constraints.upper_bounds,
    )


def test_the_equations_are_scaled_by_the_configured_scales() -> None:
    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            "lower_bounds": [0.0, 2.0],
            "upper_bounds": [1.0, 4.0],
            "scales": [2.0, 4.0],
        }
    )
    assert np.allclose(constraints.coefficients, [[0.5, 0.0, 0.0], [0.0, 0.25, 0.25]])
    assert np.allclose(constraints.lower_bounds, [0.0, 0.5])
    assert np.allclose(constraints.upper_bounds, [0.5, 1.0])


def test_auto_scale_normalizes_the_largest_coefficient() -> None:
    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": [[10.0, -20.0, 0.0], [0.0, 1.0, 4.0]],
            "lower_bounds": [-np.inf, -np.inf],
            "upper_bounds": [0.0, 0.0],
            "auto_scale": True,
        }
    )
    assert np.allclose(constraints.scales, [20.0, 4.0])
    assert np.allclose(constraints.coefficients, [[0.5, -1.0, 0.0], [0.0, 0.25, 1.0]])


def test_auto_scale_takes_the_bounds_into_account() -> None:
    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "lower_bounds": [-100.0, 0.0],
            "upper_bounds": [0.0, 50.0],
            "auto_scale": True,
        }
    )
    assert np.allclose(constraints.scales, [100.0, 50.0])
    assert np.allclose(constraints.lower_bounds, [-1.0, 0.0])
    assert np.allclose(constraints.upper_bounds, [0.0, 1.0])


def test_auto_scale_ignores_infinite_bounds() -> None:
    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": [[2.0, 0.0, 0.0]],
            "lower_bounds": [-np.inf],
            "upper_bounds": [np.inf],
            "auto_scale": True,
        }
    )
    assert np.allclose(constraints.scales, [2.0])


def test_auto_scale_ignores_fixed_columns() -> None:
    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": [[1.0, 100.0, 0.0]],
            "lower_bounds": [0.0],
            "upper_bounds": [0.0],
            "auto_scale": True,
        },
        variables={"variable_count": 3, "mask": [True, False, True]},
    )
    # The second column is eliminated before the optimizer sees the problem, so
    # it must not inflate the scale.
    assert np.allclose(constraints.scales, [1.0])


def test_auto_scale_clamps_an_empty_equation() -> None:
    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": [[0.0, 0.0, 0.0]],
            "lower_bounds": [0.0],
            "upper_bounds": [0.0],
            "auto_scale": True,
        },
    )
    # Dividing by an estimate of zero would turn the row into NaN.
    assert np.allclose(constraints.scales, [1.0])
    assert np.allclose(constraints.coefficients, [[0.0, 0.0, 0.0]])


def test_auto_scale_composes_with_the_configured_scales() -> None:
    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": [[10.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            "lower_bounds": [0.0, 0.0],
            "upper_bounds": [1.0, 1.0],
            "scales": [2.0, 0.5],
            "auto_scale": True,
        }
    )
    assert np.allclose(constraints.scales, [20.0, 2.0])


def test_the_equations_are_scaled_after_the_change_of_variables() -> None:
    constraints = _linear_constraints(
        linear_constraints={
            "coefficients": [[1.0, 0.0, 0.0]],
            "lower_bounds": [0.0],
            "upper_bounds": [np.inf],
            "auto_scale": True,
        },
        variables={"variable_count": 3, "scales": [4.0, 1.0, 1.0]},
    )
    # The estimate is 4, the scaled coefficient, not 1, the original one.
    assert np.allclose(constraints.scales, [4.0])
    assert np.allclose(constraints.coefficients, [[1.0, 0.0, 0.0]])


def test_the_equation_scales_default_to_one() -> None:
    constraints = _linear_constraints()
    assert np.allclose(constraints.scales, [1.0, 1.0])
    assert np.allclose(constraints.coefficients, [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])


@pytest.mark.parametrize(
    ("target_range", "expected_scales", "expected_offsets"),
    [
        ((0.0, 1.0), [1.0, 500.0], [0.0, 100.0]),
        ((-1.0, 1.0), [0.5, 250.0], [0.5, 350.0]),
        ((0.0, 2.0), [0.5, 250.0], [0.0, 100.0]),
    ],
)
def test_scales_and_offsets_map_the_bounds_onto_the_target_range(
    target_range: tuple[float, float],
    expected_scales: list[float],
    expected_offsets: list[float],
) -> None:
    lower_bounds = np.array([0.0, 100.0])
    upper_bounds = np.array([1.0, 600.0])
    scales, offsets = scales_and_offsets_from_bounds(
        lower_bounds, upper_bounds, target_range
    )
    assert np.allclose(scales, expected_scales)
    assert np.allclose(offsets, expected_offsets)

    config = VariablesConfig.model_validate(
        {
            "variable_count": 2,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "scales": scales,
            "offsets": offsets,
        }
    )
    context = EnOptContext.model_validate({"variables": config})
    assert np.allclose(context.variables.lower_bounds, target_range[0])
    assert np.allclose(context.variables.upper_bounds, target_range[1])


def test_scales_and_offsets_broadcast_a_shared_bound() -> None:
    scales, offsets = scales_and_offsets_from_bounds(0.0, [1.0, 2.0])
    assert np.allclose(scales, [1.0, 2.0])
    assert np.allclose(offsets, [0.0, 0.0])


@pytest.mark.parametrize(
    ("bounds", "target_range", "message"),
    [
        (([0.0], [np.inf]), (0.0, 1.0), "The variable bounds must be finite."),
        (([-np.inf], [0.0]), (0.0, 1.0), "The variable bounds must be finite."),
        (
            ([1.0], [1.0]),
            (0.0, 1.0),
            "The variable bounds must define a non-empty range.",
        ),
        (([0.0], [1.0]), (1.0, 1.0), "The target range must be non-empty."),
    ],
)
def test_scales_and_offsets_reject_a_degenerate_range(
    bounds: tuple[list[float], list[float]],
    target_range: tuple[float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        scales_and_offsets_from_bounds(*bounds, target_range)


# The `scaled` flag on the result-collecting handlers selects whether the
# values are unscaled before being stored. Variable scales and offsets make the
# difference visible: the optimizer works with (x - o)/s.

_SCALED_CONFIG: dict[str, Any] = {
    "variables": {"variable_count": 2, "scales": [2.0, 4.0], "offsets": [1.0, 2.0]},
    "objectives": {"weights": [1.0]},
    "realizations": {"weights": [1.0]},
    "optimizer": {"max_functions": 1},
}


def _objective(
    variables: NDArray[np.float64], _context: EvaluationFunctionContext
) -> float:
    return float(np.sum(variables**2))


def _initial() -> NDArray[np.float64]:
    return np.array([3.0, 6.0])


@pytest.mark.parametrize("scaled", [False, True])
def test_the_history_handler_stores_scaled_values_on_request(*, scaled: bool) -> None:
    history = HistoryHandler(scaled=scaled)
    optimize(_SCALED_CONFIG, _initial(), _objective, handlers=[history])
    first = history.results[0]
    assert isinstance(first, FunctionResults)
    variables = first.evaluations.variables
    # (3, 6) is (1, 1) once scaled, since scales are (2, 4) and offsets (1, 2).
    expected = [1.0, 1.0] if scaled else [3.0, 6.0]
    assert np.allclose(variables, expected)


def test_the_history_handler_unscales_by_default() -> None:
    default = HistoryHandler()
    explicit = HistoryHandler(scaled=False)
    optimize(_SCALED_CONFIG, _initial(), _objective, handlers=[default, explicit])
    first, second = default.results[0], explicit.results[0]
    assert isinstance(first, FunctionResults)
    assert isinstance(second, FunctionResults)
    assert np.allclose(first.evaluations.variables, second.evaluations.variables)


@pytest.mark.parametrize("scaled", [False, True])
def test_the_results_handler_stores_scaled_values_on_request(*, scaled: bool) -> None:
    handler = ResultsHandler(scaled=scaled)
    optimize(_SCALED_CONFIG, _initial(), _objective, handlers=[handler])
    assert handler.result is not None
    variables = handler.result.evaluations.variables
    unscaled = variables * [2.0, 4.0] + [1.0, 2.0]
    # Whichever way it is stored, unscaling recovers the reported values.
    reference = ResultsHandler()
    optimize(_SCALED_CONFIG, _initial(), _objective, handlers=[reference])
    assert reference.result is not None
    assert np.allclose(
        variables if not scaled else unscaled,
        reference.result.evaluations.variables,
    )
