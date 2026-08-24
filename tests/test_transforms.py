"""Tests for chained transforms.

Transforms are configured as an ordered chain applied to all elements: the
forward direction runs the chain in order, the inverse direction runs it in
reverse. Every chain here uses affine transforms with *both* a scale and a
shift, because pure scaling commutes and would make the order assertions
vacuous. The `..._do_not_commute` tests guard that property.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from ropt.config import (
    NonlinearConstraintTransformConfig,
    ObjectiveTransformConfig,
    VariableTransformConfig,
)
from ropt.context import EnOptContext
from ropt.enums import PerturbationType
from ropt.results import (
    ConstraintInfo,
    FunctionEvaluations,
    FunctionResults,
    Realizations,
)
from ropt.transforms import (
    NonlinearConstraintTransform,
    ObjectiveTransform,
    VariableTransform,
)
from ropt.transforms.default import (
    DefaultNonlinearConstraintTransform,
    DefaultObjectiveTransform,
    DefaultVariableTransform,
)


class _AffineVariableTransform(VariableTransform):
    """Affine map with a separately parameterized diff map.

    The diff map gets its own shift so that chaining two of these does not
    commute; the default scaler only multiplies diffs by a scale, which does.
    """

    def __init__(self, scale: float, shift: float, diff_shift: float) -> None:
        self._scale = scale
        self._shift = shift
        self._diff_shift = diff_shift
        self._mask: NDArray[np.bool_] | None = None

    def set_free_mask(self, mask: NDArray[np.bool_]) -> None:
        self._mask = mask

    def _keep_fixed(
        self, values: NDArray[np.float64], transformed: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._mask is None:
            return transformed
        return np.where(self._mask, transformed, values)

    def to_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._keep_fixed(values, (values - self._shift) / self._scale)

    def from_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._keep_fixed(values, values * self._scale + self._shift)

    def magnitudes_to_optimizer(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return values / self._scale

    def bound_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            lower_diffs * self._scale + self._diff_shift,
            upper_diffs * self._scale + self._diff_shift,
        )

    def linear_constraints_to_optimizer(
        self,
        coefficients: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return (
            coefficients * self._scale,
            lower_bounds - self._shift,
            upper_bounds - self._shift,
        )

    def linear_constraints_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            lower_diffs * self._scale + self._diff_shift,
            upper_diffs * self._scale + self._diff_shift,
        )


class _AffineObjectiveTransform(ObjectiveTransform):
    def __init__(self, scale: float, shift: float) -> None:
        self._scale = scale
        self._shift = shift

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return (objectives - self._shift) / self._scale

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives * self._scale + self._shift


class _AffineConstraintTransform(NonlinearConstraintTransform):
    def __init__(self, scale: float, shift: float) -> None:
        self._scale = scale
        self._shift = shift

    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return (constraints - self._shift) / self._scale

    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints * self._scale + self._shift

    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            (lower_bounds - self._shift) / self._scale,
            (upper_bounds - self._shift) / self._scale,
        )

    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            lower_diffs * self._scale + self._shift,
            upper_diffs * self._scale + self._shift,
        )


def _variable_chain() -> list[_AffineVariableTransform]:
    return [
        _AffineVariableTransform(2.0, 1.0, 0.5),
        _AffineVariableTransform(3.0, 13.0, 7.0),
    ]


def _objective_chain() -> list[_AffineObjectiveTransform]:
    return [_AffineObjectiveTransform(2.0, 1.0), _AffineObjectiveTransform(3.0, 13.0)]


def _constraint_chain() -> list[_AffineConstraintTransform]:
    return [_AffineConstraintTransform(2.0, 1.0), _AffineConstraintTransform(3.0, 13.0)]


def _context(**fields: Any) -> EnOptContext:
    return EnOptContext.model_validate({"variables": {"variable_count": 3}} | fields)


def _function_results(
    variables: NDArray[np.float64],
    objectives: NDArray[np.float64] | None = None,
    constraints: NDArray[np.float64] | None = None,
    constraint_info: ConstraintInfo | None = None,
) -> FunctionResults:
    return FunctionResults(
        batch_id=0,
        metadata={},
        names={},
        evaluations=FunctionEvaluations(
            variables=variables,
            objectives=np.zeros((1, 2)) if objectives is None else objectives,
            constraints=constraints,
        ),
        realizations=Realizations(evaluated_realizations=np.array([True])),
        functions=None,
        constraint_info=constraint_info,
    )


def test_chained_variable_transforms_do_not_commute() -> None:
    first, second = _variable_chain()
    values = np.array([0.5, 1.5, 2.5])
    assert not np.allclose(
        second.to_optimizer(first.to_optimizer(values)),
        first.to_optimizer(second.to_optimizer(values)),
    )
    diffs = np.array([0.25, 0.5, 0.75])
    assert not np.allclose(
        first.bound_constraint_diffs_from_optimizer(
            *second.bound_constraint_diffs_from_optimizer(diffs, diffs)
        )[0],
        second.bound_constraint_diffs_from_optimizer(
            *first.bound_constraint_diffs_from_optimizer(diffs, diffs)
        )[0],
    )


def test_chained_objective_transforms_do_not_commute() -> None:
    first, second = _objective_chain()
    values = np.array([[3.0, 5.0]])
    assert not np.allclose(
        second.to_optimizer(first.to_optimizer(values)),
        first.to_optimizer(second.to_optimizer(values)),
    )


def test_chained_constraint_transforms_do_not_commute() -> None:
    first, second = _constraint_chain()
    values = np.array([[3.0, 5.0]])
    assert not np.allclose(
        second.to_optimizer(first.to_optimizer(values)),
        first.to_optimizer(second.to_optimizer(values)),
    )


def test_variable_chain_transforms_bounds_in_configured_order() -> None:
    lower_bounds = np.array([-2.0, -3.0, -4.0])
    upper_bounds = np.array([5.0, 6.0, 7.0])
    context = _context(
        variables={
            "variable_count": 3,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
        },
        variable_transforms=_variable_chain(),
    )
    first, second = _variable_chain()
    assert np.allclose(
        context.variables.lower_bounds,
        second.to_optimizer(first.to_optimizer(lower_bounds)),
    )
    assert np.allclose(
        context.variables.upper_bounds,
        second.to_optimizer(first.to_optimizer(upper_bounds)),
    )


def test_variable_chain_applies_every_magnitude_transform() -> None:
    magnitudes = np.array([0.1, 0.2, 0.3])
    context = _context(
        variables={"variable_count": 3, "perturbation_magnitudes": magnitudes},
        variable_transforms=_variable_chain(),
    )
    first, second = _variable_chain()
    assert np.allclose(
        context.variables.perturbation_magnitudes,
        second.magnitudes_to_optimizer(first.magnitudes_to_optimizer(magnitudes)),
    )


def test_relative_perturbation_magnitudes_are_left_untransformed() -> None:
    magnitudes = np.array([0.1, 0.2, 0.3])
    context = _context(
        variables={
            "variable_count": 3,
            "lower_bounds": [0.0, 0.0, 0.0],
            "upper_bounds": [1.0, 1.0, 1.0],
            "perturbation_magnitudes": magnitudes,
            "perturbation_types": [
                PerturbationType.ABSOLUTE,
                PerturbationType.RELATIVE,
                PerturbationType.ABSOLUTE,
            ],
        },
        variable_transforms=_variable_chain(),
    )
    first, second = _variable_chain()
    scaled = second.magnitudes_to_optimizer(first.magnitudes_to_optimizer(magnitudes))
    assert np.allclose(
        context.variables.perturbation_magnitudes,
        [scaled[0], magnitudes[1], scaled[2]],
    )


def test_variable_chain_from_optimizer_restores_user_values() -> None:
    user_variables = np.array([0.5, 1.5, 2.5])
    first, second = _variable_chain()
    optimizer_variables = second.to_optimizer(first.to_optimizer(user_variables))

    context = _context(variable_transforms=_variable_chain())
    transformed = _function_results(optimizer_variables).transform_from_optimizer(
        context
    )
    assert np.allclose(transformed.evaluations.variables, user_variables)


def test_objective_chain_from_optimizer_restores_user_values() -> None:
    user_objectives = np.array([[3.0, 5.0]])
    first, second = _objective_chain()
    optimizer_objectives = second.to_optimizer(first.to_optimizer(user_objectives))

    context = _context(
        objectives={"weights": [0.5, 0.5]}, objective_transforms=_objective_chain()
    )
    transformed = _function_results(
        np.zeros(3), objectives=optimizer_objectives
    ).transform_from_optimizer(context)
    assert np.allclose(transformed.evaluations.objectives, user_objectives)


def test_nonlinear_constraint_chain_from_optimizer_restores_user_values() -> None:
    user_constraints = np.array([[3.0, 5.0]])
    first, second = _constraint_chain()
    optimizer_constraints = second.to_optimizer(first.to_optimizer(user_constraints))

    context = _context(
        nonlinear_constraints={"lower_bounds": [0.0, 0.0], "upper_bounds": [1.0, 1.0]},
        nonlinear_constraint_transforms=_constraint_chain(),
    )
    transformed = _function_results(
        np.zeros(3), constraints=optimizer_constraints
    ).transform_from_optimizer(context)
    assert transformed.evaluations.constraints is not None
    assert np.allclose(transformed.evaluations.constraints, user_constraints)


def test_bound_constraint_diffs_are_inverted_in_reverse_chain_order() -> None:
    lower_diffs = np.array([0.25, 0.5, 0.75])
    upper_diffs = np.array([-0.25, -0.5, -0.75])
    context = _context(variable_transforms=_variable_chain())
    transformed = _function_results(
        np.zeros(3),
        constraint_info=ConstraintInfo(
            bound_lower=lower_diffs, bound_upper=upper_diffs
        ),
    ).transform_from_optimizer(context)

    first, second = _variable_chain()
    expected = first.bound_constraint_diffs_from_optimizer(
        *second.bound_constraint_diffs_from_optimizer(lower_diffs, upper_diffs)
    )
    assert transformed.constraint_info is not None
    assert transformed.constraint_info.bound_lower is not None
    assert transformed.constraint_info.bound_upper is not None
    assert np.allclose(transformed.constraint_info.bound_lower, expected[0])
    assert np.allclose(transformed.constraint_info.bound_upper, expected[1])


def test_linear_constraint_diffs_are_inverted_in_reverse_chain_order() -> None:
    lower_diffs = np.array([0.25, 0.5])
    upper_diffs = np.array([-0.25, -0.5])
    context = _context(variable_transforms=_variable_chain())
    transformed = _function_results(
        np.zeros(3),
        constraint_info=ConstraintInfo(
            linear_lower=lower_diffs, linear_upper=upper_diffs
        ),
    ).transform_from_optimizer(context)

    first, second = _variable_chain()
    expected = first.linear_constraints_diffs_from_optimizer(
        *second.linear_constraints_diffs_from_optimizer(lower_diffs, upper_diffs)
    )
    assert transformed.constraint_info is not None
    assert transformed.constraint_info.linear_lower is not None
    assert transformed.constraint_info.linear_upper is not None
    assert np.allclose(transformed.constraint_info.linear_lower, expected[0])
    assert np.allclose(transformed.constraint_info.linear_upper, expected[1])


def test_nonlinear_constraint_diffs_are_inverted_in_reverse_chain_order() -> None:
    lower_diffs = np.array([0.25, 0.5])
    upper_diffs = np.array([-0.25, -0.5])
    context = _context(
        nonlinear_constraints={"lower_bounds": [0.0, 0.0], "upper_bounds": [1.0, 1.0]},
        nonlinear_constraint_transforms=_constraint_chain(),
    )
    transformed = _function_results(
        np.zeros(3),
        constraint_info=ConstraintInfo(
            nonlinear_lower=lower_diffs, nonlinear_upper=upper_diffs
        ),
    ).transform_from_optimizer(context)

    first, second = _constraint_chain()
    expected = first.nonlinear_constraint_diffs_from_optimizer(
        *second.nonlinear_constraint_diffs_from_optimizer(lower_diffs, upper_diffs)
    )
    assert transformed.constraint_info is not None
    assert transformed.constraint_info.nonlinear_lower is not None
    assert transformed.constraint_info.nonlinear_upper is not None
    assert np.allclose(transformed.constraint_info.nonlinear_lower, expected[0])
    assert np.allclose(transformed.constraint_info.nonlinear_upper, expected[1])


def test_constraint_transforms_without_constraints_are_rejected() -> None:
    with pytest.raises(ValidationError, match="nonlinear constraint transforms need"):
        _context(nonlinear_constraint_transforms=_constraint_chain())


def test_single_variable_transform_round_trips() -> None:
    def scaler() -> DefaultVariableTransform:
        return DefaultVariableTransform(
            VariableTransformConfig.model_validate(
                {"method": "scaler", "options": {"scales": [2.0, 4.0, 5.0]}}
            )
        )

    user_variables = np.array([0.5, 1.5, 2.5])
    context = _context(variable_transforms=[scaler()])
    transformed = _function_results(
        scaler().to_optimizer(user_variables)
    ).transform_from_optimizer(context)
    assert np.allclose(transformed.evaluations.variables, user_variables)


@pytest.mark.parametrize("mask", [[True, False, True], [False, False, True]])
def test_free_mask_is_passed_to_every_transform_in_the_chain(mask: Any) -> None:
    context = _context(
        variables={"variable_count": 3, "mask": mask},
        variable_transforms=_variable_chain(),
    )
    values = np.array([1.0, 1.0, 1.0])
    transformed = values
    for transform in context.variable_transforms:
        transformed = transform.to_optimizer(transformed)
    fixed = ~np.asarray(mask)
    assert np.allclose(transformed[fixed], values[fixed])
    assert not np.allclose(transformed[~fixed], values[~fixed])


def _default_variable_scaler(scales: Any, mask: Any = None) -> DefaultVariableTransform:
    return DefaultVariableTransform(
        VariableTransformConfig.model_validate(
            {"method": "scaler", "options": {"scales": scales}, "mask": mask}
        )
    )


def _default_objective_scaler(
    scales: Any, mask: Any = None
) -> DefaultObjectiveTransform:
    return DefaultObjectiveTransform(
        ObjectiveTransformConfig.model_validate(
            {"method": "scaler", "options": {"scales": scales}, "mask": mask}
        )
    )


def _default_constraint_scaler(
    scales: Any, mask: Any = None
) -> DefaultNonlinearConstraintTransform:
    return DefaultNonlinearConstraintTransform(
        NonlinearConstraintTransformConfig.model_validate(
            {"method": "scaler", "options": {"scales": scales}, "mask": mask}
        )
    )


def test_variable_transform_mask_excludes_masked_variables() -> None:
    context = _context(
        variable_transforms=[
            _default_variable_scaler([2.0, 2.0, 2.0], mask=[True, False, True])
        ]
    )
    scaled = context.variable_transforms[0].to_optimizer(np.array([2.0, 2.0, 2.0]))
    assert np.allclose(scaled, [1.0, 2.0, 1.0])


def test_variable_transform_mask_is_combined_with_free_mask() -> None:
    context = _context(
        variables={"variable_count": 3, "mask": [True, False, True]},
        variable_transforms=[
            _default_variable_scaler([2.0, 2.0, 2.0], mask=[True, True, False])
        ],
    )
    # Only variables free *and* selected by the transform mask are scaled.
    scaled = context.variable_transforms[0].to_optimizer(np.array([2.0, 2.0, 2.0]))
    assert np.allclose(scaled, [1.0, 2.0, 2.0])


def test_objective_transform_mask_excludes_masked_objectives() -> None:
    scaler = _default_objective_scaler([2.0, 2.0], mask=[True, False])
    assert np.allclose(scaler.to_optimizer(np.array([2.0, 2.0])), [1.0, 2.0])
    assert np.allclose(scaler.from_optimizer(np.array([1.0, 2.0])), [2.0, 2.0])


def test_constraint_transform_mask_excludes_masked_constraints() -> None:
    scaler = _default_constraint_scaler([2.0, 2.0], mask=[True, False])
    assert np.allclose(scaler.to_optimizer(np.array([2.0, 2.0])), [1.0, 2.0])
    assert np.allclose(scaler.from_optimizer(np.array([1.0, 2.0])), [2.0, 2.0])


def test_objective_transform_mask_survives_update() -> None:
    scaler = _default_objective_scaler([2.0, 2.0], mask=[True, False])
    scaler.update([4.0, 4.0])
    assert np.allclose(scaler.to_optimizer(np.array([4.0, 4.0])), [1.0, 4.0])


def test_constraint_transform_mask_survives_update() -> None:
    scaler = _default_constraint_scaler([2.0, 2.0], mask=[True, False])
    scaler.update([4.0, 4.0])
    assert np.allclose(scaler.to_optimizer(np.array([4.0, 4.0])), [1.0, 4.0])


def test_transform_without_mask_applies_everywhere() -> None:
    scaler = _default_objective_scaler([2.0, 2.0])
    assert np.allclose(scaler.to_optimizer(np.array([2.0, 2.0])), [1.0, 1.0])
