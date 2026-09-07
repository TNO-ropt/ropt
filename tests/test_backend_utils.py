"""Tests for the helpers that backends use to read the linear constraints."""

from typing import Any

import numpy as np

from ropt.backend.utils import get_linear_constraints, split_linear_constraints
from ropt.context import EnOptContext


def _context(linear_constraints: Any, **variables: Any) -> EnOptContext:
    return EnOptContext.model_validate(
        {
            "variables": {"variable_count": 3} | variables,
            "linear_constraints": linear_constraints,
        }
    )


def test_linear_constraints_without_a_finite_bound_are_dropped() -> None:
    context = _context(
        {
            "coefficients": [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            "lower_bounds": [1.0, -np.inf],
            "upper_bounds": [2.0, np.inf],
        }
    )
    coefficients, lower_bounds, upper_bounds, equality = get_linear_constraints(
        context, np.zeros(3)
    )
    assert np.allclose(coefficients, [[1.0, 0.0, 1.0]])
    assert np.allclose(lower_bounds, [1.0])
    assert np.allclose(upper_bounds, [2.0])
    assert equality.tolist() == [False]


def test_linear_constraints_without_a_free_coefficient_are_dropped() -> None:
    context = _context(
        {
            "coefficients": [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            "lower_bounds": [1.0, 1.0],
            "upper_bounds": [1.0, 2.0],
        },
        mask=[True, False, True],
    )
    coefficients, lower_bounds, upper_bounds, equality = get_linear_constraints(
        context, np.array([0.0, 3.0, 0.0])
    )
    # The fixed variable contributes 3 to the surviving equation.
    assert np.allclose(coefficients, [[1.0, 1.0]])
    assert np.allclose(lower_bounds, [-2.0])
    assert np.allclose(upper_bounds, [-1.0])
    assert equality.tolist() == [False]


def test_the_equality_flags_follow_the_surviving_constraints() -> None:
    context = _context(
        {
            "coefficients": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "lower_bounds": [-np.inf, 1.0, 1.0],
            "upper_bounds": [np.inf, 1.0, 2.0],
        }
    )
    *_, equality = get_linear_constraints(context, np.zeros(3))
    assert equality.tolist() == [True, False]


def test_split_linear_constraints_measures_against_zero() -> None:
    coefficients, offsets, equality = split_linear_constraints(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0, -np.inf]),
        np.array([1.0, 2.0]),
        np.array([True, False]),
    )
    # The equality keeps its sign, the upper bound is negated so that both are
    # satisfied when they are non-negative.
    assert np.allclose(coefficients, [[1.0, 0.0], [0.0, -1.0]])
    assert np.allclose(offsets, [1.0, -2.0])
    assert equality.tolist() == [True, False]
