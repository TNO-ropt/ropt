"""Serving the first batch from values supplied with the run."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ropt._scaling import scale, unscale_value

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.context import EnOptContext
    from ropt.evaluation import (
        EvaluationBatchCallback,
        EvaluationBatchContext,
        EvaluationBatchResult,
    )
    from ropt.results import FunctionResults, GradientResults


_POINT_TOLERANCE = 1e-15


class _InitialValueFiller:
    def __init__(
        self,
        context: EnOptContext,
        evaluator: EvaluationBatchCallback,
        f0: FunctionResults | None,
        g0: GradientResults | None,
    ) -> None:
        _check_shapes(context, f0, g0)
        self._context = context
        self._evaluator = evaluator
        self._f0 = f0
        self._g0 = g0
        realization_count = context.realizations.weights.size
        self._function_covered = _function_coverage(f0, realization_count)
        self._gradient_covered = _gradient_coverage(
            g0, realization_count, context.gradient.number_of_perturbations
        )
        self._function_pending = False
        self._gradient_pending = False

    def apply_to_perturbations(
        self,
        variables: NDArray[np.float64],
        perturbed_variables: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._g0 is None:
            return perturbed_variables
        unscaled = unscale_value(
            variables,
            self._context.variables.scales,
            self._context.variables.offsets,
        )
        if not _same_point(unscaled, self._g0.variables):
            msg = "The initial gradient results are not at the evaluated variables"
            raise ValueError(msg)
        # Latched here so that the dispatch and the gradient built from it agree
        # on which perturbations carry supplied values.
        self._gradient_pending = True
        supplied = scale(
            self._g0.perturbed_variables,
            self._context.variables.scales,
            self._context.variables.offsets,
        )
        rows, columns = np.nonzero(self._gradient_covered)
        perturbed_variables = perturbed_variables.copy()
        perturbed_variables[rows, columns, :] = supplied[rows, columns, :]
        return perturbed_variables

    def __call__(
        self, variables: NDArray[np.float64], context: EvaluationBatchContext
    ) -> EvaluationBatchResult:
        realizations = context.realizations
        perturbations = context.perturbations
        unperturbed = perturbations < 0
        function_rows = self._function_rows(variables, realizations, unperturbed)
        gradient_rows = self._gradient_rows(realizations, perturbations, unperturbed)
        supplied = function_rows | gradient_rows
        result = self._evaluator(
            variables, replace(context, active=context.active & ~supplied)
        )
        return self._fill(
            result, realizations, perturbations, function_rows, gradient_rows
        )

    def _function_rows(
        self,
        variables: NDArray[np.float64],
        realizations: NDArray[np.intc],
        unperturbed: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        rows = np.zeros(realizations.shape, dtype=np.bool_)
        if self._f0 is None or not unperturbed.any():
            return rows
        if np.unique(realizations[unperturbed]).size != int(unperturbed.sum()):
            msg = "Initial function results need a single variable vector"
            raise ValueError(msg)
        if not _same_point(variables[unperturbed, :], self._f0.variables):
            msg = "The initial function results are not at the evaluated variables"
            raise ValueError(msg)
        self._function_pending = True
        rows[unperturbed] = self._function_covered[realizations[unperturbed]]
        return rows

    def _gradient_rows(
        self,
        realizations: NDArray[np.intc],
        perturbations: NDArray[np.intc],
        unperturbed: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        rows = np.zeros(realizations.shape, dtype=np.bool_)
        if not self._gradient_pending:
            return rows
        perturbed = ~unperturbed
        rows[perturbed] = self._gradient_covered[
            realizations[perturbed], perturbations[perturbed]
        ]
        return rows

    def _fill(
        self,
        result: EvaluationBatchResult,
        realizations: NDArray[np.intc],
        perturbations: NDArray[np.intc],
        function_rows: NDArray[np.bool_],
        gradient_rows: NDArray[np.bool_],
    ) -> EvaluationBatchResult:
        objectives = np.array(result.objectives, dtype=np.float64)
        constraints = (
            None
            if result.constraints is None
            else np.array(result.constraints, dtype=np.float64)
        )
        if self._function_pending:
            assert self._f0 is not None
            evaluations = self._f0.evaluations
            index = realizations[function_rows]
            objectives[function_rows, :] = evaluations.objectives[index, :]
            if constraints is not None:
                assert evaluations.constraints is not None
                constraints[function_rows, :] = evaluations.constraints[index, :]
            self._f0 = None
            self._function_pending = False
        if self._gradient_pending:
            assert self._g0 is not None
            perturbed = self._g0.evaluations
            index = realizations[gradient_rows]
            columns = perturbations[gradient_rows]
            objectives[gradient_rows, :] = perturbed.perturbed_objectives[
                index, columns, :
            ]
            if constraints is not None:
                assert perturbed.perturbed_constraints is not None
                constraints[gradient_rows, :] = perturbed.perturbed_constraints[
                    index, columns, :
                ]
            self._g0 = None
            self._gradient_pending = False
        return replace(result, objectives=objectives, constraints=constraints)


def _same_point(variables: NDArray[np.float64], point: NDArray[np.float64]) -> bool:
    return bool(np.allclose(variables, point, rtol=0.0, atol=_POINT_TOLERANCE))


def _function_coverage(
    f0: FunctionResults | None, realization_count: int
) -> NDArray[np.bool_]:
    covered = np.zeros(realization_count, dtype=np.bool_)
    if f0 is None:
        return covered
    evaluated = f0.realizations.evaluated_realizations
    overlap = min(evaluated.size, f0.evaluations.objectives.shape[0], realization_count)
    covered[:overlap] = evaluated[:overlap]
    return covered


def _gradient_coverage(
    g0: GradientResults | None, realization_count: int, perturbation_count: int
) -> NDArray[np.bool_]:
    covered = np.zeros((realization_count, perturbation_count), dtype=np.bool_)
    if g0 is None:
        return covered
    evaluated = g0.realizations.evaluated_realizations
    stored_realizations, stored_perturbations = (
        g0.evaluations.perturbed_objectives.shape[:2]
    )
    rows = min(evaluated.size, stored_realizations, realization_count)
    columns = min(stored_perturbations, perturbation_count)
    covered[:rows, :columns] = evaluated[:rows, np.newaxis]
    return covered


def _check_shapes(
    context: EnOptContext, f0: FunctionResults | None, g0: GradientResults | None
) -> None:
    objective_count = context.objectives.weights.size
    constraint_count = (
        0
        if context.nonlinear_constraints is None
        else context.nonlinear_constraints.lower_bounds.size
    )
    variable_count = context.variables.variable_count
    if f0 is not None:
        _check_axis(f0.variables.shape[-1], variable_count, "function", "variables")
        _check_values(
            f0.evaluations.objectives,
            f0.evaluations.constraints,
            objective_count,
            constraint_count,
            "function",
        )
    if g0 is not None:
        _check_axis(g0.variables.shape[-1], variable_count, "gradient", "variables")
        _check_axis(
            g0.perturbed_variables.shape[-1],
            variable_count,
            "gradient",
            "perturbed variables",
        )
        _check_values(
            g0.evaluations.perturbed_objectives,
            g0.evaluations.perturbed_constraints,
            objective_count,
            constraint_count,
            "gradient",
        )


def _check_values(
    objectives: NDArray[np.float64],
    constraints: NDArray[np.float64] | None,
    objective_count: int,
    constraint_count: int,
    kind: str,
) -> None:
    _check_axis(objectives.shape[-1], objective_count, kind, "objectives")
    if constraints is None:
        if constraint_count > 0:
            msg = f"The initial {kind} results have no constraints"
            raise ValueError(msg)
    elif constraint_count == 0:
        msg = f"The initial {kind} results have constraints that are not configured"
        raise ValueError(msg)
    else:
        _check_axis(constraints.shape[-1], constraint_count, kind, "constraints")


def _check_axis(found: int, expected: int, kind: str, name: str) -> None:
    if found != expected:
        msg = f"The initial {kind} results have {found} {name}, expected {expected}"
        raise ValueError(msg)
