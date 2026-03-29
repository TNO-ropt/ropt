from collections.abc import Generator
from dataclasses import InitVar, dataclass
from itertools import zip_longest
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ropt.context import EnOptContext
from ropt.evaluator import EvaluatorCallback, EvaluatorContext


@dataclass(slots=True)
class _FunctionEvaluatorResults:
    batch_id: int | None
    objectives: NDArray[np.float64]
    constraints: NDArray[np.float64] | None
    evaluation_info: dict[str, NDArray[Any]]

    def __post_init__(self) -> None:
        self.objectives, self.constraints = _propagate_nan_values(
            self.objectives, self.constraints
        )
        for key, value in self.evaluation_info.items():
            if value.ndim != 1 and value.size != self.objectives.shape[1]:
                msg = f"Evaluation info has incorrect size: {key}"
                raise ValueError(msg)


@dataclass(slots=True)
class _GradientEvaluatorResults:
    batch_id: int | None
    perturbed_objectives: NDArray[np.float64]
    perturbed_constraints: NDArray[np.float64] | None
    evaluation_info: dict[str, NDArray[Any]]
    realization_count: InitVar[int]
    perturbation_count: InitVar[int]

    def __post_init__(self, realization_count: int, perturbation_count: int) -> None:
        self.perturbed_objectives, self.perturbed_constraints = _propagate_nan_values(
            self.perturbed_objectives, self.perturbed_constraints
        )

        shape = (realization_count, perturbation_count, -1)
        self.perturbed_objectives = self.perturbed_objectives.reshape(shape)
        self.perturbed_constraints = (
            None
            if self.perturbed_constraints is None
            else self.perturbed_constraints.reshape(shape)
        )
        for key, value in self.evaluation_info.items():
            if value.ndim != 1 and value.size != realization_count * perturbation_count:
                msg = f"Evaluation info has incorrect size: {key}"
                raise ValueError(msg)
        self.evaluation_info = {
            key: value.reshape(shape[:2]) for key, value in self.evaluation_info.items()
        }


def _propagate_nan_values(
    objective_results: NDArray[np.float64],
    constraint_results: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    failures = None
    if objective_results is not None:
        failures = np.logical_or.reduce(np.isnan(objective_results), axis=-1)
    if constraint_results is not None:
        constraint_failures = np.logical_or.reduce(
            np.isnan(constraint_results), axis=-1
        )
        if failures is None:
            failures = constraint_failures
        else:
            failures |= constraint_failures
    if objective_results is not None:
        objective_results = objective_results.copy()
        objective_results[failures, :] = np.nan
    if constraint_results is not None:
        constraint_failures = constraint_failures.copy()
        constraint_results[failures, :] = np.nan
    return objective_results, constraint_results


def _get_active_realizations(
    context: EnOptContext,
    *,
    objective_weights: NDArray[np.float64] | None = None,
    constraint_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.bool_]:
    if objective_weights is None:
        return np.abs(context.realizations.weights) > 0
    active_realizations: NDArray[np.bool_] = np.any(
        np.abs(objective_weights) > 0, axis=0
    )
    if constraint_weights is None:
        return active_realizations
    return np.logical_or(
        active_realizations, np.any(np.abs(constraint_weights) > 0, axis=0)
    )


def _get_function_results(
    context: EnOptContext,
    evaluator: EvaluatorCallback,
    variables: NDArray[np.float64],
    active_realizations: NDArray[np.bool_],
) -> Generator[tuple[int, _FunctionEvaluatorResults], None, None]:
    realization_num = context.realizations.weights.size
    realizations = np.tile(
        np.arange(realization_num, dtype=np.intc), variables.shape[0]
    )
    evaluator_context = EvaluatorContext(
        context=context,
        realizations=realizations,
        active=active_realizations[realizations],
    )
    for variable_transform in context.variable_transforms:
        variables = variable_transform.from_optimizer(variables)
    evaluator_result = evaluator(
        np.repeat(variables, realization_num, axis=0), evaluator_context
    )
    objectives = evaluator_result.objectives
    constraints = evaluator_result.constraints
    for objective_transform in context.objective_transforms:
        objectives = objective_transform.to_optimizer(objectives)
    if constraints is not None:
        for constraint_transform in context.nonlinear_constraint_transforms:
            constraints = constraint_transform.to_optimizer(constraints)
    split_objectives = np.vsplit(objectives, variables.shape[0])
    split_constraints = (
        [] if constraints is None else np.vsplit(constraints, variables.shape[0])
    )
    split_infos = {
        key: np.split(value, variables.shape[0])
        for key, value in evaluator_result.evaluation_info.items()
    }
    for idx, (objectives, constraints) in enumerate(
        zip_longest(split_objectives, split_constraints)
    ):
        yield (
            idx,
            _FunctionEvaluatorResults(
                batch_id=evaluator_result.batch_id,
                objectives=objectives,
                constraints=constraints,
                evaluation_info={key: value[idx] for key, value in split_infos.items()},
            ),
        )


def _get_gradient_results(
    context: EnOptContext,
    evaluator: EvaluatorCallback,
    perturbed_variables: NDArray[np.float64],
    active_realizations: NDArray[np.bool_],
) -> _GradientEvaluatorResults:
    realization_num = context.realizations.weights.size
    perturbation_num = context.gradient.number_of_perturbations
    realizations = np.repeat(
        np.arange(realization_num, dtype=np.intc), perturbation_num
    )
    evaluator_context = EvaluatorContext(
        context=context,
        realizations=realizations,
        perturbations=np.tile(np.arange(perturbation_num), realization_num),
        active=active_realizations[realizations],
    )
    variables: NDArray[np.float64] = perturbed_variables.reshape(
        -1, perturbed_variables.shape[-1]
    )
    for variable_transform in context.variable_transforms:
        variables = variable_transform.from_optimizer(variables)
    evaluator_result = evaluator(variables, evaluator_context)
    objectives = evaluator_result.objectives
    constraints = evaluator_result.constraints
    for objective_transform in context.objective_transforms:
        objectives = objective_transform.to_optimizer(objectives)
    if constraints is not None:
        for constraint_transform in context.nonlinear_constraint_transforms:
            constraints = constraint_transform.to_optimizer(constraints)
    return _GradientEvaluatorResults(
        batch_id=evaluator_result.batch_id,
        perturbed_objectives=objectives,
        perturbed_constraints=constraints,
        evaluation_info=evaluator_result.evaluation_info,
        realization_count=context.realizations.weights.size,
        perturbation_count=context.gradient.number_of_perturbations,
    )


def _get_function_and_gradient_results(
    context: EnOptContext,
    evaluator: EvaluatorCallback,
    variables: NDArray[np.float64],
    perturbed_variables: NDArray[np.float64],
    active_realizations: NDArray[np.bool_],
) -> tuple[_FunctionEvaluatorResults, _GradientEvaluatorResults]:
    realization_num = context.realizations.weights.size
    perturbation_num = context.gradient.number_of_perturbations
    realizations = np.arange(realization_num, dtype=np.intc)
    realizations = np.hstack(
        (
            realizations,
            np.repeat(realizations, perturbation_num),
        ),
    )
    evaluator_context = EvaluatorContext(
        context=context,
        realizations=realizations,
        perturbations=np.hstack(
            (
                np.full(realization_num, -1),
                np.tile(np.arange(perturbation_num), realization_num),
            )
        ),
        active=active_realizations[realizations],
    )
    all_variables = np.vstack(
        (
            np.repeat(variables[np.newaxis, ...], realization_num, axis=0),
            perturbed_variables.reshape(-1, perturbed_variables.shape[-1]),
        ),
    )
    for variable_transform in context.variable_transforms:
        all_variables = variable_transform.from_optimizer(all_variables)
    evaluator_result = evaluator(all_variables, evaluator_context)
    objectives = evaluator_result.objectives
    constraints = evaluator_result.constraints
    for objective_transform in context.objective_transforms:
        objectives = objective_transform.to_optimizer(objectives)
    if constraints is not None:
        for constraint_transform in context.nonlinear_constraint_transforms:
            constraints = constraint_transform.to_optimizer(constraints)
    return (
        _FunctionEvaluatorResults(
            batch_id=evaluator_result.batch_id,
            objectives=objectives[:realization_num],
            constraints=None if constraints is None else constraints[:realization_num],
            evaluation_info={
                key: value[:realization_num]
                for key, value in evaluator_result.evaluation_info.items()
            },
        ),
        _GradientEvaluatorResults(
            batch_id=evaluator_result.batch_id,
            perturbed_objectives=objectives[realization_num:, :],
            perturbed_constraints=(
                None if constraints is None else constraints[realization_num:, :]
            ),
            evaluation_info={
                key: value[realization_num:]
                for key, value in evaluator_result.evaluation_info.items()
            },
            realization_count=context.realizations.weights.size,
            perturbation_count=context.gradient.number_of_perturbations,
        ),
    )
