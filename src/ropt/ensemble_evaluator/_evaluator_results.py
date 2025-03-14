from dataclasses import InitVar, dataclass
from itertools import zip_longest
from typing import Any, Generator

import numpy as np
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig
from ropt.evaluator import Evaluator, EvaluatorContext
from ropt.transforms import OptModelTransforms


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
    config: EnOptConfig,
    *,
    objective_weights: NDArray[np.float64] | None = None,
    constraint_weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.bool_] | None, NDArray[np.bool_] | None]:
    if objective_weights is None:
        active_realizations = np.abs(config.realizations.weights) > 0
        if np.all(active_realizations):
            return None, None
        active_objectives = np.broadcast_to(
            active_realizations,
            (config.objectives.weights.size, active_realizations.size),
        )
        active_constraints = (
            None
            if config.nonlinear_constraints is None
            else np.broadcast_to(
                active_realizations,
                (
                    config.nonlinear_constraints.lower_bounds.size,
                    active_realizations.size,
                ),
            )
        )
        return active_objectives, active_constraints
    active_objectives = np.abs(objective_weights) > 0
    active_constraints = (
        None if constraint_weights is None else np.abs(constraint_weights) > 0
    )
    return active_objectives, active_constraints


def _get_function_results(  # noqa: PLR0913
    config: EnOptConfig,
    transforms: OptModelTransforms | None,
    evaluator: Evaluator,
    variables: NDArray[np.float64],
    active_objectives: NDArray[np.bool_] | None,
    active_constraints: NDArray[np.bool_] | None,
) -> Generator[tuple[int, _FunctionEvaluatorResults], None, None]:
    realization_num = config.realizations.weights.size
    context = EvaluatorContext(
        config=config,
        realizations=np.tile(
            np.arange(realization_num, dtype=np.intc), variables.shape[0]
        ),
        active_objectives=active_objectives,
        active_constraints=active_constraints,
    )
    if transforms is not None and transforms.variables:
        variables = transforms.variables.from_optimizer(variables)
    evaluator_result = evaluator(np.repeat(variables, realization_num, axis=0), context)
    if transforms is not None:
        if transforms.objectives is not None:
            evaluator_result.objectives = transforms.objectives.to_optimizer(
                evaluator_result.objectives
            )
        if (
            evaluator_result.constraints is not None
            and transforms.nonlinear_constraints is not None
        ):
            evaluator_result.constraints = (
                transforms.nonlinear_constraints.to_optimizer(
                    evaluator_result.constraints
                )
            )
    split_objectives = np.vsplit(evaluator_result.objectives, variables.shape[0])
    split_constraints = (
        []
        if evaluator_result.constraints is None
        else np.vsplit(evaluator_result.constraints, variables.shape[0])
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


def _get_gradient_results(  # noqa: PLR0913
    config: EnOptConfig,
    transforms: OptModelTransforms | None,
    evaluator: Evaluator,
    perturbed_variables: NDArray[np.float64],
    active_objectives: NDArray[np.bool_] | None,
    active_constraints: NDArray[np.bool_] | None,
) -> _GradientEvaluatorResults:
    realization_num = config.realizations.weights.size
    perturbation_num = config.gradient.number_of_perturbations
    context = EvaluatorContext(
        config=config,
        realizations=np.repeat(np.arange(realization_num), perturbation_num),
        perturbations=np.tile(np.arange(perturbation_num), realization_num),
        active_objectives=active_objectives,
        active_constraints=active_constraints,
    )
    variables = perturbed_variables.reshape(-1, perturbed_variables.shape[-1])
    if transforms is not None and transforms.variables:
        variables = transforms.variables.from_optimizer(variables)
    evaluator_result = evaluator(variables, context)
    if transforms is not None:
        if transforms.objectives is not None:
            evaluator_result.objectives = transforms.objectives.to_optimizer(
                evaluator_result.objectives
            )
        if (
            evaluator_result.constraints is not None
            and transforms.nonlinear_constraints is not None
        ):
            evaluator_result.constraints = (
                transforms.nonlinear_constraints.to_optimizer(
                    evaluator_result.constraints
                )
            )
    return _GradientEvaluatorResults(
        batch_id=evaluator_result.batch_id,
        perturbed_objectives=evaluator_result.objectives,
        perturbed_constraints=evaluator_result.constraints,
        evaluation_info=evaluator_result.evaluation_info,
        realization_count=config.realizations.weights.size,
        perturbation_count=config.gradient.number_of_perturbations,
    )


def _get_function_and_gradient_results(  # noqa: PLR0913
    config: EnOptConfig,
    transforms: OptModelTransforms | None,
    evaluator: Evaluator,
    variables: NDArray[np.float64],
    perturbed_variables: NDArray[np.float64],
    active_objectives: NDArray[np.bool_] | None,
    active_constraints: NDArray[np.bool_] | None,
) -> tuple[_FunctionEvaluatorResults, _GradientEvaluatorResults]:
    realization_num = config.realizations.weights.size
    perturbation_num = config.gradient.number_of_perturbations
    realizations = np.arange(realization_num)
    context = EvaluatorContext(
        config=config,
        realizations=np.hstack(
            (
                realizations,
                np.repeat(realizations, perturbation_num),
            ),
        ),
        perturbations=np.hstack(
            (
                np.full(realization_num, -1),
                np.tile(np.arange(perturbation_num), realization_num),
            )
        ),
        active_objectives=active_objectives,
        active_constraints=active_constraints,
    )
    all_variables = np.vstack(
        (
            np.repeat(variables[np.newaxis, ...], realization_num, axis=0),
            perturbed_variables.reshape(-1, perturbed_variables.shape[-1]),
        ),
    )
    if transforms is not None and transforms.variables:
        all_variables = transforms.variables.from_optimizer(all_variables)
    evaluator_result = evaluator(all_variables, context)
    if transforms is not None:
        if transforms.objectives is not None:
            evaluator_result.objectives = transforms.objectives.to_optimizer(
                evaluator_result.objectives
            )
        if (
            evaluator_result.constraints is not None
            and transforms.nonlinear_constraints is not None
        ):
            evaluator_result.constraints = (
                transforms.nonlinear_constraints.to_optimizer(
                    evaluator_result.constraints
                )
            )
    return (
        _FunctionEvaluatorResults(
            batch_id=evaluator_result.batch_id,
            objectives=evaluator_result.objectives[:realization_num],
            constraints=(
                None
                if evaluator_result.constraints is None
                else evaluator_result.constraints[:realization_num]
            ),
            evaluation_info={
                key: value[:realization_num]
                for key, value in evaluator_result.evaluation_info.items()
            },
        ),
        _GradientEvaluatorResults(
            batch_id=evaluator_result.batch_id,
            perturbed_objectives=evaluator_result.objectives[realization_num:, :],
            perturbed_constraints=(
                None
                if evaluator_result.constraints is None
                else evaluator_result.constraints[realization_num:, :]
            ),
            evaluation_info={
                key: value[realization_num:]
                for key, value in evaluator_result.evaluation_info.items()
            },
            realization_count=config.realizations.weights.size,
            perturbation_count=config.gradient.number_of_perturbations,
        ),
    )
