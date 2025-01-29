from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.config.enopt import EnOptConfig, EnOptContext
from ropt.config.enopt.constants import DEFAULT_SEED
from ropt.enums import ConstraintType, EventType, OptimizerExitCode
from ropt.plan import BasicOptimizer
from ropt.results import FunctionResults, GradientResults
from ropt.transforms import Transforms, VariableScaler
from ropt.transforms.base import NonLinearConstraintTransform, ObjectiveTransform

if TYPE_CHECKING:
    from ropt.plan import Event


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {
            "tolerance": 1e-5,
            "max_functions": 20,
        },
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_max_functions_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def track_results(_: Event) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 2
    enopt_config["optimizer"]["max_functions"] = max_functions
    optimizer = BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, track_results
    )
    optimizer.run()
    assert last_evaluation == max_functions + 1
    assert optimizer.exit_code == OptimizerExitCode.MAX_FUNCTIONS_REACHED


def test_max_functions_not_exceeded(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def track_results(_: Event) -> None:
        nonlocal last_evaluation
        last_evaluation += 1

    max_functions = 100
    enopt_config["optimizer"]["max_functions"] = max_functions
    enopt_config["optimizer"]["split_evaluations"] = True
    optimizer = BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, track_results
    )
    optimizer.run()
    assert last_evaluation + 1 < 2 * max_functions
    assert optimizer.exit_code == OptimizerExitCode.OPTIMIZER_STEP_FINISHED


def test_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    def _observer(event: Event) -> None:
        assert isinstance(event.data["results"][0], FunctionResults)
        assert event.data["results"][0].functions is None

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(enopt_config, evaluator(functions)).add_observer(
        EventType.FINISHED_EVALUATION, _observer
    )
    optimizer.run()
    assert optimizer.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS


def test_all_failed_realizations_not_supported(
    enopt_config: Any, evaluator: Any
) -> None:
    enopt_config["realizations"] = {"realization_min_success": 0}

    functions = [lambda _0, _1: np.array(1.0), lambda _0, _1: np.array(np.nan)]
    optimizer = BasicOptimizer(
        enopt_config,
        evaluator(functions),
    )
    optimizer.run()
    assert optimizer.exit_code == OptimizerExitCode.TOO_FEW_REALIZATIONS


def test_user_abort(enopt_config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def _observer(_: Event) -> None:
        nonlocal last_evaluation

        last_evaluation += 1
        if last_evaluation == 1:
            optimizer.abort_optimization()

    optimizer = BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, _observer
    )
    optimizer.run()
    assert optimizer.results is not None
    assert last_evaluation == 1
    assert optimizer.exit_code == OptimizerExitCode.USER_ABORT


def test_single_perturbation(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["gradient"] = {
        "number_of_perturbations": 1,
        "merge_realizations": True,
    }
    enopt_config["realizations"] = {"weights": 5 * [1]}
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


@pytest.mark.parametrize("speculative", [True, False])
def test_objective_auto_scale(
    enopt_config: Any, evaluator: Any, test_functions: Any, speculative: bool
) -> None:
    check_count = 0

    def check_value(event: Event, value: float) -> None:
        nonlocal check_count
        for item in event.data["results"]:
            if isinstance(item, FunctionResults) and check_count:
                check_count -= 1
                assert item.functions is not None
                assert item.functions.scaled_objectives is not None
                assert np.allclose(item.functions.scaled_objectives[-1], value)

    def check_scale(event: Event, scale: float, value: float) -> None:
        nonlocal check_count
        for item in event.data["results"]:
            if isinstance(item, FunctionResults) and check_count:
                check_count -= 1
                assert item.functions is not None
                assert item.functions.scaled_objectives is not None
                assert np.allclose(item.functions.objectives[-1], scale)
                assert np.allclose(item.functions.scaled_objectives[-1], value)

    enopt_config["optimizer"]["speculative"] = speculative
    config = EnOptConfig.model_validate(enopt_config)
    init1 = test_functions[1](config.variables.initial_values, None)

    enopt_config["objectives"]["scales"] = [1.0, init1]
    enopt_config["objectives"]["auto_scale"] = False
    optimizer = BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, partial(check_value, value=1.0)
    )
    check_count = 1
    manual_result = optimizer.run().variables
    assert manual_result is not None

    enopt_config["objectives"]["scales"] = [1.0, 1.0]
    enopt_config["objectives"]["auto_scale"] = [False, True]
    optimizer = (
        BasicOptimizer(enopt_config, evaluator())
        .add_observer(EventType.FINISHED_EVALUATION, partial(check_value, value=1.0))
        .add_observer(
            EventType.FINISHED_EVALUATOR_STEP,
            partial(check_scale, scale=init1, value=1.0),
        )
    )
    check_count = 2
    variables = optimizer.run().variables
    assert variables is not None
    assert np.allclose(variables, manual_result)

    enopt_config["objectives"]["scales"] = [1.0, 2.0 * init1]
    enopt_config["objectives"]["auto_scale"] = False
    optimizer = BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, partial(check_value, value=0.5)
    )
    check_count = 1
    manual_result = optimizer.run().variables
    assert manual_result is not None

    enopt_config["objectives"]["scales"] = [1.0, 2.0]
    enopt_config["objectives"]["auto_scale"] = [False, True]
    optimizer = (
        BasicOptimizer(enopt_config, evaluator())
        .add_observer(EventType.FINISHED_EVALUATION, partial(check_value, value=0.5))
        .add_observer(
            EventType.FINISHED_EVALUATOR_STEP,
            partial(check_scale, scale=init1, value=0.5),
        )
    )
    check_count = 2
    variables = optimizer.run().variables
    assert variables is not None
    assert np.allclose(variables, manual_result)


class ObjectiveScaler(ObjectiveTransform):
    def __init__(self, scales: NDArray[np.float64]) -> None:
        self._scales = scales

    def forward(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives / self._scales

    def backward(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives * self._scales


@pytest.mark.parametrize("speculative", [True, False])
def test_objective_with_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any, speculative: bool
) -> None:
    checked = False

    def check_value(event: Event, value: float) -> None:
        nonlocal checked
        for item in event.data["results"]:
            if isinstance(item, FunctionResults) and not checked:
                checked = True
                assert item.functions is not None
                assert item.functions.objectives is not None
                assert np.allclose(item.functions.objectives[-1], value)

    enopt_config["optimizer"]["speculative"] = speculative
    config = EnOptConfig.model_validate(enopt_config)

    results = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results is not None
    assert results.functions is not None
    variables = results.evaluations.variables
    objectives = results.functions.objectives
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)
    assert np.allclose(objectives, [0.5, 4.5], atol=0.02)

    init1 = test_functions[1](config.variables.initial_values, None)
    transforms = Transforms(objectives=ObjectiveScaler(np.array([init1, init1])))

    optimizer = BasicOptimizer(
        enopt_config, evaluator(transforms=transforms)
    ).add_observer(EventType.FINISHED_EVALUATION, partial(check_value, value=1.0))
    scaled_results = optimizer.run().results
    assert scaled_results is not None
    scaled_variables = scaled_results.evaluations.variables
    assert np.allclose(scaled_variables, variables, atol=0.02)
    unscaled_results = scaled_results.transform_back(transforms)
    assert unscaled_results.functions is not None
    assert np.allclose(objectives, unscaled_results.functions.objectives, atol=0.025)


@pytest.mark.parametrize("speculative", [True, False])
def test_constraint_auto_scale(
    enopt_config: Any, evaluator: Any, test_functions: Any, speculative: bool
) -> None:
    enopt_config["optimizer"]["speculative"] = speculative
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4,
        "types": ConstraintType.GE,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: cast(NDArray[np.float64], variables[0] + variables[2]),
    )

    config = EnOptConfig.model_validate(enopt_config)
    scales = np.fabs(test_functions[-1](config.variables.initial_values, None))

    check_count = 0

    def check_value(event: Event) -> None:
        nonlocal check_count
        for item in event.data["results"]:
            if isinstance(item, FunctionResults) and check_count:
                check_count -= 1
                assert item.functions is not None
                assert item.functions.scaled_constraints is not None
                assert np.allclose(item.functions.scaled_constraints, 1.0)

    def check_scale(event: Event, scale: float) -> None:
        nonlocal check_count
        for item in event.data["results"]:
            if isinstance(item, FunctionResults) and check_count:
                check_count -= 1
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert item.functions.scaled_constraints is not None
                assert np.allclose(item.functions.constraints[-1], scale)
                assert np.allclose(item.functions.scaled_constraints[-1], 1.0)

    enopt_config["nonlinear_constraints"]["scales"] = scales
    enopt_config["nonlinear_constraints"]["auto_scale"] = False
    check_count = 1
    BasicOptimizer(enopt_config, evaluator(test_functions)).add_observer(
        EventType.FINISHED_EVALUATION, check_value
    ).run()

    check_count = 2
    enopt_config["nonlinear_constraints"]["scales"] = 1.0
    enopt_config["nonlinear_constraints"]["auto_scale"] = True
    BasicOptimizer(enopt_config, evaluator(test_functions)).add_observer(
        EventType.FINISHED_EVALUATION, check_value
    ).add_observer(
        EventType.FINISHED_EVALUATOR_STEP,
        partial(
            check_scale,
            scale=(
                config.variables.initial_values[0] + config.variables.initial_values[2]
            ),
        ),
    ).run()


def _flip_type(constraint_type: ConstraintType) -> ConstraintType:
    if constraint_type == ConstraintType.GE:
        return ConstraintType.LE
    if constraint_type == ConstraintType.LE:
        return ConstraintType.GE
    return constraint_type


class ConstraintScaler(NonLinearConstraintTransform):
    def __init__(self, scales: NDArray[np.float64]) -> None:
        self._scales = scales

    def transform_rhs_values(
        self, rhs_values: NDArray[np.float64], types: NDArray[np.ubyte]
    ) -> tuple[NDArray[np.float64], NDArray[np.ubyte]]:
        rhs_values = rhs_values / self._scales
        types = np.fromiter(
            (
                _flip_type(type_) if scale < 0 else type_
                for type_, scale in zip(types, self._scales, strict=False)
            ),
            np.ubyte,
        )
        return rhs_values, types

    def forward(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints / self._scales

    def backward(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints * self._scales


@pytest.mark.parametrize("speculative", [True, False])
def test_constraint_with_scaler(
    enopt_config: Any, evaluator: Any, test_functions: Any, speculative: bool
) -> None:
    enopt_config["optimizer"]["speculative"] = speculative
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4,
        "types": ConstraintType.GE,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: cast(NDArray[np.float64], variables[0] + variables[2]),
    )

    scales = np.array(
        test_functions[-1](enopt_config["variables"]["initial_values"], None), ndmin=1
    )
    transforms = Transforms(nonlinear_constraints=ConstraintScaler(scales))
    context = EnOptContext(transforms=transforms)
    config = EnOptConfig.model_validate(enopt_config, context=context)
    assert config.nonlinear_constraints is not None
    assert config.nonlinear_constraints.rhs_values == 0.4 / scales

    check = True

    def check_constraints(event: Event) -> None:
        nonlocal check
        for item in event.data["results"]:
            if isinstance(item, FunctionResults) and check:
                check = False
                assert item.functions is not None
                assert item.functions.constraints is not None
                assert np.allclose(item.functions.constraints, 1.0)

    BasicOptimizer(
        config, evaluator(test_functions, transforms=transforms)
    ).add_observer(EventType.FINISHED_EVALUATION, check_constraints).run()


@pytest.mark.parametrize("offsets", [None, np.array([1.0, 1.1, 1.2])])
@pytest.mark.parametrize("scales", [None, np.array([2.0, 2.1, 2.2])])
def test_variables_scale(
    enopt_config: Any,
    evaluator: Any,
    offsets: NDArray[np.float64] | None,
    scales: NDArray[np.float64] | None,
) -> None:
    initial_values = np.array(enopt_config["variables"]["initial_values"])
    lower_bounds = np.array([-2.0, -np.inf, -3.0])
    upper_bounds = np.array([np.inf, 1.0, 4.0])

    enopt_config["optimizer"]["max_iterations"] = 20
    enopt_config["variables"]["lower_bounds"] = lower_bounds
    enopt_config["variables"]["upper_bounds"] = upper_bounds

    if offsets is not None:
        enopt_config["variables"]["offsets"] = offsets
    if scales is not None:
        enopt_config["variables"]["scales"] = scales

    results = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results is not None

    if offsets is not None:
        initial_values = initial_values - offsets
        lower_bounds = lower_bounds - offsets
        upper_bounds = upper_bounds - offsets
    if scales is not None:
        initial_values = initial_values / scales
        lower_bounds = lower_bounds / scales
        upper_bounds = upper_bounds / scales
    config = EnOptConfig.model_validate(enopt_config)
    assert np.allclose(config.variables.initial_values, initial_values)
    assert np.allclose(config.variables.lower_bounds, lower_bounds)
    assert np.allclose(config.variables.upper_bounds, upper_bounds)
    result = results.evaluations.scaled_variables
    if result is None:
        result = results.evaluations.variables
    if scales is not None:
        result = result * scales
    if offsets is not None:
        result = result + offsets
    assert np.allclose(result, [0.0, 0.0, 0.5], atol=0.05)
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.05)


@pytest.mark.parametrize("offsets", [None, np.array([1.0, 1.1, 1.2])])
@pytest.mark.parametrize("scales", [None, np.array([2.0, 2.1, 2.2])])
def test_variables_scale_with_scaler(
    enopt_config: Any,
    evaluator: Any,
    offsets: NDArray[np.float64] | None,
    scales: NDArray[np.float64] | None,
) -> None:
    initial_values = np.array(enopt_config["variables"]["initial_values"])
    lower_bounds = np.array([-2.0, -np.inf, -3.0])
    upper_bounds = np.array([np.inf, 1.0, 4.0])

    enopt_config["optimizer"]["max_iterations"] = 20
    enopt_config["variables"]["lower_bounds"] = lower_bounds
    enopt_config["variables"]["upper_bounds"] = upper_bounds

    transforms = Transforms(variables=VariableScaler(scales, offsets))
    context = EnOptContext(transforms=transforms)
    results = (
        BasicOptimizer(
            EnOptConfig.model_validate(enopt_config, context=context),
            evaluator(transforms=transforms),
        )
        .run()
        .results
    )
    assert results is not None

    if offsets is not None:
        initial_values = initial_values - offsets
        lower_bounds = lower_bounds - offsets
        upper_bounds = upper_bounds - offsets
    if scales is not None:
        initial_values = initial_values / scales
        lower_bounds = lower_bounds / scales
        upper_bounds = upper_bounds / scales
    config = EnOptConfig.model_validate(enopt_config, context=context)
    assert np.allclose(config.variables.initial_values, initial_values)
    assert np.allclose(config.variables.lower_bounds, lower_bounds)
    assert np.allclose(config.variables.upper_bounds, upper_bounds)
    result = results.evaluations.variables
    if scales is not None:
        result = result * scales
    if offsets is not None:
        result = result + offsets
    assert np.allclose(result, [0.0, 0.0, 0.5], atol=0.05)
    results = results.transform_back(context.transforms)
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.05)


def test_variables_scale_linear_constraints(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "rhs_values": [1.0, 0.75],
        "types": [ConstraintType.EQ, ConstraintType.EQ],
    }

    offsets = np.array([1.0, 1.1, 1.2])
    scales = np.array([2.0, 2.1, 2.2])
    enopt_config["variables"]["offsets"] = offsets
    enopt_config["variables"]["scales"] = scales

    config = EnOptConfig.model_validate(enopt_config)
    assert config.linear_constraints is not None
    coefficients = config.linear_constraints.coefficients
    rhs_values = config.linear_constraints.rhs_values
    assert np.allclose(
        coefficients / scales, enopt_config["linear_constraints"]["coefficients"]
    )
    assert np.allclose(
        rhs_values
        + np.matmul(coefficients if scales is None else coefficients / scales, offsets),
        enopt_config["linear_constraints"]["rhs_values"],
    )

    results = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results is not None
    assert results.evaluations.variables is not None
    assert np.allclose(results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02)


def test_variables_scale_linear_constraints_with_scaler(
    enopt_config: Any,
    evaluator: Any,
) -> None:
    coefficients = [[1, 0, 1], [0, 1, 1]]
    rhs_values = [1.0, 0.75]

    enopt_config["linear_constraints"] = {
        "coefficients": coefficients,
        "rhs_values": rhs_values,
        "types": [ConstraintType.EQ, ConstraintType.EQ],
    }

    offsets = np.array([1.0, 1.1, 1.2])
    scales = np.array([2.0, 2.1, 2.2])

    transforms = Transforms(variables=VariableScaler(scales, offsets))
    context = EnOptContext(transforms=transforms)
    config = EnOptConfig.model_validate(enopt_config, context=context)
    assert config.linear_constraints is not None
    assert np.allclose(config.linear_constraints.coefficients, coefficients * scales)
    assert np.allclose(
        config.linear_constraints.rhs_values,
        rhs_values - np.matmul(coefficients, offsets),
    )

    results = BasicOptimizer(config, evaluator(transforms=transforms)).run().results
    assert results is not None
    assert results.evaluations.variables is not None
    results = results.transform_back(context.transforms)
    assert np.allclose(results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02)


def test_check_linear_constraints(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
        "rhs_values": [0.0, 1.0, -1.0],
        "types": [ConstraintType.EQ, ConstraintType.LE, ConstraintType.GE],
    }
    enopt_config["optimizer"]["max_functions"] = 1
    results = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results is not None

    enopt_config["linear_constraints"]["rhs_values"] = [1.0, -1.0, 1.0]
    results = BasicOptimizer(enopt_config, evaluator()).run().results
    assert results is None


def test_check_nonlinear_constraints(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": [0.0, 0.0, 0.0],
        "types": [ConstraintType.EQ, ConstraintType.LE, ConstraintType.GE],
        "scales": 10.0,
    }

    test_functions = (
        *test_functions,
        lambda variables, _: cast(NDArray[np.float64], variables[0]),
        lambda variables, _: cast(NDArray[np.float64], variables[0]),
        lambda variables, _: cast(NDArray[np.float64], variables[0]),
    )

    enopt_config["optimizer"]["max_functions"] = 1

    results = BasicOptimizer(enopt_config, evaluator(test_functions)).run().results
    assert results is not None

    enopt_config["nonlinear_constraints"]["rhs_values"] = [1.0, -1.0, 1.0]

    results = BasicOptimizer(enopt_config, evaluator(test_functions)).run().results
    assert results is None


def test_optimizer_variables_subset(enopt_config: Any, evaluator: Any) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    enopt_config["variables"]["initial_values"] = [0.0, 1.0, 0.1]
    enopt_config["variables"]["indices"] = [0, 2]

    def assert_gradient(event: Event) -> None:
        for item in event.data["results"]:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.weighted_objective[1] == 0.0
                assert np.all(item.gradients.objectives[:, 1] == 0.0)

    variables = (
        BasicOptimizer(enopt_config, evaluator())
        .add_observer(EventType.FINISHED_EVALUATION, assert_gradient)
        .run()
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.0, 1.0, 0.5], atol=0.02)


def test_optimizer_variables_subset_linear_constraints(
    enopt_config: Any, evaluator: Any
) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    enopt_config["variables"]["initial_values"] = [0.0, 1.0, 0.1]
    # The second and third constraints are dropped because they involve
    # variables that are not optimized.
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
        "rhs_values": [1.0, 1.0, 2.0],
        "types": [ConstraintType.EQ, ConstraintType.EQ, ConstraintType.EQ],
    }
    enopt_config["variables"]["indices"] = [0, 2]

    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.25, 1.0, 0.75], atol=0.02)


def test_parallelize(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"] = {
        "method": "differential_evolution",
        "max_iterations": 15,
        "options": {"seed": 123, "tol": 1e-10},
    }
    enopt_config["variables"]["lower_bounds"] = [0.15, 0.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [0.5, 0.5, 0.2]
    enopt_config["variables"]["initial_values"][0] = 0.2

    enopt_config["optimizer"]["parallel"] = False
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=3e-2)

    enopt_config["optimizer"]["parallel"] = True
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.15, 0.0, 0.2], atol=3e-2)


def test_rng(enopt_config: Any, evaluator: Any) -> None:
    variables1 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables1 is not None
    assert np.allclose(variables1, [0.0, 0.0, 0.5], atol=0.02)

    variables2 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables2 is not None
    assert np.allclose(variables2, [0.0, 0.0, 0.5], atol=0.02)
    assert np.all(variables1 == variables2)

    enopt_config["gradient"]["seed"] = (1, DEFAULT_SEED)
    variables3 = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables3 is not None
    assert np.allclose(variables3, [0.0, 0.0, 0.5], atol=0.02)
    assert not np.all(variables3 == variables1)
