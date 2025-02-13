from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.evaluator import Evaluator, EvaluatorContext, EvaluatorResult

_Function = Callable[[NDArray[np.float64], Any], float]


def pytest_addoption(parser: Any) -> Any:
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config: Any, items: Sequence[Any]) -> None:
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@dataclass(slots=True)
class FunctionContext:
    realization: int
    index: int


def _function_runner(
    variables: NDArray[np.float64],
    evaluator_context: EvaluatorContext,
    functions: list[_Function],
) -> EvaluatorResult:
    objective_count = evaluator_context.config.objectives.weights.size
    constraint_count = (
        0
        if evaluator_context.config.nonlinear_constraints is None
        else evaluator_context.config.nonlinear_constraints.lower_bounds.size
    )
    objective_results = np.zeros(
        (variables.shape[0], objective_count),
        dtype=np.float64,
    )
    constraint_results = (
        np.zeros((variables.shape[0], constraint_count), dtype=np.float64)
        if constraint_count > 0
        else None
    )
    for sim, realization in enumerate(evaluator_context.realizations):
        context = FunctionContext(realization=realization, index=sim)
        for idx in range(objective_count):
            if (
                evaluator_context.active_objectives is None
                or evaluator_context.active_objectives[idx, realization]
            ):
                function = functions[idx]
                objective_results[sim, idx] = function(variables[sim, :], context)
        for idx in range(constraint_count):
            if (
                evaluator_context.active_constraints is None
                or evaluator_context.active_constraints[idx, realization]
            ):
                function = functions[idx + objective_count]
                assert constraint_results is not None
                constraint_results[sim, idx] = function(variables[sim, :], context)

    return EvaluatorResult(
        objectives=objective_results,
        constraints=constraint_results,
    )


def _compute_distance_squared(
    variables: NDArray[np.float64],
    context: Any,  # noqa: ARG001
    target: NDArray[np.float64],
) -> float:
    return float(((variables - target) ** 2).sum())


@pytest.fixture(name="test_functions", scope="session")
def fixture_test_functions() -> tuple[_Function, _Function]:
    return (
        partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5])),
    )


@pytest.fixture(scope="session")
def evaluator(test_functions: Any) -> Callable[[list[_Function]], Evaluator]:
    def _evaluator(
        test_functions: list[_Function] = test_functions,
    ) -> Evaluator:
        return partial(_function_runner, functions=test_functions)

    return _evaluator


@pytest.fixture(scope="session")
def assert_equal_dicts() -> Callable[[Any, Any], None]:
    def _assert_equal_dicts(value1: Any, value2: Any) -> None:
        match value1:
            case dict():
                assert isinstance(value2, dict)
                for key, item1 in value1.items():
                    assert key in value2
                    _assert_equal_dicts(item1, value2[key])
            case list():
                assert isinstance(value2, list)
                for item1, item2 in zip(value1, value2, strict=False):
                    _assert_equal_dicts(item1, item2)
            case tuple():
                assert isinstance(value2, tuple)
                for item1, item2 in zip(value1, value2, strict=False):
                    _assert_equal_dicts(item1, item2)
            case np.ndarray():
                assert isinstance(value2, np.ndarray)
                assert np.allclose(value1, value2)
            case _:
                assert value1 == value2

    return _assert_equal_dicts
