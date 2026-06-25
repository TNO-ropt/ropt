from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.workflow.evaluators import (
    EvaluationFunctionContext,
    EvaluationFunctionResult,
    FunctionEvaluator,
)

_Function = Callable[[NDArray[np.float64], EvaluationFunctionContext], float]


def pytest_addoption(parser: Any) -> Any:
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests with external optimizers",
    )


def pytest_collection_modifyitems(config: Any, items: Sequence[Any]) -> None:
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-external"):
        skip_external = pytest.mark.skip(reason="need --run-external option to run")
        for item in items:
            if "external" in item.keywords:
                item.add_marker(skip_external)


def _compute_distance_squared(
    variables: NDArray[np.float64],
    _: EvaluationFunctionContext,
    target: NDArray[np.float64],
) -> float:
    return float(((variables - target) ** 2).sum())


@pytest.fixture(name="test_functions", scope="session")
def fixture_test_functions() -> tuple[_Function, _Function]:
    return (
        partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5])),
    )


def _function(
    variables: NDArray[np.float64],
    context: EvaluationFunctionContext,
    *,
    objective_functions: list[_Function],
    constraint_functions: list[_Function] | None = None,
) -> EvaluationFunctionResult:
    return EvaluationFunctionResult(
        objectives=np.fromiter(
            (func(variables, context) for func in objective_functions), dtype=np.float64
        ),
        constraints=np.fromiter(
            (func(variables, context) for func in constraint_functions),
            dtype=np.float64,
        )
        if constraint_functions is not None
        else None,
    )


@pytest.fixture(scope="session")
def evaluator(test_functions: Any, constraint_functions: Any | None = None) -> Any:
    def _evaluator(
        objective_functions: list[_Function] = test_functions,
        constraint_functions: list[_Function] | None = constraint_functions,
    ) -> Any:
        return FunctionEvaluator(
            function=partial(
                _function,
                objective_functions=objective_functions,
                constraint_functions=constraint_functions,
            )
        )

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
