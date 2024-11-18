import pytest

pytest.importorskip("parsl")

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, cast, no_type_check

import numpy as np
from numpy.typing import NDArray
from parsl.app.app import python_app

from ropt.evaluator import EvaluatorContext
from ropt.evaluator.parsl import ParslEvaluator, State, Task
from ropt.plan import BasicOptimizer


@dataclass(slots=True)
class ParslTestTask(Task):
    def get_objectives(self) -> Optional[NDArray[np.float64]]:
        assert self.future is not None
        result = self.future.result()
        if result is None:
            return None
        return cast(NDArray[np.float64], result)


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "max_functions": 3,
        },
        "objective_functions": {
            "weights": [0.75, 0.25],
        },
        "realizations": {
            "weights": [1.0] * 3,
            "realization_min_success": 1,
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
            "perturbation_min_success": 2,
        },
    }


@python_app
@no_type_check
def _run_functions(
    functions: tuple[Callable[..., Any], ...],
    variables: NDArray[np.float64],
    fail: bool = False,
) -> NDArray[np.float64]:
    result = np.zeros(len(functions), dtype=np.float64)
    for idx, function in enumerate(functions):
        if fail:
            raise RuntimeError
        result[idx] = function(variables, None)
    return result


def parsl_function(
    batch_id: int,  # noqa: ARG001
    idx: int,
    variables: NDArray[np.float64],
    context: EvaluatorContext,  # noqa: ARG001
    functions: tuple[Callable[..., Any], ...],
    fail_index: int = -1,
) -> list[ParslTestTask]:
    return [
        ParslTestTask(
            future=_run_functions(functions, variables, fail=idx == fail_index)
        ),
    ]


def parsl_monitor(batch_id: int, jobs: dict[int, list[ParslTestTask]]) -> None:
    for job_idx, tasks in jobs.items():
        for task_idx, task in enumerate(tasks):
            if task.state == State.FAILED:
                print(f"error in job {job_idx}")  # noqa: T201
            if task.state == State.SUCCESS:
                print(  # noqa: T201
                    f"batch: {batch_id}, job: {job_idx}, "
                    f"task: {task_idx} has finished"
                )


def test_parsl(enopt_config: Any, test_functions: Any, tmp_path: Any) -> None:
    os.chdir(tmp_path)
    with ParslEvaluator(
        function=partial(parsl_function, functions=test_functions)
    ) as evaluator:
        variables = BasicOptimizer(enopt_config, evaluator).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_parsl_dummy_htex(
    enopt_config: Any, test_functions: Any, tmp_path: Any
) -> None:
    os.chdir(tmp_path)
    with ParslEvaluator(
        function=partial(parsl_function, functions=test_functions)
    ).with_htex(provider=None) as evaluator:
        variables = BasicOptimizer(enopt_config, evaluator).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_parsl_monitor(
    enopt_config: Any, test_functions: Any, tmp_path: Any, capsys: Any
) -> None:
    os.chdir(tmp_path)
    with ParslEvaluator(
        function=partial(parsl_function, functions=test_functions)
    ).with_monitor(
        parsl_monitor,
    ) as evaluator:
        variables = BasicOptimizer(enopt_config, evaluator).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)
    captured = capsys.readouterr()
    assert "batch: 0, job: 0, task: 0 has finished" in captured.out


def test_parsl_exception(
    enopt_config: Any, test_functions: Any, tmp_path: Any, capsys: Any
) -> None:
    os.chdir(tmp_path)
    evaluator = ParslEvaluator(
        function=partial(parsl_function, functions=test_functions, fail_index=2),
    ).with_monitor(
        parsl_monitor,
    )
    with evaluator:
        variables = BasicOptimizer(enopt_config, evaluator).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)
    captured = capsys.readouterr()
    assert "error in job 2" in captured.out


def test_parsl_no_with(enopt_config: Any, test_functions: Any, tmp_path: Any) -> None:
    os.chdir(tmp_path)
    evaluator = ParslEvaluator(
        function=partial(parsl_function, functions=test_functions, fail_index=2),
    )
    with pytest.raises(RuntimeError):
        BasicOptimizer(enopt_config, evaluator).run()
