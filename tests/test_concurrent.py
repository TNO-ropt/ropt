from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ropt.evaluator import ConcurrentEvaluator, ConcurrentTask, EvaluatorContext
from ropt.optimization import EnsembleOptimizer


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> Dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "tolerance": 1e-5,
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


def _run_functions(
    functions: Tuple[Callable[..., Any], ...],
    variables: NDArray[np.float64],
    fail: bool,
) -> NDArray[np.float64]:
    result = np.zeros(len(functions), dtype=np.float64)
    for idx, function in enumerate(functions):
        if fail:
            raise RuntimeError
        result[idx] = function(variables, None)
    return result


@dataclass
class TaskTestEvaluator(ConcurrentTask):
    future: Any

    def get_objectives(self) -> Optional[NDArray[np.float64]]:
        return cast(NDArray[np.float64], self.future.result())


class ConcurrentTestEvaluator(ConcurrentEvaluator):
    def __init__(
        self, functions: Tuple[Callable[..., Any], ...], fail_index: int = -1
    ) -> None:
        super().__init__(enable_cache=True, polling=0.0)

        self._executor = ThreadPoolExecutor(max_workers=4)
        self._functions = functions
        self._fail_index = fail_index
        self._tasks: Dict[int, ConcurrentTask]

    def launch(
        self,
        batch_id: int,  # noqa: ARG002
        variables: NDArray[np.float64],
        context: EvaluatorContext,  # noqa: ARG002
        active: Optional[NDArray[np.bool_]],
    ) -> Dict[int, ConcurrentTask]:
        self._tasks = {
            idx: TaskTestEvaluator(
                future=self._executor.submit(
                    _run_functions,
                    self._functions,
                    variables[idx, :],
                    self._fail_index == idx,
                ),
            )
            for idx in range(variables.shape[0])
            if active is None or active[idx]
        }
        return self._tasks

    def monitor(self) -> None:
        for idx, task in self._tasks.items():
            if task.future is not None and task.future.exception() is not None:
                print(f"error in evaluation {idx}")  # noqa: T201

    def disable_functions(self) -> None:
        self._functions = (lambda _0, _1: 0.0, lambda _0, _1: 0.0)


def test_concurrent(enopt_config: Any, test_functions: Any) -> None:
    evaluator = ConcurrentTestEvaluator(test_functions)
    optimizer = EnsembleOptimizer(evaluator)
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)


def test_concurrent_exception(
    enopt_config: Any, test_functions: Any, capsys: Any
) -> None:
    evaluator = ConcurrentTestEvaluator(test_functions, fail_index=2)
    optimizer = EnsembleOptimizer(evaluator)
    results = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results is not None
    assert np.allclose(results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)
    captured = capsys.readouterr()
    assert "error in evaluation 2" in captured.out


def test_concurrent_cache(enopt_config: Any, test_functions: Any) -> None:
    evaluator = ConcurrentTestEvaluator(test_functions)

    optimizer = EnsembleOptimizer(evaluator)
    results1 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results1 is not None
    assert np.allclose(results1.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02)

    # Disable the functions, not the evaluator fully relies on its cache:
    evaluator.disable_functions()

    optimizer = EnsembleOptimizer(evaluator)
    results2 = optimizer.start_optimization(
        plan=[
            {"config": enopt_config},
            {"optimizer": {"id": "opt"}},
            {"tracker": {"id": "optimum", "source": "opt"}},
        ],
    )
    assert results2 is not None
    assert np.all(
        np.equal(results1.evaluations.variables, results2.evaluations.variables)
    )
