"""Tests for supplying results at the initial variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ropt import HistoryHandler, optimize, optimize_many
from ropt.context import EnOptContext
from ropt.core import EnsembleEvaluator
from ropt.results import FunctionResults, GradientResults

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt.components.evaluators import EvaluationFunctionContext

_INITIAL = np.array([0.0, 0.0, 0.1])
_REALIZATIONS = 2
_PERTURBATIONS = 3


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "optimizer": {"max_functions": 2},
        "backend": {"method": "slsqp"},
        "variables": {
            "variable_count": _INITIAL.size,
            "perturbation_magnitudes": 0.01,
            "seed": 11,
        },
        "realizations": {"weights": [1.0, 1.0]},
        "gradient": {"number_of_perturbations": _PERTURBATIONS},
    }


def _objective(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    # Realization-dependent, so that changing the weights changes the aggregate.
    target = 0.5 + 0.25 * context.realization
    return float(((variables - target) ** 2).sum())


class _Recorder:
    def __init__(self) -> None:
        self.rows: list[tuple[int, int, int]] = []

    def __call__(
        self, variables: NDArray[np.float64], context: EvaluationFunctionContext
    ) -> float:
        self.rows.append((context.batch_id, context.realization, context.perturbation))
        return _objective(variables, context)

    def batches(self) -> list[list[tuple[int, int]]]:
        order: list[int] = []
        grouped: dict[int, list[tuple[int, int]]] = {}
        for batch_id, realization, perturbation in self.rows:
            if batch_id not in grouped:
                order.append(batch_id)
                grouped[batch_id] = []
            grouped[batch_id].append((realization, perturbation))
        return [sorted(grouped[batch_id]) for batch_id in order]


def _record(config: dict[str, Any]) -> tuple[FunctionResults, GradientResults]:
    history = HistoryHandler()
    optimize(config, _INITIAL, _objective, handlers=[history])
    results = history["results"]
    assert results is not None
    f0 = next(item for item in results if isinstance(item, FunctionResults))
    g0 = next(item for item in results if isinstance(item, GradientResults))
    return f0, g0


def _shifted(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    # Offset from the recorded run, so a value matching the record can only have
    # come from the supplied results.
    return _objective(variables, context) + 1.0


def _first_function_result(history: HistoryHandler) -> FunctionResults:
    results = history["results"]
    assert results is not None
    return next(item for item in results if isinstance(item, FunctionResults))


def test_initial_values_remove_the_initial_evaluations(config: Any) -> None:
    f0, g0 = _record(config)
    plain = _Recorder()
    optimize(config, _INITIAL, plain)
    restarted = _Recorder()
    optimize(config, _INITIAL, restarted, f0=f0, g0=g0)
    assert len(plain.rows) - len(restarted.rows) == _REALIZATIONS * (1 + _PERTURBATIONS)


def test_initial_function_results_leave_only_the_perturbations(config: Any) -> None:
    f0, _ = _record(config)
    recorder = _Recorder()
    optimize(config, _INITIAL, recorder, f0=f0)
    assert recorder.batches()[0] == [
        (realization, perturbation)
        for realization in range(_REALIZATIONS)
        for perturbation in range(_PERTURBATIONS)
    ]


def test_initial_gradient_results_leave_only_the_unperturbed_rows(config: Any) -> None:
    _, g0 = _record(config)
    recorder = _Recorder()
    optimize(config, _INITIAL, recorder, g0=g0)
    assert recorder.batches()[0] == [(0, -1), (1, -1)]
    assert recorder.batches()[1] == [(0, -1), (1, -1)]


def test_initial_values_reproduce_the_recorded_aggregation(config: Any) -> None:
    f0, g0 = _record(config)
    history = HistoryHandler()
    optimize(config, _INITIAL, _shifted, handlers=[history], f0=f0, g0=g0)
    restarted = _first_function_result(history)
    assert f0.functions is not None
    assert restarted.functions is not None
    assert np.allclose(restarted.evaluations.objectives, f0.evaluations.objectives)
    assert np.allclose(restarted.functions.objectives, f0.functions.objectives)


def test_changed_realization_weights_reaggregate_the_same_values(config: Any) -> None:
    f0, _ = _record(config)
    config["realizations"]["weights"] = [3.0, 1.0]
    history = HistoryHandler()
    optimize(config, _INITIAL, _shifted, handlers=[history], f0=f0)
    restarted = _first_function_result(history)
    assert f0.functions is not None
    assert restarted.functions is not None
    assert np.allclose(restarted.evaluations.objectives, f0.evaluations.objectives)
    assert not np.allclose(restarted.functions.objectives, f0.functions.objectives)


def test_added_realization_is_the_only_one_evaluated(config: Any) -> None:
    f0, _ = _record(config)
    config["realizations"]["weights"] = [1.0, 1.0, 1.0]
    recorder = _Recorder()
    optimize(config, _INITIAL, recorder, f0=f0)
    assert recorder.batches()[0] == [(2, -1)]


def test_added_perturbation_is_the_only_one_evaluated(config: Any) -> None:
    _, g0 = _record(config)
    config["gradient"]["number_of_perturbations"] = _PERTURBATIONS + 1
    recorder = _Recorder()
    optimize(config, _INITIAL, recorder, g0=g0)
    assert recorder.batches()[1] == [(0, _PERTURBATIONS), (1, _PERTURBATIONS)]


def test_initial_gradient_results_keep_the_recorded_perturbed_variables(
    config: Any,
) -> None:
    _, g0 = _record(config)
    # A different seed draws different perturbations, so matching the recorded
    # points can only come from the supplied ones.
    config["variables"]["seed"] = 99
    history = HistoryHandler()
    optimize(config, _INITIAL, _objective, handlers=[history], g0=g0)
    results = history["results"]
    assert results is not None
    restarted = next(item for item in results if isinstance(item, GradientResults))
    assert np.allclose(restarted.perturbed_variables, g0.perturbed_variables)
    assert np.allclose(
        restarted.evaluations.perturbed_objectives,
        g0.evaluations.perturbed_objectives,
    )


def test_initial_values_cover_a_combined_batch(config: Any, evaluator: Any) -> None:
    f0, g0 = _record(config)
    recorder = _Recorder()
    ensemble = EnsembleEvaluator(
        EnOptContext.model_validate(config),
        evaluator([recorder]).eval,
        f0=f0,
        g0=g0,
    )
    ensemble.calculate(_INITIAL, compute_functions=True, compute_gradients=True)
    assert recorder.rows == []


def test_initial_gradient_results_leave_the_combined_batch_unperturbed(
    config: Any, evaluator: Any
) -> None:
    _, g0 = _record(config)
    recorder = _Recorder()
    ensemble = EnsembleEvaluator(
        EnOptContext.model_validate(config), evaluator([recorder]).eval, g0=g0
    )
    ensemble.calculate(_INITIAL, compute_functions=True, compute_gradients=True)
    assert recorder.batches() == [[(0, -1), (1, -1)]]


def test_initial_values_do_not_serve_later_batches(config: Any) -> None:
    f0, g0 = _record(config)
    recorder = _Recorder()
    optimize(config, _INITIAL, recorder, f0=f0, g0=g0)
    # Only the batch after the initial point reaches the evaluation function.
    assert recorder.batches() == [[(0, -1), (1, -1)]]


def test_initial_function_results_with_wrong_objective_count_raise(
    config: Any,
) -> None:
    f0, _ = _record(config)
    config["objectives"] = {"weights": [1.0, 1.0]}
    with pytest.raises(ValueError, match="1 objectives, expected 2"):
        optimize(config, _INITIAL, _objective, f0=f0)


def test_initial_gradient_results_with_wrong_variable_count_raise(
    config: Any,
) -> None:
    _, g0 = _record(config)
    config["variables"]["variable_count"] = 2
    with pytest.raises(ValueError, match="gradient results have 3 variables"):
        optimize(config, np.zeros(2), _objective, g0=g0)


def test_initial_function_results_without_configured_constraints_raise(
    config: Any,
) -> None:
    f0, _ = _record(config)
    config["nonlinear_constraints"] = {"lower_bounds": 0.0, "upper_bounds": 0.4}
    with pytest.raises(ValueError, match="function results have no constraints"):
        optimize(config, _INITIAL, _objective, f0=f0)


def test_initial_function_results_at_another_point_raise(config: Any) -> None:
    f0, _ = _record(config)
    with pytest.raises(
        ValueError, match="function results are not at the evaluated variables"
    ):
        optimize(config, _INITIAL + 1.0, _objective, f0=f0)


def test_initial_gradient_results_at_another_point_raise(config: Any) -> None:
    _, g0 = _record(config)
    with pytest.raises(
        ValueError, match="gradient results are not at the evaluated variables"
    ):
        optimize(config, _INITIAL + 1.0, _objective, g0=g0)


def test_initial_function_results_reject_several_variable_vectors(
    config: Any, evaluator: Any
) -> None:
    f0, _ = _record(config)
    ensemble = EnsembleEvaluator(
        EnOptContext.model_validate(config), evaluator([_objective]).eval, f0=f0
    )
    variables = np.repeat(_INITIAL[np.newaxis, :], 2, axis=0)
    with pytest.raises(ValueError, match="need a single variable vector"):
        ensemble.calculate(variables, compute_functions=True, compute_gradients=False)


def test_optimize_many_spreads_initial_values_over_the_runs(config: Any) -> None:
    f0, g0 = _record(config)
    first = _Recorder()
    second = _Recorder()
    optimize_many(
        [config, config],
        _INITIAL,
        [first, second],
        f0=[f0, None],
        g0=[g0, None],
    )
    assert len(second.rows) - len(first.rows) == _REALIZATIONS * (1 + _PERTURBATIONS)


def test_optimize_many_rejects_a_mismatched_initial_value_sequence(
    config: Any,
) -> None:
    f0, _ = _record(config)
    with pytest.raises(ValueError, match="f0 sequence length"):
        optimize_many([config, config], _INITIAL, _objective, f0=[f0])
