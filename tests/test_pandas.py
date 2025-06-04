from typing import Any

import numpy as np
import pytest

from ropt.enums import AxisName
from ropt.results import (
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Gradients,
    Realizations,
)

pandas = pytest.importorskip("pandas")
from ropt.config import EnOptConfig
from ropt.results._pandas import _to_series


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0],
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "realizations": {
            "weights": [1.0] * 3,
        },
        "gradient": {
            "number_of_perturbations": 5,
        },
        "names": {
            AxisName.VARIABLE: ("va", "vb"),
            AxisName.REALIZATION: ("ra", "rb", "rc"),
            AxisName.OBJECTIVE: ("fa", "fb"),
        },
    }


@pytest.fixture(name="function_result")
def function_result_fixture(enopt_config: dict[str, Any]) -> FunctionResults:
    evaluations = FunctionEvaluations.create(
        variables=np.array([1.0, 2.0]),
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    realizations = Realizations(
        objective_weights=np.arange(6, dtype=np.float64).reshape((2, 3)),
        active_realizations=np.ones(3, dtype=np.bool_),
        failed_realizations=np.zeros(3, dtype=np.bool_),
    )
    functions = Functions.create(
        weighted_objective=np.array(1.0), objectives=np.array([1.0, 2.0])
    )
    config = EnOptConfig.model_validate(enopt_config)
    return FunctionResults(
        batch_id=1,
        metadata={},
        names=config.names,
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )


@pytest.fixture(name="gradient_result")
def gradient_result_fixture(enopt_config: dict[str, Any]) -> GradientResults:
    evaluations = GradientEvaluations(
        variables=np.array([1.0, 2.0]),
        perturbed_variables=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        perturbed_objectives=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        evaluation_info={"foo": np.arange(15, dtype=np.float64).reshape((3, 5))},
    )
    gradients = Gradients(
        weighted_objective=np.array([1.0, 2.0]),
        objectives=np.arange(4, dtype=np.float64).reshape((2, 2)),
    )
    config = EnOptConfig.model_validate(enopt_config)
    return GradientResults(
        batch_id=1,
        metadata={},
        names=config.names,
        evaluations=evaluations,
        realizations=Realizations(
            active_realizations=np.ones(36, dtype=np.bool_),
            failed_realizations=np.zeros(36, dtype=np.bool_),
        ),
        gradients=gradients,
    )


def test__to_series(gradient_result: GradientResults) -> None:
    series = _to_series(
        gradient_result.evaluations, "perturbed_variables", None, gradient_result.names
    )
    assert series is not None
    assert len(series) == gradient_result.evaluations.perturbed_variables.size
    assert series.index.names == [
        "realization",
        "perturbation",
        "variable",
    ]
    for v_idx, var in enumerate(gradient_result.names[AxisName.VARIABLE]):
        for r_idx, real in enumerate(gradient_result.names[AxisName.REALIZATION]):
            for pert in range(gradient_result.evaluations.perturbed_variables.shape[1]):
                assert (
                    series.loc[(real, pert, var)]
                    == gradient_result.evaluations.perturbed_variables[
                        r_idx, pert, v_idx
                    ]
                )


def test__to_series_evaluation_info(gradient_result: GradientResults) -> None:
    series = _to_series(
        gradient_result.evaluations,
        "evaluation_info",
        "foo",
        gradient_result.names,
    )
    assert series is not None
    info = np.array(gradient_result.evaluations.evaluation_info["foo"])
    assert len(series) == info.size
    assert series.index.names == [
        "realization",
        "perturbation",
    ]
    for r_idx, real in enumerate(gradient_result.names[AxisName.REALIZATION]):
        for pert in range(gradient_result.evaluations.perturbed_variables.shape[1]):
            assert series.loc[(real, pert)] == info[r_idx, pert]


def test_to_dataframe_function(function_result: FunctionResults) -> None:
    frame = function_result.to_dataframe(
        "functions",
        [
            "weighted_objective",
            "objectives",
        ],
    )
    assert len(frame) == 2
    assert frame.index.names == ["batch_id", "objective"]
    assert frame.index[0] == (1, "fa")
    assert frame.index[1] == (1, "fb")


def test_to_dataframe_gradient(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_dataframe(
        "evaluations",
        [
            "variables",
            "perturbed_variables",
            "perturbed_objectives",
            "evaluation_info.foo",
        ],
    )
    assert len(frame) == gradient_result.evaluations.perturbed_variables.size * 2
    assert frame.index.names == [
        "batch_id",
        "variable",
        "realization",
        "perturbation",
        "objective",
    ]
    idx = 0
    for var in gradient_result.names[AxisName.VARIABLE]:
        for real in gradient_result.names[AxisName.REALIZATION]:
            for pert in range(gradient_result.evaluations.perturbed_variables.shape[1]):
                for fnc in gradient_result.names[AxisName.OBJECTIVE]:
                    assert frame.index[idx] == (1, var, real, pert, fnc)
                    idx += 1


def test_to_dataframe_unstack1(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_dataframe(
        "evaluations",
        select=["perturbed_variables"],
        unstack=[AxisName.REALIZATION, AxisName.VARIABLE],
    )
    assert frame.index.names == ["batch_id", "perturbation"]
    assert list(frame.columns.values) == [
        ("perturbed_variables", "ra", "va"),
        ("perturbed_variables", "ra", "vb"),
        ("perturbed_variables", "rb", "va"),
        ("perturbed_variables", "rb", "vb"),
        ("perturbed_variables", "rc", "va"),
        ("perturbed_variables", "rc", "vb"),
    ]


def test_to_dataframe_unstack2(gradient_result: GradientResults) -> None:
    assert gradient_result.gradients is not None
    frame = gradient_result.to_dataframe(
        "gradients",
        select=["objectives", "weighted_objective"],
        unstack=[AxisName.OBJECTIVE, AxisName.VARIABLE],
    )
    assert list(frame.columns.values) == [
        ("objectives", "fa", "va"),
        ("objectives", "fa", "vb"),
        ("objectives", "fb", "va"),
        ("objectives", "fb", "vb"),
        ("weighted_objective", "va"),
        ("weighted_objective", "vb"),
    ]


def test_to_dataframe_unstack_only_variable(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_dataframe(
        "evaluations",
        select=["perturbed_objectives", "perturbed_variables"],
        unstack=[AxisName.VARIABLE],
    )
    assert frame.index.names == [
        "batch_id",
        "realization",
        "perturbation",
        "objective",
    ]
    assert list(frame.columns.values) == [
        "perturbed_objectives",
        ("perturbed_variables", "va"),
        ("perturbed_variables", "vb"),
    ]


def test_to_dataframe_join(function_result: FunctionResults) -> None:
    frame1 = function_result.to_dataframe("evaluations", ["variables", "objectives"])
    frame2 = function_result.to_dataframe(
        "functions", ["weighted_objective", "objectives"]
    )
    frame1.columns = pandas.Index(
        "_".join(column) if isinstance(column, tuple) else column
        for column in frame1.columns.to_numpy()
    )
    frame2.columns = pandas.Index(
        "_".join(column) if isinstance(column, tuple) else column
        for column in frame2.columns.to_numpy()
    )
    frame = frame1.join(frame2, how="inner", lsuffix="_eval", rsuffix="_func")
    assert not frame.empty
    assert len(frame.columns) == len(frame1.columns) + len(frame2.columns)
