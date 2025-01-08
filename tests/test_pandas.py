from typing import Any

import numpy as np
import pytest

from ropt.config.enopt import EnOptConfig
from ropt.enums import ResultAxis
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
from ropt.results._pandas import _to_series


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "names": ["x", "y"],
            "initial_values": [0.0, 0.0],
        },
        "objectives": {
            "names": ["f1", "f2"],
            "weights": [0.75, 0.25],
        },
        "realizations": {
            "names": ["r1", "r2", "r3"],
            "weights": [1.0] * 3,
        },
        "gradient": {
            "number_of_perturbations": 5,
        },
    }


@pytest.fixture(name="function_result")
def function_result_fixture(enopt_config: Any) -> FunctionResults:
    config = EnOptConfig.model_validate(enopt_config)
    evaluations = FunctionEvaluations.create(
        config=config,
        variables=np.array([1.0, 2.0]),
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    realizations = Realizations(
        objective_weights=np.arange(6, dtype=np.float64).reshape((2, 3)),
        failed_realizations=np.zeros(3, dtype=np.bool_),
    )
    functions = Functions.create(
        config=config, weighted_objective=np.array(1.0), objectives=np.array([1.0, 2.0])
    )
    return FunctionResults(
        plan_id=(0,),
        result_id=0,
        batch_id=1,
        metadata={},
        config=config,
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )


@pytest.fixture(name="gradient_result")
def gradient_result_fixture(enopt_config: Any) -> GradientResults:
    config = EnOptConfig.model_validate(enopt_config)
    evaluations = GradientEvaluations(
        variables=np.array([1.0, 2.0]),
        perturbed_variables=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        perturbed_objectives=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
    )
    gradients = Gradients(
        weighted_objective=np.array([1.0, 2.0]),
        objectives=np.arange(4, dtype=np.float64).reshape((2, 2)),
    )
    return GradientResults(
        plan_id=(0,),
        result_id=0,
        batch_id=1,
        metadata={},
        config=config,
        evaluations=evaluations,
        realizations=Realizations(
            failed_realizations=np.zeros(36, dtype=np.bool_),
        ),
        gradients=gradients,
    )


def test__to_series(gradient_result: GradientResults) -> None:
    names = {
        ResultAxis.VARIABLE: ("v1", "v2"),
        ResultAxis.REALIZATION: ("ra", "rb", "rc"),
    }
    series = _to_series(
        gradient_result.evaluations, (0,), 0, 1, "perturbed_variables", names
    )
    assert series is not None
    assert len(series) == gradient_result.evaluations.perturbed_variables.size
    assert series.index.names == [
        "plan_id",
        "result_id",
        "batch_id",
        "realization",
        "perturbation",
        "variable",
    ]
    for v_idx, var in enumerate(names[ResultAxis.VARIABLE]):
        for r_idx, real in enumerate(names[ResultAxis.REALIZATION]):
            for pert in range(gradient_result.evaluations.perturbed_variables.shape[1]):
                assert (
                    series.loc[((0,), 0, 1, real, pert, var)]
                    == gradient_result.evaluations.perturbed_variables[
                        r_idx,
                        pert,
                        v_idx,
                    ]
                )


def test_to_dataframe(gradient_result: GradientResults) -> None:
    names = {
        ResultAxis.VARIABLE: ("v1", "v2"),
        ResultAxis.REALIZATION: ("ra", "rb", "rc"),
        ResultAxis.OBJECTIVE: ("fa", "fb"),
    }
    frame = gradient_result.to_dataframe("evaluations", names=names)
    assert len(frame) == gradient_result.evaluations.perturbed_variables.size * 2
    assert frame.index.names == [
        "plan_id",
        "result_id",
        "batch_id",
        "variable",
        "realization",
        "perturbation",
        "objective",
    ]
    idx = 0
    for var in names[ResultAxis.VARIABLE]:
        for real in names[ResultAxis.REALIZATION]:
            for pert in range(gradient_result.evaluations.perturbed_variables.shape[1]):
                for fnc in names[ResultAxis.OBJECTIVE]:
                    assert frame.index[idx] == ((0,), 0, 1, var, real, pert, fnc)
                    idx += 1


def test_to_dataframe_formatter(
    enopt_config: Any, function_result: FunctionResults
) -> None:
    enopt_config["variables"]["names"] = [("x", 0), ("x", 1)]
    config = EnOptConfig.model_validate(enopt_config)
    function_result.config = config
    frame = function_result.to_dataframe("evaluations", select=["variables"])
    assert (
        tuple(frame.index.get_level_values("variable"))
        == config.variables.get_formatted_names()
    )
    assert list(frame.index.get_level_values("variable")) == ["x:0", "x:1"]


def test_to_dataframe_unstack(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_dataframe(
        "evaluations",
        select=["perturbed_variables"],
        unstack=[ResultAxis.REALIZATION, ResultAxis.VARIABLE],
    )
    assert frame.index.names == ["plan_id", "result_id", "batch_id", "perturbation"]
    assert list(frame.columns.values) == [
        ("perturbed_variables", "r1", "x"),
        ("perturbed_variables", "r1", "y"),
        ("perturbed_variables", "r2", "x"),
        ("perturbed_variables", "r2", "y"),
        ("perturbed_variables", "r3", "x"),
        ("perturbed_variables", "r3", "y"),
    ]


def test_to_dataframe_unstack2(gradient_result: GradientResults) -> None:
    assert gradient_result.gradients is not None
    frame = gradient_result.to_dataframe(
        "gradients",
        select=["objectives", "weighted_objective"],
        unstack=[ResultAxis.OBJECTIVE, ResultAxis.VARIABLE],
    )
    assert list(frame.columns.values) == [
        ("objectives", "f1", "x"),
        ("objectives", "f1", "y"),
        ("objectives", "f2", "x"),
        ("objectives", "f2", "y"),
        ("weighted_objective", "x"),
        ("weighted_objective", "y"),
    ]


def test_to_dataframe_unstack_only_variable(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_dataframe(
        "evaluations",
        select=["perturbed_objectives", "perturbed_variables"],
        unstack=[ResultAxis.VARIABLE],
    )
    assert frame.index.names == [
        "plan_id",
        "result_id",
        "batch_id",
        "realization",
        "perturbation",
        "objective",
    ]
    assert list(frame.columns.values) == [
        "perturbed_objectives",
        ("perturbed_variables", "x"),
        ("perturbed_variables", "y"),
    ]


def test_to_dataframe_join(function_result: FunctionResults) -> None:
    frame1 = function_result.to_dataframe("evaluations")
    frame2 = function_result.to_dataframe("functions")
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
