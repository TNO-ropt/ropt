from itertools import product
from typing import Any

import numpy as np
import pytest

from ropt.context import EnOptContext
from ropt.enums import AxisName
from ropt.results import (
    FunctionEvaluations,
    FunctionResults,
    Functions,
    GradientEvaluations,
    GradientResults,
    Gradients,
    Realizations,
    ScaledFunctionResults,
    ScaledGradientResults,
)
from ropt.results._frame_core import _get_field_data

pandas = pytest.importorskip("pandas")

initial_values = [0.0, 0.0]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
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
def function_result_fixture(config: dict[str, Any]) -> FunctionResults:
    evaluations = FunctionEvaluations.create(
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    realizations = Realizations(
        objective_weights=np.arange(6, dtype=np.float64).reshape((2, 3)),
        evaluated_realizations=np.ones(3, dtype=np.bool_),
    )
    functions = Functions(
        objectives=np.array([1.0, 2.0]),
    )
    context = EnOptContext.model_validate(config)
    return FunctionResults(
        batch_id=1,
        metadata={},
        names=context.names,
        variables=np.array([1.0, 2.0]),
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
        target_objective=np.array(1.0),
        scaled=ScaledFunctionResults(
            variables=np.array([1.0, 2.0]),
            functions=Functions(objectives=np.array([1.0, 2.0])),
        ),
    )


@pytest.fixture(name="gradient_result")
def gradient_result_fixture(config: dict[str, Any]) -> GradientResults:
    evaluations = GradientEvaluations(
        perturbed_objectives=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        metadata={"foo": np.arange(15, dtype=np.float64).reshape((3, 5))},
    )
    gradients = Gradients(
        objectives=np.arange(4, dtype=np.float64).reshape((2, 2)),
    )
    context = EnOptContext.model_validate(config)
    return GradientResults(
        batch_id=1,
        metadata={},
        names=context.names,
        variables=np.array([1.0, 2.0]),
        perturbed_variables=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(36, dtype=np.bool_),
        ),
        gradients=gradients,
        target_gradient=np.array([1.0, 2.0]),
        scaled=ScaledGradientResults(
            variables=np.array([1.0, 2.0]),
            perturbed_variables=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
            gradients=Gradients(
                objectives=np.arange(4, dtype=np.float64).reshape((2, 2)),
            ),
        ),
    )


def test__get_field_data(gradient_result: GradientResults) -> None:
    field_data = _get_field_data(
        gradient_result, "perturbed_variables", gradient_result.names
    )
    assert field_data is not None
    assert field_data.name == "perturbed_variables"
    assert [axis.value for axis in field_data.axes] == [
        "realization",
        "perturbation",
        "variable",
    ]
    assert len(field_data.data) == gradient_result.perturbed_variables.size
    values = dict(zip(product(*field_data.labels), field_data.data, strict=True))
    for v_idx, var in enumerate(gradient_result.names[AxisName.VARIABLE]):
        for r_idx, real in enumerate(gradient_result.names[AxisName.REALIZATION]):
            for pert in range(gradient_result.perturbed_variables.shape[1]):
                assert (
                    values[real, pert, var]
                    == gradient_result.perturbed_variables[r_idx, pert, v_idx]
                )


def test__get_field_data_metadata(gradient_result: GradientResults) -> None:
    field_data = _get_field_data(
        gradient_result, "evaluations.metadata.foo", gradient_result.names
    )
    assert field_data is not None
    assert field_data.name == "evaluations.metadata.foo"
    info = np.array(gradient_result.evaluations.metadata["foo"])
    assert len(field_data.data) == info.size
    assert [axis.value for axis in field_data.axes] == [
        "realization",
        "perturbation",
    ]
    values = dict(zip(product(*field_data.labels), field_data.data, strict=True))
    for r_idx, real in enumerate(gradient_result.names[AxisName.REALIZATION]):
        for pert in range(gradient_result.perturbed_variables.shape[1]):
            assert values[real, pert] == info[r_idx, pert]


def test_to_pandas_function(function_result: FunctionResults) -> None:
    frame = function_result.to_pandas(["functions.objectives"])
    assert len(frame) == 2
    assert frame.index.names == ["batch_id", "objective"]
    assert frame.index[0] == (1, "fa")
    assert frame.index[1] == (1, "fb")


def test_to_pandas_value_field(function_result: FunctionResults) -> None:
    frame = function_result.to_pandas(["target_objective"])
    assert list(frame.columns.values) == ["target_objective"]
    assert frame.index.names == ["batch_id"]
    assert frame["target_objective"].to_list() == [1.0]


def test_to_pandas_gradient(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_pandas(
        [
            "evaluations.perturbed_objectives",
            "evaluations.metadata.foo",
        ],
    )
    assert len(frame) == gradient_result.evaluations.perturbed_objectives.size
    assert frame.index.names == [
        "batch_id",
        "realization",
        "perturbation",
        "objective",
    ]
    idx = 0
    for real in gradient_result.names[AxisName.REALIZATION]:
        for pert in range(gradient_result.perturbed_variables.shape[1]):
            for fnc in gradient_result.names[AxisName.OBJECTIVE]:
                assert frame.index[idx] == (1, real, pert, fnc)
                idx += 1


def test_to_pandas_unstack1(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_pandas(
        ["perturbed_variables"],
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


def test_to_pandas_unstack2(gradient_result: GradientResults) -> None:
    assert gradient_result.scaled.gradients is not None
    frame = gradient_result.to_pandas(
        ["scaled.gradients.objectives"],
        unstack=[AxisName.OBJECTIVE, AxisName.VARIABLE],
    )
    assert list(frame.columns.values) == [
        ("scaled.gradients.objectives", "fa", "va"),
        ("scaled.gradients.objectives", "fa", "vb"),
        ("scaled.gradients.objectives", "fb", "va"),
        ("scaled.gradients.objectives", "fb", "vb"),
    ]


def test_to_pandas_unstack_only_variable(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_pandas(
        ["perturbed_variables"],
        unstack=[AxisName.VARIABLE],
    )
    assert frame.index.names == [
        "batch_id",
        "realization",
        "perturbation",
    ]
    assert list(frame.columns.values) == [
        ("perturbed_variables", "va"),
        ("perturbed_variables", "vb"),
    ]


def test_to_pandas_join(function_result: FunctionResults) -> None:
    frame1 = function_result.to_pandas(["evaluations.objectives"])
    frame2 = function_result.to_pandas(["functions.objectives"])
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
