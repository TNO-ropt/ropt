import math
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
    Results,
)

pytest.importorskip("polars")

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
        variables=np.array([1.0, 2.0]),
        objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
    )
    realizations = Realizations(
        objective_weights=np.arange(6, dtype=np.float64).reshape((2, 3)),
        evaluated_realizations=np.ones(3, dtype=np.bool_),
    )
    functions = Functions.create(
        target_objective=np.array(1.0), objectives=np.array([1.0, 2.0])
    )
    context = EnOptContext.model_validate(config)
    return FunctionResults(
        batch_id=1,
        metadata={},
        names=context.names,
        evaluation_point=evaluations.variables,
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )


@pytest.fixture(name="gradient_result")
def gradient_result_fixture(config: dict[str, Any]) -> GradientResults:
    evaluations = GradientEvaluations(
        variables=np.array([1.0, 2.0]),
        perturbed_variables=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        perturbed_objectives=np.arange(30, dtype=np.float64).reshape((3, 5, 2)),
        metadata={"foo": np.arange(15, dtype=np.float64).reshape((3, 5))},
    )
    gradients = Gradients(
        target_objective=np.array([1.0, 2.0]),
        objectives=np.arange(4, dtype=np.float64).reshape((2, 2)),
    )
    context = EnOptContext.model_validate(config)
    return GradientResults(
        batch_id=1,
        metadata={},
        names=context.names,
        evaluation_point=evaluations.variables,
        evaluations=evaluations,
        realizations=Realizations(
            evaluated_realizations=np.ones(36, dtype=np.bool_),
        ),
        gradients=gradients,
    )


def test_to_polars_scalar(function_result: FunctionResults) -> None:
    frame = function_result.to_polars("functions", ["target_objective"])
    assert frame.columns == ["batch_id", "target_objective"]
    assert frame.height == 1
    assert frame["batch_id"].to_list() == [1]
    assert frame["target_objective"].to_list() == [1.0]


def test_to_polars_function(function_result: FunctionResults) -> None:
    frame = function_result.to_polars(
        "functions",
        [
            "target_objective",
            "objectives",
        ],
    )
    assert frame.height == 2
    assert frame.columns == [
        "batch_id",
        "objective",
        "target_objective",
        "objectives",
    ]
    assert frame["batch_id"].to_list() == [1, 1]
    assert frame["objective"].to_list() == ["fa", "fb"]


def test_to_polars_gradient(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        "evaluations",
        [
            "variables",
            "perturbed_variables",
            "perturbed_objectives",
            "metadata.foo",
        ],
    )
    assert frame.height == gradient_result.evaluations.perturbed_variables.size * 2
    assert frame.columns[:5] == [
        "batch_id",
        "variable",
        "realization",
        "perturbation",
        "objective",
    ]
    rows = frame.select("batch_id", "variable", "realization", "perturbation").rows()
    idx = 0
    for var in gradient_result.names[AxisName.VARIABLE]:
        for real in gradient_result.names[AxisName.REALIZATION]:
            for pert in range(gradient_result.evaluations.perturbed_variables.shape[1]):
                for _ in gradient_result.names[AxisName.OBJECTIVE]:
                    assert rows[idx] == (1, var, real, pert)
                    idx += 1


def test_to_polars_unstack1(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        "evaluations",
        select=["perturbed_variables"],
        unstack=[AxisName.REALIZATION, AxisName.VARIABLE],
    )
    assert frame.columns == [
        "batch_id",
        "perturbation",
        "perturbed_variables,ra,va",
        "perturbed_variables,ra,vb",
        "perturbed_variables,rb,va",
        "perturbed_variables,rb,vb",
        "perturbed_variables,rc,va",
        "perturbed_variables,rc,vb",
    ]


def test_to_polars_unstack2(gradient_result: GradientResults) -> None:
    assert gradient_result.gradients is not None
    frame = gradient_result.to_polars(
        "gradients",
        select=["objectives", "target_objective"],
        unstack=[AxisName.OBJECTIVE, AxisName.VARIABLE],
    )
    assert frame.columns == [
        "batch_id",
        "objectives,fa,va",
        "objectives,fa,vb",
        "objectives,fb,va",
        "objectives,fb,vb",
        "target_objective,va",
        "target_objective,vb",
    ]


def test_to_polars_unstack_only_variable(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        "evaluations",
        select=["perturbed_objectives", "perturbed_variables"],
        unstack=[AxisName.VARIABLE],
    )
    assert frame.columns == [
        "batch_id",
        "realization",
        "perturbation",
        "objective",
        "perturbed_objectives",
        "perturbed_variables,va",
        "perturbed_variables,vb",
    ]


def test_to_polars_sep(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        "evaluations",
        select=["perturbed_variables"],
        unstack=[AxisName.REALIZATION, AxisName.VARIABLE],
        sep="::",
    )
    assert frame.columns[2:] == [
        "perturbed_variables::ra::va",
        "perturbed_variables::ra::vb",
        "perturbed_variables::rb::va",
        "perturbed_variables::rb::vb",
        "perturbed_variables::rc::va",
        "perturbed_variables::rc::vb",
    ]


def test_to_polars_unnamed_axis(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        "evaluations",
        select=["metadata.foo"],
        unstack=[AxisName.PERTURBATION],
    )
    assert frame.columns == [
        "batch_id",
        "realization",
        "metadata.foo,0",
        "metadata.foo,1",
        "metadata.foo,2",
        "metadata.foo,3",
        "metadata.foo,4",
    ]


def test_to_polars_missing_field(function_result: FunctionResults) -> None:
    assert function_result.to_polars("functions", ["constraints"]).is_empty()
    assert function_result.to_polars("functions", []).is_empty()


def test_to_polars_invalid_field(function_result: FunctionResults) -> None:
    with pytest.raises(AttributeError, match="Invalid result field: nonexistent"):
        function_result.to_polars("nonexistent", ["objectives"])
    with pytest.raises(ValueError, match="Not a field name: nonexistent"):
        function_result.to_polars("functions", ["nonexistent"])
    with pytest.raises(ValueError, match=r"Not a correct field name: objectives\."):
        function_result.to_polars("functions", ["objectives."])


_PARITY_CASES = [
    ("function_result", "functions", ["target_objective"], None),
    ("function_result", "functions", ["target_objective", "objectives"], None),
    ("function_result", "functions", ["objectives", "target_objective"], None),
    (
        "function_result",
        "functions",
        ["target_objective", "objectives"],
        [AxisName.OBJECTIVE],
    ),
    ("function_result", "evaluations", ["variables", "objectives"], None),
    (
        "function_result",
        "evaluations",
        ["variables", "objectives"],
        [AxisName.VARIABLE],
    ),
    (
        "function_result",
        "realizations",
        ["objective_weights", "evaluated_realizations"],
        None,
    ),
    (
        "gradient_result",
        "evaluations",
        ["variables", "perturbed_variables", "perturbed_objectives", "metadata.foo"],
        None,
    ),
    (
        "gradient_result",
        "evaluations",
        ["perturbed_variables"],
        [AxisName.REALIZATION, AxisName.VARIABLE],
    ),
    (
        "gradient_result",
        "evaluations",
        ["perturbed_variables"],
        [AxisName.VARIABLE, AxisName.REALIZATION],
    ),
    (
        "gradient_result",
        "evaluations",
        ["perturbed_objectives", "perturbed_variables"],
        [AxisName.VARIABLE],
    ),
    (
        "gradient_result",
        "gradients",
        ["objectives", "target_objective"],
        [AxisName.OBJECTIVE, AxisName.VARIABLE],
    ),
    ("gradient_result", "gradients", ["target_objective", "objectives"], None),
    (
        "gradient_result",
        "evaluations",
        ["metadata.foo"],
        [AxisName.PERTURBATION],
    ),
    (
        "gradient_result",
        "evaluations",
        ["perturbed_variables", "metadata.foo"],
        [AxisName.VARIABLE],
    ),
]


@pytest.mark.parametrize("case", _PARITY_CASES)
@pytest.mark.parametrize("sep", [",", "::"])
def test_to_polars_pandas_parity(
    request: pytest.FixtureRequest,
    case: tuple[str, str, list[str], list[AxisName] | None],
    sep: str,
) -> None:
    pytest.importorskip("pandas")

    fixture, field, select, unstack = case
    result: Results = request.getfixturevalue(fixture)
    pandas_frame = result.to_dataframe(field, select, unstack).reset_index()
    pandas_frame.columns = [
        sep.join(str(part) for part in column)
        if isinstance(column, tuple)
        else str(column)
        for column in pandas_frame.columns
    ]
    polars_frame = result.to_polars(field, select, unstack, sep=sep)

    assert polars_frame.columns == list(pandas_frame.columns)
    assert polars_frame.height == len(pandas_frame)
    for column in pandas_frame.columns:
        expected = list(pandas_frame[column])
        actual = polars_frame[column].to_list()
        for lhs, rhs in zip(actual, expected, strict=True):
            if isinstance(rhs, float) and math.isnan(rhs):
                assert math.isnan(lhs)
            else:
                assert lhs == rhs
