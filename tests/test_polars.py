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
    ScaledFunctionResults,
    ScaledGradientResults,
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


def test_to_polars_scalar(function_result: FunctionResults) -> None:
    frame = function_result.to_polars(["target_objective"])
    assert frame.columns == ["batch_id", "target_objective"]
    assert frame.height == 1
    assert frame["batch_id"].to_list() == [1]
    assert frame["target_objective"].to_list() == [1.0]


def test_to_polars_function(function_result: FunctionResults) -> None:
    frame = function_result.to_polars(["functions.objectives"])
    assert frame.height == 2
    assert frame.columns == [
        "batch_id",
        "objective",
        "functions.objectives",
    ]
    assert frame["batch_id"].to_list() == [1, 1]
    assert frame["objective"].to_list() == ["fa", "fb"]


def test_to_polars_gradient(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        [
            "evaluations.perturbed_objectives",
            "evaluations.metadata.foo",
        ],
    )
    assert frame.height == gradient_result.evaluations.perturbed_objectives.size
    assert frame.columns[:4] == [
        "batch_id",
        "realization",
        "perturbation",
        "objective",
    ]
    rows = frame.select("batch_id", "realization", "perturbation").rows()
    idx = 0
    for real in gradient_result.names[AxisName.REALIZATION]:
        for pert in range(gradient_result.perturbed_variables.shape[1]):
            for _ in gradient_result.names[AxisName.OBJECTIVE]:
                assert rows[idx] == (1, real, pert)
                idx += 1


def test_to_polars_unstack1(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        ["perturbed_variables"],
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
    assert gradient_result.scaled.gradients is not None
    frame = gradient_result.to_polars(
        ["scaled.gradients.objectives"],
        unstack=[AxisName.OBJECTIVE, AxisName.VARIABLE],
    )
    assert frame.columns == [
        "batch_id",
        "scaled.gradients.objectives,fa,va",
        "scaled.gradients.objectives,fa,vb",
        "scaled.gradients.objectives,fb,va",
        "scaled.gradients.objectives,fb,vb",
    ]


def test_to_polars_unstack_only_variable(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        ["perturbed_variables"],
        unstack=[AxisName.VARIABLE],
    )
    assert frame.columns == [
        "batch_id",
        "realization",
        "perturbation",
        "perturbed_variables,va",
        "perturbed_variables,vb",
    ]


def test_to_polars_sep(gradient_result: GradientResults) -> None:
    frame = gradient_result.to_polars(
        ["perturbed_variables"],
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
        ["evaluations.metadata.foo"],
        unstack=[AxisName.PERTURBATION],
    )
    assert frame.columns == [
        "batch_id",
        "realization",
        "evaluations.metadata.foo,0",
        "evaluations.metadata.foo,1",
        "evaluations.metadata.foo,2",
        "evaluations.metadata.foo,3",
        "evaluations.metadata.foo,4",
    ]


def test_to_polars_missing_field(function_result: FunctionResults) -> None:
    assert function_result.to_polars(["functions.constraints"]).is_empty()
    assert function_result.to_polars([]).is_empty()


def test_to_polars_invalid_field(function_result: FunctionResults) -> None:
    with pytest.raises(ValueError, match="Not a field name: nonexistent"):
        function_result.to_polars(["nonexistent"])
    with pytest.raises(ValueError, match=r"Not a field name: functions\.nonexistent"):
        function_result.to_polars(["functions.nonexistent"])
    with pytest.raises(ValueError, match=r"Not a correct field name: functions\."):
        function_result.to_polars(["functions."])
    with pytest.raises(ValueError, match="Field holds sub-fields, not a value"):
        function_result.to_polars(["functions"])
    with pytest.raises(ValueError, match="Field holds a mapping, add a key"):
        function_result.to_polars(["evaluations.metadata"])


def test_to_polars_unknown_unstack_axis_is_rejected(
    function_result: FunctionResults,
) -> None:
    with pytest.raises(ValueError, match="Unknown axes to unstack: nonexistent"):
        function_result.to_polars(["variables"], unstack=["nonexistent"])


def _with_array_metadata(
    config: dict[str, Any], metadata: dict[str, Any]
) -> FunctionResults:
    context = EnOptContext.model_validate(config)
    return FunctionResults(
        batch_id=1,
        metadata={},
        names=context.names,
        variables=np.array([1.0, 2.0]),
        evaluations=FunctionEvaluations.create(
            objectives=np.arange(6, dtype=np.float64).reshape((3, 2)),
            metadata=metadata,
        ),
        realizations=Realizations(
            objective_weights=np.arange(6, dtype=np.float64).reshape((2, 3)),
            evaluated_realizations=np.ones(3, dtype=np.bool_),
        ),
        functions=Functions(objectives=np.array([1.0, 2.0])),
        target_objective=np.array(1.0),
        scaled=ScaledFunctionResults(
            variables=np.array([1.0, 2.0]),
            functions=Functions(objectives=np.array([1.0, 2.0])),
        ),
    )


def test_to_polars_array_metadata_spans_a_labeled_user_axis(
    config: dict[str, Any],
) -> None:
    config["names"]["pair"] = ("lo", "hi")
    result = _with_array_metadata(
        config, {"pair": np.arange(6, dtype=np.float64).reshape((3, 2))}
    )
    frame = result.to_polars(["evaluations.metadata.pair"], unstack=["pair"])
    assert frame.columns == [
        "batch_id",
        "realization",
        "evaluations.metadata.pair,lo",
        "evaluations.metadata.pair,hi",
    ]
    assert frame["evaluations.metadata.pair,lo"].to_list() == [0.0, 2.0, 4.0]
    assert frame["evaluations.metadata.pair,hi"].to_list() == [1.0, 3.0, 5.0]


def test_to_polars_unlabeled_user_axis_falls_back_to_indices(
    config: dict[str, Any],
) -> None:
    result = _with_array_metadata(
        config, {"pair": np.arange(6, dtype=np.float64).reshape((3, 2))}
    )
    frame = result.to_polars(["evaluations.metadata.pair"], unstack=["pair"])
    assert frame.columns[2:] == [
        "evaluations.metadata.pair,0",
        "evaluations.metadata.pair,1",
    ]


def test_to_polars_user_axis_stays_stacked_without_unstack(
    config: dict[str, Any],
) -> None:
    result = _with_array_metadata(
        config, {"pair": np.arange(6, dtype=np.float64).reshape((3, 2))}
    )
    frame = result.to_polars(["evaluations.metadata.pair"])
    assert frame.columns == [
        "batch_id",
        "realization",
        "pair",
        "evaluations.metadata.pair",
    ]
    assert frame["evaluations.metadata.pair"].to_list() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


_PARITY_CASES = [
    ("function_result", ["functions.objectives"], None),
    ("function_result", ["functions.objectives"], [AxisName.OBJECTIVE]),
    ("function_result", ["evaluations.objectives"], None),
    ("function_result", ["variables"], [AxisName.VARIABLE]),
    ("function_result", ["target_objective"], None),
    (
        "function_result",
        ["realizations.objective_weights", "realizations.evaluated_realizations"],
        None,
    ),
    (
        "gradient_result",
        ["evaluations.perturbed_objectives", "evaluations.metadata.foo"],
        None,
    ),
    (
        "gradient_result",
        ["perturbed_variables"],
        [AxisName.REALIZATION, AxisName.VARIABLE],
    ),
    (
        "gradient_result",
        ["perturbed_variables"],
        [AxisName.VARIABLE, AxisName.REALIZATION],
    ),
    ("gradient_result", ["perturbed_variables"], [AxisName.VARIABLE]),
    (
        "gradient_result",
        ["scaled.gradients.objectives"],
        [AxisName.OBJECTIVE, AxisName.VARIABLE],
    ),
    ("gradient_result", ["scaled.gradients.objectives"], None),
    ("gradient_result", ["target_gradient"], [AxisName.VARIABLE]),
    ("gradient_result", ["evaluations.metadata.foo"], [AxisName.PERTURBATION]),
    (
        "gradient_result",
        ["evaluations.perturbed_objectives", "evaluations.metadata.foo"],
        [AxisName.OBJECTIVE],
    ),
    (
        "gradient_result",
        ["gradients.objectives", "target_gradient"],
        [AxisName.OBJECTIVE, AxisName.VARIABLE],
    ),
]


@pytest.mark.parametrize("case", _PARITY_CASES)
@pytest.mark.parametrize("sep", [",", "::"])
def test_to_polars_pandas_parity(
    request: pytest.FixtureRequest,
    case: tuple[str, list[str], list[AxisName] | None],
    sep: str,
) -> None:
    pytest.importorskip("pandas")

    fixture, select, unstack = case
    result: Results = request.getfixturevalue(fixture)
    pandas_frame = result.to_pandas(select, unstack).reset_index()
    pandas_frame.columns = [
        sep.join(str(part) for part in column)
        if isinstance(column, tuple)
        else str(column)
        for column in pandas_frame.columns
    ]
    polars_frame = result.to_polars(select, unstack, sep=sep)

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
