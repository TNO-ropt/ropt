from typing import Any

import numpy as np
import pytest

from ropt.config.enopt import EnOptConfig
from ropt.results import FunctionEvaluations, FunctionResults, Functions, Realizations

pytest.importorskip("xarray")


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
        config=config,
        weighted_objective=np.array(1.0),
        objectives=np.array([1.0, 2.0]),
    )
    return FunctionResults(
        plan_id=(0,),
        result_id=0,
        batch_id=1,
        metadata={"foo": "bar"},
        evaluations=evaluations,
        realizations=realizations,
        functions=functions,
    )


def test_to_dataset(enopt_config: Any, function_result: FunctionResults) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    dataset = function_result.to_dataset(config, "evaluations")
    for field in dataset:
        assert np.all(
            dataset[field].to_numpy()
            == getattr(function_result.evaluations, str(field)),
        )
    assert (
        tuple(dataset.coords["variable-axis"].values)
        == config.variables.get_formatted_names()
    )
    assert tuple(dataset.coords["realization-axis"].values) == config.realizations.names
    assert tuple(dataset.coords["objective-axis"].values) == config.objectives.names


@pytest.mark.parametrize("add_metadata", [True, False])
def test_to_dataset_with_attrs(
    enopt_config: Any, function_result: FunctionResults, add_metadata: bool
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    dataset = function_result.to_dataset(
        config, "evaluations", add_metadata=add_metadata
    )
    assert dataset.attrs["result_id"] == 0
    assert dataset.attrs["batch_id"] == 1
    if add_metadata:
        assert dataset.attrs["metadata"] == {"foo": "bar"}
    else:
        assert "metadata" not in dataset.attrs


def test_to_dataset_formatter(
    enopt_config: Any, function_result: FunctionResults
) -> None:
    enopt_config["variables"]["names"] = [("x", 0), ("x", 1)]
    config = EnOptConfig.model_validate(enopt_config)
    dataset = function_result.to_dataset(config, "evaluations", select=["variables"])
    assert dataset.coords["variable-axis"].to_numpy().tolist() == ["x:0", "x:1"]
