import copy
from typing import Any, Callable

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.config.enopt import (
    EnOptConfig,
    GradientConfig,
    LinearConstraintsConfig,
    NonlinearConstraintsConfig,
    ObjectiveFunctionsConfig,
    RealizationsConfig,
    VariablesConfig,
)
from ropt.enums import BoundaryType, ConstraintType, PerturbationType, VariableType


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": np.array([1, 2]),
        },
        "objectives": {
            "weights": [1.0],
        },
        "optimizer": {
            "method": "dummy",
        },
    }


def test_check_variable_names() -> None:
    config = {"names": ["a", "b"], "initial_values": np.array([1, 2])}
    variables = VariablesConfig.model_validate(config)
    assert variables.names == ("a", "b")

    config = {"names": [1, "b"], "initial_values": np.array([1, 2])}
    variables = VariablesConfig.model_validate(config)
    assert variables.names is not None
    assert isinstance(variables.names[0], int)
    assert isinstance(variables.names[1], str)

    config = {"names": [("a",), ("b",)], "initial_values": np.array([1, 2])}
    variables = VariablesConfig.model_validate(config)
    assert variables.names == (("a",), ("b",))

    config = {"names": [(1, "a"), "b"], "initial_values": np.array([1, 2])}
    variables = VariablesConfig.model_validate(config)
    assert variables.names is not None
    assert isinstance(variables.names[0], tuple)
    assert isinstance(variables.names[0][0], int)
    assert isinstance(variables.names[0][1], str)
    assert isinstance(variables.names[1], str)

    config = {"names": ["a", "a", "b", "b", "c"], "initial_values": 0}
    with pytest.raises(ValueError, match="duplicate names: a, b"):
        VariablesConfig.model_validate(config)


def test_check_variable_arrays() -> None:
    config = {"initial_values": np.array([1, 2]), "lower_bounds": np.array([0.0, 0.0])}

    for key in ["initial_values", "lower_bounds", "upper_bounds"]:
        config_copy = copy.deepcopy(config)

        variables = VariablesConfig.model_validate(config_copy)
        assert getattr(variables, key).ndim == 1
        assert len(getattr(variables, key)) == 2
        with pytest.raises(ValueError):  # noqa: PT011
            getattr(variables, key)[0] = 0

        config_copy[key] = np.array(0.0)
        variables = VariablesConfig.model_validate(config)
        assert getattr(variables, key).ndim == 1
        assert len(getattr(variables, key)) == 2

        config_copy[key] = np.array([0.0])
        variables = VariablesConfig.model_validate(config)
        assert getattr(variables, key).ndim == 1
        assert len(getattr(variables, key)) == 2

        config_copy[key] = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValidationError):
            VariablesConfig.model_validate(config_copy)


def test_check_variable_convert_array() -> None:
    config: dict[str, Any] = {"initial_values": [1, 2], "lower_bounds": [0, 0]}

    for key in ["initial_values", "lower_bounds", "upper_bounds"]:
        config_copy = copy.deepcopy(config)

        variables = VariablesConfig.model_validate(config_copy)
        assert getattr(variables, key).ndim == 1
        assert len(getattr(variables, key)) == 2
        with pytest.raises(ValueError):  # noqa: PT011
            getattr(variables, key)[0] = 0

        config_copy[key] = 0
        variables = VariablesConfig.model_validate(config)
        assert getattr(variables, key).ndim == 1
        assert len(getattr(variables, key)) == 2

        config_copy[key] = [0]
        variables = VariablesConfig.model_validate(config)
        assert getattr(variables, key).ndim == 1
        assert len(getattr(variables, key)) == 2

        config_copy[key] = [0, 0, 0]
        with pytest.raises(ValidationError):
            VariablesConfig.model_validate(config_copy)


def test_check_variable_arrays_types() -> None:
    config: dict[str, Any] = {"initial_values": np.array([1, 2])}
    variables = VariablesConfig.model_validate(config)
    assert variables.types is None

    config["types"] = VariableType.INTEGER
    variables = VariablesConfig.model_validate(config)
    assert variables.types is not None
    assert np.all(variables.types == [VariableType.INTEGER, VariableType.INTEGER])

    config["types"] = [VariableType.INTEGER, VariableType.REAL]
    variables = VariablesConfig.model_validate(config)
    assert variables.types is not None
    assert np.all(variables.types == [VariableType.INTEGER, VariableType.REAL])


def test_get_formatted_names() -> None:
    config: dict[str, Any] = {"initial_values": [0, 0]}
    variables = VariablesConfig.model_validate(config)
    joined = variables.get_formatted_names()
    assert joined is None

    config = {"names": ["a", ("b", "c"), "d"], "initial_values": 0}
    variables = VariablesConfig.model_validate(config)
    joined = variables.get_formatted_names()
    assert joined == ("a", "b:c", "d")

    config = {"names": [("a", "b", "c", "d")], "initial_values": 0}
    variables = VariablesConfig.model_validate(config)
    joined = variables.get_formatted_names()
    assert joined == ("a:b:c:d",)

    config = {"names": [("a", "b", "c", "d")], "initial_values": 0, "delimiters": "123"}
    variables = VariablesConfig.model_validate(config)
    joined = variables.get_formatted_names()
    assert joined == ("a1b2c3d",)

    config = {
        "names": [("a", "b", "c", "d")],
        "initial_values": 0,
        "delimiters": "1234",
    }
    variables = VariablesConfig.model_validate(config)
    joined = variables.get_formatted_names()
    assert joined == ("a1b2c3d",)

    config = {"names": [("a", "b", "c", "d")], "initial_values": 0, "delimiters": "12"}
    variables = VariablesConfig.model_validate(config)
    joined = variables.get_formatted_names()
    assert joined == ("a1b2c2d",)

    config = {"names": [("a", "b", "c", "d")], "initial_values": 0, "delimiters": ""}
    variables = VariablesConfig.model_validate(config)
    joined = variables.get_formatted_names()
    assert joined == ("abcd",)


def test_check_objective_function_arrays() -> None:
    config: dict[str, Any] = {
        "weights": np.array([1.0, 1.0]),
        "scales": np.array([1.0, 1.0]),
    }

    for key in ["scales", "weights"]:
        config_copy = copy.deepcopy(config)

        objectives = ObjectiveFunctionsConfig.model_validate(config_copy)
        assert getattr(objectives, key).ndim == 1
        assert len(getattr(objectives, key)) == 2
        with pytest.raises(ValueError):  # noqa: PT011
            getattr(objectives, key)[0] = 0

        config_copy[key] = np.array(0.0)
        objectives = ObjectiveFunctionsConfig.model_validate(config)
        assert getattr(objectives, key).ndim == 1
        assert len(getattr(objectives, key)) == 2

        config_copy[key] = np.array([0.0])
        objectives = ObjectiveFunctionsConfig.model_validate(config)
        assert getattr(objectives, key).ndim == 1
        assert len(getattr(objectives, key)) == 2

        config_copy[key] = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValidationError):
            ObjectiveFunctionsConfig.model_validate(config_copy)

        assert objectives.weights.sum() == 1.0


def test_check_objective_function_convert_arrays() -> None:
    config: dict[str, Any] = {"weights": [1, 1], "scales": [1, 1]}

    for key in ["scales", "weights"]:
        config_copy = copy.deepcopy(config)

        objectives = ObjectiveFunctionsConfig.model_validate(config_copy)
        assert getattr(objectives, key).ndim == 1
        assert len(getattr(objectives, key)) == 2

        config_copy[key] = 1.0
        objectives = ObjectiveFunctionsConfig.model_validate(config)
        assert getattr(objectives, key).ndim == 1
        assert len(getattr(objectives, key)) == 2

        config_copy[key] = [1.0]
        objectives = ObjectiveFunctionsConfig.model_validate(config)
        assert getattr(objectives, key).ndim == 1
        assert len(getattr(objectives, key)) == 2

        config_copy[key] = [1.0, 1.0, 1.0]
        with pytest.raises(ValidationError):
            ObjectiveFunctionsConfig.model_validate(config_copy)

        assert objectives.weights.sum() == 1.0


def test_check_linear_constraints() -> None:
    config = {
        "coefficients": np.array([[1, 2], [1, 2], [1, 2]]),
        "rhs_values": np.array([1, 2, 3]),
        "types": [ConstraintType.LE, ConstraintType.GE, ConstraintType.EQ],
    }
    linear_constraints = LinearConstraintsConfig.model_validate(config)
    assert linear_constraints.coefficients is not None
    with pytest.raises(ValueError):  # noqa: PT011
        linear_constraints.coefficients[0, 0] = 0
    with pytest.raises(ValueError):  # noqa: PT011
        linear_constraints.rhs_values[0] = 0


def test_check_linear_constraints_convert() -> None:
    config = {
        "coefficients": [[1, 2], [1, 2], [1, 2]],
        "rhs_values": [1, 2, 3],
        "types": [ConstraintType.LE, ConstraintType.GE, ConstraintType.EQ],
    }
    LinearConstraintsConfig.model_validate(config)


def test_check_linear_constraints_vector_shapes() -> None:
    config = {
        "coefficients": [[1, 2, 3], [1, 2, 3]],
        "rhs_values": [1, 2],
        "types": [ConstraintType.LE, ConstraintType.GE],
    }
    LinearConstraintsConfig.model_validate(config)

    config_copy = copy.deepcopy(config)
    config_copy["rhs_values"] = [1, 2, 3]
    with pytest.raises(
        ValueError,
        match="rhs_values cannot be broadcasted to a length of 2",
    ):
        LinearConstraintsConfig.model_validate(config_copy)

    config_copy = copy.deepcopy(config)
    config_copy["types"] = [ConstraintType.LE, ConstraintType.GE, ConstraintType.EQ]
    with pytest.raises(
        ValueError,
        match="types cannot be broadcasted to a length of 2",
    ):
        LinearConstraintsConfig.model_validate(config_copy)


def test_check_nonlinear_constraint_arrays() -> None:
    config = {
        "types": [ConstraintType.EQ, ConstraintType.LE],
        "rhs_values": np.array([1.0, 1.0]),
    }

    for key in ["rhs_values", "scales"]:
        config_copy = copy.deepcopy(config)

        nonlinear_constraints = NonlinearConstraintsConfig.model_validate(config_copy)
        assert getattr(nonlinear_constraints, key).ndim == 1
        assert len(getattr(nonlinear_constraints, key)) == 2
        with pytest.raises(ValueError):  # noqa: PT011
            getattr(nonlinear_constraints, key)[0] = 0

        config_copy[key] = np.array(1.0)
        nonlinear_constraints = NonlinearConstraintsConfig.model_validate(config)
        assert getattr(nonlinear_constraints, key).ndim == 1
        assert len(getattr(nonlinear_constraints, key)) == 2

        config_copy[key] = np.array([1.0])
        nonlinear_constraints = NonlinearConstraintsConfig.model_validate(config)
        assert getattr(nonlinear_constraints, key).ndim == 1
        assert len(getattr(nonlinear_constraints, key)) == 2

        config_copy[key] = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValidationError):
            NonlinearConstraintsConfig.model_validate(config_copy)


def test_check_nonlinear_constraint_convert_arrays() -> None:
    config = {
        "types": [ConstraintType.EQ, ConstraintType.LE],
        "rhs_values": [1.0, 1.0],
    }

    for key in ["rhs_values", "scales"]:
        config_copy = copy.deepcopy(config)

        nonlinear_constraints = NonlinearConstraintsConfig.model_validate(config_copy)
        assert getattr(nonlinear_constraints, key).ndim == 1
        assert len(getattr(nonlinear_constraints, key)) == 2

        config_copy[key] = 1.0
        nonlinear_constraints = NonlinearConstraintsConfig.model_validate(config)
        assert getattr(nonlinear_constraints, key).ndim == 1
        assert len(getattr(nonlinear_constraints, key)) == 2

        config_copy[key] = [1.0]
        nonlinear_constraints = NonlinearConstraintsConfig.model_validate(config)
        assert getattr(nonlinear_constraints, key).ndim == 1
        assert len(getattr(nonlinear_constraints, key)) == 2

        config_copy[key] = [1.0, 1.0, 1.0]
        with pytest.raises(ValidationError):
            NonlinearConstraintsConfig.model_validate(config_copy)


def test_check_realization_names() -> None:
    config = {"names": [1, "2"]}
    realizations = RealizationsConfig.model_validate(config)
    assert realizations.names is not None
    assert isinstance(realizations.names[0], int)
    assert isinstance(realizations.names[1], str)


def test_check_realization_arrays() -> None:
    config: dict[str, Any] = {"weights": np.array([1.0, 1.0])}

    realizations = RealizationsConfig.model_validate(config)
    assert realizations.weights.ndim == 1
    assert len(realizations.weights) == 2
    with pytest.raises(ValueError):  # noqa: PT011
        realizations.weights[0] = 0

    config["weights"] = np.array(1.0)
    realizations = RealizationsConfig.model_validate(config)
    assert realizations.weights.ndim == 1
    assert len(realizations.weights) == 1

    config["weights"] = np.array([1.0])
    realizations = RealizationsConfig.model_validate(config)
    assert realizations.weights.ndim == 1
    assert len(realizations.weights) == 1

    config["weights"] = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValidationError):
        RealizationsConfig.model_validate(config)


def test_check_realization_convert_arrays() -> None:
    config: dict[str, Any] = {"weights": [1, 1]}

    realizations = RealizationsConfig.model_validate(config)
    assert realizations.weights.ndim == 1
    assert len(realizations.weights) == 2
    with pytest.raises(ValueError):  # noqa: PT011
        realizations.weights[0] = 0

    config["weights"] = 1.0
    realizations = RealizationsConfig.model_validate(config)
    assert realizations.weights.ndim == 1
    assert len(realizations.weights) == 1

    config["weights"] = [1.0]
    realizations = RealizationsConfig.model_validate(config)
    assert realizations.weights.ndim == 1
    assert len(realizations.weights) == 1

    config["weights"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValidationError):
        RealizationsConfig.model_validate(config)


def test_check_perturbations() -> None:
    GradientConfig()
    gradients = GradientConfig(perturbation_magnitudes=np.array([0.1]))
    assert gradients.perturbation_magnitudes == np.array([0.1])


def test_check_config(enopt_config: Any) -> None:
    EnOptConfig.model_validate(enopt_config)


def test_check_config_linear_constraints(enopt_config: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 2, 3], [2, 3, 4]],
        "rhs_values": [1, 2],
        "types": [ConstraintType.GE, ConstraintType.GE],
    }
    with pytest.raises(
        ValueError,
        match="the coefficients matrix should have 2 columns",
    ):
        EnOptConfig.model_validate(enopt_config)


def test_check_config_perturbations(enopt_config: Any) -> None:
    enopt_config["gradient"] = {
        "perturbation_magnitudes": [1] * 2,
        "boundary_types": [BoundaryType.TRUNCATE_BOTH] * 2,
        "perturbation_types": [PerturbationType.ABSOLUTE] * 2,
    }
    EnOptConfig.model_validate(enopt_config)

    config_copy = copy.deepcopy(enopt_config)
    config_copy["gradient"]["perturbation_magnitudes"] = [1] * 3
    with pytest.raises(
        ValueError,
        match="the perturbation magnitudes cannot be broadcasted to a length of 2",
    ):
        EnOptConfig.model_validate(config_copy)

    config_copy = copy.deepcopy(enopt_config)
    config_copy["gradient"]["boundary_types"] = [BoundaryType.TRUNCATE_BOTH] * 3
    with pytest.raises(
        ValueError, match="perturbation boundary_types must have 2 items"
    ):
        EnOptConfig.model_validate(config_copy)

    config_copy = copy.deepcopy(enopt_config)
    config_copy["gradient"]["perturbation_types"] = [PerturbationType.ABSOLUTE] * 3
    with pytest.raises(ValueError, match="perturbation types must have 2 items"):
        EnOptConfig.model_validate(config_copy)


def test_check_config_min_success(enopt_config: Any) -> None:
    def gen_config(pert_min: int | None, real_min: int | None) -> dict[str, Any]:
        config: dict[str, Any] = copy.deepcopy(enopt_config)
        config["realizations"] = {"weights": 4 * [1.0]}
        config["gradient"] = {}
        if pert_min is not None:
            config["gradient"]["perturbation_min_success"] = pert_min
        if real_min is not None:
            config["realizations"]["realization_min_success"] = real_min
        return config

    pert_test_map = {None: 5, 1: 1, 4: 4, 7: 5}
    real_test_map = {None: 4, 1: 1, 3: 3, 7: 4}
    test_space = zip(pert_test_map.keys(), real_test_map.keys(), strict=False)
    for pert_in, real_in in test_space:
        config = EnOptConfig.model_validate(gen_config(pert_in, real_in))
        assert pert_test_map[pert_in] == config.gradient.perturbation_min_success
        assert real_test_map[real_in] == config.realizations.realization_min_success


def test_perturbation_types(enopt_config: Any) -> None:
    enopt_config["gradient"] = {
        "perturbation_magnitudes": [0.1, 0.01],
        "perturbation_types": [PerturbationType.ABSOLUTE, PerturbationType.RELATIVE],
    }
    enopt_config["variables"]["lower_bounds"] = [0.0, np.inf]
    enopt_config["variables"]["upper_bounds"] = [1.0, 600.0]
    with pytest.raises(
        ValueError,
        match="The variable bounds must be finite to use relative perturbations",
    ):
        config = EnOptConfig.model_validate(enopt_config)

    enopt_config["variables"]["initial_values"] = [0.0, 0.0, 0.0]
    enopt_config["variables"]["lower_bounds"] = [0.0, 100.0, 0.0]
    enopt_config["variables"]["upper_bounds"] = [np.inf, 600.0, 1.0]
    enopt_config["variables"]["scales"] = [1.0, 1.0, 50]
    enopt_config["gradient"] = {
        "perturbation_magnitudes": [0.1, 0.01, 1.0],
        "perturbation_types": [
            PerturbationType.ABSOLUTE,
            PerturbationType.RELATIVE,
            PerturbationType.SCALED,
        ],
    }
    config = EnOptConfig.model_validate(enopt_config)
    assert np.allclose(config.gradient.perturbation_magnitudes, [0.1, 5.0, 0.02])


def test_config_inputs_property(
    enopt_config: Any, assert_equal_dicts: Callable[[Any, Any], None]
) -> None:
    config = EnOptConfig.model_validate(enopt_config)
    assert_equal_dicts(enopt_config, config.original_inputs)
