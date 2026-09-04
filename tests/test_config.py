import copy
import re
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from ropt.config import (
    LinearConstraintsConfig,
    NonlinearConstraintsConfig,
    ObjectiveFunctionsConfig,
    VariablesConfig,
)
from ropt.context import EnOptContext
from ropt.enums import BoundaryType, PerturbationType

initial_values = np.array([1, 2])


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
        },
        "objectives": {
            "weights": [1.0],
        },
    }


def test_check_linear_constraints() -> None:
    config = {
        "coefficients": np.array([[1, 2], [1, 2], [1, 2]]),
        "lower_bounds": np.array([-np.inf, 2, 3]),
        "upper_bounds": np.array([1, np.inf, 3]),
    }
    linear_constraints = LinearConstraintsConfig.model_validate(config)
    assert linear_constraints.coefficients is not None
    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        linear_constraints.coefficients[0, 0] = 0
    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        linear_constraints.upper_bounds[0] = 1


def test_check_linear_constraints_convert() -> None:
    config = {
        "coefficients": [[1, 2], [1, 2], [1, 2]],
        "lower_bounds": np.array([-np.inf, 2, 3]),
        "upper_bounds": np.array([1, np.inf, 3]),
    }
    LinearConstraintsConfig.model_validate(config)


def test_check_linear_constraints_vector_shapes() -> None:
    config = {
        "coefficients": [[1, 2, 3], [1, 2, 3]],
        "lower_bounds": np.array([-np.inf, 2]),
        "upper_bounds": np.array([1, np.inf]),
    }
    LinearConstraintsConfig.model_validate(config)

    config_copy = copy.deepcopy(config)
    config_copy["lower_bounds"] = [1, 2, 3]
    with pytest.raises(
        ValueError,
        match="lower_bounds cannot be broadcasted to a length of 2",
    ):
        LinearConstraintsConfig.model_validate(config_copy)


def test_negative_weights_are_rejected(config: Any) -> None:
    # A negative weight used to be a back door to maximization; direction is
    # now set by the `maximize` field instead.
    config["objectives"]["weights"] = [0.75, -0.25]
    with pytest.raises(ValidationError, match="Weights must not be negative"):
        EnOptContext.model_validate(config)

    config["objectives"]["weights"] = [1.0, 1.0]
    config["realizations"] = {"weights": [1.0, -1.0]}
    with pytest.raises(ValidationError, match="Weights must not be negative"):
        EnOptContext.model_validate(config)


def test_weights_summing_to_zero_are_rejected(config: Any) -> None:
    config["objectives"]["weights"] = [0.0, 0.0]
    with pytest.raises(ValidationError, match="The sum of weights is not positive"):
        EnOptContext.model_validate(config)


def test_objective_scales() -> None:
    objectives = ObjectiveFunctionsConfig.model_validate({"weights": [1.0, 1.0]})
    assert np.allclose(objectives.scales, 1.0)
    assert objectives.scales.shape == (2,)
    assert not objectives.auto_scale

    objectives = ObjectiveFunctionsConfig.model_validate(
        {"weights": [1.0, 1.0], "scales": [2.0, 3.0]}
    )
    assert np.allclose(objectives.scales, [2.0, 3.0])


@pytest.mark.parametrize("scale", [0.0, -2.0])
def test_objective_scales_must_be_positive(scale: float) -> None:
    with pytest.raises(ValueError, match="scales must be positive"):
        ObjectiveFunctionsConfig.model_validate(
            {"weights": [1.0, 1.0], "scales": [1.0, scale]}
        )


def test_objective_scales_broadcast() -> None:
    with pytest.raises(
        ValueError, match="scales cannot be broadcasted to a length of 2"
    ):
        ObjectiveFunctionsConfig.model_validate(
            {"weights": [1.0, 1.0], "scales": [1.0, 2.0, 3.0]}
        )


def test_constraint_scales() -> None:
    constraints = NonlinearConstraintsConfig.model_validate(
        {"lower_bounds": [0.0, 0.0], "upper_bounds": [1.0, 1.0]}
    )
    assert np.allclose(constraints.scales, 1.0)
    assert constraints.scales.shape == (2,)
    assert not constraints.auto_scale.any()
    assert constraints.auto_scale.shape == (2,)

    constraints = NonlinearConstraintsConfig.model_validate(
        {"lower_bounds": [0.0, 0.0], "upper_bounds": [1.0, 1.0], "scales": 2.0}
    )
    assert np.allclose(constraints.scales, 2.0)


@pytest.mark.parametrize("scale", [0.0, -2.0])
def test_constraint_scales_must_be_positive(scale: float) -> None:
    with pytest.raises(ValueError, match="scales must be positive"):
        NonlinearConstraintsConfig.model_validate(
            {"lower_bounds": [0.0], "upper_bounds": [1.0], "scales": scale}
        )


def test_variable_scales_and_offsets_default_to_the_identity() -> None:
    variables = VariablesConfig.model_validate({"variable_count": 3})
    assert np.allclose(variables.scales, 1.0)
    assert np.allclose(variables.offsets, 0.0)
    assert variables.scales.shape == (3,)
    assert variables.offsets.shape == (3,)


def test_variable_scales_and_offsets_broadcast() -> None:
    variables = VariablesConfig.model_validate(
        {"variable_count": 3, "scales": 2.0, "offsets": [1.0, 2.0, 3.0]}
    )
    assert np.allclose(variables.scales, 2.0)
    assert np.allclose(variables.offsets, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("scale", [0.0, -2.0])
def test_variable_scales_must_be_positive(scale: float) -> None:
    with pytest.raises(ValueError, match="scales must be positive"):
        VariablesConfig.model_validate({"variable_count": 2, "scales": [1.0, scale]})


def test_linear_constraint_scales_default_to_one() -> None:
    linear_constraints = LinearConstraintsConfig.model_validate(
        {
            "coefficients": [[1.0, 1.0], [1.0, 0.0]],
            "lower_bounds": 0.0,
            "upper_bounds": 1.0,
        }
    )
    assert np.allclose(linear_constraints.scales, 1.0)
    assert linear_constraints.scales.shape == (2,)
    assert not linear_constraints.auto_scale


@pytest.mark.parametrize("scale", [0.0, -2.0])
def test_linear_constraint_scales_must_be_positive(scale: float) -> None:
    with pytest.raises(ValueError, match="scales must be positive"):
        LinearConstraintsConfig.model_validate(
            {
                "coefficients": [[1.0, 1.0]],
                "lower_bounds": 0.0,
                "upper_bounds": 1.0,
                "scales": scale,
            }
        )


def test_objective_maximize_defaults_and_broadcasts() -> None:
    objectives = ObjectiveFunctionsConfig.model_validate({"weights": [1.0, 1.0]})
    assert objectives.maximize.shape == (2,)
    assert not objectives.maximize.any()

    objectives = ObjectiveFunctionsConfig.model_validate(
        {"weights": [1.0, 1.0], "maximize": True}
    )
    assert objectives.maximize.tolist() == [True, True]

    objectives = ObjectiveFunctionsConfig.model_validate(
        {"weights": [1.0, 1.0], "maximize": [True, False]}
    )
    assert objectives.maximize.tolist() == [True, False]


def test_objective_maximize_broadcast_error() -> None:
    with pytest.raises(
        ValueError, match="maximize cannot be broadcasted to a length of 2"
    ):
        ObjectiveFunctionsConfig.model_validate(
            {"weights": [1.0, 1.0], "maximize": [True, False, True]}
        )


def test_check_perturbations() -> None:
    VariablesConfig.model_validate({"variable_count": 1})
    variables = VariablesConfig.model_validate(
        {"variable_count": 1, "perturbation_magnitudes": np.array([0.1])}
    )
    assert variables.perturbation_magnitudes == np.array([0.1])


def test_check_config(config: Any) -> None:
    EnOptContext.model_validate(config)


def test_check_config_linear_constraints(config: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 2, 3], [2, 3, 4]],
        "lower_bounds": [1, 2],
        "upper_bounds": [np.inf, np.inf],
    }
    with pytest.raises(
        ValueError,
        match="the coefficients matrix should have 2 columns",
    ):
        EnOptContext.model_validate(config)


def test_check_config_perturbations(config: Any) -> None:
    config["variables"].update(
        {
            "perturbation_magnitudes": [1] * 2,
            "boundary_types": [BoundaryType.TRUNCATE_BOTH] * 2,
            "perturbation_types": [PerturbationType.ABSOLUTE] * 2,
        }
    )
    EnOptContext.model_validate(config)

    config_copy = copy.deepcopy(config)
    config_copy["variables"]["perturbation_magnitudes"] = [1] * 3
    with pytest.raises(
        ValueError,
        match="perturbation_magnitudes cannot be broadcasted to a length of 2",
    ):
        EnOptContext.model_validate(config_copy)

    config_copy = copy.deepcopy(config)
    config_copy["variables"]["boundary_types"] = [BoundaryType.TRUNCATE_BOTH] * 3
    with pytest.raises(
        ValueError, match="boundary_types cannot be broadcasted to a length of 2"
    ):
        EnOptContext.model_validate(config_copy)

    config_copy = copy.deepcopy(config)
    config_copy["variables"]["perturbation_types"] = [PerturbationType.ABSOLUTE] * 3
    with pytest.raises(
        ValueError, match="perturbation_types cannot be broadcasted to a length of 2"
    ):
        EnOptContext.model_validate(config_copy)


def test_check_config_min_success(config: Any) -> None:
    def gen_config(pert_min: int | None, real_min: int | None) -> dict[str, Any]:
        config_copy: dict[str, Any] = copy.deepcopy(config)
        config_copy["realizations"] = {"weights": 4 * [1.0]}
        config_copy["gradient"] = {}
        if pert_min is not None:
            config_copy["gradient"]["perturbation_min_success"] = pert_min
        if real_min is not None:
            config_copy["realizations"]["realization_min_success"] = real_min
        return config_copy

    pert_test_map = {None: 5, 1: 1, 4: 4, 7: 5}
    real_test_map = {None: 4, 1: 1, 3: 3, 7: 4}
    test_space = zip(pert_test_map.keys(), real_test_map.keys(), strict=False)
    for pert_in, real_in in test_space:
        context = EnOptContext.model_validate(gen_config(pert_in, real_in))
        assert pert_test_map[pert_in] == context.gradient.perturbation_min_success
        assert real_test_map[real_in] == context.realizations.realization_min_success


def test_perturbation_types(config: Any) -> None:
    config["variables"].update(
        {
            "perturbation_magnitudes": [0.1, 0.01],
            "perturbation_types": [
                PerturbationType.ABSOLUTE,
                PerturbationType.RELATIVE,
            ],
        }
    )
    config["variables"]["lower_bounds"] = [0.0, 600]
    config["variables"]["upper_bounds"] = [1.0, np.inf]
    with pytest.raises(
        ValueError,
        match="The variable bounds must be finite to use relative perturbations",
    ):
        context = EnOptContext.model_validate(config)

    config["variables"]["variable_count"] = 3
    config["variables"]["lower_bounds"] = [0.0, 100.0, 0.0]
    config["variables"]["upper_bounds"] = [np.inf, 600.0, 1.0]
    config["variables"].update(
        {
            "perturbation_magnitudes": [0.1, 0.01, 1.0],
            "perturbation_types": [
                PerturbationType.ABSOLUTE,
                PerturbationType.RELATIVE,
                PerturbationType.ABSOLUTE,
            ],
        }
    )
    context = EnOptContext.model_validate(config)
    assert np.allclose(context.variables.perturbation_magnitudes, [0.1, 0.01, 1.0])


def test_perturbation_types_with_scaled_variables(config: Any) -> None:
    config["variables"].update(
        {
            "perturbation_magnitudes": [0.1, 0.01],
            "perturbation_types": [
                PerturbationType.ABSOLUTE,
                PerturbationType.RELATIVE,
            ],
        }
    )
    config["variables"]["lower_bounds"] = [0.0, 600]
    config["variables"]["upper_bounds"] = [1.0, np.inf]
    with pytest.raises(
        ValueError,
        match="The variable bounds must be finite to use relative perturbations",
    ):
        context = EnOptContext.model_validate(config)

    config["variables"]["variable_count"] = 3
    config["variables"]["lower_bounds"] = [0.0, 100.0, 0.0]
    config["variables"]["upper_bounds"] = [np.inf, 600.0, 1.0]
    config["variables"]["scales"] = [1.0, 1.0, 50.0]

    config["variables"].update(
        {
            "perturbation_magnitudes": [0.1, 0.01, 1.0],
            "perturbation_types": [
                PerturbationType.ABSOLUTE,
                PerturbationType.RELATIVE,
                PerturbationType.ABSOLUTE,
            ],
        }
    )
    context = EnOptContext.model_validate(config)
    assert np.allclose(context.variables.perturbation_magnitudes, [0.1, 0.01, 0.02])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", object(), "Value must be a Backend, BackendConfig, or dict."),
        ("samplers", (object(),), "Value must be a Sampler, SamplerConfig, or dict."),
        (
            "realization_filters",
            (object(),),
            "Value must be a RealizationFilter, RealizationFilterConfig, or dict.",
        ),
        (
            "function_estimators",
            (object(),),
            "Value must be a FunctionEstimator, FunctionEstimatorConfig, or dict.",
        ),
    ],
)
def test_plugin_field_rejects_unknown_value(
    config: dict[str, Any], field: str, value: Any, message: str
) -> None:
    config[field] = value
    with pytest.raises(ValidationError, match=re.escape(message)):
        EnOptContext.model_validate(config)


def test_components_given_as_a_list_are_keyed_by_position(
    config: dict[str, Any],
) -> None:
    config["samplers"] = [{}, {}]
    config["variables"]["samplers"] = [0, 1]
    context = EnOptContext.model_validate(config)
    assert list(context.samplers) == ["0", "1"]
    assert context.variables.samplers == ("0", "1")


def test_components_can_be_selected_by_name(config: dict[str, Any]) -> None:
    config["samplers"] = {"coarse": {}, "fine": {}}
    config["variables"]["samplers"] = ["fine", "coarse"]
    context = EnOptContext.model_validate(config)
    assert list(context.samplers) == ["coarse", "fine"]
    assert context.variables.samplers == ("fine", "coarse")


def test_a_single_key_is_broadcast_over_the_elements(config: dict[str, Any]) -> None:
    config["samplers"] = {"only": {}}
    config["variables"]["samplers"] = "only"
    context = EnOptContext.model_validate(config)
    assert context.variables.samplers == ("only", "only")


def test_a_null_reference_selects_no_realization_filter(
    config: dict[str, Any],
) -> None:
    config["objectives"]["realization_filters"] = None
    context = EnOptContext.model_validate(config)
    assert context.objectives.realization_filters == (None,)


def test_an_unknown_key_names_the_defined_keys(config: dict[str, Any]) -> None:
    config["samplers"] = {"coarse": {}}
    config["variables"]["samplers"] = "fine"
    with pytest.raises(
        ValidationError,
        match=re.escape("variables.samplers: unknown key 'fine'; defined keys are"),
    ):
        EnOptContext.model_validate(config)


def test_the_former_sentinel_is_now_an_unknown_key(config: dict[str, Any]) -> None:
    config["objectives"]["realization_filters"] = -1
    with pytest.raises(
        ValidationError,
        match=re.escape("objectives.realization_filters: unknown key '-1'"),
    ):
        EnOptContext.model_validate(config)
