"""The `ropt.config.enopt` module provides configuration classes for optimization workflows.

This module defines a set of classes that are used to configure various aspects
of an optimization process, including variables, objectives, constraints,
realizations, samplers, and more.

The central configuration class is
[`EnOptConfig`][ropt.config.enopt.EnOptConfig], which encapsulates the complete
configuration for a single optimization step. It is designed to be flexible and
extensible, allowing users to customize the optimization process to their
specific needs.

These configuration classes are built using
[`pydantic`](https://docs.pydantic.dev/), which provides robust data validation
and parsing capabilities. This ensures that the configuration data is consistent
and adheres to the expected structure.

Configuration objects are typically created from dictionaries of configuration
values using the `model_validate` method provided by `pydantic`.

**Key Features**:

- **Modular Design:** The configuration is broken down into smaller, manageable
  components, each represented by a dedicated class.
- **Validation:** `pydantic` ensures that the configuration data is valid and
  consistent.
- **Extensibility:** The modular design allows for easy extension and
  customization of the optimization process.
- **Centralized Configuration:** The
  [`EnOptConfig`][ropt.config.enopt.EnOptConfig] class provides a single point
  of entry for configuring an optimization step.

**Classes:**

- [`EnOptConfig`][ropt.config.enopt.EnOptConfig]: The main configuration class
  for an optimization step.
- [`VariablesConfig`][ropt.config.enopt.VariablesConfig]: Configuration for
  variables.
- [`ObjectiveFunctionsConfig`][ropt.config.enopt.ObjectiveFunctionsConfig]:
  Configuration for objective functions.
- [`LinearConstraintsConfig`][ropt.config.enopt.LinearConstraintsConfig]:
  Configuration for linear constraints.
- [`NonlinearConstraintsConfig`][ropt.config.enopt.NonlinearConstraintsConfig]:
  Configuration for non-linear constraints.
- [`RealizationsConfig`][ropt.config.enopt.RealizationsConfig]: Configuration
  for realizations.
- [`OptimizerConfig`][ropt.config.enopt.OptimizerConfig]: Configuration for the
  optimizer.
- [`GradientConfig`][ropt.config.enopt.GradientConfig]: Configuration for
  gradient calculations.
- [`FunctionEstimatorConfig`][ropt.config.enopt.FunctionEstimatorConfig]:
  Configuration for function estimators.
- [`RealizationFilterConfig`][ropt.config.enopt.RealizationFilterConfig]:
  Configuration for realization filters.
- [`SamplerConfig`][ropt.config.enopt.SamplerConfig]: Configuration for
  samplers.
"""

from ._enopt_config import EnOptConfig
from ._function_estimator_config import FunctionEstimatorConfig
from ._gradient_config import GradientConfig
from ._linear_constraints_config import LinearConstraintsConfig
from ._nonlinear_constraints_config import NonlinearConstraintsConfig
from ._objective_functions_config import ObjectiveFunctionsConfig
from ._optimizer_config import OptimizerConfig
from ._realization_filter_config import RealizationFilterConfig
from ._realizations_config import RealizationsConfig
from ._sampler_config import SamplerConfig
from ._variables_config import VariablesConfig

__all__ = [
    "EnOptConfig",
    "FunctionEstimatorConfig",
    "GradientConfig",
    "LinearConstraintsConfig",
    "NonlinearConstraintsConfig",
    "ObjectiveFunctionsConfig",
    "OptimizerConfig",
    "RealizationFilterConfig",
    "RealizationsConfig",
    "SamplerConfig",
    "VariablesConfig",
]
