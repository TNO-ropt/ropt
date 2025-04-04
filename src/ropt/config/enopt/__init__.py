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
- **Domain Transformations:**  The optimization process supports domain
  transformations, as detailed in the [`ropt.transforms`][ropt.transforms]
  module. These transformations map variables, objectives, and constraints
  between the user-defined domain and the domain used by the optimizer. This
  capability is valuable for operations such as scaling, shifting, or other
  adjustments that can enhance the performance and stability of the optimization
  algorithm. Domain transformations are implemented through a set of classes
  that define the necessary mappings. When creating an `EnOptConfig` object,
  transformation objects can be provided to automatically apply these
  transformations during configuration validation.


**Parsing and Validation**

The configuration classes are built using
[`pydantic`](https://docs.pydantic.dev/), which provides robust data validation.
The primary configuration class is
[`EnOptConfig`][ropt.config.enopt.EnOptConfig], and it contains nested
configuration classes for various aspects of the optimization. To parse a
configuration from a dictionary, use the
[`model_validate`][pydantic.BaseModel.model_validate] class method:

```py
from ropt.config.enopt import EnOptConfig

config_dict = {
    "variables": {
        "initial_values": [10.0, 10.0],
    }
}
config = EnOptConfig.model_validate(config_dict)
config.variables.initial_values  # [10.0, 10.0]
```

Domain transformation objects from the [`ropt.transforms`][ropt.transforms]
module can be passed to the `model_validate` method via the `context` parameter:

```py
from ropt.config.enopt import EnOptConfig
from ropt.transforms import OptModelTransforms, VariableScaler

config_dict = {
    "variables": {
        "initial_values": [10.0, 10.0],
    }
}
scaler = VariableScaler([10.0, 5.0], None)
config = EnOptConfig.model_validate(
    config_dict, context=OptModelTransforms(variables=scaler)
)
config.variables.initial_values  # [1.0, 2.0]
```


Classes:
    EnOptConfig:                The main configuration class for an optimization step.
    VariablesConfig:            Configuration for variables.
    ObjectiveFunctionsConfig:   Configuration for objective functions.
    LinearConstraintsConfig:    Configuration for linear constraints.
    NonlinearConstraintsConfig: Configuration for non-linear constraints.
    RealizationsConfig:         Configuration for realizations.
    OptimizerConfig:            Configuration for the optimizer.
    GradientConfig:             Configuration for gradient calculations.
    FunctionEstimatorConfig:    Configuration for function estimators.
    RealizationFilterConfig:    Configuration for realization filters.
    SamplerConfig:              Configuration for samplers.
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
