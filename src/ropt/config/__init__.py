"""The `ropt.config` module provides configuration classes for optimization workflows.

This module defines a set of classes that are used to configure various aspects
of an optimization process, including variables, objectives, constraints,
realizations, samplers, and more.

The central configuration class for optimization is
[`EnOptConfig`][ropt.config.EnOptConfig], which encapsulates the complete
configuration for a single optimization run. It is designed to be flexible and
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
  [`EnOptConfig`][ropt.config.EnOptConfig] class provides a single point
  of entry for configuring an optimization run.

**Parsing and Validation**

The configuration classes are built using
[`pydantic`](https://docs.pydantic.dev/), which provides robust data validation.
The primary configuration class is [`EnOptConfig`][ropt.config.EnOptConfig], and
it contains nested configuration classes for various aspects of the
optimization. To parse a configuration from a dictionary, use the
[`model_validate`][pydantic.BaseModel.model_validate] class method.
```
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
