"""Public API for function estimator implementations.

Function estimators define how realization-level results are aggregated into the
values used by the optimizer. In ensemble-based workflows, each realization
contributes objective/constraint function values and gradients; estimators
combine those per-realization arrays with realization weights to produce one
representative function value and one representative gradient.

**Core Interface**

All estimator implementations inherit from the
[`FunctionEstimator`][ropt.function_estimator.FunctionEstimator] base class,
which defines the estimator lifecycle (`__init__`, `init`) and the two required
aggregation methods (`calculate_function`, `calculate_gradient`).

**Integration with Optimization**

Function estimators are accessed via an
[`EnOptContext`][ropt.context.EnOptContext] object through its
`function_estimators` field, a tuple of function estimator instances. Estimators
are instantiated either directly as objects or via
[`FunctionEstimatorConfig`][ropt.config.FunctionEstimatorConfig] objects, which
are used by the plugin system to create instances based on the configured method
string (e.g., `"mean"` or `"stddev"`).

**Built-in and Custom Estimators**

The
[`DefaultFunctionEstimator`][ropt.function_estimator.default.DefaultFunctionEstimator]
class provides two commonly used aggregation strategies:

- `mean`: Weighted average of realization values/gradients.
- `stddev`: Weighted standard deviation of values with chain-rule gradients.

Users can implement custom estimators by subclassing `FunctionEstimator`. Those
subclasses can be instantiated directly and passed into an
[`EnOptContext`][ropt.context.EnOptContext] object through its
`function_estimators` field. Registering a custom estimator with the plugin
system is optional and only required when the estimator should be selected and
configured via `FunctionEstimatorConfig` objects instead of being instantiated
explicitly by the user.
"""

from ._base import FunctionEstimator

__all__ = [
    "FunctionEstimator",
]
