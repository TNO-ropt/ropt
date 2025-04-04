"""Domain Transformation Framework.

This module provides a flexible framework for transforming optimization
variables, objectives, and constraints between user-defined domains and the
domains used internally by the optimizer. These transformations are essential
for:

- **Improving Optimizer Performance:** Scaling, shifting, and other
  transformations can significantly enhance the efficiency, stability, and
  convergence of optimization algorithms.
- **Implementing Custom Mappings:**  Beyond simple scaling, this framework
  supports complex, user-defined mappings between domains, allowing for
  tailored problem representations.
- **Handling Diverse Units and Scales:** Transformations enable the optimizer
  to work with variables and functions that may have vastly different units
  or scales, improving numerical stability.

**Key Components:**

- **Abstract Base Classes:** Transform classes derive from abstract base classes
  that define the specific mapping logic between domains.
    - **[`VariableTransform`][ropt.transforms.base.VariableTransform]:**
      Defines the interface for transforming variables between user and
      optimizer domains.
    - **[`ObjectiveTransform`][ropt.transforms.base.ObjectiveTransform]:**
      Defines the interface for transforming objective values between user
      and optimizer domains.
    - **[`NonLinearConstraintTransform`][ropt.transforms.base.NonLinearConstraintTransform]:**
      Defines the interface for transforming non-linear constraint values
      between user and optimizer domains.
- **[`OptModelTransforms`][ropt.transforms.OptModelTransforms]:**
  A container class for conveniently grouping and
  passing multiple transformation objects (variable, objective, and
  nonlinear constraint).

**Workflow and Integration:**

1.  **Configuration:** Transformation objects are passed to the
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] during configuration
    validation, using an
    [`OptModelTransforms`][ropt.transforms.OptModelTransforms] instance. This
    ensures that the entire optimization process is aware of and configured for
    the transformed space.
2.  **Optimization Plan:** The same transformation objects are passed to the
    relevant optimization steps within the [`Plan`][ropt.plan.Plan]. (See, for
    example, the default implementation of an optimizer step in
    [`DefaultOptimizerStep.run`][ropt.plugins.plan.optimizer.DefaultOptimizerStep.run]).
3.  **Evaluation:** When the optimizer requests an evaluation of a variable
    vector, the following occurs:
      -  **Transformation to the User Domain:** The variable vector is
         transformed from the optimizer
          domain back to the user domain using the `from_optimizer` method of
          the `VariableTransform`.
      -  **Function Evaluation:** Objective and constraint values are calculated
          in the user domain.
      -  **Transformation to the Optimizer Domain:** The resulting objective and
         constraint values are
          transformed to the optimizer domain using the `to_optimizer` methods
          of the `ObjectiveTransform` and `NonLinearConstraintTransform`.
4.  **Optimization:** The optimizer proceeds using the transformed values.
5.  **Results:** The [`Results`][ropt.results.Results] objects produced during
    optimization hold values in the optimizer domain. To obtain results in the
    user domain, the
    [`transform_from_optimizer`][ropt.results.Results.transform_from_optimizer]
    method is used to create new `Results` objects with the transformed values.
    For example,
    [`DefaultOptimizerStep.run`][ropt.plugins.plan.optimizer.DefaultOptimizerStep.run]
    emits events that include a dictionary with a `"results"` key That contains
    `Results` objects in the  optimizer domain. To obtain results in the user
    domain they must be converted using the
    [`transform_from_optimizer`][ropt.results.Results.transform_from_optimizer]
    method.


Classes:
    OptModelTransforms: A data class for conveniently grouping and passing
                        multiple transformation objects.
    VariableScaler:     A concrete implementation of `VariableTransform`
                        that performs linear scaling and shifting.
"""

from ._transforms import OptModelTransforms
from .variable_scaler import VariableScaler

__all__ = [
    "OptModelTransforms",
    "VariableScaler",
]
