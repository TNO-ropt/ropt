"""Public API for domain transforms.

Provides the base class for transforming variables between a user-defined
domain and the optimizer's internal domain:

- [`VariableTransform`][ropt.transforms.VariableTransform]

See [Variable Transforms](../optimizer_setup/variable_transforms.md) for usage,
configuration, and implementation guidance.
"""

from .base import VariableTransform

__all__ = [
    "VariableTransform",
]
