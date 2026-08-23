"""Plugin support for domain transforms.

A transform converts values between the user domain and the optimizer domain.
The [`VariableTransformPlugin`][ropt.plugins.transforms.VariableTransformPlugin],
[`ObjectiveTransformPlugin`][ropt.plugins.transforms.ObjectiveTransformPlugin]
and
[`NonlinearConstraintTransformPlugin`][ropt.plugins.transforms.NonlinearConstraintTransformPlugin]
classes are factories that create the transform objects doing the actual work,
which the [`PluginManager`][ropt.plugins.manager.PluginManager] discovers
through the `ropt.plugins.transforms` entry point group.

See [Transforms](../optimizer_setup/transforms.md) for usage, and
[Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.
"""

from ._base import (
    NonlinearConstraintTransformPlugin,
    ObjectiveTransformPlugin,
    VariableTransformPlugin,
)

__all__ = [
    "NonlinearConstraintTransformPlugin",
    "ObjectiveTransformPlugin",
    "VariableTransformPlugin",
]
