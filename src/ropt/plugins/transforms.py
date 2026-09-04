"""Plugin support for domain transforms.

A transform converts variables between the user domain and the optimizer domain.
The [`VariableTransformPlugin`][ropt.plugins.transforms.VariableTransformPlugin]
class is a factory that creates the transform objects doing the actual work,
which the [`PluginManager`][ropt.plugins.manager.PluginManager] discovers
through the `ropt.plugins.transforms` entry point group.

See [Transforms](../optimizer_setup/variable_transforms.md) for usage, and
[Writing a Plugin](../utilities/writing_plugins.md) for a walkthrough.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config import VariableTransformConfig
    from ropt.transforms import VariableTransform


class VariableTransformPlugin(Plugin):
    """Abstract base class for variable transform plugins (factories).

    Creates [`VariableTransform`][ropt.transforms.VariableTransform] instances;
    concrete plugins implement `create` as a factory for their own
    `VariableTransform` subclass.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: VariableTransformConfig,
    ) -> VariableTransform:
        """Create a VariableTransform instance.

        Called by the [`PluginManager`][ropt.plugins.manager.PluginManager]
        when an optimization requires variable transformations from this plugin.

        Args:
            config: The variable transform configuration object.

        Returns:
            An initialized VariableTransform object ready for use.
        """
