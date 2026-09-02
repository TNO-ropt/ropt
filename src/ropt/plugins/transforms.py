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

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config import (
        NonlinearConstraintTransformConfig,
        ObjectiveTransformConfig,
        VariableTransformConfig,
    )
    from ropt.transforms import (
        NonlinearConstraintTransform,
        ObjectiveTransform,
        VariableTransform,
    )


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


class ObjectiveTransformPlugin(Plugin):
    """Abstract base class for objective transform plugins (factories).

    Creates [`ObjectiveTransform`][ropt.transforms.ObjectiveTransform]
    instances; concrete plugins implement `create` as a factory for their own
    `ObjectiveTransform` subclass.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: ObjectiveTransformConfig,
    ) -> ObjectiveTransform:
        """Create an ObjectiveTransform instance.

        Called by the [`PluginManager`][ropt.plugins.manager.PluginManager]
        when an optimization requires objective transformations from this plugin.

        Args:
            config: The objective transform configuration object.

        Returns:
            An initialized ObjectiveTransform object ready for use.
        """


class NonlinearConstraintTransformPlugin(Plugin):
    """Abstract base class for nonlinear constraint transform plugins (factories).

    Creates
    [`NonlinearConstraintTransform`][ropt.transforms.NonlinearConstraintTransform]
    instances; concrete plugins implement `create` as a factory for their own
    `NonlinearConstraintTransform` subclass.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: NonlinearConstraintTransformConfig,
    ) -> NonlinearConstraintTransform:
        """Create a NonlinearConstraintTransform instance.

        Called by the [`PluginManager`][ropt.plugins.manager.PluginManager]
        when an optimization requires nonlinear constraint transformations from
        this plugin.

        Args:
            config:   The nonlinear constraint transform configuration object.

        Returns:
            An initialized NonlinearConstraintTransform object ready for use.
        """
