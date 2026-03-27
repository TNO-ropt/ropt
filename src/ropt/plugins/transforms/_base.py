"""This module defines the abstract base class for variable transform plugins."""

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
    """Abstract Base Class for Variable Transform Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`VariableTransform`][ropt.transforms.VariableTransform] instances. These
    plugins act as factories for specific variable transformation algorithms or
    strategies.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: VariableTransformConfig,
    ) -> VariableTransform:
        """Factory method to create a concrete VariableTransform instance.

        This abstract class method serves as a factory for creating concrete
        [`VariableTransform`][ropt.transforms.VariableTransform] objects. Plugin
        implementations must override this method to return an instance of their
        specific `VariableTransform` subclass.

        The [`PluginManager`][ropt.plugins.manager.PluginManager] calls this
        method when an optimization requires variable transformations provided by this plugin.

        Args:
            config: The variable transform configuration object.

        Returns:
            An initialized VariableTransform object ready for use.
        """


class ObjectiveTransformPlugin(Plugin):
    """Abstract Base Class for Objective Transform Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`ObjectiveTransform`][ropt.transforms.ObjectiveTransform] instances. These
    plugins act as factories for specific objective transformation algorithms or
    strategies.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: ObjectiveTransformConfig,
    ) -> ObjectiveTransform:
        """Factory method to create a concrete ObjectiveTransform instance.

        This abstract class method serves as a factory for creating concrete
        [`ObjectiveTransform`][ropt.transforms.ObjectiveTransform] objects. Plugin
        implementations must override this method to return an instance of their
        specific `ObjectiveTransform` subclass.

        The [`PluginManager`][ropt.plugins.manager.PluginManager] calls this
        method when an optimization requires objective transformations provided
        by this plugin.

        Args:
            config: The objective transform configuration object.

        Returns:
            An initialized ObjectiveTransform object ready for use.
        """


class NonlinearConstraintTransformPlugin(Plugin):
    """Abstract Base Class for Nonlinear Constraint Transform Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`NonlinearConstraintTransform`][ropt.transforms.NonlinearConstraintTransform]
    instances. These plugins act as factories for specific nonlinear constraint
    transformation algorithms or strategies.
    """

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: NonlinearConstraintTransformConfig,
    ) -> NonlinearConstraintTransform:
        """Factory method to create a concrete NonlinearConstraintTransform instance.

        This abstract class method serves as a factory for creating concrete
        [`NonlinearConstraintTransform`][ropt.transforms.NonlinearConstraintTransform]
        objects. Plugin implementations must override this method to return an
        instance of their specific `NonlinearConstraintTransform` subclass.

        The [`PluginManager`][ropt.plugins.manager.PluginManager] calls this
        method when an optimization requires nonlinear constraint
        transformations provided by this plugin.

        Args:
            config:   The nonlinear constraint transform configuration object.

        Returns:
            An initialized NonlinearConstraintTransform object ready for use.
        """
