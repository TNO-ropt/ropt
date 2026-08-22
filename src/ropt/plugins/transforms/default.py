"""This module implements the SciPy sampler plugin."""

from ropt.config import (
    NonlinearConstraintTransformConfig,
    ObjectiveTransformConfig,
    VariableTransformConfig,
)
from ropt.transforms.default import (
    DEFAULT_NONLINEAR_CONSTRAINT_TRANSFORM_METHODS,
    DEFAULT_OBJECTIVE_TRANSFORM_METHODS,
    DEFAULT_VARIABLE_TRANSFORM_METHODS,
    DefaultNonlinearConstraintTransform,
    DefaultObjectiveTransform,
    DefaultVariableTransform,
)

from ._base import (
    NonlinearConstraintTransformPlugin,
    ObjectiveTransformPlugin,
    VariableTransformPlugin,
)


class DefaultVariableTransformPlugin(VariableTransformPlugin):
    """Default variable transform plugin class."""

    @classmethod
    def create(
        cls,
        config: VariableTransformConfig,
    ) -> DefaultVariableTransform:
        """Create a DefaultVariableTransform instance.

        Args:
            config: The variable transform configuration.

        Returns:
            A new `DefaultVariableTransform`.
        """
        return DefaultVariableTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in (DEFAULT_VARIABLE_TRANSFORM_METHODS | {"default"})


class DefaultObjectiveTransformPlugin(ObjectiveTransformPlugin):
    """Default objective transform plugin class."""

    @classmethod
    def create(
        cls,
        config: ObjectiveTransformConfig,
    ) -> DefaultObjectiveTransform:
        """Create a DefaultObjectiveTransform instance.

        Args:
            config: The objective transform configuration.

        Returns:
            A new `DefaultObjectiveTransform`.
        """
        return DefaultObjectiveTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in (DEFAULT_OBJECTIVE_TRANSFORM_METHODS | {"default"})


class DefaultNonlinearConstraintTransformPlugin(NonlinearConstraintTransformPlugin):
    """Default nonlinear constraint transform plugin class."""

    @classmethod
    def create(
        cls,
        config: NonlinearConstraintTransformConfig,
    ) -> DefaultNonlinearConstraintTransform:
        """Create a DefaultNonlinearConstraintTransform instance.

        Args:
            config: The nonlinear constraint transform configuration.

        Returns:
            A new `DefaultNonlinearConstraintTransform`.
        """
        return DefaultNonlinearConstraintTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # ruff: ignore[undocumented-public-method]
        return method.lower() in (
            DEFAULT_NONLINEAR_CONSTRAINT_TRANSFORM_METHODS | {"default"}
        )
