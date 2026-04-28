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
    def create(  # noqa: D102
        cls,
        config: VariableTransformConfig,
    ) -> DefaultVariableTransform:
        return DefaultVariableTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # noqa: D102
        return method.lower() in (DEFAULT_VARIABLE_TRANSFORM_METHODS | {"default"})


class DefaultObjectiveTransformPlugin(ObjectiveTransformPlugin):
    """Default objective transform plugin class."""

    @classmethod
    def create(  # noqa: D102
        cls,
        config: ObjectiveTransformConfig,
    ) -> DefaultObjectiveTransform:
        return DefaultObjectiveTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # noqa: D102
        return method.lower() in (DEFAULT_OBJECTIVE_TRANSFORM_METHODS | {"default"})


class DefaultNonlinearConstraintTransformPlugin(NonlinearConstraintTransformPlugin):
    """Default nonlinear constraint transform plugin class."""

    @classmethod
    def create(  # noqa: D102
        cls,
        config: NonlinearConstraintTransformConfig,
    ) -> DefaultNonlinearConstraintTransform:
        return DefaultNonlinearConstraintTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:  # noqa: D102
        return method.lower() in (
            DEFAULT_NONLINEAR_CONSTRAINT_TRANSFORM_METHODS | {"default"}
        )
