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
        """Initialize the variable transform plugin.

        See the [ropt.plugins.sampler.SamplerPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return DefaultVariableTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in (DEFAULT_VARIABLE_TRANSFORM_METHODS | {"default"})


class DefaultObjectiveTransformPlugin(ObjectiveTransformPlugin):
    """Default objective transform plugin class."""

    @classmethod
    def create(
        cls,
        config: ObjectiveTransformConfig,
    ) -> DefaultObjectiveTransform:
        """Initialize the objective transform plugin.

        See the [ropt.plugins.sampler.SamplerPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return DefaultObjectiveTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in (DEFAULT_OBJECTIVE_TRANSFORM_METHODS | {"default"})


class DefaultNonlinearConstraintTransformPlugin(NonlinearConstraintTransformPlugin):
    """Default nonlinear constraint transform plugin class."""

    @classmethod
    def create(
        cls,
        config: NonlinearConstraintTransformConfig,
    ) -> DefaultNonlinearConstraintTransform:
        """Initialize the nonlinear constraint transform plugin.

        See the [ropt.plugins.sampler.SamplerPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return DefaultNonlinearConstraintTransform(config)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in (
            DEFAULT_NONLINEAR_CONSTRAINT_TRANSFORM_METHODS | {"default"}
        )
