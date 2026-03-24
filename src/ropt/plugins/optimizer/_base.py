"""Base class for optimzers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ropt.plugins.base import Plugin

if TYPE_CHECKING:
    from ropt.config import EnOptConfig
    from ropt.core import OptimizerCallback
    from ropt.optimizer import Optimizer


class OptimizerPlugin(Plugin):
    """Abstract Base Class for Optimizer Plugins (Factories).

    This class defines the interface for plugins responsible for creating
    [`Optimizer`][ropt.optimizer.Optimizer] instances. These plugins
    act as factories for specific optimization algorithms or backends.
    """

    @classmethod
    @abstractmethod
    def create(
        cls, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> Optimizer:
        """Create an Optimizer instance.

        This abstract class method serves as a factory for creating concrete
        [`Optimizer`][ropt.optimizer.Optimizer] objects. Plugin
        implementations must override this method to return an instance of their
        specific `Optimizer` subclass.

        The [`PluginManager`][ropt.plugins.manager.PluginManager] calls this
        method when an optimization workflow requires an optimizer provided by
        this plugin.

        Args:
            config:             The  configuration object containing the
                                optimization settings.
            optimizer_callback: The callback function used by the optimizer to
                                request evaluations.

        Returns:
            An initialized instance of an `Optimizer` subclass.
        """

    @classmethod
    def validate_options(
        cls,
        method: str,
        options: dict[str, Any] | list[str] | None,
    ) -> None:
        """Validate the optimizer-specific options for a given method.

        This class method is intended to check if the `options` dictionary,
        typically provided in the
        [`OptimizerConfig`][ropt.config.OptimizerConfig], contains valid keys
        and values for the specified optimization `method` supported by this
        plugin.

        This default implementation performs no validation. Subclasses should
        override this method to implement validation logic specific to the
        methods they support, potentially using schema validation tools like
        Pydantic.

        The raised exception must be a ValueError, or derive from a ValueError.

        Note:
            It is expected that the optimizer either receives a dictionary, or a
            list of options. This method should test if the type of the options
            is as expected, and raise a `ValueError` with an appropriate message
            if this is not the case.

        Warning: Method name with prefix
            The method string may be prefixed in the form "backend/method", take
            this into account when parsing the method name.

        Warning: Handling the default method
            The the method string may be set to "default", in which case it should
            be mapped to the correct default method of the backend.

        Args:
            method:  The specific optimization method name (e.g., "slsqp",
                     "my_optimizer/variant1").
            options: The dictionary or a list of strings of options.

        Raises:
            ValueError: If the provided options are invalid for the specified
                        method.
        """
