"""Abstract base class for optimizer backend implementations.

This module defines the interface that all concrete backend implementations
must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from ropt.backend import OptimizationProblem
    from ropt.config import BackendConfig
    from ropt.core import OptimizerCallback
    from ropt.plugins import MethodSpec


class Backend(ABC):
    """Abstract base class for optimizer backend implementations.

    A backend configures a concrete optimization algorithm, interacts with the
    `ropt` evaluation pipeline through an
    [`OptimizerCallback`][ropt.core.OptimizerCallback], and executes the main
    optimization loop. See
    [Writing plugins](../advanced/writing_plugins.md).

    **What a backend receives**

    What a backend is asked to solve arrives as an
    [`OptimizationProblem`][ropt.backend.OptimizationProblem]: already scaled,
    and already reduced to the free variables. A backend therefore neither
    scales nor masks anything; `ropt` unscales and expands results for
    reporting.

    Non-linear constraints arrive **normalized**, as values that are
    non-negative when the constraint is satisfied, so a backend compares them
    against zero and never handles a bound. See
    [`OptimizerCallbackResult`][ropt.core.OptimizerCallbackResult]. A backend
    whose algorithm expects the opposite convention negates the values and
    their gradients.

    **Process-global state**

    While a run is in progress a backend must not change the working directory,
    the environment, `sys.stdout` or `sys.stderr`, or file descriptors 1 and 2,
    since it shares them with every other run in the process. See
    [What a backend may not change](../advanced/writing_plugins.md#what-a-backend-may-not-change).
    """

    methods: ClassVar[MethodSpec]
    """The optimization algorithms this class provides.

    Either a set of names, which the registry matches case-insensitively, or a
    predicate for classes that cannot enumerate them. Include `"default"` in the
    set if this class has one. See [`MethodSpec`][ropt.plugins.MethodSpec].
    """

    @abstractmethod
    def __init__(self, backend_config: BackendConfig) -> None:
        """Create a new backend instance.

        Called during instantiation. Subclasses should store the configuration
        and perform any lightweight initialization. Validation and
        problem-dependent setup should usually be deferred to `validate_options`
        and `start`.

        Warning: Method name with prefix
            `backend_config.method` may be prefixed in the form
            `"backend/method"`. Implementations should account for this when
            parsing the method name.

        Warning: Handling the default method
            `backend_config.method` may be set to `"default"`, in which case it
            should be mapped to the backend's actual default method.

        Args:
            backend_config: Configuration specifying the method and its options.
        """

    @abstractmethod
    def start(
        self,
        problem: OptimizationProblem,
        optimizer_callback: OptimizerCallback,
        *,
        evaluation_policy: Literal["speculative", "separate", "auto"],
        output_dir: Path | None,
    ) -> None:
        """Run the optimization algorithm on the given problem.

        Starts the backend's main optimization loop. During execution, the
        implementation uses `optimizer_callback` to request any objective,
        constraint, or gradient evaluations its algorithm needs.

        Called at most once per backend instance.

        `evaluation_policy` is whether functions and gradients should be asked
        for together (`"speculative"`), in separate calls (`"separate"`), or as
        the algorithm happens to need them (`"auto"`).

        Args:
            problem:            The problem to solve, in free-variable space.
            optimizer_callback: Callback for requesting evaluations from `ropt`.
            evaluation_policy:  How functions and gradients are requested.
            output_dir:         Directory for files the optimizer writes, or `None`.
        """

    @property
    def bypasses_python_output(self) -> bool:
        """Indicate whether the optimizer prints without going through Python.

        Compiled optimizers commonly write to file descriptors 1 and 2 directly
        instead of through `sys.stdout`, which puts their output beyond reach of
        the capture `ropt` applies for the `stdout` and `stderr` settings of
        [`OptimizerConfig`][ropt.config.OptimizerConfig]. A backend wrapping
        such a library should override this to return `True`, and `ropt` then
        redirects the descriptors as well.

        The answer may differ per method, in which case return it based on the
        configured method. Return `True` for the whole backend when unsure: the
        cost is that capture briefly rewires process-global state, whereas the
        cost of being wrong the other way is output escaping to the terminal.

        See [Writing a Plugin](../advanced/writing_plugins.md#declaring-native-output).

        Returns:
            `True` if the optimizer writes output below the Python level.
        """
        return False

    @abstractmethod
    def validate_options(self) -> None:
        """Validate backend-specific options for the configured method.

        Checks that the options supplied through the
        [`BackendConfig`][ropt.config.BackendConfig] object have the expected
        type, contain only supported keys, and satisfy any method-specific
        value constraints. Options are a dictionary or a list, depending on the
        backend. The exception raised must be a `ValueError`, or derive from one.

        Raises:
            ValueError: If the provided options are invalid.
        """
