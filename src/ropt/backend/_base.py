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

    All concrete backend implementations must inherit from this class and
    implement the required lifecycle and validation methods. A backend is
    responsible for configuring a concrete optimization algorithm, interacting
    with the `ropt` evaluation pipeline through an
    [`OptimizerCallback`][ropt.core.OptimizerCallback], and executing the main
    optimization loop.

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

    **Lifecycle**

    1. Instantiation via `__init__`: Called with a backend configuration
        object.
    2. Validation via `validate_options`: Called to verify that the configured
        backend options are supported.
    3. Execution via `start`: Called with the problem to solve and the callback
        that evaluates it.

    Subclasses must implement:

    - `__init__`: Stores backend configuration and performs lightweight setup.
    - `start`: Runs the optimization algorithm.
    - `validate_options`: Verifies that backend-specific options are valid.

    Subclasses may optionally override:

    - `bypasses_python_output`: Declares that the optimizer prints below the
                                Python level.

    **Process-global state**

    A backend shares its process with everything else in the program, including
    other optimizations running at the same time. While a run is in progress it
    must therefore not change the working directory, the environment,
    `sys.stdout` or `sys.stderr`, or file descriptors 1 and 2. A backend wrapping
    a library that requires this, or that prints where it cannot be redirected
    per run, must document that it cannot run concurrently in-process and direct
    users to the [`external`][ropt.backend.external.ExternalBackend] backend. See
    [What a backend may not change](../advanced/writing_plugins.md#what-a-backend-may-not-change)
    for the full contract.
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
            backend_config: Configuration object specifying the backend method
                and any method-specific options.
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

        Args:
            problem:            The problem to solve, in free-variable space.
            optimizer_callback: Callback used to request objective, constraint,
                                and gradient evaluations from the `ropt` core.
            evaluation_policy:  Whether functions and gradients should be asked
                                for together (`"speculative"`), in separate
                                calls (`"separate"`), or as the algorithm
                                happens to need them (`"auto"`).
            output_dir:         Directory for any files the optimizer writes,
                                or `None` if none was configured.
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
        value constraints.

        Concrete backends should implement validation logic for the methods
        they support, potentially using schema-validation tools such as
        Pydantic.

        The raised exception must be a ValueError, or derive from a ValueError.

        Note:
            Backend options may be represented as a dictionary or list,
            depending on the backend. This method should verify that the type
            matches what the backend expects and raise a `ValueError` with a
            clear message when it does not.

        Raises:
            ValueError: If the provided options are invalid.
        """
