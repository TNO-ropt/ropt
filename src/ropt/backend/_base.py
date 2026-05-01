"""Abstract base class for optimizer backend implementations.

Backends are responsible for driving the optimization algorithm used by `ropt`.
They coordinate the optimizer lifecycle, request objective/constraint and
gradient evaluations through the core callback interface, and advance the
optimization from an initial variable vector toward a solution. This module
defines the interface that all concrete backend implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import BackendConfig
    from ropt.context import EnOptContext
    from ropt.core import OptimizerCallback


class Backend(ABC):
    """Abstract base class for optimizer backend implementations.

    All concrete backend implementations must inherit from this class and
    implement the required lifecycle and validation methods. A backend is
    responsible for configuring a concrete optimization algorithm, interacting
    with the `ropt` evaluation pipeline through an
    [`OptimizerCallback`][ropt.core.OptimizerCallback], and executing the main
    optimization loop.

    During optimization, the backend receives an
    [`EnOptContext`][ropt.context.EnOptContext] object describing the problem
    setup and uses the callback interface to request objective, constraint, and
    gradient evaluations as needed by the underlying algorithm.

    **Lifecycle**

    1. Instantiation via `__init__`: Called with a backend configuration
        object.
    2. Setup via `init`: Called once per optimization workflow with the
        [`EnOptContext`][ropt.context.EnOptContext] and an
        [`OptimizerCallback`][ropt.core.OptimizerCallback].
    3. Validation via `validate_options`: Called to verify that the configured
        backend options are supported.
    4. Execution via `start`: Called with the initial variable vector to run
        the optimization algorithm.

    Subclasses must implement:

    - `__init__`: Stores backend configuration and performs lightweight setup.
    - `init`: Receives the optimization context and callback interface.
    - `start`: Runs the optimization algorithm.
    - `validate_options`: Verifies that backend-specific options are valid.

    Subclasses may optionally override:

    - `allow_nan`: Indicates whether the backend can continue when evaluations
                   produce `NaN` values.
    - `is_parallel`: Indicates whether the backend may evaluate multiple
                     candidate variable vectors concurrently.
    """

    @abstractmethod
    def __init__(self, backend_config: BackendConfig) -> None:
        """Create a new backend instance.

        Called during instantiation. Subclasses should store the configuration
        and perform any lightweight initialization. Validation and
        context-dependent setup should usually be deferred to `validate_options`
        and `init`.

        Args:
            backend_config: Configuration object specifying the backend method
                and any method-specific options.
        """

    @abstractmethod
    def init(
        self, context: EnOptContext, optimizer_callback: OptimizerCallback
    ) -> None:
        """Finalize initialization after the optimization context is known.

        Called once at the start of each optimization workflow, after all
        configuration is finalized. Use this method to store the optimization
        context, retain the callback interface, and perform any setup that
        depends on the full problem definition.

        Args:
            context: The full optimization context, containing all
                configuration and state for the current workflow.
            optimizer_callback: Callback interface used to request objective,
                constraint, and gradient evaluations from the `ropt` core.
        """

    @abstractmethod
    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Run the optimization algorithm from the provided initial values.

        Starts the backend's main optimization loop using the supplied initial
        variable vector. During execution, the implementation is expected to
        use the [`OptimizerCallback`][ropt.core.OptimizerCallback] provided in
        `init` to request any required objective, constraint, or gradient
        evaluations from the `ropt` core.

        Args:
            initial_values: A 1D array of shape `(n_variables,)` containing
                the starting point for the optimization.
        """

    @property
    def allow_nan(self) -> bool:
        """Indicate whether the backend can handle `NaN` evaluation results.

        Backends that can continue after receiving `NaN` objective or
        constraint values should override this property to return `True`.

        This is particularly relevant in ensemble-based optimization where
        evaluations might fail for all realizations. When `allow_nan` is `True`,
        setting [`realization_min_success`][ropt.config.RealizationsConfig] to
        zero allows the evaluation process to return `NaN` instead of raising an
        error, enabling the optimizer to potentially continue.

        Returns:
            `True` if the backend supports `NaN` evaluation results.
        """
        return False

    @property
    def is_parallel(self) -> bool:
        """Indicate whether the backend may issue parallel evaluations.

        Backends that evaluate multiple candidate variable vectors concurrently
        should override this property to return `True`.

        This information can be used by `ropt` and related components to manage
        resources or coordinate parallel execution appropriately.

        Returns:
            `True` if the backend may perform parallel evaluations.
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

        Warning: Method name with prefix
            The method string may be prefixed in the form `"backend/method"`.
            Implementations should account for this when parsing the method
            name.

        Warning: Handling the default method
            The method string may be set to `"default"`, in which case it
            should be mapped to the backend's actual default method.

        Raises:
            ValueError: If the provided options are invalid.
        """
