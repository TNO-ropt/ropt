"""Abstract base class for function estimator implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import FunctionEstimatorConfig
    from ropt.plugins import MethodSpec


class FunctionEstimator(ABC):
    """Abstract base class for function estimator implementations.

    Subclasses must implement four methods:

    1. `__init__` — store configuration; defer heavy work to `init`.
    2. `init` — called once before the run; validate settings and pre-compute
       state here.
    3. `calculate_function` — aggregate per-realization function values.
    4. `calculate_gradient` — aggregate per-realization gradients.

    See [Function Estimators](../optimizer_setup/function_estimators.md) for examples
    and further guidance.
    """

    methods: ClassVar[MethodSpec]
    """The estimator methods this class provides.

    Either a set of names, which the registry matches case-insensitively, or a
    predicate for classes that cannot enumerate them. Include `"default"` in the
    set if this class has one. See [`MethodSpec`][ropt.plugins.MethodSpec].
    """

    @abstractmethod
    def __init__(self, estimator_config: FunctionEstimatorConfig) -> None:
        """Create a new function estimator instance.

        Store the configuration; keep initialization lightweight.
        Run-dependent setup belongs in `init`.

        Args:
            estimator_config: The estimator configuration.
        """

    @abstractmethod
    def init(self, *, merge_realizations: bool) -> None:
        """Finalize initialization before the optimization starts.

        Called once after configuration is finalized. Use for validation
        (for example compatibility with `merge_realizations`) and precomputation.

        Args:
            merge_realizations: Whether gradients arrive merged across
                                realizations, as described in
                                `calculate_gradient`.
        """

    @abstractmethod
    def calculate_function(
        self, functions: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Aggregate function values across realizations.

        The values arrive as the evaluator returned them. Scales are applied to
        the aggregate this method produces, so an implementation does not need
        to account for them.

        Args:
            functions: Shape `(n_realizations,)` — per-realization values.
            weights:   Shape `(n_realizations,)` — realization weights.

        Returns:
            Aggregated value (scalar or 1-D array).
        """

    @abstractmethod
    def calculate_gradient(
        self,
        functions: NDArray[np.float64],
        gradient: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Aggregate gradients across realizations.

        When `merge_realizations` is `False` (default), `gradient` has shape
        `(n_variables, n_realizations)` and must be combined using `weights`.
        When `True`, a single pre-merged gradient of shape `(n_variables,)` is
        passed instead — suitable only for estimators that aggregate by a
        simple weighted combination (for example the mean). Estimators that need each
        realization's own gradient (for example standard deviation, via the chain
        rule) are incompatible with merging and should raise `ValueError` from
        `init`.

        Args:
            functions: Shape `(n_realizations,)` — needed for chain-rule
                estimators (for example standard deviation).
            gradient:  Shape `(n_variables, n_realizations)` or
                `(n_variables,)` if merged.
            weights:   Shape `(n_realizations,)` — realization weights.

        Returns:
            1-D array of shape `(n_variables,)`.
        """
