"""Abstract base class for gradient calculator implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ropt.config import GradientConfig
    from ropt.context import EnOptContext
    from ropt.plugins import MethodSpec


@dataclass(frozen=True, slots=True)
class GradientData:
    """The evaluated points used to calculate gradients.

    Variables are in the optimizer's scaled domain and contain only the
    variables selected by the mask passed to
    [`GradientCalculator.init`][ropt.gradient.GradientCalculator.init]. The
    functions are the objectives followed by the nonlinear constraints, as
    evaluated. A failed evaluation is `NaN` in every function.

    Attributes:
        variables:           Shape `(n_variables,)`, the point at which
                             gradients are calculated.
        functions:           Shape `(n_realizations, n_functions)`, the
                             functions at that point.
        perturbed_variables: Shape `(n_realizations, n_perturbations,
                             n_variables)`, the points returned by
                             [`perturbed_variables`][ropt.gradient.GradientCalculator.perturbed_variables].
        perturbed_functions: Shape `(n_realizations, n_perturbations,
                             n_functions)`, the functions at those points.
    """

    variables: NDArray[np.float64]
    functions: NDArray[np.float64]
    perturbed_variables: NDArray[np.float64]
    perturbed_functions: NDArray[np.float64]


class GradientCalculator(ABC):
    """Abstract base class for gradient calculator implementations.

    A gradient calculator chooses the points at which the functions are
    evaluated, and calculates the gradient of each function in each realization
    from the results. `ropt` evaluates the points, and aggregates the
    gradients over the realizations with the realization aggregators.

    See [Writing a Plugin](../advanced/writing_plugins.md#gradient-calculators).
    """

    methods: ClassVar[MethodSpec]
    """The gradient calculation methods this class provides.

    Either a set of names, which the registry matches case-insensitively, or a
    predicate for classes that cannot enumerate them. Include `"default"` in the
    set if this class has one. See [`MethodSpec`][ropt.plugins.MethodSpec].
    """

    @abstractmethod
    def __init__(self, gradient_config: GradientConfig) -> None:
        """Create a new gradient calculator instance.

        Store the configuration and validate the method options. Setup that
        depends on the optimization belongs in `init`. Raise a `ValueError` for
        invalid options.

        Warning: Method name with prefix
            `gradient_config.method` may be prefixed in the form
            `"plugin/method"`. Implementations should account for this when
            parsing the method name.

        Warning: Handling the default method
            `gradient_config.method` may be set to `"default"`, in which case
            it should be mapped to the actual default method.

        Args:
            gradient_config: Configuration specifying the method and its options.
        """

    @abstractmethod
    def init(self, context: EnOptContext, *, mask: NDArray[np.bool_]) -> None:
        """Prepare the calculator before the optimization starts.

        Called once, before any points are requested. `mask` selects the
        variables for which the calculator calculates gradients; use it instead
        of `context.variables.mask`.

        Args:
            context: The optimization context.
            mask:    Shape `(n_variables,)`, the variables to calculate
                     gradients for.
        """

    @property
    def merge_realizations(self) -> bool:
        """Indicate whether the aggregated gradient is calculated directly.

        If `True`, [`calculate`][ropt.gradient.GradientCalculator.calculate]
        returns one gradient per function, calculated from all realizations,
        and no per-realization gradients are reported. The default
        implementation returns `False`.

        Returns:
            `True` if the aggregated gradient is calculated directly.
        """
        return False

    @abstractmethod
    def perturbed_variables(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the points at which the functions are evaluated.

        The variables are in the optimizer's scaled domain, and contain all
        variables, not only those selected by the mask. Variables outside the
        mask must keep their value in the returned points.

        Args:
            variables: Shape `(n_variables,)`, the point at which gradients are
                       calculated.

        Returns:
            Shape `(n_realizations, n_perturbations, n_variables)`, where the
            calculator chooses `n_perturbations`.
        """

    @abstractmethod
    def failed_realizations(self, data: GradientData) -> NDArray[np.bool_]:
        """Identify the realizations for which no gradient can be calculated.

        Report only the failures in the evaluations of the perturbed points.
        `ropt` also marks a realization as failed if the evaluation at the
        unperturbed point failed.

        Args:
            data: The evaluated points.

        Returns:
            Shape `(n_realizations,)`, `True` for a failed realization.
        """

    @abstractmethod
    def calculate(
        self, data: GradientData, weights: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_] | None]:
        """Calculate the gradient of each function in each realization.

        The weights of each function are normalized to sum to one, and are zero
        for failed realizations.

        Args:
            data:    The evaluated points.
            weights: Shape `(n_realizations, n_functions)`, the realization
                     weights of each function.

        Returns:
            The gradients, of shape `(n_realizations, n_functions,
            n_variables)`, and a mask of shape `(n_realizations, n_functions)`
            that is `True` where a gradient was calculated. Gradients where the
            mask is `False` are not used. If
            [`merge_realizations`][ropt.gradient.GradientCalculator.merge_realizations]
            is `True`, the gradients have shape `(n_functions, n_variables)`
            and the mask is `None`.
        """
