"""This module implements the SciPy optimization plugin."""

from __future__ import annotations

import copy
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    differential_evolution,
    minimize,
)

from ropt.backend._base import Backend
from ropt.backend.utils import (
    NormalizedConstraints,
    get_masked_linear_constraints,
    validate_supported_constraints,
)
from ropt.enums import VariableType

if TYPE_CHECKING:
    from ropt.config import EnOptConfig
    from ropt.core import OptimizerCallback

SUPPORTED_SCIPY_METHODS: Final[set[str]] = {
    name.lower()
    for name in (
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "differential_evolution",
    )
}
DEFAULT_SCIPY_METHOD: Final = "slsqp"


# Categorize the methods by the types of constraint they support or require.

_CONSTRAINT_REQUIRES_BOUNDS: Final = {
    "differential_evolution",
}
_CONSTRAINT_SUPPORT_BOUNDS: Final = {
    name.lower()
    for name in [
        "Nelder-Mead",
        "Powell",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "differential_evolution",
    ]
}
_CONSTRAINT_SUPPORT_LINEAR_EQ: Final = {
    name.lower() for name in ["SLSQP", "differential_evolution"]
}
_CONSTRAINT_SUPPORT_LINEAR_INEQ: Final = {
    name.lower() for name in ["COBYLA", "SLSQP", "differential_evolution"]
}
_CONSTRAINT_SUPPORT_NONLINEAR_EQ: Final = {
    name.lower() for name in ["SLSQP", "differential_evolution"]
}
_CONSTRAINT_SUPPORT_NONLINEAR_INEQ: Final = {
    name.lower() for name in ["COBYLA", "SLSQP", "differential_evolution"]
}

# These methods do not use a gradient:
_NO_GRADIENT: Final = {
    name.lower()
    for name in ["Nelder-Mead", "Powell", "COBYLA", "differential_evolution"]
}


_ConstraintType = str | Callable[..., float] | Callable[..., NDArray[np.float64]]


class SciPyBackend(Backend):
    """SciPy optimization backend for ropt.

    This class provides an interface to several optimization algorithms from
    SciPy's
    [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
    module, enabling their use within `ropt`.

    To select an optimizer, set the `method` field within the
    [`optimizer`][ropt.config.BackendConfig] section of the
    [`EnOptConfig`][ropt.config.EnOptConfig] configuration object to the desired
    algorithm's name. Most methods support the general options defined in the
    [`EnOptConfig`][ropt.config.EnOptConfig] object. For algorithm-specific
    options, use the `options` dictionary within the
    [`optimizer`][ropt.config.BackendConfig] section.

    The table below lists the included methods together with the method-specific
    options that are supported. Click on the method name to consult the
    corresponding
    [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
    documentation:

    --8<-- "scipy.md"
    """

    _supported_constraints: ClassVar[dict[str, set[str]]] = {
        "bounds": _CONSTRAINT_SUPPORT_BOUNDS,
        "linear:eq": _CONSTRAINT_SUPPORT_LINEAR_EQ,
        "linear:ineq": _CONSTRAINT_SUPPORT_LINEAR_INEQ,
        "nonlinear:eq": _CONSTRAINT_SUPPORT_NONLINEAR_EQ,
        "nonlinear:ineq": _CONSTRAINT_SUPPORT_NONLINEAR_INEQ,
    }
    _required_constraints: ClassVar[dict[str, set[str]]] = {
        "bounds": _CONSTRAINT_REQUIRES_BOUNDS,
    }

    def __init__(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize the optimizer implemented by the SciPy plugin.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        self._optimizer_callback = optimizer_callback
        self._config = config
        _, _, self._method = self._config.backend.method.lower().rpartition("/")
        if self._method == "default":
            self._method = DEFAULT_SCIPY_METHOD
        if self._method not in SUPPORTED_SCIPY_METHODS:
            msg = f"SciPy optimizer algorithm {self._method} is not supported"
            raise NotImplementedError(msg)
        validate_supported_constraints(
            self._config,
            self._method,
            self._supported_constraints,
            self._required_constraints,
        )
        self._options = self._parse_options()
        self._parallel = (
            self._config.backend.parallel and self._method == "differential_evolution"
        )

        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._cached_gradient: NDArray[np.float64] | None = None

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        self._cached_variables = None
        self._cached_function = None
        self._cached_gradient = None

        self._bounds = self._initialize_bounds()
        self._constraints = self._initialize_constraints(initial_values)

        if self._method == "differential_evolution":
            if self._parallel:
                self._options["updating"] = "deferred"
                self._options["workers"] = 1
            assert self._bounds is not None
            differential_evolution(
                func=self._function,  # type: ignore[arg-type]
                x0=initial_values[self._config.variables.mask],
                bounds=self._bounds,
                constraints=self._constraints,  # type: ignore[arg-type]
                polish=False,
                vectorized=self._parallel,
                **self._options,
            )
        else:
            minimize(  # type: ignore[call-overload,misc]
                fun=self._function,
                x0=initial_values[self._config.variables.mask],
                tol=self._config.backend.tolerance,
                method=self._method,
                bounds=self._bounds,
                jac=(False if self._method in _NO_GRADIENT else self._gradient),
                constraints=self._constraints,
                options=self._options or None,
            )

    @property
    def allow_nan(self) -> bool:
        """Whether NaN is allowed.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        return self._method == "differential_evolution"

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.backend.Backend][] abstract base class.

        # noqa
        """
        return self._parallel

    def _initialize_bounds(self) -> Bounds | None:
        if (
            np.isfinite(self._config.variables.lower_bounds).any()
            or np.isfinite(self._config.variables.upper_bounds).any()
        ):
            lower_bounds = self._config.variables.lower_bounds[
                self._config.variables.mask
            ]
            upper_bounds = self._config.variables.upper_bounds[
                self._config.variables.mask
            ]
            return Bounds(lower_bounds, upper_bounds)
        return None

    def _get_constraint_bounds(
        self, nonlinear_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        bounds = []
        if nonlinear_bounds is not None:
            bounds.append(nonlinear_bounds)
        if (
            self._method != "differential_evolution"
            and self._linear_constraint_bounds is not None
        ):
            bounds.append(self._linear_constraint_bounds)

        if bounds:
            lower_bounds, upper_bounds = zip(*bounds, strict=True)
            return np.concatenate(lower_bounds), np.concatenate(upper_bounds)
        return None

    def _initialize_constraints(
        self, initial_values: NDArray[np.float64]
    ) -> (
        list[dict[str, _ConstraintType]] | list[NonlinearConstraint | LinearConstraint]
    ):
        self._normalized_constraints = None

        lin_coef, lin_lower, lin_upper = None, None, None
        self._linear_constraint_bounds: (
            tuple[NDArray[np.float64], NDArray[np.float64]] | None
        ) = None
        if self._config.linear_constraints is not None:
            lin_coef, lin_lower, lin_upper = get_masked_linear_constraints(
                self._config, initial_values
            )
            self._linear_constraint_bounds = (lin_lower, lin_upper)
        nonlinear_bounds = (
            None
            if self._config.nonlinear_constraints is None
            else (
                self._config.nonlinear_constraints.lower_bounds,
                self._config.nonlinear_constraints.upper_bounds,
            )
        )
        if (bounds := self._get_constraint_bounds(nonlinear_bounds)) is not None:
            self._normalized_constraints = NormalizedConstraints()
            self._normalized_constraints.set_bounds(*bounds)
        if self._method == "differential_evolution":
            return self._initialize_constraints_object(lin_coef, lin_lower, lin_upper)

        return self._initialize_constraints_dict(lin_coef)

    def _fun_dict(
        self,
        variables: NDArray[np.float64],
        index: int | None,
        lin_coef: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        assert self._normalized_constraints is not None
        functions = self._constraint_functions(variables).transpose()
        if self._normalized_constraints.constraints is None:
            constraints = []
            if self._config.nonlinear_constraints is not None:
                constraints.append(functions)
            if lin_coef is not None:
                constraints.append(np.matmul(lin_coef, variables))
            self._normalized_constraints.set_constraints(
                np.concatenate(constraints, axis=0)
            )
        assert self._normalized_constraints.constraints is not None
        return self._normalized_constraints.constraints[index, :]

    def _jac_dict(
        self,
        variables: NDArray[np.float64],
        index: int | None,
        lin_coef: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        assert self._normalized_constraints is not None
        gradients = self._constraint_gradients(variables)
        if self._normalized_constraints.gradients is None:
            constraints = []
            if self._config.nonlinear_constraints is not None:
                constraints.append(gradients)
            if lin_coef is not None:
                constraints.append(lin_coef)
            self._normalized_constraints.set_gradients(
                np.concatenate(constraints, axis=0)
            )
        assert self._normalized_constraints.gradients is not None
        return self._normalized_constraints.gradients[index, :]

    def _initialize_constraints_dict(
        self,
        lin_coef: NDArray[np.float64] | None,
    ) -> list[dict[str, _ConstraintType]]:
        if self._normalized_constraints is None:
            return []

        def _constraint_entry(type_: str, index: int) -> dict[str, _ConstraintType]:
            fun = partial(self._fun_dict, index=index, lin_coef=lin_coef)
            if self._method == "cobyla":
                return {"type": type_, "fun": fun}
            jac = partial(self._jac_dict, index=index, lin_coef=lin_coef)
            return {"type": type_, "fun": fun, "jac": jac}

        return [
            _constraint_entry("eq" if is_eq else "ineq", inx)
            for inx, is_eq in enumerate(self._normalized_constraints.is_eq)
        ]

    def _fun_object(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._normalized_constraints is not None
        self._normalized_constraints.set_constraints(
            self._constraint_functions(variables).transpose()
        )
        assert self._normalized_constraints.constraints is not None
        return self._normalized_constraints.constraints

    def _jac_object(
        self,
        variables: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert self._normalized_constraints is not None
        self._normalized_constraints.set_gradients(
            self._constraint_gradients(variables)
        )
        assert self._normalized_constraints.gradients is not None
        return self._normalized_constraints.gradients

    def _initialize_constraints_object(
        self,
        lin_coef: NDArray[np.float64] | None,
        lin_lower: NDArray[np.float64] | None,
        lin_upper: NDArray[np.float64] | None,
    ) -> list[LinearConstraint | NonlinearConstraint]:
        constraints: list[LinearConstraint | NonlinearConstraint] = []
        if self._config.linear_constraints is not None:
            assert lin_coef is not None
            assert lin_lower is not None
            assert lin_upper is not None
            constraints.append(LinearConstraint(lin_coef, lin_lower, lin_upper))
        if self._normalized_constraints is not None:
            ub = [
                0.0 if is_eq else np.inf for is_eq in self._normalized_constraints.is_eq
            ]
            constraints.append(
                NonlinearConstraint(
                    fun=self._fun_object, jac=self._jac_object, lb=0.0, ub=ub
                ),
            )
        return constraints

    def _function(self, variables: NDArray[np.float64], /) -> NDArray[np.float64]:
        if variables.ndim > 1 and variables.size == 0:
            return np.array([])
        functions, _ = self._get_function_or_gradient(
            variables, get_function=True, get_gradient=False
        )
        assert functions is not None
        if variables.ndim > 1:
            return functions[:, 0]
        return np.array(functions[0])

    def _gradient(self, variables: NDArray[np.float64], /) -> NDArray[np.float64]:
        _, gradients = self._get_function_or_gradient(
            variables, get_function=False, get_gradient=True
        )
        assert gradients is not None
        return gradients[0, :]

    def _constraint_functions(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if variables.ndim > 1 and variables.size == 0:
            return np.array([])
        functions, _ = self._get_function_or_gradient(
            variables, get_function=True, get_gradient=False
        )
        assert functions is not None
        if variables.ndim > 1:
            return functions[:, 1:]
        return np.array(functions[1:])

    def _constraint_gradients(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        _, gradients = self._get_function_or_gradient(
            variables, get_function=False, get_gradient=True
        )
        assert gradients is not None
        return gradients[1:, :]

    def _get_function_or_gradient(
        self, variables: NDArray[np.float64], *, get_function: bool, get_gradient: bool
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        if self._parallel and variables.ndim > 1:
            variables = variables.T

        if self._method in _NO_GRADIENT:
            get_gradient = False

        if (
            self._cached_variables is None
            or variables.shape != self._cached_variables.shape
            or not np.allclose(variables, self._cached_variables)
        ):
            self._cached_variables = None
            self._cached_function = None
            self._cached_gradient = None
            if self._normalized_constraints is not None:
                self._normalized_constraints.reset()

        function = self._cached_function if get_function else None
        gradient = self._cached_gradient if get_gradient else None

        compute_functions = get_function and function is None
        compute_gradients = get_gradient and gradient is None

        if compute_functions or compute_gradients:
            self._cached_variables = variables.copy()
            speculative = self._config.gradient.evaluation_policy == "speculative"
            compute_functions = compute_functions or speculative
            compute_gradients = compute_gradients or speculative
            new_function, new_gradient = self._compute_functions_and_gradients(
                variables,
                compute_functions=compute_functions,
                compute_gradients=compute_gradients,
            )
            if compute_functions:
                assert new_function is not None
                self._cached_function = new_function.copy()
                if get_function:
                    function = new_function
            if compute_gradients:
                assert new_gradient is not None
                self._cached_gradient = new_gradient.copy()
                if get_gradient:
                    gradient = new_gradient

        return function, gradient

    def _compute_functions_and_gradients(
        self,
        variables: NDArray[np.float64],
        *,
        compute_functions: bool,
        compute_gradients: bool,
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        new_function = None
        new_gradient = None
        if (
            compute_functions
            and compute_gradients
            and self._config.gradient.evaluation_policy == "separate"
        ):
            callback_result = self._optimizer_callback(
                variables,
                return_functions=True,
                return_gradients=False,
            )
            new_function = callback_result.functions
            callback_result = self._optimizer_callback(
                variables,
                return_functions=False,
                return_gradients=True,
            )
            new_gradient = callback_result.gradients
        else:
            callback_result = self._optimizer_callback(
                variables,
                return_functions=compute_functions,
                return_gradients=compute_gradients,
            )
            new_function = callback_result.functions
            new_gradient = callback_result.gradients

        # The optimizer callback may change non-linear constraint bounds:
        if (
            self._normalized_constraints is not None
            and callback_result.nonlinear_constraint_bounds is not None
        ):
            bounds = self._get_constraint_bounds(
                callback_result.nonlinear_constraint_bounds
            )
            assert bounds is not None
            self._normalized_constraints.set_bounds(*bounds)

        if self.allow_nan and new_function is not None:
            new_function = np.where(np.isnan(new_function), np.inf, new_function)
        return new_function, new_gradient

    def _parse_options(self) -> dict[str, Any]:
        options = (
            copy.deepcopy(self._config.backend.options)
            if isinstance(self._config.backend.options, dict)
            else {}
        )
        # The maximum number of iterations is passed as an option to ropt.
        # Setting maxiter directly as an entry in the options dict will also
        # work, but iterations will override it.
        iterations = self._config.backend.max_iterations
        if iterations is not None:
            if self._method == "tnc":
                options["maxfun"] = iterations
            else:
                options["maxiter"] = iterations
        # We switch on display if there is an output folder.
        if self._config.backend.output_dir is not None:
            options["disp"] = True

        if self._method == "differential_evolution" and "integrality" not in options:
            options["integrality"] = (
                self._config.variables.types == VariableType.INTEGER
            )

        return options
