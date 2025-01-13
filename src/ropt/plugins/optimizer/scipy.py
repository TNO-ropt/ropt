"""This module implements the SciPy optimization plugin."""

import copy
import os
import sys
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    TextIO,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    differential_evolution,
    minimize,
)

from ropt.config.enopt import EnOptConfig
from ropt.enums import ConstraintType, VariableType
from ropt.plugins.optimizer.utils import (
    create_output_path,
    filter_linear_constraints,
    validate_supported_constraints,
)

from .base import Optimizer, OptimizerCallback, OptimizerPlugin

_SUPPORTED_METHODS: Final[set[str]] = {
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

_OUTPUT_FILE: Final = "optimizer_output"

_ConstraintType = str | Callable[..., float] | Callable[..., NDArray[np.float64]]


class SciPyOptimizer(Optimizer):
    """Plugin class for optimization via SciPy.

    This class implements several optimizers provided by SciPy in the
    [`scipy.optimize`](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
    package:

    - Nelder-Mead
    - Powell
    - CG
    - BFGS
    - Newton-CG
    - L-BFGS-B
    - TNC
    - COBYLA
    - SLSQP
    - differential_evolution

    The optimizer to use is selected by setting the `method` field in the
    [`optimizer`][ropt.config.enopt.OptimizerConfig] field of
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] to the name of the algorithm.
    Most of these methods support the general options set in the
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. However, specific
    options that are normally passed as arguments in the SciPy functions can be
    provided via the `options` dictionary in the configuration object. Consult
    the
    [`scipy.optimize`](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
    manual for details on these options.

    Not all constraints are supported by all optimizers:

    - Bound constraints: Nelder-Mead, L-BFGS-B, SLSQP, TNC,
      differential_evolution
    - Linear constraints: SLSQP, differential_evolution
    - Nonlinear constraints: COBYLA (only inequality), SLSQP,
      differential_evolution

    Info:
        - The Nelder-Mead algorithm only supports bound constraints if SciPy
          version >= 1.7.
        - Some SciPy algorithms that require a Hessian or a Hessian-vector
          product are not supported. These include dogleg, trust-ncg,
          trust-exact, and trust-krylov.
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

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._optimizer_callback = optimizer_callback
        self._config = config
        _, _, self._method = self._config.optimizer.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "slsqp"
        if self._method not in _SUPPORTED_METHODS:
            msg = f"SciPy optimizer algorithm {self._method} is not supported"
            raise NotImplementedError(msg)
        validate_supported_constraints(
            self._config,
            self._method,
            self._supported_constraints,
            self._required_constraints,
        )
        self._bounds = self._initialize_bounds()

        self._constraints: (
            tuple[LinearConstraint, ...] | list[dict[str, _ConstraintType]]
        )

        if self._method == "differential_evolution":
            self._constraints = (
                self._initialize_linear_constraint_object()
                + self._initialize_nonlinear_constraint_object()
            )
        else:
            self._constraints = (
                self._initialize_linear_constraints()
                + self._initialize_nonlinear_constraints()
            )
        self._options = self._parse_options()
        self._parallel = (
            self._config.optimizer.parallel and self._method == "differential_evolution"
        )

        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._cached_gradient: NDArray[np.float64] | None = None
        self._stdout: TextIO

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._cached_variables = None
        self._cached_function = None
        self._cached_gradient = None

        variable_indices = self._config.variables.indices
        if variable_indices is not None:
            initial_values = np.take(initial_values, variable_indices, axis=-1)

        output_dir = self._config.optimizer.output_dir
        output_file: str | Path
        if output_dir is None:
            output_file = os.devnull
        else:
            output_file = create_output_path(_OUTPUT_FILE, output_dir, suffix=".txt")

        self._stdout = sys.stdout
        with (
            Path(output_file).open("a", encoding="utf-8") as output,
            redirect_stdout(
                output,
            ),
        ):
            if self._method == "differential_evolution":
                if self._parallel:
                    self._options["updating"] = "deferred"
                    self._options["workers"] = 1
                differential_evolution(
                    func=self._function,
                    x0=initial_values,
                    bounds=self._bounds,
                    constraints=self._constraints,
                    polish=False,
                    vectorized=self._parallel,
                    **self._options,
                )
            else:
                minimize(
                    fun=self._function,
                    x0=initial_values,
                    tol=self._config.optimizer.tolerance,
                    method=self._method,
                    bounds=self._bounds,
                    jac=(False if self._method in _NO_GRADIENT else self._gradient),
                    constraints=self._constraints,
                    options=self._options if self._options else None,
                )

    @property
    def allow_nan(self) -> bool:
        """Whether NaN is allowed.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return self._method == "differential_evolution"

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return self._parallel

    def _initialize_bounds(self) -> Bounds | None:
        if (
            np.isfinite(self._config.variables.lower_bounds).any()
            or np.isfinite(self._config.variables.upper_bounds).any()
        ):
            lower_bounds = self._config.variables.lower_bounds
            upper_bounds = self._config.variables.upper_bounds
            variable_indices = self._config.variables.indices
            if variable_indices is not None:
                lower_bounds = lower_bounds[variable_indices]
                upper_bounds = upper_bounds[variable_indices]
            return Bounds(lower_bounds, upper_bounds)
        return None

    def _initialize_linear_constraints(self) -> list[dict[str, _ConstraintType]]:
        constraints: list[dict[str, _ConstraintType]] = []

        linear_constraints_config = self._config.linear_constraints
        if linear_constraints_config is None:
            return constraints

        if self._config.variables.indices is not None:
            linear_constraints_config = filter_linear_constraints(
                linear_constraints_config, self._config.variables.indices
            )

        types = linear_constraints_config.types
        coefficients = linear_constraints_config.coefficients
        rhs_values = linear_constraints_config.rhs_values

        eq_idx = types == ConstraintType.EQ
        if np.any(eq_idx):
            assert coefficients is not None
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: (
                        np.matmul(coefficients[eq_idx, :], x) - rhs_values[eq_idx]
                    ),
                    "jac": lambda _: coefficients[eq_idx, :],
                },
            )

        ineq_constraints = []
        ineq_rhs_values = []
        ge_idx = types == ConstraintType.GE
        if np.any(ge_idx):
            assert coefficients is not None
            assert rhs_values is not None
            ineq_constraints.append(coefficients[ge_idx, :])
            ineq_rhs_values.append(rhs_values[ge_idx])
        le_idx = types == ConstraintType.LE
        if np.any(le_idx):
            assert coefficients is not None
            assert rhs_values is not None
            ineq_constraints.append(-coefficients[le_idx, :])
            ineq_rhs_values.append(-rhs_values[le_idx])
        if ineq_constraints:
            coefficients_matrix = np.vstack(tuple(ineq_constraints))
            rhs_vector = np.hstack(tuple(ineq_rhs_values))
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: (np.matmul(coefficients_matrix, x) - rhs_vector),
                    "jac": lambda _: coefficients_matrix,
                },
            )

        return constraints

    def _initialize_linear_constraint_object(self) -> tuple[LinearConstraint, ...]:
        linear_constraints_config = self._config.linear_constraints
        if linear_constraints_config is None:
            return ()

        if self._config.variables.indices is not None:
            linear_constraints_config = filter_linear_constraints(
                linear_constraints_config, self._config.variables.indices
            )

        lower_bounds = linear_constraints_config.rhs_values.copy()
        upper_bounds = linear_constraints_config.rhs_values.copy()

        ge_idx = linear_constraints_config.types == ConstraintType.GE
        le_idx = linear_constraints_config.types == ConstraintType.LE
        if np.any(le_idx):
            lower_bounds[le_idx] = -np.inf
        if np.any(ge_idx):
            upper_bounds[ge_idx] = np.inf

        return (
            LinearConstraint(
                linear_constraints_config.coefficients, lower_bounds, upper_bounds
            ),
        )

    def _initialize_nonlinear_constraints(self) -> list[dict[str, _ConstraintType]]:
        constraints: list[dict[str, _ConstraintType]] = []

        if self._config.nonlinear_constraints is None:
            return constraints

        for inx, constraint_type in enumerate(self._config.nonlinear_constraints.types):
            constr = "eq" if constraint_type == ConstraintType.EQ else "ineq"
            fun = partial(self._constraint_function, index=inx)
            if self._method == "cobyla":
                constraints.append({"type": constr, "fun": fun})
            else:
                jac = partial(self._constraint_gradient, index=inx)
                constraints.append({"type": constr, "fun": fun, "jac": jac})
        return constraints

    def _initialize_nonlinear_constraint_object(
        self,
    ) -> tuple[NonlinearConstraint, ...]:
        if self._config.nonlinear_constraints is None:
            return ()

        constraints_count = self._config.nonlinear_constraints.rhs_values.size

        def fun(variables: NDArray[np.float64]) -> NDArray[np.float64]:
            if variables.ndim == 1:
                result = np.empty(constraints_count, dtype=np.float64)
                for inx in range(constraints_count):
                    result[inx] = partial(self._constraint_function, index=inx)(
                        variables
                    )
            else:
                result = np.empty(
                    (constraints_count, variables.shape[1]), dtype=np.float64
                )
                for inx in range(constraints_count):
                    result[inx, :] = partial(self._constraint_function, index=inx)(
                        variables
                    )
            return result

        def jac(variables: NDArray[np.float64]) -> NDArray[np.float64]:
            result = np.empty((constraints_count, variables.size), dtype=np.float64)
            for inx in range(constraints_count):
                result[inx, :] = partial(self._constraint_gradient, index=inx)(
                    variables
                )
            return result

        return (NonlinearConstraint(fun=fun, jac=jac, lb=-np.inf, ub=0.0),)

    def _function(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        if variables.ndim > 1 and variables.size == 0:
            return np.array([])
        functions, _ = self._get_function_or_gradient(
            variables, get_function=True, get_gradient=False
        )
        assert functions is not None
        if variables.ndim > 1:
            return functions[:, 0]
        return np.array(functions[0])

    def _gradient(self, variables: NDArray[np.float64]) -> NDArray[np.float64]:
        _, gradients = self._get_function_or_gradient(
            variables, get_function=False, get_gradient=True
        )
        assert gradients is not None
        return gradients[0, :]

    def _constraint_function(
        self, variables: NDArray[np.float64], index: int
    ) -> NDArray[np.float64]:
        if variables.ndim > 1 and variables.size == 0:
            return np.array([])
        functions, _ = self._get_function_or_gradient(
            variables, get_function=True, get_gradient=False
        )
        assert functions is not None
        if variables.ndim > 1:
            return -functions[:, index + 1]
        return np.array(-functions[index + 1])

    def _constraint_gradient(
        self, variables: NDArray[np.float64], index: int
    ) -> NDArray[np.float64]:
        _, gradients = self._get_function_or_gradient(
            variables, get_function=False, get_gradient=True
        )
        assert gradients is not None
        return -gradients[index + 1, :]

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

        function = self._cached_function if get_function else None
        gradient = self._cached_gradient if get_gradient else None

        compute_functions = get_function and function is None
        compute_gradients = get_gradient and gradient is None

        if compute_functions or compute_gradients:
            self._cached_variables = variables.copy()
            compute_functions = compute_functions or self._config.optimizer.speculative
            compute_gradients = compute_gradients or self._config.optimizer.speculative
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
            and self._config.optimizer.split_evaluations
        ):
            with redirect_stdout(self._stdout):
                new_function, _ = self._optimizer_callback(
                    variables,
                    return_functions=True,
                    return_gradients=False,
                )
                _, new_gradient = self._optimizer_callback(
                    variables,
                    return_functions=False,
                    return_gradients=True,
                )
        else:
            with redirect_stdout(self._stdout):
                new_function, new_gradient = self._optimizer_callback(
                    variables,
                    return_functions=compute_functions,
                    return_gradients=compute_gradients,
                )
        if self.allow_nan:
            new_function = np.where(np.isnan(new_function), np.inf, new_function)
        return new_function, new_gradient

    def _parse_options(self) -> dict[str, Any]:
        if not isinstance(self._config.optimizer.options, dict):
            return {}
        options = copy.deepcopy(self._config.optimizer.options)
        # The maximum number of iterations is passed as an option to ropt.
        # Setting maxiter directly as an entry in the options dict will also
        # work, but iterations will override it.
        iterations = self._config.optimizer.max_iterations
        if iterations is not None:
            if self._method == "tnc":
                options["maxfun"] = iterations
            else:
                options["maxiter"] = iterations
        # We switch on display if there is an output folder.
        if self._config.optimizer.output_dir is not None:
            options["disp"] = True

        if (
            self._method == "differential_evolution"
            and self._config.variables.types is not None
            and "integrality" not in options
        ):
            options["integrality"] = (
                self._config.variables.types == VariableType.INTEGER
            )

        return options


class SciPyOptimizerPlugin(OptimizerPlugin):
    """Default filter transform plugin class."""

    def create(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> SciPyOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return SciPyOptimizer(config, optimizer_callback)

    def is_supported(self, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.base.Plugin][] abstract base class.

        # noqa
        """
        return method.lower() in (_SUPPORTED_METHODS | {"default"})
