"""This module implements the SciPy optimization plugin."""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    differential_evolution,
    minimize,
)

from ropt.config.options import OptionsSchemaModel
from ropt.enums import VariableType
from ropt.plugins.optimizer.utils import validate_supported_constraints

from .base import Optimizer, OptimizerCallback, OptimizerPlugin
from .utils import NormalizedConstraints, get_masked_linear_constraints

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig

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


_ConstraintType = str | Callable[..., float] | Callable[..., NDArray[np.float64]]


class SciPyOptimizer(Optimizer):
    """SciPy optimization backend for ropt.

    This class provides an interface to several optimization algorithms from
    SciPy's
    [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
    module, enabling their use within `ropt`.

    To select an optimizer, set the `method` field within the
    [`optimizer`][ropt.config.enopt.OptimizerConfig] section of the
    [`EnOptConfig`][ropt.config.enopt.EnOptConfig] configuration object to the
    desired algorithm's name. Most methods support the general options defined
    in the [`EnOptConfig`][ropt.config.enopt.EnOptConfig] object. For
    algorithm-specific options, use the `options` dictionary within the
    [`optimizer`][ropt.config.enopt.OptimizerConfig] section.

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
        self._constraints = self._initialize_constraints()
        self._options = self._parse_options()
        self._parallel = (
            self._config.optimizer.parallel and self._method == "differential_evolution"
        )

        self._cached_variables: NDArray[np.float64] | None = None
        self._cached_function: NDArray[np.float64] | None = None
        self._cached_gradient: NDArray[np.float64] | None = None

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._cached_variables = None
        self._cached_function = None
        self._cached_gradient = None

        if self._config.variables.mask is not None:
            initial_values = initial_values[..., self._config.variables.mask]

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
            if self._config.variables.mask is not None:
                lower_bounds = lower_bounds[self._config.variables.mask]
                upper_bounds = upper_bounds[self._config.variables.mask]
            return Bounds(lower_bounds, upper_bounds)
        return None

    def _initialize_constraints(
        self,
    ) -> list[dict[str, _ConstraintType] | NonlinearConstraint | LinearConstraint]:
        self._normalized_constraints = None

        lin_coef, lin_lower, lin_upper = None, None, None
        if self._config.linear_constraints is not None:
            lin_coef, lin_lower, lin_upper = get_masked_linear_constraints(self._config)

        if self._method == "differential_evolution":
            return self._initialize_constraints_object(lin_coef, lin_lower, lin_upper)

        lower_bounds = []
        upper_bounds = []
        if self._config.nonlinear_constraints is not None:
            lower_bounds.append(self._config.nonlinear_constraints.lower_bounds)
            upper_bounds.append(self._config.nonlinear_constraints.upper_bounds)
        if lin_lower is not None and lin_upper is not None:
            lower_bounds.append(lin_lower)
            upper_bounds.append(lin_upper)
        if lower_bounds:
            self._normalized_constraints = NormalizedConstraints(
                np.concatenate(lower_bounds), np.concatenate(upper_bounds)
            )
            return self._initialize_constraints_dict(lin_coef)

        return []

    def _fun(
        self,
        variables: NDArray[np.float64],
        index: int,
        lin_coef: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        assert self._normalized_constraints is not None
        if self._normalized_constraints.constraints is None:
            constraints = []
            if self._config.nonlinear_constraints is not None:
                constraints.append(self._constraint_functions(variables).transpose())
            if lin_coef is not None:
                constraints.append(np.matmul(lin_coef, variables))
            self._normalized_constraints.set_constraints(
                np.concatenate(constraints, axis=0)
            )
        assert self._normalized_constraints.constraints is not None
        return self._normalized_constraints.constraints[index, :]

    def _jac(
        self,
        variables: NDArray[np.float64],
        index: int,
        lin_coef: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        assert self._normalized_constraints is not None
        if self._normalized_constraints.gradients is None:
            gradients = []
            if self._config.nonlinear_constraints is not None:
                gradients.append(self._constraint_gradients(variables))
            if lin_coef is not None:
                gradients.append(lin_coef)
            self._normalized_constraints.set_gradients(
                np.concatenate(gradients, axis=0)
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
            fun = partial(self._fun, index=index, lin_coef=lin_coef)
            if self._method == "cobyla":
                return {"type": type_, "fun": fun}
            jac = partial(self._jac, index=index, lin_coef=lin_coef)
            return {"type": type_, "fun": fun, "jac": jac}

        return [
            _constraint_entry("eq" if is_eq else "ineq", inx)
            for inx, is_eq in enumerate(self._normalized_constraints.is_eq)
        ]

    def _initialize_constraints_object(
        self,
        lin_coef: NDArray[np.float64] | None,
        lin_lower: NDArray[np.float64] | None,
        lin_upper: NDArray[np.float64] | None,
    ) -> list[LinearConstraint | NonlinearConstraint]:
        constraints = []
        if self._config.linear_constraints is not None:
            constraints.append(LinearConstraint(lin_coef, lin_lower, lin_upper))

        if self._config.nonlinear_constraints is not None:
            lower_bounds = self._config.nonlinear_constraints.lower_bounds
            upper_bounds = self._config.nonlinear_constraints.upper_bounds

            def _fun(variables: NDArray[np.float64]) -> NDArray[np.float64]:
                functions = self._constraint_functions(variables)
                return functions.transpose()

            def _jac(variables: NDArray[np.float64]) -> NDArray[np.float64]:
                return self._constraint_gradients(variables)

            constraints.append(
                NonlinearConstraint(
                    fun=_fun, jac=_jac, lb=lower_bounds, ub=upper_bounds
                ),
            )
        return constraints

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
    """The SciPY optimizer plugin class."""

    @classmethod
    def create(
        cls, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> SciPyOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return SciPyOptimizer(config, optimizer_callback)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return method.lower() in (_SUPPORTED_METHODS | {"default"})

    @classmethod
    def validate_options(
        cls, method: str, options: dict[str, Any] | list[str] | None
    ) -> None:
        """Validate the options of a given method.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        if options is not None:
            OptionsSchemaModel.model_validate(_OPTIONS_SCHEMA).get_options_model(
                method
            ).model_validate(options)


_OPTIONS_SCHEMA: dict[str, Any] = {
    "methods": {
        "Nelder-Mead": {
            "options": {
                "disp": bool,
                "maxiter": int,
                "maxfev": int,
                "xatol": float,
                "fatol": float,
                "adaptive": bool,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html",
        },
        "Powell": {
            "options": {
                "disp": bool,
                "maxiter": int,
                "maxfev": int,
                "xtol": float,
                "ftol": float,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html",
        },
        "CG": {
            "options": {
                "disp": bool,
                "maxiter": int,
                "gtol": float,
                "norm": float,
                "eps": float,
                "finite_diff_rel_step": float,
                "c1": float,
                "c2": float,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html",
        },
        "BFGS": {
            "options": {
                "disp": bool,
                "maxiter": int,
                "gtol": float,
                "norm": float,
                "eps": float,
                "finite_diff_rel_step": float,
                "xrtol": float,
                "c1": float,
                "c2": float,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html",
        },
        "Newton-CG": {
            "options": {
                "disp": bool,
                "maxiter": int,
                "xtol": float,
                "eps": float,
                "c1": float,
                "c2": float,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html",
        },
        "L-BFGS-B": {
            "options": {
                "disp": int,
                "maxiter": int,
                "maxcor": int,
                "ftol": float,
                "gtol": float,
                "eps": float,
                "maxfun": int,
                "iprint": int,
                "maxls": int,
                "finite_diff_rel_step": float,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html",
        },
        "TNC": {
            "options": {
                "disp": bool,
                "maxfun": int,
                "eps": float,
                "scale": list[float],
                "offset": float,
                "maxCGit": int,
                "eta": float,
                "stepmx": float,
                "accuracy": float,
                "minfev": float,
                "ftol": float,
                "xtol": float,
                "gtol": float,
                "rescale": float,
                "finite_diff_rel_step": float,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html",
        },
        "COBYLA": {
            "options": {
                "disp": bool,
                "maxiter": int,
                "rhobeg": float,
                "tol": float,
                "catol": float,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html",
        },
        "SLSQP": {
            "options": {
                "disp": bool,
                "maxiter": int,
                "ftol": float,
                "eps": float,
                "finite_diff_rel_step": float,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html",
        },
        "differential_evolution": {
            "options": {
                "disp": bool,
                "maxiter": int,
                "strategy": Literal[
                    "best1bin"
                    "best1exp"
                    "rand1bin"
                    "rand1exp"
                    "rand2bin"
                    "rand2exp"
                    "randtobest1bin"
                    "randtobest1exp"
                    "currenttobest1bin"
                    "currenttobest1exp"
                    "best2exp"
                    "best2bin"
                ],
                "popsize": int,
                "tol": float,
                "mutation": float | tuple[float, float],
                "recombination": float,
                "seed": int,
                "polish": bool,
                "init": Literal["latinhypercube", "sobol", "haltonrandom"],
                "atol": float,
                "updating": Literal["immediate", "deferred"],
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html",
        },
    },
}


if __name__ == "__main__":
    from ropt.config.options import gen_options_table

    with Path("scipy.md").open("w", encoding="utf-8") as fp:
        fp.write(gen_options_table(_OPTIONS_SCHEMA))
