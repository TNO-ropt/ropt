"""SciPy optimizer plugin implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ropt.backend.scipy import (
    DEFAULT_SCIPY_METHOD,
    SUPPORTED_SCIPY_METHODS,
    SciPyBackend,
)
from ropt.config.options import OptionsSchemaModel

from ._base import BackendPlugin

if TYPE_CHECKING:
    from ropt.config import BackendConfig


class SciPyBackendPlugin(BackendPlugin):
    """The SciPy backend plugin class."""

    @classmethod
    def create(cls, backend_config: BackendConfig) -> SciPyBackend:
        """Initialize the backend plugin.

        See the [ropt.plugins.backend.BackendPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return SciPyBackend(backend_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.backend.BackendPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC201
        return method.lower() in (SUPPORTED_SCIPY_METHODS | {"default"})

    @classmethod
    def validate_options(
        cls, method: str, options: dict[str, Any] | list[str] | None
    ) -> None:
        """Validate the options of a given method.

        See the [ropt.plugins.backend.BackendPlugin][] abstract base class.

        # noqa
        """  # noqa: DOC501
        if options is not None:
            if not isinstance(options, dict):
                msg = "SciPy backend options must be a dictionary"
                raise ValueError(msg)
            *_, method = method.rpartition("/")
            OptionsSchemaModel.model_validate(SCIPY_OPTIONS_SCHEMA).get_options_model(
                DEFAULT_SCIPY_METHOD if method == "default" else method
            ).model_validate(options)


SCIPY_OPTIONS_SCHEMA: dict[str, Any] = {
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
        "COBYQA": {
            "options": {
                "disp": bool,
                "maxfev": int,
                "maxiter": int,
                "f_target": float,
                "feasibility_tol": float,
                "initial_tr_radius": float,
                "final_tr_radius": float,
                "scale": bool,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyqa.html",
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
        "trust-constr": {
            "options": {
                "gtol": float,
                "xtol": float,
                "barrier_tol": float,
                "sparse_jacobian": bool,
                "initial_tr_radius": float,
                "initial_constr_penalty": float,
                "initial_barrier_parameter": float,
                "initial_barrier_tolerance": float,
                "factorization_method": Literal[
                    "NormalEquation",
                    "AugmentedSystem",
                    "QRFactorization",
                    "SVDFactorization",
                ],
                "finite_diff_rel_step": float,
                "maxiter": int,
                "verbose": int,
                "disp": bool,
            },
            "url": "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html",
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
                "rng": int,
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

    Path("scipy.md").write_text(
        gen_options_table(SCIPY_OPTIONS_SCHEMA), encoding="utf-8"
    )
