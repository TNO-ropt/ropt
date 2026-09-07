"""Utility functions for use by optimizer backend plugins.

This module provides helpers for constraint validation, reduction of the linear
constraints to the problem the optimizer solves, unique output path
construction, and splitting linear constraints into the same normalized form as
the non-linear constraints delivered through the callback.

Every array these helpers accept or return is scaled, as is everything else a
backend sees; see [`Backend`][ropt.backend.Backend].
"""

import io
import os
import sys
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray

from ropt._native_streams import flush_native_streams
from ropt._utils import split_constraints
from ropt.context import EnOptContext
from ropt.exceptions import UnsupportedError

_MESSAGES = {
    "bounds": "bound constraints",
    "linear:eq": "linear equality constraints",
    "linear:ineq": "linear inequality constraints",
    "nonlinear:eq": "non-linear equality constraints",
    "nonlinear:ineq": "non-linear inequality constraints",
}


def resolve_verbosity(*, verbose: bool | int | None) -> int | None:
    """Resolve how much the optimizer should report.

    Normalizes the `verbose` field of
    [`BackendConfig`][ropt.config.BackendConfig] into a single value, so that
    backends need not distinguish `True` from `1` themselves:

    | Result | Meaning                                                    |
    | ------ | ---------------------------------------------------------- |
    | `None` | Report at the optimizer's own default level.               |
    | `0`    | Do not report.                                             |
    | `n`    | Report at level `n`, clamped to what the optimizer offers. |

    This says nothing about where the output goes: that is decided by the
    `stdout` and `stderr` settings of
    [`OptimizerConfig`][ropt.config.OptimizerConfig].

    Args:
        verbose: The `verbose` field of the backend configuration.

    Returns:
        The reporting level, or `None` for the optimizer's own default.
    """
    if verbose is None or verbose is False:
        return 0
    return None if verbose is True else verbose


def collect_native_output(run: Callable[[], None]) -> str:
    """Collect the output a callable writes below the Python level.

    Runs `run` with `sys.stdout` and `sys.stderr` replaced, so that everything
    written through Python is diverted, and with file descriptors 1 and 2
    pointing at a temporary file. Whatever reaches that file was therefore
    written without going through Python.

    Use this to check a backend's
    [`bypasses_python_output`][ropt.backend.Backend.bypasses_python_output]
    declaration from its test suite: a non-empty result means the declaration
    must be `True` for the method that was run.

    Args:
        run: A callable that runs an optimization with the backend under test.

    Returns:
        Everything the callable wrote below the Python level.
    """
    with TemporaryDirectory() as directory:
        path = Path(directory) / "native-output.txt"
        sys.stdout.flush()
        sys.stderr.flush()
        flush_native_streams()
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        try:
            with path.open("w") as handle:
                os.dup2(handle.fileno(), 1)
                os.dup2(handle.fileno(), 2)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    run()
                flush_native_streams()
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
        return path.read_text()


def validate_supported_constraints(
    context: EnOptContext,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    """Raise if the context's constraints are incompatible with the chosen method.

    Checks bounds, linear, and non-linear constraints in `context` against the
    sets of method names that support or require each constraint type. Constraint
    types are identified by the keys `"bounds"`, `"linear:eq"`, `"linear:ineq"`,
    `"nonlinear:eq"`, and `"nonlinear:ineq"`.

    Raises `UnsupportedError` if a constraint present in `context` is not
    supported by the method, or if a constraint required by the method is absent
    from `context`.

    Args:
        context:               The optimization context to inspect.
        method:                The name of the optimization method being used.
        supported_constraints: Maps each constraint type to the methods that
                               support it.
        required_constraints:  Maps each constraint type to the methods that
                               require it.
    """
    _validate_bounds(context, method, supported_constraints, required_constraints)
    _validate_linear_constraints(
        context, method, supported_constraints, required_constraints
    )
    _validate_nonlinear_constraints(
        context, method, supported_constraints, required_constraints
    )


def _check_constraint(
    constraint_type: str,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
    *,
    have_constraint: bool,
) -> None:
    supported = {
        algo.lower() for algo in supported_constraints.get(constraint_type, set())
    }
    required = {
        algo.lower() for algo in required_constraints.get(constraint_type, set())
    }
    msg = _MESSAGES[constraint_type]
    if have_constraint and method.lower() not in supported:
        msg = f"Optimizer '{method}' does not support {msg}."
        raise UnsupportedError(msg)
    if not have_constraint and method.lower() in required:
        msg = f"Optimizer '{method}' requires {msg}."
        raise UnsupportedError(msg)


def _validate_bounds(
    context: EnOptContext,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    _check_constraint(
        "bounds",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool(
            np.isfinite(context.variables.lower_bounds).any()
            or np.isfinite(context.variables.upper_bounds).any(),
        ),
    )


def _validate_linear_constraints(
    context: EnOptContext,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    if context.linear_constraints is None:
        return

    equality = context._linear_equality  # ruff: ignore[private-member-access]
    assert equality is not None

    _check_constraint(
        "linear:ineq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool((~equality).any()),
    )

    _check_constraint(
        "linear:eq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool(equality.all()),
    )


def _validate_nonlinear_constraints(
    context: EnOptContext,
    method: str,
    supported_constraints: dict[str, set[str]],
    required_constraints: dict[str, set[str]],
) -> None:
    if context.nonlinear_constraints is None:
        return

    equality = context._nonlinear_equality  # ruff: ignore[private-member-access]
    assert equality is not None

    _check_constraint(
        "nonlinear:ineq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool((~equality).any()),
    )

    _check_constraint(
        "nonlinear:eq",
        method,
        supported_constraints,
        required_constraints,
        have_constraint=bool(equality.all()),
    )


def create_output_path(
    base_name: str,
    base_dir: Path | None = None,
    name: str | None = None,
    suffix: str | None = None,
) -> Path:
    """Construct a unique output path, appending an index if necessary.

    Builds a path from the provided components. If the resulting path already
    exists on disk, a three-digit counter suffix (for example `-001`, `-002`) is
    appended or incremented until a non-existing path is found.

    The path is assembled as:
    `<base_dir>/<base_name>[-<name>][-<index>][<suffix>]`

    `base_dir` is created (including parents) if it does not exist.

    Args:
        base_name: Base file or directory name.
        base_dir:  Parent directory. If `None`, the path is relative to the
                   current working directory.
        name:      Optional label appended to `base_name` with a `-` separator.
        suffix:    Optional file extension including the leading dot

    Returns:
        A `pathlib.Path` that does not currently exist on disk.
    """
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
    if name is not None:
        base_name += f"-{name}"
    output = base_dir / base_name if base_dir is not None else Path(base_name)
    if suffix is not None:
        output = output.with_suffix(suffix)
    while output.exists():
        fields = base_name.split("-")
        if fields[-1].strip().isdigit():
            index = int(fields[-1]) + 1
            base_name = "-".join(fields[:-1]) + f"-{index:03}"
        else:
            base_name = f"{base_name}-001"
        output = base_dir / base_name if base_dir is not None else Path(base_name)
        if suffix is not None:
            output = output.with_suffix(suffix)
    return output


def get_nonlinear_equalities(context: EnOptContext) -> NDArray[np.bool_] | None:
    """Return which nonlinear constraint values are equalities.

    One flag per value delivered through the
    [`OptimizerCallback`][ropt.core.OptimizerCallback]: a constraint with a
    finite lower and a finite upper bound delivers two values, an equality one,
    and a constraint with no finite bound none. The split follows the bounds as
    configured, so scaling cannot change it.

    The values themselves arrive through the callback, which is why this is all
    a backend needs to ask for; the linear counterpart is
    [`get_linear_constraints`][ropt.backend.utils.get_linear_constraints].

    Args:
        context: The [`EnOptContext`][ropt.context.EnOptContext] object to inspect.

    Returns:
        A flag per value, or `None` if there are no nonlinear constraints.
    """
    if context.nonlinear_constraints is None:
        return None
    equality = context._nonlinear_equality  # ruff: ignore[private-member-access]
    assert equality is not None
    constraint_index, _ = split_constraints(
        context.nonlinear_constraints.lower_bounds,
        context.nonlinear_constraints.upper_bounds,
        equality,
    )
    return equality[constraint_index]


def get_linear_constraints(
    context: EnOptContext, initial_values: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]
]:
    """Reduce the linear constraints to the problem the optimizer solves.

    Rows that cannot constrain the optimization are dropped: a row is kept only
    if it has a non-zero coefficient on a variable that `context.variables.mask`
    leaves free, and at least one finite bound. The columns of the fixed
    variables are removed, and the contribution those variables make at their
    `initial_values` is subtracted from the bounds.

    The four returned arrays are aligned, with one entry per surviving
    constraint. `equality` marks the constraints whose bounds coincide; it
    follows the bounds as configured, so scaling cannot change it.

    Args:
        context:        The [`EnOptContext`][ropt.context.EnOptContext] object
                        containing the variable mask and linear constraints.
        initial_values: The initial values to use.

    Returns:
        A tuple of `(coefficients, lower_bounds, upper_bounds, equality)`.
    """
    assert context.linear_constraints is not None
    equality = context._linear_equality  # ruff: ignore[private-member-access]
    assert equality is not None
    mask = context.variables.mask
    coefficients = context.linear_constraints.coefficients
    lower_bounds = context.linear_constraints.lower_bounds
    upper_bounds = context.linear_constraints.upper_bounds

    keep = np.any(coefficients[:, mask] != 0, axis=1) & (
        np.isfinite(lower_bounds) | np.isfinite(upper_bounds)
    )
    coefficients = coefficients[keep, :]
    offsets = np.matmul(coefficients[:, ~mask], initial_values[~mask])
    return (
        coefficients[:, mask],
        lower_bounds[keep] - offsets,
        upper_bounds[keep] - offsets,
        equality[keep],
    )


def split_linear_constraints(
    coefficients: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    equality: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Split linear constraints into values compared against zero.

    Produces the same form as the non-linear constraint values delivered
    through the [`OptimizerCallback`][ropt.core.OptimizerCallback], so that a
    backend that handles both together can concatenate them. Entry `k` evaluates
    to `coefficients[k] @ variables - offsets[k]`, which is non-negative when the
    constraint is satisfied. A constraint with a finite lower and a finite upper
    bound contributes two entries, and an equality one.

    The arguments are those returned by
    [`get_linear_constraints`][ropt.backend.utils.get_linear_constraints].

    Args:
        coefficients: The constraint coefficients.
        lower_bounds: The lower bounds.
        upper_bounds: The upper bounds.
        equality:     Which constraints have coinciding bounds.

    Returns:
        A tuple of `(coefficients, offsets, equality)`, with one entry per value.
    """
    constraint_index, use_lower_bound = split_constraints(
        lower_bounds, upper_bounds, equality
    )
    rows = coefficients[constraint_index, :]
    return (
        np.where(use_lower_bound[:, np.newaxis], rows, -rows),
        np.where(
            use_lower_bound,
            lower_bounds[constraint_index],
            -upper_bounds[constraint_index],
        ),
        equality[constraint_index],
    )
