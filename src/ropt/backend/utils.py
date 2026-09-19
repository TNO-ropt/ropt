"""Utility functions for use by optimizer backend plugins.

This module provides helpers for resolving the requested reporting level,
checking what the optimizer writes below the Python level, unique output path
construction, and splitting linear constraints into the same normalized form as
the non-linear constraints delivered through the callback.

The problem itself, including its constraints, arrives as an
[`OptimizationProblem`][ropt.backend.OptimizationProblem]. Every array these
helpers accept or return is scaled, as is everything else a backend sees.
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

    This does not determine where the output goes: that is set by the
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
        base_dir:  Parent directory, or `None` for the working directory.
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
    [`linear_constraints`][ropt.backend.OptimizationProblem.linear_constraints].

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
