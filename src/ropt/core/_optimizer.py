"""Ensemble optimizer class."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ropt._logging import get_logger
from ropt._native_streams import flush_native_streams
from ropt.enums import ExitCode
from ropt.exceptions import (
    ExecutorStopped,
    OptimizerStop,
    TooFewRealizations,
    WorkflowError,
)
from ropt.results import FunctionResults, GradientResults

from ._callback import OptimizerCallbackResult

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from typing import TextIO

    from numpy.typing import NDArray

    from ropt.context import EnOptContext
    from ropt.results import Functions, Gradients, Results

    from ._evaluator import EnsembleEvaluator


_logger = get_logger(__name__)


# Output capture rewires process-global state, so at most one run may hold it.
_capture_lock = threading.Lock()
_capture_active = False


def _claim_capture() -> None:
    global _capture_active  # ruff: ignore[global-statement]
    with _capture_lock:
        if _capture_active:
            msg = (
                "Optimizer output is already being captured in this process. "
                "Only one optimization at a time can capture output; remove "
                "`stdout` and `stderr` from the optimizer settings of the "
                "concurrent runs, or run them in separate processes."
            )
            raise WorkflowError(msg)
        _capture_active = True


def _release_capture() -> None:
    global _capture_active  # ruff: ignore[global-statement]
    with _capture_lock:
        _capture_active = False


def _resolve_output_path(path: Path | None, output_dir: Path | None) -> Path | None:
    if path is None or path.is_absolute() or output_dir is None:
        return path
    return output_dir / path


class SignalEvaluationCallback(Protocol):
    """Protocol for a callback to signal the start and end of an evaluation.

    This callback is invoked before and after each evaluation, allowing for
    custom handling or tracking of evaluation events.
    """

    def __call__(self, results: tuple[Results, ...] | None = None, /) -> None:
        """Callback protocol for signaling the start and end of evaluations.

        This callback is invoked by the ensemble optimizer before and after
        each evaluation. Before the evaluation starts, the callback is called
        with `results` set to `None`. After the evaluation completes, the
        callback is called again, this time with `results` containing the
        output of the evaluation.

        Args:
            results: The results produced by the evaluation, or `None` if the
                     evaluation has not yet started.
        """


class EnsembleOptimizer:
    """Backend for ensemble-based optimizations.

    The [`EnsembleOptimizer`][ropt.core.EnsembleOptimizer] class provides the
    core functionality for running ensemble-based optimizations. Direct use of
    this class is generally discouraged. Instead, use the high-level
    [`optimize`][ropt.simple.optimize] API or build a custom workflow
    containing the optimization steps.
    """

    def __init__(
        self,
        context: EnOptContext,
        ensemble_evaluator: EnsembleEvaluator,
        signal_evaluation: SignalEvaluationCallback | None = None,
    ) -> None:
        """Initialize the EnsembleOptimizer.

        This class orchestrates ensemble-based optimizations. It requires an
        optimization context object and an evaluator to function.

        The `EnsembleOptimizer` needs the following to define a single
        optimization run:

        1.  An [`EnOptContext`][ropt.context.EnOptContext] object: This contains
            all necessary information for the optimization.
        2.  An [`EnsembleEvaluator`][ropt.core.EnsembleEvaluator]
            object: This object is responsible for evaluating functions.

        Additionally, an optional
        [`callback`][ropt.core.SignalEvaluationCallback] can be provided that is
        invoked before and after each function evaluation.

        Args:
            context:            The ensemble optimization context.
            ensemble_evaluator: The evaluator for function evaluations.
            signal_evaluation:  Optional callback to signal evaluations.
        """
        self._context = context
        self._function_evaluator = ensemble_evaluator
        self._signal_evaluation = signal_evaluation

        # This stores the values of the fixed variable
        self._initial_variables: NDArray[np.float64]

        # For implementing max_functions:
        self._completed_functions = 0
        self._completed_batches = 0

        self._backend = self._context.backend
        self._backend.init(self._context, self._optimizer_callback)

        # Optional capture of the optimizer's output:
        self._capture = _OutputCapture(
            self._context,
            bypasses_python_output=self._backend.bypasses_python_output,
        )

    @property
    def is_parallel(self) -> bool:
        """Determine if the optimization supports parallel evaluations.

        The underlying optimization algorithm may request function evaluations
        via a callback. Parallel optimization, in this context, means that the
        algorithm may request multiple function evaluations in a single
        callback.

        Returns:
            `True` if the optimization supports parallel evaluations, `False`
            otherwise.
        """
        return self._backend.is_parallel

    def start(self, variables: NDArray[np.float64]) -> ExitCode:
        """Start the optimization process.

        This method initiates the optimization process using the provided
        initial variables. The optimization will continue until a stopping
        criterion is met or an error occurs.

        Args:
            variables: The initial variables for the optimization.

        Returns:
            An [`ExitCode`][ropt.enums.ExitCode] describing the reason for termination.
        """
        self._initial_variables = variables.copy()
        exit_code = ExitCode.OPTIMIZER_FINISHED
        try:
            with self._capture.capture():
                self._backend.start(variables)
        except TooFewRealizations:
            exit_code = ExitCode.TOO_FEW_REALIZATIONS
        except ExecutorStopped:
            exit_code = ExitCode.EXECUTOR_STOPPED
        except OptimizerStop as exc:
            exit_code = exc.exit_code
        return exit_code

    def _optimizer_callback(
        self,
        variables: NDArray[np.float64],
        *,
        return_functions: bool,
        return_gradients: bool,
    ) -> OptimizerCallbackResult:
        assert return_functions or return_gradients

        if return_functions and return_gradients:
            _logger.debug("Optimizer callback: requesting functions and gradients")
        elif return_functions:
            _logger.debug("Optimizer callback: requesting functions")
        else:
            _logger.debug("Optimizer callback: requesting gradients")

        self._check_stopping_criteria()

        variables = self._get_completed_variables(variables)

        results = self._run_evaluations(
            variables,
            compute_functions=return_functions,
            compute_gradients=return_gradients,
        )

        functions = None
        if return_functions:
            # Functions might be parallelized hence we need potentially to
            # process a list of function results:
            functions_list = [
                self._functions_from_results(item.functions)
                for item in results
                if isinstance(item, FunctionResults)
            ]
            self._completed_functions += len(functions_list)
            functions = (
                np.vstack(functions_list) if variables.ndim > 1 else functions_list[0]
            )

        gradients = None
        if return_gradients:
            # Gradients cannot be parallelized, there is at most one gradient:
            gradients = self._gradients_from_results(
                next(
                    item.gradients
                    for item in results
                    if isinstance(item, GradientResults)
                ),
                self._context.variables.mask,
            )

        self._completed_batches += 1

        return OptimizerCallbackResult(
            functions=functions,
            gradients=gradients,
            nonlinear_constraint_bounds=self._context.get_nonlinear_constraint_bounds(),
        )

    def _get_completed_variables(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        mask = self._context.variables.mask
        if variables.ndim > 1:
            tmp_variables = np.repeat(
                self._initial_variables[np.newaxis, :], variables.shape[0], axis=0
            )
            tmp_variables[:, mask] = variables
        else:
            tmp_variables = self._initial_variables.copy()
            tmp_variables[mask] = variables
        return tmp_variables

    def _check_stopping_criteria(self) -> None:
        max_functions = self._context.optimizer.max_functions
        if max_functions is not None and self._completed_functions >= max_functions:
            _logger.info(
                "Stopping: Maximum number of function evaluations reached (%d)",
                max_functions,
            )
            raise OptimizerStop(ExitCode.MAX_FUNCTIONS_REACHED)
        max_batches = self._context.optimizer.max_batches
        if max_batches is not None and self._completed_batches >= max_batches:
            _logger.info(
                "Stopping: Maximum number of evaluation batches reached (%d)",
                max_batches,
            )
            raise OptimizerStop(ExitCode.MAX_BATCHES_REACHED)

    def _run_evaluations(
        self,
        variables: NDArray[np.float64],
        *,
        compute_functions: bool = False,
        compute_gradients: bool = False,
    ) -> tuple[Results, ...]:
        with self._capture.release():
            assert compute_functions or compute_gradients
            if self._signal_evaluation:
                self._signal_evaluation()
            results = self._function_evaluator.calculate(
                variables,
                compute_functions=compute_functions,
                compute_gradients=compute_gradients,
            )

            too_few = False
            for result in results:
                assert isinstance(result, FunctionResults | GradientResults)
                if (
                    isinstance(result, FunctionResults) and result.functions is None
                ) or (isinstance(result, GradientResults) and result.gradients is None):
                    too_few = True
                    break

            if self._signal_evaluation:
                self._signal_evaluation(results)

            if too_few:
                raise TooFewRealizations

        return results

    @staticmethod
    def _functions_from_results(functions: Functions | None) -> NDArray[np.float64]:
        assert functions is not None
        return (
            np.array(functions.target_objective, ndmin=1)
            if functions.constraints is None
            else np.append(functions.target_objective, functions.constraints)
        )

    @staticmethod
    def _gradients_from_results(
        gradients: Gradients | None, mask: NDArray[np.bool_] | None
    ) -> NDArray[np.float64]:
        assert gradients is not None
        target_objective_gradient = (
            gradients.target_objective.copy()
            if mask is None
            else gradients.target_objective[mask]
        )
        constraint_gradients = (
            None
            if gradients.constraints is None
            else (
                gradients.constraints.copy()
                if mask is None
                else gradients.constraints[:, mask]
            )
        )
        return (
            np.expand_dims(target_objective_gradient, axis=0)
            if constraint_gradients is None
            else np.vstack((target_objective_gradient, constraint_gradients))
        )


class _OutputCapture:
    def __init__(self, context: EnOptContext, *, bypasses_python_output: bool) -> None:
        output_dir = context.optimizer.output_dir
        stdout = _resolve_output_path(context.optimizer.stdout, output_dir)
        stderr = _resolve_output_path(context.optimizer.stderr, output_dir)
        # Configuring only `stdout` sends both streams to the same file.
        if stdout is not None and stderr is None:
            stderr = stdout

        self._stdout_path = stdout
        self._stderr_path = stderr
        self._redirect_descriptors = bypasses_python_output
        self._enabled = stdout is not None or stderr is not None

        self._stdout_file: TextIO | None = None
        self._stderr_file: TextIO | None = None
        self._saved_stdout: TextIO | None = None
        self._saved_stderr: TextIO | None = None
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._installed = False

    @contextmanager
    def capture(self) -> Generator[None]:
        if not self._enabled:
            yield
            return
        _claim_capture()
        try:
            with self._open_files():
                self._install()
                try:
                    yield
                finally:
                    self._uninstall()
        finally:
            _release_capture()

    @contextmanager
    def release(self) -> Generator[None]:
        if not self._installed:
            yield
            return
        self._uninstall()
        try:
            yield
        finally:
            self._install()

    @contextmanager
    def _open_files(self) -> Generator[None]:
        opened: dict[Path, TextIO] = {}
        try:
            for path in (self._stdout_path, self._stderr_path):
                if path is not None and path not in opened:
                    opened[path] = path.open("w", buffering=1)
            self._stdout_file = (
                None if self._stdout_path is None else opened[self._stdout_path]
            )
            self._stderr_file = (
                None if self._stderr_path is None else opened[self._stderr_path]
            )
            yield
        finally:
            self._stdout_file = None
            self._stderr_file = None
            for handle in opened.values():
                handle.close()

    def _install(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        if self._redirect_descriptors:
            flush_native_streams()

        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        if self._stdout_file is not None:
            sys.stdout = self._stdout_file
        if self._stderr_file is not None:
            sys.stderr = self._stderr_file

        if self._redirect_descriptors:
            if self._stdout_file is not None:
                self._saved_stdout_fd = os.dup(1)
                os.dup2(self._stdout_file.fileno(), 1)
            if self._stderr_file is not None:
                self._saved_stderr_fd = os.dup(2)
                os.dup2(self._stderr_file.fileno(), 2)

        self._installed = True

    def _uninstall(self) -> None:
        for handle in (self._stdout_file, self._stderr_file):
            if handle is not None:
                handle.flush()
        if self._redirect_descriptors:
            flush_native_streams()

        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
            os.close(self._saved_stdout_fd)
            self._saved_stdout_fd = None
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)
            self._saved_stderr_fd = None

        if self._saved_stdout is not None:
            sys.stdout = self._saved_stdout
        if self._saved_stderr is not None:
            sys.stderr = self._saved_stderr

        self._installed = False
