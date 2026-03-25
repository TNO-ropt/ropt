"""Ensemble optimizer class."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ropt.enums import ExitCode
from ropt.exceptions import ComputeStepAborted
from ropt.plugins.manager import get_plugin
from ropt.results import FunctionResults, GradientResults

from ._callback import OptimizerCallbackResult

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from ropt.config import EnOptConfig
    from ropt.optimizer import Optimizer
    from ropt.results import Functions, Gradients, Results

    from ._evaluator import EnsembleEvaluator


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
    """Optimizer for ensemble-based optimizations.

    The [`EnsembleOptimizer`][ropt.core.EnsembleOptimizer] class provides the
    core functionality for running ensemble-based optimizations. Direct use of
    this class is generally discouraged. Instead, use the
    [`BasicOptimizer`][ropt.workflow.BasicOptimizer] class or build a workflow
    containing the optimization steps.
    """

    def __init__(
        self,
        enopt_config: EnOptConfig,
        ensemble_evaluator: EnsembleEvaluator,
        signal_evaluation: SignalEvaluationCallback | None = None,
    ) -> None:
        """Initialize the EnsembleOptimizer.

        This class orchestrates ensemble-based optimizations. It requires an
        optimization configuration, an evaluator, and a plugin manager to
        function.

        The `EnsembleOptimizer` needs the following to define a single
        optimization run:

        1.  An [`EnOptConfig`][ropt.config.EnOptConfig] object: This contains
            all configuration settings for the optimization.
        2.  An [`EnsembleEvaluator`][ropt.core.EnsembleEvaluator]
            object: This object is responsible for evaluating functions.

        Additionally, an optional callbacks can be provided that is invoked
        before and after each function evaluation.:
        [`SignalEvaluationCallback`][ropt.core.SignalEvaluationCallback]:

        The optimizer plugins are used by the ensemble optimizer to implement
        the actual optimization process. The `EnsembleOptimizer` class provides
        the callback function to these plugins needed (see
        [OptimizerCallback][ropt.core.OptimizerCallback])

        Args:
            enopt_config:       The ensemble optimization configuration.
            ensemble_evaluator: The evaluator for function evaluations.
            signal_evaluation:  Optional callback to signal evaluations.
        """
        self._enopt_config = enopt_config
        self._function_evaluator = ensemble_evaluator
        self._signal_evaluation = signal_evaluation

        # This stores the values of the fixed variable
        self._initial_variables: NDArray[np.float64]

        # For implementing max_functions:
        self._completed_functions = 0
        self._completed_batches = 0

        # Whether NaN values are allowed:
        self._allow_nan = False

        plugin = get_plugin("optimizer", method=self._enopt_config.optimizer.method)

        # Validate the optimizer options:
        plugin.validate_options(
            self._enopt_config.optimizer.method, self._enopt_config.optimizer.options
        )

        self._optimizer: Optimizer = plugin.create(
            self._enopt_config, self._optimizer_callback
        )
        self._allow_nan = self._optimizer.allow_nan

        # Optional redirection of standard output:
        self._redirector = _Redirector(self._enopt_config)

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
        return self._optimizer.is_parallel

    def start(self, variables: NDArray[np.float64]) -> ExitCode:
        """Start the optimization process.

        This method initiates the optimization process using the provided
        initial variables. The optimization will continue until a stopping
        criterion is met or an error occurs.

        Args:
            variables: The initial variables for the optimization.

        Returns:
            An [`ExitCode`][ropt.enums.ExitCode] indicating the reason for
            termination.
        """
        self._initial_variables = variables.copy()
        exit_code = ExitCode.OPTIMIZER_FINISHED
        try:
            with self._redirector.start():
                self._optimizer.start(variables)
        except ComputeStepAborted as exc:
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
                self._enopt_config.variables.mask,
            )

        self._completed_batches += 1

        return OptimizerCallbackResult(
            functions=functions,
            gradients=gradients,
            nonlinear_constraint_bounds=self._get_nonlinear_constraint_bounds(),
        )

    def _get_completed_variables(
        self, variables: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        mask = self._enopt_config.variables.mask
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
        max_functions = self._enopt_config.optimizer.max_functions
        if max_functions is not None and self._completed_functions >= max_functions:
            raise ComputeStepAborted(exit_code=ExitCode.MAX_FUNCTIONS_REACHED)
        max_batches = self._enopt_config.optimizer.max_batches
        if max_batches is not None and self._completed_batches >= max_batches:
            raise ComputeStepAborted(exit_code=ExitCode.MAX_BATCHES_REACHED)

    def _run_evaluations(
        self,
        variables: NDArray[np.float64],
        *,
        compute_functions: bool = False,
        compute_gradients: bool = False,
    ) -> tuple[Results, ...]:
        with self._redirector.suspend():
            assert compute_functions or compute_gradients
            if self._signal_evaluation:
                self._signal_evaluation()
            results = self._function_evaluator.calculate(
                variables,
                compute_functions=compute_functions,
                compute_gradients=compute_gradients,
            )

            # If the configuration allows for zero successful realizations, there
            # will always be results. However, they may all be equal to `np.nan`. If
            # the optimizer does not set the allow_nan flag, it cannot handle such a
            # case, and we need to check for it:
            assert self._enopt_config.realizations.realization_min_success is not None
            check_failures = (
                self._enopt_config.realizations.realization_min_success < 1
                and not self._allow_nan
            )
            exit_code: ExitCode | None = None
            for result in results:
                assert isinstance(result, FunctionResults | GradientResults)
                no_functions = (
                    isinstance(result, FunctionResults) and result.functions is None
                )
                no_gradient = (
                    isinstance(result, GradientResults) and result.gradients is None
                )
                all_failures = check_failures and np.all(
                    result.realizations.failed_realizations
                )
                if no_functions or no_gradient or all_failures:
                    exit_code = ExitCode.TOO_FEW_REALIZATIONS
                    break

            if self._signal_evaluation:
                self._signal_evaluation(results)

            if exit_code is not None:
                raise ComputeStepAborted(exit_code=exit_code)

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

    def _get_nonlinear_constraint_bounds(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        if self._enopt_config.nonlinear_constraints is None:
            return None
        lower_bounds = self._enopt_config.nonlinear_constraints.lower_bounds
        upper_bounds = self._enopt_config.nonlinear_constraints.upper_bounds
        for transform in self._enopt_config.nonlinear_constraint_transform_instances:
            lower_bounds, upper_bounds = transform.bounds_to_optimizer(
                lower_bounds, upper_bounds
            )
        return lower_bounds, upper_bounds


class _Redirector:
    def __init__(self, config: EnOptConfig) -> None:
        output_dir = config.optimizer.output_dir
        stdout = config.optimizer.stdout
        stderr = config.optimizer.stderr

        if stdout is not None:
            self._redirect = True
            if stderr is None:
                stderr = stdout
            if not stdout.is_absolute() and output_dir is not None:
                stdout = output_dir / stdout
            if not stderr.is_absolute() and output_dir is not None:
                stderr = output_dir / stderr
            sys.stdout.flush()
            sys.stderr.flush()
            self._old_stdout = os.dup(1)
            self._old_stderr = os.dup(2)
            self._new_stdout = os.open(stdout, os.O_WRONLY | os.O_CREAT)
            self._new_stderr = os.open(stderr, os.O_WRONLY | os.O_CREAT)
        else:
            self._redirect = False

    @contextmanager
    def start(self) -> Generator[None]:
        if self._redirect:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(self._new_stdout, 1)
                os.dup2(self._new_stderr, 2)
                yield
            finally:
                os.dup2(self._old_stdout, 1)
                os.dup2(self._old_stderr, 2)
                os.close(self._new_stdout)
                os.close(self._new_stderr)
        else:
            yield

    @contextmanager
    def suspend(self) -> Generator[None]:
        if self._redirect:
            try:
                os.fsync(self._new_stdout)
                os.fsync(self._new_stderr)
                os.dup2(self._old_stdout, 1)
                os.dup2(self._old_stderr, 2)
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(self._new_stdout, 1)
                os.dup2(self._new_stderr, 2)
        else:
            yield
