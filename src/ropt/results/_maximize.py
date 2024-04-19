import copy
from functools import singledispatch

from ._function_results import FunctionResults
from ._gradient_results import GradientResults
from ._results import Results


@singledispatch
def convert_to_maximize(results: Results) -> Results:
    """Convert results to maximization results.

    The optimization process always assumes that the objective functions must be
    minimized. To perform a maximization, the negative of the objective
    functions can be passed instead. However, the result objects will then
    contain those negative values and all objective function values derived from
    it. Since this may be inconvenient for the application, this convenience
    function can be used make a copy of a result where all objective values are
    multiplied by -1, which would correspond to a maximization.

    Args:
        results: The result to convert

    Returns:
        A copy of the result, modified to reflect maximization.
    """
    msg = f"`results` type not supported: {type(results).__name__}"
    raise TypeError(msg)


@convert_to_maximize.register
def _(results: FunctionResults) -> FunctionResults:
    results = copy.deepcopy(results)
    results.evaluations.objectives.setflags(write=True)
    results.evaluations.objectives *= -1.0
    results.evaluations.objectives.setflags(write=False)
    if results.evaluations.scaled_objectives is not None:
        results.evaluations.scaled_objectives.setflags(write=True)
        results.evaluations.scaled_objectives *= -1.0
        results.evaluations.scaled_objectives.setflags(write=False)
    if results.functions is not None:
        results.functions.weighted_objective *= -1.0
        results.functions.objectives.setflags(write=True)
        results.functions.objectives *= -1.0
        results.functions.objectives.setflags(write=False)
        if results.functions.scaled_objectives is not None:
            results.functions.scaled_objectives.setflags(write=True)
            results.functions.scaled_objectives *= -1.0
            results.functions.scaled_objectives.setflags(write=False)
    return results


@convert_to_maximize.register
def _(results: GradientResults) -> GradientResults:
    results = copy.deepcopy(results)
    results.evaluations.perturbed_objectives.setflags(write=True)
    results.evaluations.perturbed_objectives *= -1.0
    results.evaluations.perturbed_objectives.setflags(write=False)
    if results.gradients is not None:
        results.gradients.weighted_objective *= -1.0
        results.gradients.objectives.setflags(write=True)
        results.gradients.objectives *= -1.0
        results.gradients.objectives.setflags(write=False)
    return results
