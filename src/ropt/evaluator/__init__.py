r"""Function Evaluations.

An optimizer will repeatedly request function and gradient evaluations. In each
iteration, it may require values for multiple variable vectors, and the
gradients may be calculated from various perturbed variable vectors. To address
this, the optimizer requires a function evaluator callback upon initialization,
conforming to the [Evaluator][ropt.evaluator.Evaluator] signature.

The callback accepts a matrix containing multiple variable vectors to evaluate
together with a [`EvaluatorContext`][ropt.evaluator.EvaluatorContext] object
providing information usable by the evaluator. It returns an
[`EvaluatorResult`][ropt.evaluator.EvaluatorResult] object with the calculated
values of the objective and constraint functions for all variable vectors and
realizations, and optionally some additional metadata.

Tip: Reusing Objective and Non-linear Constraint Values
    When defining multiple objectives, there might be an intention to reuse the
    same objective value multiple times. For example, a total objective could
    consist of the mean of the objectives for each realization, plus the
    standard deviation of the same values. This can be implemented by defining
    two objectives: the first calculated as the mean of the realizations, and
    the second using a function transform to compute the standard deviations.
    The optimizer is unaware that both objectives use the same set of
    realizations. To prevent redundant calculations, the evaluator should
    compute the results of the realizations once and return them for both
    objectives.

    Non-linear constraint values may potentially appear multiple times in the
    `constraint_results` matrix. For instance, to express the constraint $a <
    F_c \le b$, two constraints must be defined: $F_c \ge a$ and $F_c \le b$,
    sharing the same function value $F_c$ but differing in types and
    right-hand-sides (`ConstraintType.GE`/`ConstraintType.LE` and $a$/$b$). The
    run method should ensure that the calculated value of $F_c$ is the same in
    both cases, which is most efficiently achieved by evaluating $F_c$ only
    once.
"""

from ._concurrent import ConcurrentEvaluator, ConcurrentTask
from ._ensemble_evaluator import EnsembleEvaluator
from ._evaluator import Evaluator, EvaluatorContext, EvaluatorResult

__all__ = [
    "ConcurrentEvaluator",
    "ConcurrentTask",
    "EnsembleEvaluator",
    "Evaluator",
    "EvaluatorContext",
    "EvaluatorResult",
]
