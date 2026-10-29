"""Draw noise in a `START_EVALUATION` handler and pass it through event metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ropt import CallbackHandler, optimize
from ropt.enums import EnOptEventType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ropt import EvaluationFunctionContext
    from ropt.events import EnOptEvent

CONFIG: dict[str, object] = {
    "variables": {
        "variable_count": 1,
        "perturbation_magnitudes": 1e-6,
        "seed": 11,
    },
    "realizations": {"weights": [1.0, 1.0, 1.0]},
}


# --8<-- [start:objective]
def objective(
    variables: NDArray[np.float64], context: EvaluationFunctionContext
) -> float:
    """Evaluate one realization with the noise drawn for this batch.

    Args:
        variables: The variable vector to evaluate.
        context:   Identifies the realization and provides the batch noise.

    Returns:
        The objective value for this realization.
    """
    noise = context.metadata["noise"][context.realization]
    return float(np.sum((variables - 1.0) ** 2) + noise)


# --8<-- [end:objective]


def main() -> None:
    """Optimize with noise drawn once per evaluation batch."""
    # --8<-- [start:draw_noise]
    rng = np.random.default_rng(42)

    def draw_noise(event: EnOptEvent) -> None:
        realizations = event.context.realizations.weights.size
        event.metadata["noise"] = rng.normal(scale=1e-3, size=realizations)

    # --8<-- [end:draw_noise]

    # --8<-- [start:run]
    handler = CallbackHandler(
        event_types={EnOptEventType.START_EVALUATION}, callback=draw_noise
    )
    optimize(CONFIG, [0.0], objective, handlers=[handler])
    # --8<-- [end:run]


if __name__ == "__main__":
    main()
