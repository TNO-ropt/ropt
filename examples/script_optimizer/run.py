"""Script to run the rosenbrock optimization."""

# ruff: noqa: ERA001

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from numpy.random import default_rng

from ropt.apps.script_optimizer import ScriptOptimizer
from ropt.enums import EventType
from ropt.plan import Event
from ropt.results import FunctionResults

# Number of realizations:
REALIZATIONS = 3


ENOPT_CONFIG: Dict[str, Any] = {
    "variables": {
        "names": [("x", 1), ("y", 1)],
        "initial_values": [0.75, 1.25],
    },
    "optimizer": {
        "output_dir": "optimizer_output",
        "max_functions": 20,
        "speculative": True,
    },
    "objective_functions": {
        "names": ["rosenbrock"],
    },
    "realizations": {
        "names": [idx + 1 for idx in range(REALIZATIONS)],
        "weights": 1.0,
    },
    "gradient": {
        "perturbation_magnitudes": 1e-6,
    },
}

tasks = {
    "rosenbrock": """python ${work_dir}/../rosenbrock.py \
        --vars variables.json \
        --realization ${realization} \
        --coefficients ${work_dir}/../coefficients.json \
        --out rosenbrock""",
}

# Make data files for the problem:
rng = default_rng(seed=5)
a = rng.normal(loc=1.0, scale=0.1, size=REALIZATIONS)
b = rng.normal(loc=100.0, scale=10.0, size=REALIZATIONS)
coefficients = {
    realization: (a[idx], b[idx])
    for idx, realization in enumerate(ENOPT_CONFIG["realizations"]["names"])
}
with Path("coefficients.json").open("w", encoding="utf-8") as file_obj:
    json.dump(coefficients, file_obj, sort_keys=True, indent=4)


def report(event: Event) -> None:
    """Report results of an evaluation.

    Args:
        event: event data
    """
    assert event.results is not None
    for item in event.results:
        if isinstance(item, FunctionResults) and item.functions is not None:
            print(f"result: {item.result_id}")
            print(f"  variables: {item.evaluations.variables}")
            print(f"  objective: {item.functions.weighted_objective}\n")


# Running locally:
provider = None

# # Example of running on a Slurm cluster:
# from parsl.providers import SlurmProvider
# provider = SlurmProvider(
#     max_blocks=16, exclusive=False, partition="defq", walltime="08:00:00"
# )


def main() -> None:
    """Run the example and check the result."""
    optimal_result = (
        ScriptOptimizer(
            config={
                "work_dir": "work",
            },
            plan={
                "variables": {
                    "enopt_config": ENOPT_CONFIG,
                    "optimal": None,
                },
                "steps": [
                    {
                        "run": "optimizer",
                        "with": {"config": "$enopt_config", "tag": "opt"},
                    },
                ],
                "results": [
                    {"init": "tracker", "with": {"var": "optimal", "tags": "opt"}},
                ],
            },
            tasks=tasks,
        )
        .add_observer(EventType.FINISHED_EVALUATION, report)
        .run(provider)["optimal"]
    )
    if optimal_result is not None and optimal_result.functions is not None:
        print(f"BEST RESULT: {optimal_result.result_id}")
        print(f"  variables: {optimal_result.evaluations.variables}")
        print(f"  objective: {optimal_result.functions.weighted_objective}\n")
    assert optimal_result is not None
    assert optimal_result.functions is not None
    assert np.allclose(optimal_result.functions.weighted_objective, 0, atol=0.05)
    assert np.allclose(optimal_result.evaluations.variables, 1, atol=0.005)


if __name__ == "__main__":
    main()
