"""The optimization plan configuration class."""

from __future__ import annotations

from typing import Tuple

from pydantic import RootModel

from ._step_config import StepConfig


class PlanConfig(RootModel[Tuple[StepConfig, ...]]):
    """Configuration for a single optimization plan.

    A plan is represented as a sequence of
    [`StepConfig`][ropt.config.plan.StepConfig] objects. Each of these can be
    any optimization step, including another plan, enabling the representation
    of a directed acyclic graph of steps.
    """

    root: Tuple[StepConfig, ...]
