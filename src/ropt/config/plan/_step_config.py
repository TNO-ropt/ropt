"""The optimization step configuration classes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StepConfig(BaseModel):
    """The core class for an optimization step.

    This class acts as a generic container for the configuration of optimization
    steps. The only field processed by this class is the `backend` field, used
    by the plugin system to locate the code responsible for parsing the
    configuration and generating the optimization step. The class is configured
    to allow extra fields that carry the specific step configuration passed to
    that code.

    Attributes:
        backend: Name of the backend containing the step code.
    """

    backend: str = "default"

    model_config = ConfigDict(
        extra="allow",
        validate_default=True,
    )
