"""Base model class for `EnOptConfig` fields."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EnOptBaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        str_min_length=1,
        str_strip_whitespace=True,
        validate_default=True,
    )
