from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from ropt.enums import ResultAxis

from ._utils import _get_axis_names

if TYPE_CHECKING:
    from ropt.config.enopt import EnOptConfig


TypeResultField = TypeVar("TypeResultField", bound="ResultField")


@dataclass(slots=True)
class ResultField:
    """Base class for `Results` fields."""

    @classmethod
    def get_axes(cls, name: str) -> tuple[ResultAxis, ...]:
        """Return the axes of a field in the given field class or object.

        When used with the class or an instance of that class for one of the
        fields of a `Results` object, retrieve the axes of the stored `numpy`
        array from the metadata and return them.

        Args:
            name: The name of the sub-field in the instance or class.

        Raises:
            ValueError: Raised if an unknown field name is passed.

        Returns:
            A tuple listing the names of the axes.
        """
        metadata = next(
            (item.metadata for item in fields(cls) if item.name == name), None
        )
        if metadata is None:
            msg = f"Unknown field name: {name}"
            raise ValueError(msg)
        return metadata.get("__axes__", ())

    def to_dict(
        self, config: EnOptConfig, name: str, axis: ResultAxis | None = None
    ) -> dict[str | int, Any]:
        """Convert a field of ResultsField to a dictionary.

        The keys of the output correspond to the indices of the axis of the
        field given by `axis`. The values are a slice of the field value at each
        index. If the given `config` object contains the names of the objects
        stored along the axis, these are used as keys in the output dictionary
        rather than the numerical indices.

        If `axis` is `None` (the default), a default is chosen using the following logic:

        1. If the field has a `ResultAxis.OBJECTIVE` axis, is used.
        2. If the field has a `ResultAxis.NONLINEAR_CONSTRAINT` axis, it is used.
        3. Otherwise, use the last axis.

        Args:
            config: The configuration object.
            name:   Name of the field to export.
            axis:   Axis to use as the keys.

        Returns:
            The field converted to a dictionary.
        """
        data = getattr(self, name)
        axes = self.get_axes(name)
        if axis is None:
            if ResultAxis.OBJECTIVE in axes:
                axis = ResultAxis.OBJECTIVE
            elif ResultAxis.NONLINEAR_CONSTRAINT in axes:
                axis = ResultAxis.NONLINEAR_CONSTRAINT
            else:
                axis = axes[-1]
        elif axis not in axes:
            msg = f"invalid key: {axis}"
            raise ValueError(msg)
        data = np.swapaxes(data, axes.index(axis), 0)
        names = _get_axis_names(config, axis)
        return (
            {idx: data[idx] for idx in range(data.shape[0])}
            if names is None
            else {name: data[idx, ...] for idx, name in enumerate(names)}
        )
